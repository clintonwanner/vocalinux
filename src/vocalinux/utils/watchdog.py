"""systemd watchdog pings driven by the hotkey monitor.

The AppImage/launcher service can set C{WatchdogSec=}. This module sends
C{WATCHDOG=1} to C{$NOTIFY_SOCKET} at most every five seconds while the
keyboard monitor is alive, so a wedged hotkey listener is detected and
restarted by systemd instead of leaving the user with a dead tray icon.

Liveness rules:

* The ping interval is capped at five seconds and derived from
  C{WATCHDOG_USEC} (systemd recommends half the watchdog timeout).
* Nothing is sent unless C{NOTIFY_SOCKET} and C{WATCHDOG_USEC} are valid and
  C{WATCHDOG_PID}, when present, matches this process.
* C{note_monitor_beat()} is called from the evdev monitor loop and pings
  when due. If a monitor has ever run, pings stop once its last beat is
  older than C{MONITOR_STALE_SECONDS}; if no monitor has run at all (no
  keyboard, no evdev backend, external activation), a background thread
  keeps pinging so a service with C{TimeoutStartSec} is not restarted
  before the listener can exist.
* A hotkey callback stuck for more than C{CALLBACK_STALL_SECONDS} stops
  pings, unless it is inside an explicit C{long_operation()} block. Cold
  model loads and transcription make a started session legitimately slow,
  so the tray wraps those engine calls in that block; the detector still
  catches a wedged gesture/callback worker that never returns.

Abstract sockets (C{@name}) are supported as well as filesystem paths.
"""

import faulthandler
import logging
import os
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

#: Upper bound on the gap between pings, as systemd expects a healthy feed.
PING_INTERVAL_SECONDS = 5.0

#: A callback running longer than this stops the watchdog feed.
CALLBACK_STALL_SECONDS = 30.0

#: A monitor whose last beat is older than this is considered wedged.
MONITOR_STALE_SECONDS = 30.0


def _parse_notify_socket(raw: str) -> Optional[str]:
    """Translate NOTIFY_SOCKET into an address usable by AF_UNIX datagrams."""
    if not raw:
        return None
    if raw.startswith("@") and len(raw) > 1:
        # Linux abstract namespace: the leading NUL is part of the address.
        return "\0" + raw[1:]
    if raw.startswith("/"):
        return raw
    return None


def _watchdog_interval_usec(raw: Optional[str]) -> Optional[float]:
    """Watchdog timeout in seconds, or None when disabled or malformed."""
    if not raw:
        return None
    try:
        usec = int(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid WATCHDOG_USEC=%r; watchdog pings disabled", raw)
        return None
    if usec <= 0:
        return None
    return usec / 1_000_000.0


def _watchdog_pid_matches(raw: Optional[str]) -> bool:
    """False when WATCHDOG_PID names a different process than this one."""
    if not raw:
        return True
    try:
        return int(raw.strip()) == os.getpid()
    except (TypeError, ValueError):
        logger.warning("Invalid WATCHDOG_PID=%r; ignoring it", raw)
        return True


class WatchdogPinger:
    """Send WATCHDOG=1 while the hotkey path is healthy.

    Args:
        env: Environment mapping. Defaults to C{os.environ}.
        clock: Monotonic clock. Injectable for tests.
        socket_factory: Builds a datagram socket. Injectable for tests.
        thread_factory: Builds the background pinger thread. Injectable.
    """

    def __init__(
        self,
        env: Optional[Mapping[str, str]] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        socket_factory: Callable[[], Any] = lambda: socket.socket(
            socket.AF_UNIX, socket.SOCK_DGRAM
        ),
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        source = os.environ if env is None else env
        self._clock = clock
        self._socket_factory = socket_factory
        self._thread_factory = thread_factory
        self._address = _parse_notify_socket(source.get("NOTIFY_SOCKET", ""))
        timeout = _watchdog_interval_usec(source.get("WATCHDOG_USEC"))
        pid_ok = _watchdog_pid_matches(source.get("WATCHDOG_PID"))
        self.interval = (
            0.0 if timeout is None else max(0.1, min(PING_INTERVAL_SECONDS, timeout / 2.0))
        )
        self.enabled = self._address is not None and timeout is not None and pid_ok
        if self._address is not None and timeout is not None and not pid_ok:
            logger.info("Watchdog is for another process (WATCHDOG_PID); not pinging")
        elif self.enabled:
            logger.info(
                "systemd watchdog enabled: pinging every %.1fs (WatchdogSec=%.1fs)",
                self.interval,
                timeout,
            )

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._last_beat: Optional[float] = None
        self._last_ping: Optional[float] = None
        self._callback_since: Optional[float] = None
        self._callbacks_running = 0
        self._long_operations = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._send_failed = False

    # -- feeds -------------------------------------------------------------

    def note_monitor_beat(self) -> None:
        """Record a hotkey-monitor iteration and ping when one is due."""
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            self._last_beat = now
        self.ping_if_due(now)

    def note_monitor_stopped(self) -> None:
        """Drop the last monitor beat (intentional listener shutdown)."""
        if not self.enabled:
            return
        with self._lock:
            self._last_beat = None

    def callback_started(self) -> None:
        """Record that a hotkey callback began running.

        Counted rather than flagged: push-to-talk press and release callbacks
        can overlap on separate threads, so the stall clock runs from the
        first still-running callback until the last one returns.
        """
        if not self.enabled:
            return
        with self._lock:
            self._callbacks_running += 1
            if self._callbacks_running == 1:
                self._callback_since = self._clock()

    def callback_finished(self) -> None:
        """Record that a hotkey callback returned."""
        if not self.enabled:
            return
        with self._lock:
            self._callbacks_running = max(0, self._callbacks_running - 1)
            if self._callbacks_running == 0:
                self._callback_since = None

    @contextmanager
    def long_operation(self, reason: str = "") -> Iterator[None]:
        """Mark work that may legitimately exceed the callback stall limit.

        Cold model loads and bounded transcription joins run inside hotkey
        callbacks. Without this marker the watchdog would restart a healthy
        process just because the speech model took over 30 seconds to load.
        """
        if not self.enabled:
            yield
            return
        with self._lock:
            self._long_operations += 1
        try:
            yield
        finally:
            with self._lock:
                self._long_operations = max(0, self._long_operations - 1)

    # -- health and pings --------------------------------------------------

    def is_healthy(self, now: Optional[float] = None) -> bool:
        """True when neither the monitor nor a callback is stuck."""
        if not self.enabled:
            return False
        now = self._clock() if now is None else now
        with self._lock:
            if self._long_operations == 0 and self._callback_since is not None:
                if now - self._callback_since > CALLBACK_STALL_SECONDS:
                    return False
            if self._last_beat is not None and now - self._last_beat > MONITOR_STALE_SECONDS:
                return False
        return True

    def ping_if_due(self, now: Optional[float] = None) -> bool:
        """Send WATCHDOG=1 when due and healthy. Returns True if sent."""
        if not self.enabled:
            return False
        now = self._clock() if now is None else now
        with self._lock:
            if self._last_ping is not None and now - self._last_ping < self.interval:
                return False
            if not self.is_healthy(now):
                return False
            self._last_ping = now
        return self._send()

    def _send(self) -> bool:
        assert self._address is not None
        try:
            sock = self._socket_factory()
        except OSError as exc:
            self._log_send_failure(exc)
            return False
        try:
            sock.sendto(b"WATCHDOG=1", self._address)
        except OSError as exc:
            self._log_send_failure(exc)
            return False
        finally:
            try:
                sock.close()
            except OSError:
                pass
        self._send_failed = False
        return True

    def _log_send_failure(self, exc: OSError) -> None:
        if not self._send_failed:
            logger.warning("Could not send systemd watchdog ping: %s", exc)
            self._send_failed = True

    # -- background thread -------------------------------------------------

    def start(self) -> bool:
        """Start the background pinger. Idempotent; False when disabled."""
        if not self.enabled:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop = False
            thread = self._thread_factory(
                target=self._run,
                name="vocalinux-watchdog",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return True

    def stop(self) -> None:
        """Stop the background pinger (tests and orderly shutdown)."""
        with self._lock:
            self._stop = True
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(self) -> None:
        # The first ping is due immediately: a slow startup must still answer
        # systemd well inside WatchdogSec.
        self.ping_if_due()
        while True:
            with self._condition:
                if self._stop:
                    return
                self._condition.wait(timeout=self.interval)
                if self._stop:
                    return
            self.ping_if_due()


_pinger: Optional[WatchdogPinger] = None
_pinger_lock = threading.Lock()


def get_pinger() -> WatchdogPinger:
    """Return the process-wide pinger, creating it from the environment once."""
    global _pinger
    if _pinger is None:
        with _pinger_lock:
            if _pinger is None:
                _pinger = WatchdogPinger()
    return _pinger


def install_faulthandler() -> None:
    """Dump every thread's stack on fatal signals and on SIGUSR1.

    systemd's watchdog action is SIGABRT, so enabling faulthandler makes a
    watchdog kill leave a full Python stack dump in the journal instead of a
    bare signal; SIGUSR1 keeps the on-demand dump for manual investigation.
    """
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except (OSError, RuntimeError, ValueError):
        logger.debug("Could not enable faulthandler", exc_info=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    except (OSError, RuntimeError, ValueError, AttributeError):
        logger.debug("Could not register SIGUSR1 stack dumps", exc_info=True)


def start_from_environment() -> bool:
    """Install fault dumps and start the pinger when systemd configured one."""
    install_faulthandler()
    return get_pinger().start()


def note_monitor_beat() -> None:
    """Feed the watchdog from the hotkey monitor loop."""
    get_pinger().note_monitor_beat()


def note_monitor_stopped() -> None:
    """Forget the monitor so the no-monitor fallback feeds the watchdog.

    Called only for an intentional listener shutdown (external activation,
    backend switch). An unexpected monitor exit leaves its last beat in place
    on purpose, so pings stop and systemd restarts the wedged process.
    """
    get_pinger().note_monitor_stopped()


def callback_started() -> None:
    get_pinger().callback_started()


def callback_finished() -> None:
    get_pinger().callback_finished()


@contextmanager
def long_operation(reason: str = "") -> Iterator[None]:
    """Context manager marking known-slow work (model load, transcription)."""
    with get_pinger().long_operation(reason):
        yield


def reset_for_tests() -> None:
    """Drop the process-wide pinger (tests only)."""
    global _pinger
    with _pinger_lock:
        old = _pinger
        _pinger = None
    if old is not None:
        old.stop()


__all__ = [
    "CALLBACK_STALL_SECONDS",
    "MONITOR_STALE_SECONDS",
    "PING_INTERVAL_SECONDS",
    "WatchdogPinger",
    "callback_finished",
    "callback_started",
    "get_pinger",
    "install_faulthandler",
    "long_operation",
    "note_monitor_beat",
    "note_monitor_stopped",
    "reset_for_tests",
    "start_from_environment",
]
