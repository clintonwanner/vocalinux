"""Tests for the systemd watchdog feed and its hotkey-path integration."""

import os
import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from vocalinux.ui.keyboard_backends import evdev_backend
from vocalinux.utils import watchdog


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _pinger(tmp_path, clock=None, usec="30000000", pid=None, socket_factory=None):
    env = {"NOTIFY_SOCKET": str(tmp_path / "notify.sock"), "WATCHDOG_USEC": usec}
    if pid is not None:
        env["WATCHDOG_PID"] = str(pid)
    return watchdog.WatchdogPinger(
        env,
        clock=clock or FakeClock(),
        socket_factory=socket_factory or (lambda: MagicMock()),
    )


class TestWatchdogParsing:
    def test_disabled_without_notify_socket(self) -> None:
        pinger = watchdog.WatchdogPinger({"WATCHDOG_USEC": "30000000"})
        assert pinger.enabled is False
        assert pinger.start() is False
        assert pinger.ping_if_due() is False

    def test_disabled_without_watchdog_usec(self, tmp_path) -> None:
        pinger = watchdog.WatchdogPinger({"NOTIFY_SOCKET": str(tmp_path / "s")})
        assert pinger.enabled is False

    def test_disabled_for_another_pid(self, tmp_path) -> None:
        pinger = _pinger(tmp_path, pid=os.getpid() + 1)
        assert pinger.enabled is False

    def test_enabled_for_own_pid(self, tmp_path) -> None:
        pinger = _pinger(tmp_path, pid=os.getpid())
        assert pinger.enabled is True

    def test_interval_is_capped_at_five_seconds(self, tmp_path) -> None:
        assert _pinger(tmp_path, usec="30000000").interval == 5.0
        assert _pinger(tmp_path, usec="2000000").interval == 1.0

    def test_invalid_usec_disables(self, tmp_path) -> None:
        pinger = watchdog.WatchdogPinger(
            {"NOTIFY_SOCKET": str(tmp_path / "s"), "WATCHDOG_USEC": "soon"}
        )
        assert pinger.enabled is False

    def test_invalid_pid_is_ignored(self, tmp_path) -> None:
        pinger = watchdog.WatchdogPinger(
            {
                "NOTIFY_SOCKET": str(tmp_path / "s"),
                "WATCHDOG_USEC": "30000000",
                "WATCHDOG_PID": "not-a-pid",
            }
        )
        assert pinger.enabled is True


class TestWatchdogHealth:
    def test_pings_at_most_once_per_interval(self, tmp_path) -> None:
        clock = FakeClock()
        sent: list[bytes] = []

        def factory():
            sock = MagicMock()
            sock.sendto.side_effect = lambda data, addr: sent.append(data)
            return sock

        pinger = _pinger(tmp_path, clock=clock, socket_factory=factory)
        assert pinger.ping_if_due() is True
        clock.now += 2.0
        assert pinger.ping_if_due() is False
        clock.now += 3.0
        assert pinger.ping_if_due() is True
        assert sent == [b"WATCHDOG=1", b"WATCHDOG=1"]

    def test_slow_callback_stops_pings_until_it_finishes(self, tmp_path) -> None:
        clock = FakeClock()
        pinger = _pinger(tmp_path, clock=clock)
        pinger.callback_started()
        clock.now += watchdog.CALLBACK_STALL_SECONDS + 1
        assert pinger.ping_if_due() is False
        pinger.callback_finished()
        clock.now += 6.0
        assert pinger.ping_if_due() is True

    def test_long_operation_exempts_cold_model_load(self, tmp_path) -> None:
        clock = FakeClock()
        pinger = _pinger(tmp_path, clock=clock)
        pinger.callback_started()
        clock.now += watchdog.CALLBACK_STALL_SECONDS + 30
        with pinger.long_operation("cold model load"):
            assert pinger.ping_if_due() is True
        clock.now += 6.0
        # Back in ordinary callback work: now the stall rule applies again.
        assert pinger.ping_if_due() is False

    def test_stale_monitor_beat_stops_pings(self, tmp_path) -> None:
        clock = FakeClock()
        pinger = _pinger(tmp_path, clock=clock)
        assert pinger.note_monitor_beat() is None
        clock.now += watchdog.MONITOR_STALE_SECONDS + 1
        assert pinger.ping_if_due() is False
        pinger.note_monitor_beat()
        clock.now += 6.0
        assert pinger.ping_if_due() is True

    def test_intentional_monitor_stop_restores_the_fallback(self, tmp_path) -> None:
        clock = FakeClock()
        pinger = _pinger(tmp_path, clock=clock)
        pinger.note_monitor_beat()
        pinger.note_monitor_stopped()
        clock.now += 10_000
        assert pinger.ping_if_due() is True

    def test_overlapping_callbacks_track_the_first_still_running(self, tmp_path) -> None:
        clock = FakeClock()
        pinger = _pinger(tmp_path, clock=clock)
        pinger.callback_started()
        clock.now += 5
        pinger.callback_started()
        pinger.callback_finished()
        clock.now += watchdog.CALLBACK_STALL_SECONDS
        assert pinger.ping_if_due() is False
        pinger.callback_finished()
        clock.now += 6
        assert pinger.ping_if_due() is True

    def test_no_monitor_ever_keeps_pinging(self, tmp_path) -> None:
        clock = FakeClock()
        pinger = _pinger(tmp_path, clock=clock)
        clock.now += 10_000
        assert pinger.ping_if_due() is True

    def test_send_failure_is_logged_once_and_retried(self, tmp_path) -> None:
        clock = FakeClock()
        sock = MagicMock()
        sock.sendto.side_effect = OSError("no listener")
        pinger = _pinger(tmp_path, clock=clock, socket_factory=lambda: sock)
        assert pinger.ping_if_due() is False
        clock.now += 6.0
        sock.sendto.side_effect = None
        assert pinger.ping_if_due() is True


class TestWatchdogSockets:
    def test_filesystem_datagram_socket_receives_watchdog(self, tmp_path) -> None:
        path = str(tmp_path / "notify.sock")
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(path)
        receiver.settimeout(2.0)
        try:
            pinger = watchdog.WatchdogPinger(
                {"NOTIFY_SOCKET": path, "WATCHDOG_USEC": "30000000"},
                clock=FakeClock(),
            )
            assert pinger.ping_if_due() is True
            assert receiver.recv(64) == b"WATCHDOG=1"
        finally:
            receiver.close()

    def test_abstract_socket_is_supported(self) -> None:
        name = f"vocalinux-watchdog-test-{os.getpid()}-{time.monotonic_ns()}"
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind("\0" + name)
        receiver.settimeout(2.0)
        try:
            pinger = watchdog.WatchdogPinger(
                {"NOTIFY_SOCKET": "@" + name, "WATCHDOG_USEC": "30000000"},
                clock=FakeClock(),
            )
            assert pinger.enabled is True
            assert pinger.ping_if_due() is True
            assert receiver.recv(64) == b"WATCHDOG=1"
        finally:
            receiver.close()

    def test_background_thread_pings_immediately(self, tmp_path) -> None:
        path = str(tmp_path / "notify.sock")
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(path)
        receiver.settimeout(2.0)
        try:
            pinger = watchdog.WatchdogPinger(
                {"NOTIFY_SOCKET": path, "WATCHDOG_USEC": "200000"},
            )
            assert pinger.start() is True
            try:
                assert receiver.recv(64) == b"WATCHDOG=1"
            finally:
                pinger.stop()
        finally:
            receiver.close()

    def test_slow_model_load_does_not_starve_the_thread(self, tmp_path) -> None:
        path = str(tmp_path / "notify.sock")
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(path)
        receiver.settimeout(3.0)
        try:
            pinger = watchdog.WatchdogPinger(
                {"NOTIFY_SOCKET": path, "WATCHDOG_USEC": "200000"},
            )
            pinger.callback_started()
            with pinger.long_operation("cold model load"):
                assert pinger.start() is True
                try:
                    assert receiver.recv(64) == b"WATCHDOG=1"
                finally:
                    pinger.stop()
        finally:
            receiver.close()


class TestWatchdogIntegration:
    def test_faulthandler_installed_for_sigusr1_and_fatal_signals(self) -> None:
        with (
            patch.object(watchdog.faulthandler, "enable") as enable,
            patch.object(watchdog.faulthandler, "register") as register,
        ):
            watchdog.install_faulthandler()
        enable.assert_called_once()
        register.assert_called_once()
        args, kwargs = register.call_args
        assert args[0] == watchdog.signal.SIGUSR1
        assert kwargs["all_threads"] is True

    def test_evdev_monitor_feeds_the_watchdog(self) -> None:
        # conftest deliberately refuses to start threads named
        # _monitor_devices, so drive one iteration on this thread and stop it
        # at the no-device sleep.
        class _StopMonitor(Exception):
            pass

        hub = evdev_backend.EvdevDeviceHub()
        hub.running = True
        with (
            patch.object(watchdog, "note_monitor_beat") as beat,
            patch(
                "vocalinux.ui.keyboard_backends.evdev_backend.time.sleep",
                side_effect=_StopMonitor,
            ),
        ):
            with pytest.raises(_StopMonitor):
                hub._monitor_devices(generation=hub._generation)
        assert beat.called

    def test_detector_worker_reports_callback_start_and_finish(self) -> None:
        from vocalinux.ui.keyboard_backends.hold_or_double_tap import (
            HoldOrDoubleTapDetector,
        )

        ran = threading.Event()
        detector = HoldOrDoubleTapDetector(ran.set, lambda: None, lambda: None, lambda: None)
        with (
            patch.object(watchdog, "callback_started") as started,
            patch.object(watchdog, "callback_finished") as finished,
        ):
            detector.press("kbd")
            assert ran.wait(2.0)
            deadline = time.monotonic() + 2.0
            while not finished.called and time.monotonic() < deadline:
                time.sleep(0.005)
        started.assert_called_once()
        finished.assert_called_once()

    def test_reset_for_tests_stops_the_singleton(self) -> None:
        with patch.object(watchdog, "WatchdogPinger") as pinger_cls:
            pinger_cls.return_value.start.return_value = True
            watchdog.reset_for_tests()
            assert watchdog.start_from_environment() is True
            pinger_cls.return_value.start.assert_called_once()
            watchdog.reset_for_tests()
            pinger_cls.return_value.stop.assert_called_once()
