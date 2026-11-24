"""Tests for the opt-in ydotool ASCII typing path.

VOCALINUX_YDOTOOL_TYPE_ASCII=1 routes printable ASCII through
`ydotool type`; everything else (non-ASCII, control characters) keeps using
the clipboard paste chord.

Command shape verified against the Debian 0.1.8-3 source:
Tools/Type/Type.cpp registers `--key-delay` with boost::program_options and a
positional `extra-args` option, and boost::program_options treats `--` as the
positional terminator, so text starting with `-` is typed literally. ydotool
0.1.8 TypeText() stops at the first character its key table cannot map, hence
the printable-ASCII-only guard.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from vocalinux.text_injection.text_injector import TextInjector


def _injector(clipboard_ok: bool = True) -> TextInjector:
    injector = object.__new__(TextInjector)
    injector.wayland_tool = "ydotool"
    injector._ensure_ydotoold = MagicMock(return_value=True)
    injector._inject_via_clipboard_paste = MagicMock(return_value=clipboard_ok)
    injector._abort_injections = threading.Event()
    injector._wait_for_modifiers_released = MagicMock()
    return injector


def _commands(run_mock: MagicMock) -> list[list[str]]:
    return [call.args[0] for call in run_mock.call_args_list]


class TestYdotoolTypeAsciiSwitch:
    def test_disabled_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("VOCALINUX_YDOTOOL_TYPE_ASCII", raising=False)
        assert _injector()._ydotool_type_ascii_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_truthy_values_enable(self, monkeypatch, value) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", value)
        assert _injector()._ydotool_type_ascii_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "2"])
    def test_other_values_keep_it_off(self, monkeypatch, value) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", value)
        assert _injector()._ydotool_type_ascii_enabled() is False

    def test_printable_ascii_is_typeable(self) -> None:
        assert TextInjector._ascii_typeable_by_ydotool("Hello, world! 42 {a-b}") is True

    @pytest.mark.parametrize("text", ["Grüße", "new\nline", "tab\there", "🙂"])
    def test_non_ascii_and_control_characters_are_not_typeable(self, text) -> None:
        assert TextInjector._ascii_typeable_by_ydotool(text) is False


class TestYdotoolInjectionRouting:
    def test_ascii_uses_ydotool_type_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", "1")
        monkeypatch.setenv("VOCALINUX_YDOTOOL_KEY_DELAY", "2")
        injector = _injector()
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool("Hello world")
        injector._inject_via_clipboard_paste.assert_not_called()
        assert _commands(run) == [["ydotool", "type", "--key-delay", "2", "--", "Hello world"]]

    def test_dash_leading_text_goes_after_the_terminator(self, monkeypatch) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", "1")
        injector = _injector()
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool("-n looks like a flag")
        command = _commands(run)[0]
        assert command[-2:] == ["--", "-n looks like a flag"]

    def test_non_ascii_still_uses_the_clipboard(self, monkeypatch) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", "1")
        injector = _injector()
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool("Grüße")
        injector._inject_via_clipboard_paste.assert_called_once_with("Grüße")
        run.assert_not_called()

    def test_newlines_still_use_the_clipboard(self, monkeypatch) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", "1")
        injector = _injector()
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool("first\nsecond")
        injector._inject_via_clipboard_paste.assert_called_once_with("first\nsecond")
        run.assert_not_called()

    def test_disabled_ascii_uses_the_clipboard(self, monkeypatch) -> None:
        monkeypatch.delenv("VOCALINUX_YDOTOOL_TYPE_ASCII", raising=False)
        injector = _injector()
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool("Hello world")
        injector._inject_via_clipboard_paste.assert_called_once_with("Hello world")
        run.assert_not_called()

    def test_clipboard_failure_falls_back_to_typing_when_disabled(self, monkeypatch) -> None:
        monkeypatch.delenv("VOCALINUX_YDOTOOL_TYPE_ASCII", raising=False)
        injector = _injector(clipboard_ok=False)
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool("Hello")
        assert _commands(run)[0][:2] == ["ydotool", "type"]

    def test_long_ascii_text_is_chunked(self, monkeypatch) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", "1")
        injector = _injector()
        text = "a" * 450
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool(text)
        commands = _commands(run)
        assert [len(command[-1]) for command in commands] == [200, 200, 50]
        assert all(command[2:5] == ["--key-delay", "2", "--"] for command in commands)

    def test_key_delay_is_still_configurable(self, monkeypatch) -> None:
        monkeypatch.setenv("VOCALINUX_YDOTOOL_TYPE_ASCII", "1")
        monkeypatch.setenv("VOCALINUX_YDOTOOL_KEY_DELAY", "7")
        injector = _injector()
        with patch("vocalinux.text_injection.text_injector.subprocess.run") as run:
            injector._inject_with_wayland_tool("abc")
        assert _commands(run)[0][2:5] == ["--key-delay", "7", "--"]
