import msvcrt
import tkinter

from eval_rag import hidden_api_key_input, sanitize_api_key


def test_sanitize_api_key_removes_console_control_characters():
    assert sanitize_api_key("  sk-\x16fake-probe\r\n") == "sk-fake-probe"


def test_sanitize_api_key_keeps_printable_ascii_characters():
    assert sanitize_api_key("sk-test_123.!-") == "sk-test_123.!-"


def test_hidden_api_key_input_reads_clipboard_on_ctrl_v(monkeypatch):
    keys = iter(["\x16", "\r"])

    class FakeRoot:
        def withdraw(self):
            pass

        def clipboard_get(self):
            return "sk-from-clipboard"

        def destroy(self):
            pass

    monkeypatch.setattr(msvcrt, "getwch", lambda: next(keys))
    monkeypatch.setattr(tkinter, "Tk", FakeRoot)

    assert hidden_api_key_input("Key：") == "sk-from-clipboard"
