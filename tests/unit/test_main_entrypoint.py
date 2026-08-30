"""Unit tests for module entrypoint behavior."""

import runpy
import sys

import pytest

from opengate_gate_tree import cli


def test_main_module_calls_cli_main_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running opengate_gate_tree.__main__ should call cli.main and exit with its code."""
    call_count = {"main": 0}

    def fake_main() -> int:
        call_count["main"] += 1
        return 7

    def fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("opengate_gate_tree.__main__", run_name="__main__")

    assert exc_info.value.code == 7
    assert call_count["main"] == 1
