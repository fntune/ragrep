import sys

from ragrep import cli


def test_pipeline_invocation_accepts_documented_commands() -> None:
    assert cli._is_pipeline_invocation(["scrape"])
    assert cli._is_pipeline_invocation(["ingest", "--force"])
    assert cli._is_pipeline_invocation(["query", "-q", "auth"])
    assert cli._is_pipeline_invocation(["stats"])


def test_pipeline_invocation_accepts_main_global_options_before_command() -> None:
    assert cli._is_pipeline_invocation(["--config", "config.toml", "scrape"])
    assert cli._is_pipeline_invocation(["--log-level=DEBUG", "ingest"])
    assert cli._is_pipeline_invocation(["--config", "config.toml", "--log-level", "DEBUG", "stats"])


def test_pipeline_invocation_leaves_search_terms_on_search_surface() -> None:
    assert not cli._is_pipeline_invocation(["auth flow"])
    assert not cli._is_pipeline_invocation(["--server", "http://localhost:8321", "auth"])
    assert not cli._is_pipeline_invocation(["--", "scrape"])


def test_entrypoint_dispatches_pipeline_commands(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "main", lambda: calls.append("main"))
    monkeypatch.setattr(cli, "grep", lambda: calls.append("grep"))
    monkeypatch.setattr(sys, "argv", ["ragrep", "scrape"])

    cli.entrypoint()

    assert calls == ["main"]


def test_entrypoint_dispatches_search_terms(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "main", lambda: calls.append("main"))
    monkeypatch.setattr(cli, "grep", lambda: calls.append("grep"))
    monkeypatch.setattr(sys, "argv", ["ragrep", "auth flow"])

    cli.entrypoint()

    assert calls == ["grep"]
