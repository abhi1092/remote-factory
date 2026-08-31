"""Tests for CodexRunner — command building, AGENTS.md handling, and output parsing."""

from __future__ import annotations

from pathlib import Path

from factory.models import AgentRunRequest
from factory.runners.codex import CodexRunner, _parse_codex_usage


class TestMetadata:
    def test_name(self) -> None:
        meta = CodexRunner.metadata()
        assert meta.name == "codex"

    def test_binary(self) -> None:
        meta = CodexRunner.metadata()
        assert meta.binary == "codex"

    def test_requires_openai_key(self) -> None:
        meta = CodexRunner.metadata()
        assert "OPENAI_API_KEY" in meta.required_env_vars

    def test_no_session_support(self) -> None:
        meta = CodexRunner.metadata()
        assert meta.supports_session_name is False
        assert meta.supports_session_resume is False

    def test_install_hint(self) -> None:
        meta = CodexRunner.metadata()
        assert "@openai/codex" in meta.install_hint


class TestBuildCommand:
    def _make_request(self, tmp_path: Path, **overrides: object) -> AgentRunRequest:
        defaults: dict[str, object] = {
            "prompt": "You are a helpful assistant.",
            "task": "Fix the bug",
            "cwd": tmp_path,
            "role": "builder",
        }
        defaults.update(overrides)
        return AgentRunRequest(**defaults)  # type: ignore[arg-type]

    def test_basic_command_structure(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path)
        cmd, env, temp_files = runner.build_command(req)
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--json" in cmd
        assert "never" in cmd
        assert cmd[-1] == "Fix the bug"

    def test_model_flag(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path, model="o3")
        cmd, _env, _temp = runner.build_command(req)
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "o3"

    def test_cd_flag(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path)
        cmd, _env, _temp = runner.build_command(req)
        idx = cmd.index("--cd")
        assert cmd[idx + 1] == str(tmp_path)

    def test_no_model_flag_when_none(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = self._make_request(tmp_path)
        cmd, _env, _temp = runner.build_command(req)
        assert "--model" not in cmd


class TestAgentsMd:
    def test_creates_agents_md(self, tmp_path: Path) -> None:
        runner = CodexRunner()
        req = AgentRunRequest(
            prompt="System instructions here.",
            task="Do something",
            cwd=tmp_path,
            role="builder",
        )
        _cmd, _env, temp_files = runner.build_command(req)
        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()
        assert agents_md.read_text() == "System instructions here."
        assert agents_md in temp_files

    def test_preserves_existing_agents_md(self, tmp_path: Path) -> None:
        existing = tmp_path / "AGENTS.md"
        existing.write_text("Existing project instructions.")
        runner = CodexRunner()
        req = AgentRunRequest(
            prompt="Factory instructions.",
            task="Do something",
            cwd=tmp_path,
            role="builder",
        )
        _cmd, _env, temp_files = runner.build_command(req)
        assert existing.read_text() == "Existing project instructions."
        assert existing not in temp_files


class TestParseUsage:
    def test_openai_token_fields(self) -> None:
        data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "model": "o3",
        }
        usage = _parse_codex_usage(data)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == "o3"

    def test_standard_token_fields(self) -> None:
        data = {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 75,
            },
        }
        usage = _parse_codex_usage(data)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 75

    def test_empty_usage(self) -> None:
        usage = _parse_codex_usage({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestRegistration:
    def test_codex_in_runners(self) -> None:
        from factory.runners import get_available_runners

        runners = get_available_runners()
        assert "codex" in runners

    def test_get_runner_codex(self) -> None:
        from factory.runners import get_runner

        runner = get_runner("codex")
        assert isinstance(runner, CodexRunner)

    def test_runner_choices_include_codex(self) -> None:
        from factory.runners import get_runner_choices

        choices = get_runner_choices()
        assert "codex" in choices
