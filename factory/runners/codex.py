"""CodexRunner — OpenAI Codex CLI backend implementation.

Codex CLI (https://github.com/openai/codex) is OpenAI's open-source
agentic coding tool. Key interface differences from Claude Code:

- Headless mode uses ``codex exec "<prompt>"`` (positional arg to exec subcommand)
- System prompt is injected via an ``AGENTS.md`` file in the project directory
  (no --append-system-prompt-file flag)
- JSON output via ``--json`` (JSONL to stdout)
- Approval bypass via ``--ask-for-approval never``
- Working directory via ``--cd <path>``
- Auth via OPENAI_API_KEY env var
- No session management (--name, --resume, --session-id)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from factory.runners._subprocess import run_subprocess

if TYPE_CHECKING:
    from factory.models import AgentRunRequest, AgentRunResult, AgentUsage
    from factory.runners.protocol import RunnerMeta

log = structlog.get_logger()


def _parse_codex_usage(data: dict) -> AgentUsage:
    """Extract AgentUsage from Codex JSON output."""
    from factory.models import AgentUsage

    usage_block = data.get("usage", {})
    return AgentUsage(
        input_tokens=usage_block.get("input_tokens", 0)
        or usage_block.get("prompt_tokens", 0),
        output_tokens=usage_block.get("output_tokens", 0)
        or usage_block.get("completion_tokens", 0),
        cache_read_tokens=usage_block.get("cache_read_input_tokens", 0),
        cache_creation_tokens=usage_block.get("cache_creation_input_tokens", 0),
        total_cost_usd=data.get("total_cost_usd", 0.0) or 0.0,
        duration_ms=data.get("duration_ms", 0.0) or 0.0,
        num_turns=data.get("num_turns", 0) or 0,
        model=data.get("model", ""),
    )


class CodexRunner:
    """Runner implementation for OpenAI Codex CLI."""

    name: str = "codex"

    @classmethod
    def metadata(cls) -> RunnerMeta:
        from factory.runners.protocol import RunnerMeta

        return RunnerMeta(
            name="codex",
            display_name="OpenAI Codex CLI",
            binary="codex",
            install_hint="npm install -g @openai/codex",

            supports_usage_telemetry=False,
            supports_session_name=False,
            supports_session_resume=False,
            supports_background=False,
            supports_interactive=True,
            supports_streaming=True,
        )

    def _write_agents_md(self, cwd: Path, prompt: str) -> Path | None:
        """Write system prompt to AGENTS.md in the project directory.

        Returns the path if a file was created (caller must clean up),
        or None if an existing AGENTS.md was found (left untouched).
        """
        agents_md = cwd / "AGENTS.md"
        if agents_md.exists():
            log.debug("codex_agents_md_exists", path=str(agents_md))
            return None
        agents_md.write_text(prompt)
        return agents_md

    def build_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        """Build the Codex CLI command, env dict, and temp files."""
        temp_files: list[Path] = []

        cwd = Path(request.cwd)
        agents_md_path = self._write_agents_md(cwd, request.prompt)
        if agents_md_path is not None:
            temp_files.append(agents_md_path)

        cmd = [
            "codex",
            "exec",
            "--json",
            "--ask-for-approval",
            "never",
        ]
        if request.cwd:
            cmd.extend(["--cd", str(request.cwd)])
        if request.model:
            cmd.extend(["--model", request.model])
        cmd.append(request.task)

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        if request.cwd:
            env["PROJECT_PATH"] = str(Path(request.cwd).resolve())
        if request.model:
            env["FACTORY_MODEL"] = request.model

        return cmd, env, temp_files

    async def headless(self, request: AgentRunRequest) -> AgentRunResult:
        """Run a headless Codex CLI invocation."""
        from factory.models import AgentRunResult

        cmd, env, temp_files = self.build_command(request)
        try:
            log.info("codex_headless", cwd=str(request.cwd), model=request.model)

            result = await run_subprocess(
                cmd,
                cwd=str(request.cwd),
                env=env,
                timeout=request.timeout,
                runner_name="codex",
                role=request.role,
                sanitize=True,
            )

            usage = None
            result_text = result.stdout
            metadata: dict[str, object] = {**result.metadata}

            data: dict[str, object] | None = None
            for line in reversed(result.stdout.strip().splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict) and ("result" in parsed or "message" in parsed):
                    data = parsed
                    break

            if data is not None:
                result_value = data.get("result", data.get("message", result.stdout))
                result_text = result_value if isinstance(result_value, str) else result.stdout
                usage = _parse_codex_usage(data)

            return AgentRunResult(
                stdout=result_text,
                return_code=result.return_code,
                usage=usage,
                metadata=metadata,
            )
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)

    def interactive_run(self, request: AgentRunRequest) -> int:
        """Run an interactive Codex session as a subprocess."""
        cwd = Path(request.cwd)
        agents_md_path = self._write_agents_md(cwd, request.prompt)
        temp_files: list[Path] = []
        if agents_md_path is not None:
            temp_files.append(agents_md_path)

        cmd = ["codex"]
        if request.cwd:
            cmd.extend(["--cd", str(request.cwd)])
        if request.model:
            cmd.extend(["--model", request.model])

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        if request.cwd:
            env["PROJECT_PATH"] = str(Path(request.cwd).resolve())
        if request.model:
            env["FACTORY_MODEL"] = request.model

        try:
            log.info("codex_interactive", cwd=str(request.cwd))
            result = subprocess.run(cmd, cwd=request.cwd, env=env)
            return result.returncode
        finally:
            for f in temp_files:
                f.unlink(missing_ok=True)
