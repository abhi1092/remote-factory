"""GlaudeRunner — Glaude (GLM-5.2 via RITS) CLI backend implementation.

Glaude is a wrapper around Claude Code that routes through a CONNECT proxy
to the RITS GLM-5.2 endpoint. It has the same CLI interface as claude —
all flags, output formats, and session management work identically.

The RITS endpoint serves a single model, so any model override from workflow
node configs (e.g. "sonnet", "opus") is stripped — glaude's own default
model injection handles it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from factory.runners.claude import ClaudeRunner

if TYPE_CHECKING:
    from factory.models import AgentRunRequest
    from factory.runners.protocol import RunnerMeta


class GlaudeRunner(ClaudeRunner):
    """Runner implementation for Glaude (GLM-5.2 via RITS).

    Inherits all behavior from ClaudeRunner. Key differences:
    - Binary is 'glaude' instead of 'claude'
    - Strips any --model flag so glaude injects its own default model
      (the RITS endpoint only serves one model)
    """

    name: str = "glaude"

    @classmethod
    def metadata(cls) -> RunnerMeta:
        from factory.runners.protocol import RunnerMeta

        return RunnerMeta(
            name="glaude",
            display_name="Glaude (GLM-5.2 via RITS)",
            binary="glaude",
            install_hint="Run install-glaude-rits.sh to install ~/.local/bin/glaude",
            supports_usage_telemetry=True,
            supports_session_name=True,
            supports_session_resume=True,
            supports_background=True,
            supports_model_override=False,
        )

    @staticmethod
    def _strip_model_flag(cmd: list[str]) -> list[str]:
        """Remove --model / --model=VALUE from the command so glaude uses its default."""
        result: list[str] = []
        skip_next = False
        for arg in cmd:
            if skip_next:
                skip_next = False
                continue
            if arg == "--model":
                skip_next = True
                continue
            if arg.startswith("--model="):
                continue
            result.append(arg)
        return result

    def build_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        cmd, env, temp_files = super().build_command(request)
        cmd = self._strip_model_flag(cmd)
        cmd[0] = "glaude"
        return cmd, env, temp_files

    def build_interactive_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        cmd, env, temp_files = super().build_interactive_command(request)
        cmd = self._strip_model_flag(cmd)
        cmd[0] = "glaude"
        return cmd, env, temp_files
