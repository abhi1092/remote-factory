"""GlaudeRunner — Glaude (GLM-5.2 via RITS) CLI backend implementation.

Glaude is a wrapper around Claude Code that routes through a CONNECT proxy
to the RITS GLM-5.2 endpoint. It has the same CLI interface as claude —
all flags, output formats, and session management work identically.
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

    Inherits all behavior from ClaudeRunner — the only differences are the
    binary name ('glaude' instead of 'claude') and metadata.
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
        )

    def build_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        cmd, env, temp_files = super().build_command(request)
        cmd[0] = "glaude"
        return cmd, env, temp_files

    def build_interactive_command(
        self, request: AgentRunRequest
    ) -> tuple[list[str], dict[str, str], list[Path]]:
        cmd, env, temp_files = super().build_interactive_command(request)
        cmd[0] = "glaude"
        return cmd, env, temp_files
