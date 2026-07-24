"""InnerLoop — model-like wrapper for mode + evaluator that an outer-loop optimizer calls.

Usage:
    evaluator = CirclePackingEvaluator(evaluator_path, initial_program_path)
    loop = InnerLoop(project_dir, mode="evolve", evaluator=evaluator)

    for i in range(budget):
        result = loop.step()
        if result.score_end > target:
            break
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from factory.cycle_analyzer import CycleAnalyzer, CycleRecord
from factory.workflow.primitives import Workflow


@dataclass
class EvalResult:
    """Structured evaluator output."""

    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    artifacts: list[str] = field(default_factory=list)


@runtime_checkable
class Evaluator(Protocol):
    """Hook for plugging in different evaluators."""

    def evaluate(self, code: str) -> EvalResult: ...

    def get_info(self) -> dict: ...


class CirclePackingEvaluator:
    """Wraps skydiscover's circle packing evaluator for direct use."""

    def __init__(self, evaluator_path: Path, initial_program_path: Path | None = None) -> None:
        self.evaluator_path = Path(evaluator_path)
        self.eval_fn = self._load_evaluator(self.evaluator_path)
        self.initial_program: str | None = None
        if initial_program_path:
            self.initial_program = Path(initial_program_path).read_text()

    def evaluate(self, code: str) -> EvalResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = self.eval_fn(tmp_path)
            if not isinstance(result, dict):
                return EvalResult(score=0.0, valid=False)
            return EvalResult(
                score=float(result.get("combined_score", 0.0)),
                metrics={k: float(v) for k, v in result.items() if isinstance(v, (int, float))},
                valid=result.get("validity", 0.0) == 1.0,
                artifacts=[tmp_path],
            )
        except Exception as e:
            return EvalResult(score=0.0, valid=False, metrics={"error": 0.0})

    def get_info(self) -> dict:
        return {
            "benchmark": self.evaluator_path.parent.name,
            "evaluator_path": str(self.evaluator_path),
            "initial_program": self.initial_program,
        }

    @staticmethod
    def _load_evaluator(evaluator_path: Path):
        evaluator_path = evaluator_path.resolve()
        eval_dir = str(evaluator_path.parent)
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        module_name = f"_eval_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "evaluate"):
            raise AttributeError(f"No evaluate() function in {evaluator_path}")
        return module.evaluate


class InnerLoop:
    """Wraps a factory mode + evaluator. Optimizer calls loop.step()."""

    def __init__(
        self,
        project_dir: Path,
        mode: str = "evolve",
        evaluator: Evaluator | None = None,
        workflow: Workflow | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.factory_dir = self.project_dir / ".factory"
        self.mode = mode
        self.evaluator = evaluator
        self.workflow = workflow
        self._step_count = 0
        self._history: list[CycleRecord] = []

    def step(self, directives: dict[str, Any] | None = None) -> CycleRecord:
        """Run one inner-loop cycle and return structured results."""
        if directives:
            self._write_directives(directives)

        subprocess.run(
            [sys.executable, "-m", "factory", "ceo", str(self.project_dir),
             "--mode", self.mode, "--no-worktree"],
            cwd=self.project_dir,
        )

        analyzer = CycleAnalyzer(self.factory_dir, workflow=self.workflow)
        record = analyzer.latest()
        if record is None:
            record = CycleRecord(
                cycle_number=self._step_count + 1,
                mode=self.mode,
                started_at=None,
                ended_at=None,
                duration_s=0,
                score_start=None,
                score_end=None,
                score_delta=None,
            )

        record.cycle_number = self._step_count + 1

        if self.evaluator:
            best = self.current_best()
            if best:
                eval_result = self.evaluator.evaluate(best)
                record.score_end = eval_result.score
                record.eval_artifacts = eval_result.artifacts

        self._step_count += 1
        self._history.append(record)
        return record

    def evaluate(self, code: str | Path) -> EvalResult:
        """Evaluate a solution directly, outside the mode cycle."""
        if not self.evaluator:
            raise RuntimeError("No evaluator configured")
        if isinstance(code, Path):
            code = code.read_text()
        return self.evaluator.evaluate(code)

    def current_best(self) -> str | None:
        """Return the current best solution code."""
        candidates = [
            self.factory_dir / "evolve" / "current_best.py",
            self.factory_dir / "evolve" / "candidate.py",
        ]
        for p in candidates:
            if p.exists():
                return p.read_text()
        return None

    def score_trajectory(self) -> list[float]:
        """Score history across all steps."""
        if self._history:
            return [r.score_end for r in self._history if r.score_end is not None]
        analyzer = CycleAnalyzer(self.factory_dir, workflow=self.workflow)
        return analyzer.trajectory()

    def total_cost(self) -> float:
        """Cumulative cost across all steps."""
        return sum(r.total_cost_usd for r in self._history)

    def history(self) -> list[CycleRecord]:
        """All cycle records from this session."""
        return list(self._history)

    def _write_directives(self, directives: dict[str, Any]) -> None:
        """Write outer-loop directives as a factory message."""
        msg_dir = self.factory_dir / "messages"
        msg_dir.mkdir(parents=True, exist_ok=True)
        msg_id = f"outer-loop-{self._step_count:04d}"
        msg_path = msg_dir / f"{msg_id}.md"

        lines = ["# Outer Loop Directives\n"]
        for key, value in directives.items():
            if isinstance(value, list):
                lines.append(f"- **{key}:** {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"- **{key}:** {value}")

        msg_path.write_text("\n".join(lines) + "\n")
