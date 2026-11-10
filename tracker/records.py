"""Shared experiment data definitions."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptCase:
    sample_id: str
    dataset: str
    label: int  # 0 = benign, 1 = injection
    system: str
    user: str
    instruction: str
    # Exact meaningful data strings, in the order they appear in the formatted chat.
    data_parts: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    dataset: str
    label: int
    focus_score: float
    risk_score: float
    rejected: bool
    metadata: dict[str, Any]
    baseline_response: str | None = None
    defended_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "label": self.label,
            "focus_score": self.focus_score,
            "risk_score": self.risk_score,
            "rejected": self.rejected,
            "correct": int(self.rejected) == self.label,
            **self.metadata,
            "baseline_response": self.baseline_response,
            "defended_response": self.defended_response,
        }
