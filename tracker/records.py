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
    # Optional ordered attention spans for prompts with separated trusted text.
    attention_parts: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    dataset: str
    label: int
    focus_score: float
    formatted_text: str
    metadata: dict[str, Any]
    instruction_token_count: int
    total_token_count: int
    baseline_response: str | None = None
    attack_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "label": self.label,
            "focus_score": self.focus_score,
            **self.metadata,
            "formatted_text": self.formatted_text,
            "instruction_token_count": self.instruction_token_count,
            "total_token_count": self.total_token_count,
            "baseline_response": self.baseline_response,
            "attack_success": self.attack_success,
        }
