"""MMLU dataset loading and prompt formatting.

MMLU is multiple-choice with ground-truth answers, so accuracy is
exact-match — no LLM-as-judge required (zero API cost).
"""

from dataclasses import dataclass

CHOICE_LETTERS = ["A", "B", "C", "D"]


@dataclass
class MMLUQuestion:
    question: str
    choices: list[str]
    answer_idx: int  # 0-3
    subject: str

    @property
    def answer_letter(self) -> str:
        return CHOICE_LETTERS[self.answer_idx]

    def to_prompt(self) -> str:
        lines = [f"Question: {self.question}"]
        for letter, choice in zip(CHOICE_LETTERS, self.choices):
            lines.append(f"{letter}. {choice}")
        lines.append("Answer:")
        return "\n".join(lines)


def load_mmlu(n_questions: int = 300, seed: int = 42) -> list[MMLUQuestion]:
    """Load a balanced random sample of MMLU test questions.

    Uses the HuggingFace 'cais/mmlu' dataset ('all' config, test split).
    Downloaded once, cached locally afterwards.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))

    questions = []
    for row in ds:
        questions.append(
            MMLUQuestion(
                question=row["question"],
                choices=list(row["choices"]),
                answer_idx=int(row["answer"]),
                subject=row.get("subject", ""),
            )
        )
    return questions
