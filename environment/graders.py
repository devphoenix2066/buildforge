from typing import Dict
from .tasks import Component, Status


def grade_easy(components: Dict[str, Component], steps_taken: int, max_steps: int) -> float:
    total = len(components)
    done = sum(1 for c in components.values() if c.status == Status.DONE)
    failed = sum(1 for c in components.values() if c.status == Status.FAILED)

    completion_score = done / total
    speed_bonus = max(0.0, 1.0 - (steps_taken / max_steps)) * 0.3
    failure_penalty = failed * 0.15

    score = completion_score + speed_bonus - failure_penalty
    return round(min(max(score, 0.01), 0.99), 3)


def grade_medium(components: Dict[str, Component], steps_taken: int, max_steps: int,
                 recoveries: int) -> float:
    total = len(components)
    done = sum(1 for c in components.values() if c.status == Status.DONE)
    failed = sum(1 for c in components.values() if c.status == Status.FAILED)

    completion_score = (done / total) * 0.65
    recovery_bonus = min(recoveries * 0.12, 0.25)
    cascade_penalty = max(0.0, (failed - 1) * 0.1)
    speed_bonus = max(0.0, 1.0 - (steps_taken / max_steps)) * 0.25

    score = completion_score + recovery_bonus + speed_bonus - cascade_penalty
    return round(min(max(score, 0.01), 0.99), 3)


def grade_hard(components: Dict[str, Component], steps_taken: int, max_steps: int,
               recoveries: int, correct_pauses: int) -> float:
    total = len(components)
    done = sum(1 for c in components.values() if c.status == Status.DONE)
    failed = sum(1 for c in components.values() if c.status == Status.FAILED)

    completion_score = (done / total) * 0.55
    recovery_bonus = min(recoveries * 0.10, 0.20)
    pause_bonus = min(correct_pauses * 0.06, 0.15)
    cascade_penalty = max(0.0, (failed - 1) * 0.12)
    speed_bonus = max(0.0, 1.0 - (steps_taken / max_steps)) * 0.25

    score = completion_score + recovery_bonus + pause_bonus + speed_bonus - cascade_penalty
    return round(min(max(score, 0.01), 0.99), 3)


def compute_grade(difficulty: str, components: Dict[str, Component],
                  steps_taken: int, max_steps: int,
                  recoveries: int = 0, correct_pauses: int = 0) -> float:
    if difficulty == "easy":
        return grade_easy(components, steps_taken, max_steps)
    elif difficulty == "medium":
        return grade_medium(components, steps_taken, max_steps, recoveries)
    else:
        return grade_hard(components, steps_taken, max_steps, recoveries, correct_pauses)