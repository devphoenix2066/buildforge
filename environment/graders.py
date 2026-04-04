from typing import Dict
from .tasks import Component, Status


def grade_easy(components: Dict[str, Component], steps_taken: int, max_steps: int) -> float:
    """
    Easy task grader — 3 components, no failures expected.
    Score based on:
    - Completion rate of components
    - Speed (how quickly everything finished)
    - Penalty for any failures
    """
    total_components = len(components)
    done = sum(1 for c in components.values() if c.status == Status.DONE)
    failed = sum(1 for c in components.values() if c.status == Status.FAILED)

    # Base score — completion ratio
    completion_score = done / total_components

    # Speed bonus — finishing faster gives higher score
    speed_bonus = max(0.0, 1.0 - (steps_taken / max_steps)) * 0.2

    # Failure penalty
    failure_penalty = failed * 0.2

    score = completion_score + speed_bonus - failure_penalty
    return round(min(max(score, 0.0), 1.0), 3)


def grade_medium(components: Dict[str, Component], steps_taken: int, max_steps: int, 
                 recoveries: int) -> float:
    """
    Medium task grader — 5 components, failures expected.
    Score based on:
    - Completion rate
    - Recovery ability (did agent recover from failures?)
    - Cascade prevention (did failures spread?)
    - Speed
    """
    total_components = len(components)
    done = sum(1 for c in components.values() if c.status == Status.DONE)
    failed = sum(1 for c in components.values() if c.status == Status.FAILED)

    # Base score
    completion_score = (done / total_components) * 0.6

    # Recovery bonus — agent gets credit for restarting failed components
    recovery_bonus = min(recoveries * 0.1, 0.2)

    # Cascade penalty — multiple failures mean orchestrator didn't act
    cascade_penalty = max(0.0, (failed - 1) * 0.15)

    # Speed bonus
    speed_bonus = max(0.0, 1.0 - (steps_taken / max_steps)) * 0.2

    score = completion_score + recovery_bonus + speed_bonus - cascade_penalty
    return round(min(max(score, 0.0), 1.0), 3)


def grade_hard(components: Dict[str, Component], steps_taken: int, max_steps: int,
               recoveries: int, correct_pauses: int) -> float:
    """
    Hard task grader — 5 components, multiple simultaneous failures.
    Score based on:
    - Completion rate
    - Recovery speed and accuracy
    - Correct pause decisions (pausing blocked components)
    - Cascade prevention
    - Overall efficiency
    """
    total_components = len(components)
    done = sum(1 for c in components.values() if c.status == Status.DONE)
    failed = sum(1 for c in components.values() if c.status == Status.FAILED)

    # Base score
    completion_score = (done / total_components) * 0.5

    # Recovery bonus
    recovery_bonus = min(recoveries * 0.08, 0.15)

    # Correct pause bonus — pausing blocked tasks is smart orchestration
    pause_bonus = min(correct_pauses * 0.05, 0.15)

    # Cascade penalty — harsher on hard difficulty
    cascade_penalty = max(0.0, (failed - 1) * 0.2)

    # Speed bonus
    speed_bonus = max(0.0, 1.0 - (steps_taken / max_steps)) * 0.2

    score = completion_score + recovery_bonus + pause_bonus + speed_bonus - cascade_penalty
    return round(min(max(score, 0.0), 1.0), 3)


def compute_grade(difficulty: str, components: Dict[str, Component],
                  steps_taken: int, max_steps: int,
                  recoveries: int = 0, correct_pauses: int = 0) -> float:
    """
    Single entry point for grading.
    Called at episode end by the environment.
    """
    if difficulty == "easy":
        return grade_easy(components, steps_taken, max_steps)
    elif difficulty == "medium":
        return grade_medium(components, steps_taken, max_steps, recoveries)
    else:
        return grade_hard(components, steps_taken, max_steps, recoveries, correct_pauses)