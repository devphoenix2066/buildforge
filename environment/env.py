import numpy as np
from typing import Dict, Optional, Tuple
from pydantic import BaseModel
from .tasks import Component, Status, make_components, DEPENDENCIES
from .graders import compute_grade


# ---------- Pydantic Models (OpenEnv spec requires typed models) ----------

class ComponentState(BaseModel):
    status: str
    progress: float
    reward_so_far: float


class Observation(BaseModel):
    components: Dict[str, ComponentState]
    blocked_components: list
    critical_failure: bool
    time_elapsed: int


class Action(BaseModel):
    action_type: str        # "boost" | "pause" | "resume" | "restart" | "noop"
    target: Optional[str] = None   # which component to act on


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


# ---------- Core Environment ----------

class BuildForgeEnv:

    DIFFICULTIES = ["easy", "medium", "hard"]
    MAX_STEPS = {
        "easy":   20,
        "medium": 30,
        "hard":   40,
    }

    def __init__(self, difficulty: str = "easy"):
        assert difficulty in self.DIFFICULTIES
        self.difficulty = difficulty
        self.max_steps = self.MAX_STEPS[difficulty]
        self.components: Dict[str, Component] = {}
        self.steps_taken = 0
        self.recoveries = 0
        self.correct_pauses = 0
        self.done = False

    # ---------- OpenEnv Interface ----------

    def reset(self) -> StepResult:
        self.components = make_components(self.difficulty)
        self.steps_taken = 0
        self.recoveries = 0
        self.correct_pauses = 0
        self.done = False
        return StepResult(
            observation=self._get_observation(),
            reward=0.0,
            done=False,
            info={"message": f"BuildForge {self.difficulty} task started"}
        )

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"message": "Episode already done"}
            )

        # Apply agent action
        action_reward = self._apply_action(action)

        # Step all components forward
        step_reward = 0.0
        for comp in self.components.values():
            step_reward += comp.step(self.components)

        # Total reward this step
        total_reward = round(action_reward + step_reward, 4)
        # Ensure action rewards are visible even when step rewards are negative
        if action_reward > 0:
            total_reward = round(max(total_reward, action_reward * 0.5), 4)

        self.steps_taken += 1

        # Check termination
        self.done = self._check_done()

        # Build result
        obs = self._get_observation()
        info = self._build_info()

        return StepResult(
            observation=obs,
            reward=total_reward,
            done=self.done,
            info=info
        )

    def state(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "components": {
                name: {
                    "status": c.status.value,
                    "progress": round(c.progress, 3),
                    "reward_so_far": round(c.reward_so_far, 3)
                }
                for name, c in self.components.items()
            }
        }

    def close(self):
        pass

    # ---------- Internal Helpers ----------

    def _apply_action(self, action: Action) -> float:
        atype = action.action_type
        target = action.target

        # noop
        if atype == "noop" or target is None:
            return 0.0

        # Invalid target
        if target not in self.components:
            return -0.1

        comp = self.components[target]

        if atype == "boost":
            if comp.status == Status.RUNNING:
                comp.speed = min(comp.speed * 1.3, 0.3)
                return 0.3
            elif comp.status == Status.BLOCKED:
                return -0.05
            return 0.0

        elif atype == "pause":
            # Pausing a blocked component is smart
            if comp.status == Status.BLOCKED:
                comp.status = Status.PAUSED
                self.correct_pauses += 1
                return 0.2
            elif comp.status == Status.RUNNING:
                comp.status = Status.PAUSED
                return -0.05  # slight penalty for pausing running component
            return 0.0

        elif atype == "resume":
            if comp.status == Status.PAUSED:
                comp.status = Status.BLOCKED  # will re-check deps on next step
                return 0.05
            return 0.0

        elif atype == "restart":
            if comp.status == Status.FAILED:
                comp.status = Status.BLOCKED
                comp.progress = 0.0
                self.recoveries += 1
                return 0.3  # recovery bonus
            return -0.1  # penalty for restarting non-failed component

        return 0.0

    def _get_observation(self) -> Observation:
        blocked = [
            name for name, c in self.components.items()
            if c.status == Status.BLOCKED
        ]
        critical = sum(
            1 for c in self.components.values()
            if c.status == Status.FAILED
        ) >= 2

        return Observation(
            components={
                name: ComponentState(
                    status=c.status.value,
                    progress=round(c.progress, 3),
                    reward_so_far=round(c.reward_so_far, 3)
                )
                for name, c in self.components.items()
            },
            blocked_components=blocked,
            critical_failure=critical,
            time_elapsed=self.steps_taken
        )

    def _check_done(self) -> bool:
        # All done
        if all(c.status == Status.DONE for c in self.components.values()):
            return True
        # Max steps reached
        if self.steps_taken >= self.max_steps:
            return True
        # Critical cascade — 3+ failures
        if sum(1 for c in self.components.values() 
               if c.status == Status.FAILED) >= 3:
            return True
        return False

    def _build_info(self) -> dict:
        score = compute_grade(
            difficulty=self.difficulty,
            components=self.components,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps,
            recoveries=self.recoveries,
            correct_pauses=self.correct_pauses
        )
        return {
            "score": score,
            "recoveries": self.recoveries,
            "correct_pauses": self.correct_pauses,
            "steps_taken": self.steps_taken
        }