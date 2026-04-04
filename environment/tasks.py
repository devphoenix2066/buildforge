import random
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class Status(Enum):
    BLOCKED   = "blocked"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    PAUSED    = "paused"

# Dependency map — who needs who to be DONE before they can start
DEPENDENCIES = {
    "dep_resolver":    [],
    "backend_compiler": ["dep_resolver"],
    "frontend_builder": ["dep_resolver"],
    "static_analyzer":  ["dep_resolver"],
    "test_runner":      ["backend_compiler", "static_analyzer"],
}

@dataclass
class Component:
    name: str
    progress: float = 0.0          # 0.0 to 1.0
    status: Status = Status.BLOCKED
    reward_so_far: float = 0.0
    failure_prob: float = 0.05     # chance of random failure per step
    speed: float = 0.1             # how much progress per step
    
    def step(self, components: dict) -> float:
        """
        Advance this component by one step.
        Returns the reward signal for this step.
        """
        # If done or failed, nothing happens
        if self.status in (Status.DONE, Status.FAILED, Status.PAUSED):
            return 0.0

        # Check if dependencies are met
        deps = DEPENDENCIES[self.name]
        for dep in deps:
            if components[dep].status != Status.DONE:
                self.status = Status.BLOCKED
                return -0.1  # penalty for sitting idle

        # Dependencies met — set to running
        if self.status == Status.BLOCKED:
            self.status = Status.RUNNING

        # Random failure
        if random.random() < self.failure_prob:
            self.status = Status.FAILED
            return -0.3  # failure penalty

        # Make progress
        self.progress = min(1.0, self.progress + self.speed * random.uniform(0.8, 1.2))
        reward = 0.1  # progress reward

        # Completed
        if self.progress >= 1.0:
            self.status = Status.DONE
            reward += 0.5  # completion bonus

        self.reward_so_far += reward
        return reward


def make_components(difficulty: str = "easy") -> dict:
    """
    Returns a fresh set of components based on difficulty.
    
    easy   — 3 components, low failure prob
    medium — 5 components, moderate failure prob
    hard   — 5 components, high failure prob, faster cascade
    """
    if difficulty == "easy":
        return {
            "dep_resolver":     Component("dep_resolver",
                                          status=Status.RUNNING,
                                          failure_prob=0.0,
                                          speed=0.2),
            "backend_compiler": Component("backend_compiler",
                                          failure_prob=0.02,
                                          speed=0.15),
            "frontend_builder": Component("frontend_builder",
                                          failure_prob=0.02,
                                          speed=0.15),
        }

    elif difficulty == "medium":
        return {
            "dep_resolver":     Component("dep_resolver",
                                          status=Status.RUNNING,
                                          failure_prob=0.0,
                                          speed=0.15),
            "backend_compiler": Component("backend_compiler",
                                          failure_prob=0.05,
                                          speed=0.12),
            "frontend_builder": Component("frontend_builder",
                                          failure_prob=0.05,
                                          speed=0.12),
            "static_analyzer":  Component("static_analyzer",
                                          failure_prob=0.05,
                                          speed=0.12),
            "test_runner":      Component("test_runner",
                                          failure_prob=0.08,
                                          speed=0.10),
        }

    else:  # hard
        return {
            "dep_resolver":     Component("dep_resolver",
                                          status=Status.RUNNING,
                                          failure_prob=0.02,
                                          speed=0.12),
            "backend_compiler": Component("backend_compiler",
                                          failure_prob=0.12,
                                          speed=0.10),
            "frontend_builder": Component("frontend_builder",
                                          failure_prob=0.12,
                                          speed=0.10),
            "static_analyzer":  Component("static_analyzer",
                                          failure_prob=0.12,
                                          speed=0.10),
            "test_runner":      Component("test_runner",
                                          failure_prob=0.15,
                                          speed=0.08),
        }