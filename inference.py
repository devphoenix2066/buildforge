"""
BuildForge Inference Script
===========================
Hybrid LLM + rule-based agent for BuildForge hackathon.
Uses LLM decisions, but enforces critical rules for reliability.

Emits structured stdout logs in OpenEnv format:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import os
import json
import requests
import textwrap
from typing import List, Optional
from openai import OpenAI

from environment.tasks import DEPENDENCIES, Status

# ---------- Config ----------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "buildforge"
MAX_STEPS    = 40
SUCCESS_SCORE_THRESHOLD = 0.4

TASKS = [
    {"name": "simple_build",            "difficulty": "easy"},
    {"name": "cascading_failure",       "difficulty": "medium"},
    {"name": "race_condition_recovery", "difficulty": "hard"},
    {"name": "hotfix_deploy",           "difficulty": "hotfix"},
]

# ---------- Logging ----------
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------- Environment API ----------
def env_reset(difficulty: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty})
    r.raise_for_status()
    return r.json()

def env_step(action_type: str, target: Optional[str] = None) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"action_type": action_type, "target": target})
    r.raise_for_status()
    return r.json()

def env_state() -> dict:
    r = requests.get(f"{ENV_URL}/state")
    r.raise_for_status()
    return r.json()

# ---------- LLM Agent ----------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent build orchestration agent.
    You manage a parallel software build pipeline with these components:
      - dep_resolver
      - backend_compiler
      - frontend_builder
      - static_analyzer
      - test_runner

    Actions:
      boost   <component>  — speed up a RUNNING component
      pause   <component>  — pause a BLOCKED component
      resume  <component>  — resume a PAUSED component
      restart <component>  — restart a FAILED component
      noop                 — do nothing

    Critical rules:
      - Never choose noop if a component can be boosted.
      - Always restart FAILED components immediately.
      - Always resume PAUSED components if dependencies are done.

    Respond with JSON only:
    {"action_type": "restart", "target": "backend_compiler"}
    or
    {"action_type": "noop", "target": null}
""").strip()


def build_user_prompt(obs: dict, step: int, last_reward: float) -> str:
    components = obs["components"]
    blocked    = obs["blocked_components"]
    critical   = obs["critical_failure"]
    elapsed    = obs["time_elapsed"]

    comp_lines = []
    for name, data in components.items():
        comp_lines.append(
            f"  {name}: status={data['status']} progress={data['progress']:.2f} reward={data['reward_so_far']:.2f}"
        )
    comp_str = "\n".join(comp_lines)

    return textwrap.dedent(f"""
        Step: {step}/{MAX_STEPS}
        Last reward: {last_reward:.2f}
        Time elapsed: {elapsed}
        Critical failure: {critical}
        Blocked: {blocked}

        Components:
        {comp_str}

        What is your next action?
    """).strip()


def get_agent_action(client: OpenAI, obs: dict, step: int, last_reward: float) -> dict:
    prompt = build_user_prompt(obs, step, last_reward)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=50,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

        # Validate
        if parsed.get("action_type") not in ["boost","pause","resume","restart","noop"]:
            return {"action_type": "noop", "target": None}
        return parsed
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action_type": "noop", "target": None}

# ---------- Hybrid Rule Enforcement ----------
def enforce_rules(obs: dict, action: dict) -> dict:
    components = obs["components"]

    # Restart any failed component immediately
    for name, c in components.items():
        if c["status"] == Status.FAILED.value:
            return {"action_type": "restart", "target": name}

    # Resume paused components if deps done
    for name, c in components.items():
        if c["status"] == Status.PAUSED.value:
            deps_done = all(components[d]["status"] == Status.DONE.value for d in DEPENDENCIES.get(name, []))
            if deps_done:
                return {"action_type": "resume", "target": name}

    # Boost running components with progress > 0.5
    for name, c in components.items():
        if c["status"] == Status.RUNNING.value and c["progress"] > 0.5:
            return {"action_type": "boost", "target": name}

    # Otherwise, keep LLM choice
    return action

# ---------- Main Loop ----------
def run_task(client: Optional[OpenAI], task: dict):
    name       = task["name"]
    difficulty = task["difficulty"]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(difficulty)
        obs        = result["observation"]
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            if client:
                action = get_agent_action(client, obs, step, last_reward)
                action = enforce_rules(obs, action)
            else:
                # Fallback rule-based agent
                from environment.tasks import rule_based_action
                action = rule_based_action(obs)

            action_type = action.get("action_type", "noop")
            target      = action.get("target", None)

            # Step environment
            result      = env_step(action_type, target)
            obs         = result["observation"]
            reward      = result.get("reward", 0.0)
            done        = result.get("done", False)
            info        = result.get("info", {})
            error       = None

            rewards.append(reward)
            steps_taken  = step
            last_reward  = reward
            score        = info.get("score", 0.0)

            action_str = f"{action_type}({target})" if target else action_type
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    client = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            print(f"[DEBUG] LLM initialized: {MODEL_NAME}", flush=True)
        except Exception as e:
            print(f"[DEBUG] LLM init failed: {e} — using rule-based agent", flush=True)
    else:
        print("[DEBUG] No API key — using rule-based agent", flush=True)

    for task in TASKS:
        print(f"\n[DEBUG] Running {task['name']}", flush=True)
        run_task(client, task)
        print(flush=True)


if __name__ == "__main__":
    main()