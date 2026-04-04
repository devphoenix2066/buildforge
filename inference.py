"""
BuildForge Inference Script
===========================
Baseline agent that runs against the BuildForge environment.
Uses an LLM via OpenAI-compatible API to make orchestration decisions.

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

# ---------- Config ----------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "buildforge"
MAX_STEPS    = 40
SUCCESS_SCORE_THRESHOLD = 0.4

TASKS = [
    {"name": "simple_build",           "difficulty": "easy"},
    {"name": "cascading_failure",      "difficulty": "medium"},
    {"name": "race_condition_recovery","difficulty": "hard"},
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
      - dep_resolver:     resolves dependencies (must finish first)
      - backend_compiler: compiles backend code
      - frontend_builder: builds frontend
      - static_analyzer:  checks code quality
      - test_runner:      runs tests (needs backend + static_analyzer done)

    At each step you observe the status of all components and must choose ONE action:

    Actions:
      boost   <component>  — speed up a RUNNING component
      pause   <component>  — pause a BLOCKED component (saves resources)
      resume  <component>  — resume a PAUSED component
      restart <component>  — restart a FAILED component (critical!)
      noop                 — do nothing

    Rules:
      - Always restart FAILED components immediately
      - Pause BLOCKED components that are waiting on unfinished dependencies
      - Boost components that are close to completion (progress > 0.7)
      - Use noop only when everything is running smoothly

    Respond with JSON only. No explanation. Format:
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
        Step: {step}
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
            temperature=0.2,
            max_tokens=50,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps in ```json
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        # Validate
        if parsed.get("action_type") not in ["boost","pause","resume","restart","noop"]:
            return {"action_type": "noop", "target": None}
        return parsed
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"action_type": "noop", "target": None}


# ---------- Fallback Rule-Based Agent ----------
def rule_based_action(obs: dict) -> dict:
    components = obs["components"]
    from environment.tasks import DEPENDENCIES

    # Priority 1 — restart any failed component immediately
    for name, data in components.items():
        if data["status"] == "failed":
            return {"action_type": "restart", "target": name}

    # Priority 2 — resume paused components if deps are done
    for name, data in components.items():
        if data["status"] == "paused":
            deps = DEPENDENCIES.get(name, [])
            deps_done = all(
                components[d]["status"] == "done"
                for d in deps
                if d in components
            )
            if deps_done:
                return {"action_type": "resume", "target": name}

    # Priority 3 — boost any running component aggressively
    # First boost whoever is closest to completion
    best_name = None
    best_progress = -1
    for name, data in components.items():
        if data["status"] == "running" and data["progress"] > best_progress:
            best_progress = data["progress"]
            best_name = name

    if best_name and best_progress > 0.4:
        return {"action_type": "boost", "target": best_name}

    # Priority 4 — pause blocked components
    for name, data in components.items():
        if data["status"] == "blocked":
            return {"action_type": "pause", "target": name}

    # Priority 5 — boost anything running even at low progress
    if best_name:
        return {"action_type": "boost", "target": best_name}

    return {"action_type": "noop", "target": None}
    
# ---------- Main ----------
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

            # Get action from LLM or fallback
            if client:
                action = get_agent_action(client, obs, step, last_reward)
            else:
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
    # Try to init LLM client — fall back to rule-based if no API key
    client = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            print(f"[DEBUG] LLM client initialized: {MODEL_NAME}", flush=True)
        except Exception as e:
            print(f"[DEBUG] LLM init failed: {e} — using rule-based agent", flush=True)
    else:
        print("[DEBUG] No API key found — using rule-based agent", flush=True)

    for task in TASKS:
        print(f"\n[DEBUG] Running task: {task['name']}", flush=True)
        run_task(client, task)
        print(flush=True)


if __name__ == "__main__":
    main()