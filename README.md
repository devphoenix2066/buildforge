---
title: BuildForge
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# BuildForge 🏗️

A dynamic build orchestration environment for the OpenEnv Hackathon.

## Overview
BuildForge simulates a parallel software build pipeline where an AI agent 
must orchestrate interdependent build components in real time. When components 
fail, the agent must detect, recover, and prevent cascade failures.

## Tasks
| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| simple_build | Easy | 20 | 3 components, simple dependencies |
| cascading_failure | Medium | 30 | 5 components, random failures |
| race_condition_recovery | Hard | 40 | 5 components, multiple simultaneous failures |

## Components
- `dep_resolver` — resolves dependencies (must finish first)
- `backend_compiler` — compiles backend code
- `frontend_builder` — builds frontend
- `static_analyzer` — checks code quality
- `test_runner` — runs tests (needs backend + static_analyzer)

## Action Space
| Action | Target | Description |
|---|---|---|
| boost | component | Speed up a running component |
| pause | component | Pause a blocked component |
| resume | component | Resume a paused component |
| restart | component | Restart a failed component |
| noop | null | Do nothing |

## Observation Space
- `components` — status, progress, reward for each component
- `blocked_components` — list of currently blocked components
- `critical_failure` — true if 2+ components failed simultaneously
- `time_elapsed` — steps taken so far

## Reward Function
- `+0.1` progress per step
- `+0.5` component completion bonus
- `+0.3` restart recovery bonus
- `+0.2` correct pause bonus
- `-0.1` blocked component idle penalty
- `-0.3` component failure penalty

## Setup
```bash
docker build -t buildforge .
docker run -p 7860:7860 buildforge
```

## Inference
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=https://prithviRK-buildforge.hf.space
python inference.py
```

## Baseline Scores
| Task | Score |
|---|---|
| simple_build | 1.000 |
| cascading_failure | ~0.71 |
| race_condition_recovery | ~0.70 |