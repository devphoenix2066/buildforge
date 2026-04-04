import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from environment.env import BuildForgeEnv, Action, StepResult

app = FastAPI(title="BuildForge OpenEnv Server")

# Global environment instance
env: Optional[BuildForgeEnv] = None


# ---------- Request Models ----------

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"


class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None


# ---------- Routes ----------

@app.post("/reset")
def reset(req: ResetRequest):
    global env
    difficulty = req.difficulty or "easy"
    if difficulty not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
    env = BuildForgeEnv(difficulty=difficulty)
    result = env.reset()
    return result.dict()


@app.post("/step")
def step(req: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    action = Action(action_type=req.action_type, target=req.target)
    result = env.step(action)
    return result.dict()


@app.get("/state")
def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "simple_build",
                "difficulty": "easy",
                "description": "3 components, simple dependencies, no failures expected",
                "max_steps": 20,
                "score_range": [0.0, 1.0]
            },
            {
                "name": "cascading_failure",
                "difficulty": "medium",
                "description": "5 components, random failures occur, agent must recover",
                "max_steps": 30,
                "score_range": [0.0, 1.0]
            },
            {
                "name": "race_condition_recovery",
                "difficulty": "hard",
                "description": "5 components, multiple simultaneous failures, agent must prioritize and recover",
                "max_steps": 40,
                "score_range": [0.0, 1.0]
            }
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()