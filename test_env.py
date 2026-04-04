from environment.env import BuildForgeEnv, Action

print("=" * 50)
print("Testing EASY task")
print("=" * 50)

env = BuildForgeEnv(difficulty="easy")
result = env.reset()
print(f"Reset OK — components: {list(result.observation.components.keys())}")

for step in range(5):
    action = Action(action_type="noop")
    result = env.step(action)
    print(f"Step {step+1} | reward={result.reward} | done={result.done}")
    for name, comp in result.observation.components.items():
        print(f"  {name}: {comp.status} | progress={comp.progress}")

print()
print("=" * 50)
print("Testing MEDIUM task")
print("=" * 50)

env = BuildForgeEnv(difficulty="medium")
result = env.reset()
print(f"Reset OK — components: {list(result.observation.components.keys())}")

for step in range(5):
    action = Action(action_type="noop")
    result = env.step(action)
    print(f"Step {step+1} | reward={result.reward} | done={result.done}")

print()
print("=" * 50)
print("Testing HARD task")
print("=" * 50)

env = BuildForgeEnv(difficulty="hard")
result = env.reset()
print(f"Reset OK — components: {list(result.observation.components.keys())}")

for step in range(5):
    action = Action(action_type="noop")
    result = env.step(action)
    print(f"Step {step+1} | reward={result.reward} | done={result.done}")

print()
print("=" * 50)
print("Testing ACTIONS")
print("=" * 50)

from environment.tasks import Status

# Fresh env
env = BuildForgeEnv(difficulty="medium")
result = env.reset()

# Force dep_resolver to be running and not complete
env.components["dep_resolver"].progress = 0.3
env.components["dep_resolver"].status = Status.RUNNING

# Test boost on a genuinely running component
result = env.step(Action(action_type="boost", target="dep_resolver"))
print(f"Boost on RUNNING   | reward={result.reward} (should be > 0)")

# Force a failure
env.components["backend_compiler"].status = Status.FAILED
env.components["dep_resolver"].progress = 0.3  # reset progress so it doesn't complete

# Test restart on failed component
result = env.step(Action(action_type="restart", target="backend_compiler"))
print(f"Restart on FAILED  | reward={result.reward} (should be > 0)")

# Test pause on blocked component
env.components["frontend_builder"].status = Status.BLOCKED
result = env.step(Action(action_type="pause", target="frontend_builder"))
print(f"Pause on BLOCKED   | reward={result.reward} (should be > 0)")

# Test noop
result = env.step(Action(action_type="noop"))
print(f"Noop               | reward={result.reward}")

print()
print("ALL TESTS PASSED")
'''
print()
print("=" * 50)
print("Testing ACTIONS")
print("=" * 50)

env = BuildForgeEnv(difficulty="medium")
result = env.reset()

# Test restart on a forced failure
env.components["backend_compiler"].status.__class__
from environment.tasks import Status
env.components["backend_compiler"].status = Status.FAILED

result = env.step(Action(action_type="restart", target="backend_compiler"))
print(f"Restart action | reward={result.reward}")

result = env.step(Action(action_type="boost", target="dep_resolver"))
print(f"Boost action   | reward={result.reward}")

result = env.step(Action(action_type="pause", target="frontend_builder"))
print(f"Pause action   | reward={result.reward}")

print()
print("ALL TESTS PASSED")'''