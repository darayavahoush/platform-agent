"""Integration loop: ties Perception -> Planning -> Policy together each tick,
with MissionGuidance refreshed periodically (LLM calls are expensive — don't
call every tick). This file is the one place allowed to import all 4 member
packages; nobody else should cross-import.
"""
from core.env import BaseEnv, MockPlatformEnv
from core.interfaces import PerceptionModule, PlanningModule, PolicyModule, MissionModule, Frame


def run_episode(env: BaseEnv,
                 perception: PerceptionModule,
                 planning: PlanningModule,
                 policy: PolicyModule,
                 mission: MissionModule,
                 mission_text: str,
                 guidance_refresh_every: int = 30,
                 max_steps: int = 1000):
    state, frame = env.reset()
    history: list[Frame] = [frame]
    guidance = mission.interpret(state, mission_text)

    for t in range(max_steps):
        if t % guidance_refresh_every == 0 and t > 0:
            guidance = mission.interpret(state, mission_text)

        perception_out = perception.process(frame, history)
        plan = planning.plan(state, guidance)
        action = policy.act(state, plan, perception_out)

        state, frame, reward, done, info = env.step(action)
        history.append(frame)
        history = history[-8:]
        if done:
            break
    return state


if __name__ == "__main__":
    print("Wire in real implementations here once each module lands; "
          "this stub only proves the interfaces click together.")
