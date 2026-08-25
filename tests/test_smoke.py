"""Smoke test: proves the interface contracts click together end to end using
trivial dummy implementations. Run this on day 1, before any real model exists.
"""
from core.env import MockPlatformEnv
from core.interfaces import (PerceptionModule, PlanningModule, PolicyModule, MissionModule,
                              PerceptionOutput, Plan, MissionGuidance, Action, GameState, Frame)
from member4_llm_rag.orchestrator import run_episode


class DummyPerception(PerceptionModule):
    def process(self, frame, history):
        return PerceptionOutput(entities=[], player_pose=None, predicted_trajectories={}, confidence=0.0)


class DummyPlanning(PlanningModule):
    def plan(self, state: GameState, guidance=None) -> Plan:
        return Plan(waypoints=[state.goal], committed_action=Action.RIGHT, risk_score=0.0)


class DummyPolicy(PolicyModule):
    def act(self, state, plan=None, perception=None) -> Action:
        return Action.RIGHT


class DummyMission(MissionModule):
    def interpret(self, state, mission_text) -> MissionGuidance:
        return MissionGuidance(subgoal="reach goal", reward_shaping_hints={}, constraints=[], rationale="dummy")


def test_pipeline_runs():
    env = MockPlatformEnv()
    final_state = run_episode(env, DummyPerception(), DummyPlanning(), DummyPolicy(), DummyMission(),
                               mission_text="reach the goal", max_steps=50)
    assert final_state.tick <= 50


if __name__ == "__main__":
    test_pipeline_runs()
    print("smoke test passed")
