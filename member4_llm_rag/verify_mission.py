"""Manual verification for RAGMissionInterpreter, separate from
tests/test_smoke.py (which uses a DummyMission and doesn't touch Ollama or
the RAG index). Run this after setup to confirm the real implementation
loads and returns a valid MissionGuidance.

Run with:
    python -m member4_llm_rag.verify_mission
"""
from core.interfaces import Entity, GameState
from member4_llm_rag.mission import RAGMissionInterpreter


def make_mock_state() -> GameState:
    player = Entity(x=10.0, y=50.0, vx=1.0, vy=0.0, kind="platform")
    moving_platform = Entity(
        x=40.0, y=45.0, vx=0.5, vy=0.0, kind="moving_platform",
        extra={"period": 60, "amplitude_tiles": 2},
    )
    enemy = Entity(x=25.0, y=55.0, vx=0.0, vy=0.0, kind="enemy", extra={"patrol_tiles": (20, 30)})
    return GameState(
        player=player,
        entities=[moving_platform, enemy],
        goal=(100.0, 40.0),
        tick=250,
        done=False,
        info={"difficulty": 0.6},
    )


def main():
    print("Loading RAGMissionInterpreter (pulls the embedding model on first run)...")
    mission = RAGMissionInterpreter()

    print("LLM reachable:", mission.llm.is_available())

    state = make_mock_state()
    mission_text = "Reach the goal while avoiding the enemy near the moving platform."
    guidance = mission.interpret(state, mission_text)

    print()
    print("subgoal:", guidance.subgoal)
    print("reward_shaping_hints:", guidance.reward_shaping_hints)
    print("constraints:", guidance.constraints)
    print("rationale:", guidance.rationale)


if __name__ == "__main__":
    main()