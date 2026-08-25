"""Member 4: LLM/SLM + RAG for high-level mission interpretation and strategic
guidance, AND system integration/orchestration (wires modules 1-3 together
into one control loop). Implement MissionModule from core.interfaces.
"""
from core.interfaces import MissionModule, MissionGuidance, GameState


class RAGMissionInterpreter(MissionModule):
    def __init__(self, knowledge_base_path: str = "member4_llm_rag/kb"):
        # TODO: retriever (e.g. embedding index over level docs / strategy notes)
        # TODO: LLM/SLM call (local small model or API) grounded on retrieved context
        pass

    def interpret(self, state: GameState, mission_text: str) -> MissionGuidance:
        raise NotImplementedError
