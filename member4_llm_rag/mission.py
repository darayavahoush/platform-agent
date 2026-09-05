"""Member 4: LLM/SLM + RAG for high-level mission interpretation and strategic
guidance, AND system integration/orchestration (wires modules 1-3 together
into one control loop). Implement MissionModule from core.interfaces.

Approach: a locally-hosted small instruct model via Ollama, PROMPTED not
fine-tuned -- gets a working, interface-compliant MissionModule today
without a labeled dataset or training pipeline. Grounded with a small RAG
layer over `kb/*.md` (platform mechanics, hazards, curriculum notes) so
guidance reflects this project''s actual entities/mechanics instead of
generic platformer guesses.

Never raises out of interpret() -- an unreachable model or malformed
output falls back to a safe no-op guidance so a bad LLM call can''t break
the orchestrator loop or a teammate''s training run.
"""
from __future__ import annotations

from typing import Optional

from core.interfaces import GameState, MissionGuidance, MissionModule
from member4_llm_rag.llm_client import OllamaClient
from member4_llm_rag.rag import Retriever

SYSTEM_PROMPT = """You are a mission-interpretation module for a 2D \
platformer navigation agent. Given the current game state and a mission \
instruction, output ONLY a JSON object with exactly these keys:

{
  "subgoal": "<short_snake_case_label>",
  "reward_shaping_hints": {"<name>": <float between -1.0 and 1.0>, ...},
  "constraints": ["<short constraint string>", ...],
  "rationale": "<one or two sentence explanation>"
}

Rules:
- subgoal must be a short snake_case phrase, e.g. "chain_jumps_to_ledge".
- reward_shaping_hints values must stay between -1.0 and 1.0.
- constraints should be short machine-usable strings, not prose.
- Do not include any text outside the JSON object.
- Use the provided reference notes if relevant; ignore them if not.
"""


class RAGMissionInterpreter(MissionModule):
    def __init__(
        self,
        knowledge_base_path: str = "member4_llm_rag/kb",
        model: str = "qwen2.5:3b-instruct",
        top_k: int = 3,
    ):
        self.llm = OllamaClient(model=model)
        self._kb_path = knowledge_base_path
        self._top_k = top_k
        self.retriever: Optional[Retriever] = None  # lazy-loaded, see _get_retriever

    def _get_retriever(self) -> Retriever:
        # Lazy init avoids loading the embedding model + FAISS index when
        # interpret() is never called (e.g. in tests that stub this class out).
        if self.retriever is None:
            self.retriever = Retriever(self._kb_path)
        return self.retriever

    def _summarize_state(self, state: GameState) -> str:
        player = state.player
        entity_lines = [
            f"  - kind={e.kind} pos=({e.x:.1f},{e.y:.1f}) vel=({e.vx:.1f},{e.vy:.1f}) extra={e.extra}"
            for e in state.entities
        ]
        return (
            f"tick={state.tick}\n"
            f"player pos=({player.x:.1f},{player.y:.1f}) vel=({player.vx:.1f},{player.vy:.1f})\n"
            f"goal={state.goal}\n"
            f"info={state.info}\n"
            f"entities:\n" + ("\n".join(entity_lines) if entity_lines else "  (none)")
        )

    def _build_prompt(self, mission_text: str, state_summary: str, context_chunks) -> str:
        context_text = (
            "\n\n".join(f"[{c.source}]\n{c.text}" for c in context_chunks)
            if context_chunks
            else "(no relevant reference notes found)"
        )
        return (
            f"MISSION INSTRUCTION:\n{mission_text}\n\n"
            f"CURRENT STATE:\n{state_summary}\n\n"
            f"REFERENCE NOTES:\n{context_text}\n\n"
            f"Respond with the JSON object described in the system prompt."
        )

    def _parse_guidance(self, raw: dict) -> MissionGuidance:
        subgoal = str(raw.get("subgoal", "reach_goal"))

        hints = {}
        for k, v in (raw.get("reward_shaping_hints", {}) or {}).items():
            try:
                hints[str(k)] = max(-1.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                continue  # drop a malformed hint rather than fail the whole parse

        constraints = [
            str(c) for c in (raw.get("constraints", []) or []) if isinstance(c, (str, int, float))
        ]

        return MissionGuidance(
            subgoal=subgoal,
            reward_shaping_hints=hints,
            constraints=constraints,
            rationale=str(raw.get("rationale", "")),
        )

    def _fallback_guidance(self, reason: str) -> MissionGuidance:
        return MissionGuidance(
            subgoal="reach_goal",
            reward_shaping_hints={},
            constraints=[],
            rationale=f"fallback (SLM unavailable or invalid output): {reason}",
        )

    def interpret(self, state: GameState, mission_text: str) -> MissionGuidance:
        try:
            state_summary = self._summarize_state(state)
            context_chunks = self._get_retriever().retrieve(mission_text, k=self._top_k)
            prompt = self._build_prompt(mission_text, state_summary, context_chunks)
            raw = self.llm.generate_json(prompt, system=SYSTEM_PROMPT)
            return self._parse_guidance(raw)
        except Exception as e:  # noqa: BLE001 - guidance must always return something
            return self._fallback_guidance(str(e))
