"""
Shared contracts. EVERYONE imports from here and ONLY from here across module
boundaries. Nobody edits another person's package. Breaking a signature here
= a PR that all 4 people review.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    JUMP = 2
    NOOP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


@dataclass
class Entity:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    kind: str = "platform"          # platform | moving_platform | enemy | collectible | hazard
    extra: dict = field(default_factory=dict)   # e.g. patrol path, period


@dataclass
class GameState:
    """Symbolic ground-truth state (from the env, used by A*/MCTS/RL directly)."""
    player: Entity
    entities: list[Entity]
    goal: tuple[float, float]
    tick: int
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Frame:
    """Raw pixels, for the perception module only."""
    rgb: np.ndarray          # HxWx3 uint8
    tick: int


@dataclass
class PerceptionOutput:
    """CNN+LSTM -> everyone downstream. Owned by Member 1."""
    entities: list[Entity]           # detected/tracked entities, image-space or world-space (document units)
    player_pose: Entity
    predicted_trajectories: dict[str, list[tuple[float, float]]]  # entity_id -> future (x,y) over horizon H
    confidence: float
    embedding: Optional[np.ndarray] = None   # optional latent for RL/planning to consume


@dataclass
class Plan:
    """A*/MCTS -> RL and orchestrator. Owned by Member 2."""
    waypoints: list[tuple[float, float]]
    committed_action: Optional[Action]   # next primitive action MCTS recommends right now
    risk_score: float                    # 0-1, informs orchestrator/RL blending
    replan_reason: Optional[str] = None


@dataclass
class MissionGuidance:
    """LLM/RAG -> everyone. Owned by Member 4."""
    subgoal: str                         # e.g. "prioritize collectible cluster A, avoid hazard B"
    reward_shaping_hints: dict[str, float]  # term -> weight, consumed by RL reward wrapper
    constraints: list[str]               # e.g. "do not backtrack past checkpoint 2"
    rationale: str


class PerceptionModule:
    """Member 1 implements this."""
    def process(self, frame: Frame, history: list[Frame]) -> PerceptionOutput:
        raise NotImplementedError


class PlanningModule:
    """Member 2 implements this."""
    def plan(self, state: GameState, guidance: Optional[MissionGuidance] = None) -> Plan:
        raise NotImplementedError


class PolicyModule:
    """Member 3 implements this."""
    def act(self, state: GameState, plan: Optional[Plan] = None,
            perception: Optional[PerceptionOutput] = None) -> Action:
        raise NotImplementedError

    def train_step(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class MissionModule:
    """Member 4 implements this."""
    def interpret(self, state: GameState, mission_text: str) -> MissionGuidance:
        raise NotImplementedError
