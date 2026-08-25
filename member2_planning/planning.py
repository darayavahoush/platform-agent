"""Member 2: A* over a discretized level graph for global route planning +
MCTS for short-horizon tactical decisions (jump timing around moving
platforms/enemies). Implement PlanningModule from core.interfaces.
"""
from core.interfaces import PlanningModule, Plan, GameState, MissionGuidance
from typing import Optional


class AStarPlanner:
    """Coarse global path over a static/quasi-static graph of GameState.entities."""
    def build_graph(self, state: GameState):
        raise NotImplementedError

    def search(self, state: GameState, goal: tuple[float, float]) -> list[tuple[float, float]]:
        raise NotImplementedError


class MCTSPlanner:
    """Tactical search over primitive Actions given predicted trajectories
    (from PerceptionOutput.predicted_trajectories, passed in via state.info
    or a direct arg) to pick the next committed action + risk score."""
    def search(self, state: GameState, waypoint: tuple[float, float]):
        raise NotImplementedError


class HybridPlanningModule(PlanningModule):
    def __init__(self):
        self.astar = AStarPlanner()
        self.mcts = MCTSPlanner()

    def plan(self, state: GameState, guidance: Optional[MissionGuidance] = None) -> Plan:
        raise NotImplementedError
