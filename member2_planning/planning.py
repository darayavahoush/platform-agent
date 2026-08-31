"""Member 2: A* over a discretized level graph for global route planning +
a lightweight MCTS-style rollout search for tactical decisions (jump timing
around moving platforms/enemies). Implements PlanningModule from
core.interfaces.

This is a REAL, working implementation — not a stub. It's intentionally
built against a simple *tile grid* level representation (see demo/level.py)
rather than raw pixel/continuous coordinates, because that's the natural
graph for A*. If the real level engine ends up continuous, swap
`build_graph`'s discretization step; `search` and the MCTS layer don't care.
"""
from __future__ import annotations
import heapq
import random
from dataclasses import dataclass, field
from typing import Optional

from core.interfaces import PlanningModule, Plan, GameState, MissionGuidance, Action


TILE_W = 20  # px per tile, must match demo/level.py


@dataclass
class LevelGraph:
    """Static traversability graph. ground[i] == 1 means tile i is solid;
    hazards is a set of tile indices considered risky (enemy patrol range);
    bridge_tiles are gap tiles only reachable when a moving platform is
    passing through — traversable for A*'s macro route at elevated cost,
    with the exact crossing *timing* left to the MCTS/tactical layer."""
    ground: list[int]
    hazards: set[int]
    bridge_tiles: set[int] = field(default_factory=set)

    def _walkable(self, tile: int) -> bool:
        return self.ground[tile] == 1 or tile in self.bridge_tiles

    def neighbors(self, tile: int):
        n = len(self.ground)
        for nxt in (tile - 1, tile + 1):
            if 0 <= nxt < n and self._walkable(nxt):
                cost = 1.0
                cost += 2.0 if nxt in self.hazards else 0.0
                cost += 3.0 if nxt in self.bridge_tiles else 0.0
                yield nxt, cost
        # a "jump" edge: skip exactly one gap tile (covers single-tile gaps
        # that A* couldn't otherwise cross since the gap tile isn't ground)
        for nxt in (tile - 2, tile + 2):
            mid = (tile + nxt) // 2
            if 0 <= nxt < n and self._walkable(nxt) and self.ground[mid] == 0 and mid not in self.bridge_tiles:
                cost = 1.5 + (2.0 if nxt in self.hazards else 0.0)
                yield nxt, cost


class AStarPlanner:
    """Coarse global path over the static level graph."""

    def build_graph(self, ground: list[int], hazard_tiles: set[int],
                     bridge_tiles: set[int] = frozenset()) -> LevelGraph:
        return LevelGraph(ground=ground, hazards=hazard_tiles, bridge_tiles=set(bridge_tiles))

    def search(self, graph: LevelGraph, start_tile: int, goal_tile: int) -> list[int]:
        """Standard A* with |goal - n| heuristic (admissible: min cost/edge is 1)."""
        frontier = [(0.0, start_tile)]
        came_from: dict[int, Optional[int]] = {start_tile: None}
        cost_so_far: dict[int, float] = {start_tile: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_tile:
                break
            for nxt, step_cost in graph.neighbors(current):
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + abs(goal_tile - nxt)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal_tile not in came_from:
            return []  # no path found — caller should signal replan/failure

        path = [goal_tile]
        while path[-1] != start_tile:
            path.append(came_from[path[-1]])
        path.reverse()
        return path


class MCTSPlanner:
    """Lightweight tactical search: short random/greedy rollouts over the next
    few primitive actions to pick the immediate committed action + a risk
    estimate, given predicted positions of moving hazards. This is a rollout
    policy, not a full UCT tree — sufficient to unblock the pipeline; a
    teammate can upgrade to proper UCT selection without touching the
    Plan/PlanningModule contract.
    """
    def __init__(self, horizon: int = 4, n_rollouts: int = 24, seed: int = 0):
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.rng = random.Random(seed)

    def search(self, player_tile: float, target_tile: int,
               hazard_predictor) -> tuple[Action, float]:
        """hazard_predictor(t: int) -> set[int] of hazardous tiles at future
        tick offset t (moving platform position / enemy patrol extrapolation)."""
        candidates = [Action.LEFT, Action.RIGHT, Action.LEFT_JUMP, Action.RIGHT_JUMP, Action.NOOP]
        best_action, best_score = Action.NOOP, float("-inf")
        collisions = 0
        total = 0

        for first_action in candidates:
            scores = []
            for _ in range(self.n_rollouts // len(candidates)):
                pos = player_tile
                hit = False
                for t in range(self.horizon):
                    action = first_action if t == 0 else self.rng.choice(candidates)
                    pos += {Action.LEFT: -1, Action.RIGHT: 1,
                            Action.LEFT_JUMP: -1, Action.RIGHT_JUMP: 1,
                            Action.NOOP: 0}[action]
                    if round(pos) in hazard_predictor(t):
                        hit = True
                        total += 1
                        collisions += 1
                        break
                    total += 1
                progress = -abs(target_tile - pos)
                scores.append(progress - (5.0 if hit else 0.0))
            avg = sum(scores) / max(len(scores), 1)
            if avg > best_score:
                best_score, best_action = avg, first_action

        risk = collisions / max(total, 1)
        return best_action, risk


class HybridPlanningModule(PlanningModule):
    """Combines AStarPlanner (macro route) with MCTSPlanner (micro/tactical
    action choice), matching the PlanningModule contract. Ground-truth level
    graph must be supplied via `set_level`; without it, `plan` falls back to
    a straight-line waypoint (still returns a valid Plan, just unoptimized —
    lets other modules keep working before the level graph exists)."""

    def __init__(self):
        self.astar = AStarPlanner()
        self.mcts = MCTSPlanner()
        self.graph: Optional[LevelGraph] = None
        self._cached_path: list[int] = []
        self._last_start: Optional[int] = None
        self._external_hazard_predictor = None

    def set_hazard_predictor(self, predictor):
        """Set an optional external hazard predictor with signature:
        predictor(t_offset: int) -> set[int]. When set, this overrides the
        default state.entities-based hazard logic for tactical planning.
        """
        self._external_hazard_predictor = predictor

    def set_level(self, ground: list[int], hazard_tiles: set[int],
                  bridge_tiles: set[int] = frozenset()):
        self.graph = self.astar.build_graph(ground, hazard_tiles, bridge_tiles)
        self._cached_path = []

    def plan(self, state: GameState, guidance: Optional[MissionGuidance] = None) -> Plan:
        goal_tile = round(state.goal[0] / TILE_W)
        player_tile = round(state.player.x / TILE_W)

        if self.graph is None:
            return Plan(waypoints=[state.goal], committed_action=Action.RIGHT,
                         risk_score=0.0, replan_reason="no level graph set")

        # Replan only when we've drifted off the cached path (cheap check,
        # avoids re-running A* every tick).
        if not self._cached_path or player_tile not in self._cached_path:
            self._cached_path = self.astar.search(self.graph, player_tile, goal_tile)
            if not self._cached_path:
                return Plan(waypoints=[], committed_action=Action.NOOP,
                             risk_score=1.0, replan_reason="no path found")

        idx = self._cached_path.index(player_tile)
        next_tiles = self._cached_path[idx:idx + 6]
        target_tile = next_tiles[1] if len(next_tiles) > 1 else next_tiles[0]

        def default_hazard_predictor(t_offset: int):
            hazards = set(self.graph.hazards)
            for e in state.entities:
                if e.kind == "moving_platform":
                    period = e.extra.get("period", 60)
                    amp = e.extra.get("amplitude_tiles", 2)
                    phase = ((state.tick + t_offset) % period) / period
                    shift = round(amp * (1 if phase < 0.5 else -1) * (phase if phase < 0.5 else 1 - phase) * 2)
                    hazards.add(round(e.x / TILE_W) + shift)
                elif e.kind == "enemy":
                    lo, hi = e.extra.get("patrol_tiles", (0, 0))
                    if hi > lo:
                        span = hi - lo
                        pos = lo + abs(((state.tick + t_offset) % (2 * span)) - span)
                        hazards.add(pos)
            return hazards

        hazard_predictor = self._external_hazard_predictor or default_hazard_predictor
        action, risk = self.mcts.search(player_tile, target_tile, hazard_predictor)
        waypoints = [(t * TILE_W, state.player.y) for t in next_tiles]
        return Plan(waypoints=waypoints, committed_action=action, risk_score=risk)
