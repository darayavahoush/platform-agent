"""A real (if simple) tile-based BaseEnv implementation, used to generate an
actual playable trace for the frontend demo. This is NOT core/env.py's
MockPlatformEnv (that stays untouched — it's the shared contract stub every
member codes against). This is scoped to the demo only.
"""
from __future__ import annotations
import numpy as np
from core.interfaces import Action, GameState, Frame, Entity
from demo.level import (GROUND, N_TILES, TILE_W, MOVING_PLATFORM, ENEMY,
                         COLLECTIBLE_TILE, GOAL_TILE, START_TILE)


class DemoPlatformEnv:
    def __init__(self):
        self.tick = 0
        self.player_tile = float(START_TILE)
        self.player = Entity(x=START_TILE * TILE_W, y=0, kind="player")
        self.goal = (GOAL_TILE * TILE_W, 0)
        self.collected = False
        self.hit_count = 0
        self.reward_total = 0.0

    def reset(self):
        self.__init__()
        return self._state(), self._frame()

    def _moving_platform_tile(self) -> int:
        period = MOVING_PLATFORM["period"]
        amp = MOVING_PLATFORM["amplitude_tiles"]
        phase = (self.tick % period) / period
        # triangle wave, -amp..+amp
        tri = (4 * abs(phase - 0.5) - 1) * amp
        return round(MOVING_PLATFORM["home_tile"] + tri)

    def _enemy_tile(self) -> int:
        lo, hi = ENEMY["patrol_tiles"]
        span = hi - lo
        if span == 0:
            return lo
        cyc = self.tick % (2 * span)
        return lo + abs(cyc - span)

    def _entities(self) -> list[Entity]:
        mp_tile = self._moving_platform_tile()
        en_tile = self._enemy_tile()
        ents = [
            Entity(x=mp_tile * TILE_W, y=0, kind="moving_platform",
                   extra={"period": MOVING_PLATFORM["period"],
                          "amplitude_tiles": MOVING_PLATFORM["amplitude_tiles"]}),
            Entity(x=en_tile * TILE_W, y=0, kind="enemy",
                   extra={"patrol_tiles": ENEMY["patrol_tiles"]}),
        ]
        if not self.collected:
            ents.append(Entity(x=COLLECTIBLE_TILE * TILE_W, y=0, kind="collectible"))
        return ents

    def _is_ground(self, tile: int) -> bool:
        if tile < 0 or tile >= N_TILES:
            return False
        if GROUND[tile] == 1:
            return True
        return tile == self._moving_platform_tile()  # bridged by moving platform

    def step(self, action: Action):
        delta = {Action.LEFT: -1, Action.RIGHT: 1,
                 Action.LEFT_JUMP: -1, Action.RIGHT_JUMP: 1,
                 Action.NOOP: 0}[action]
        is_jump = action in (Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP)

        target_tile = round(self.player_tile) + delta
        # a jump can bridge exactly one gap tile; a walk cannot land on a gap
        landed_ok = self._is_ground(target_tile) or (
            is_jump and self._is_ground(target_tile) is False and
            self._is_ground(target_tile + (1 if delta > 0 else -1))
        )

        reward = -0.01
        if self._is_ground(target_tile):
            self.player_tile = target_tile
        elif is_jump and self._is_ground(target_tile + delta):
            self.player_tile = target_tile + delta  # cleared a 1-tile gap
        # else: blocked, stay put (small penalty already applied)

        self.player.x = self.player_tile * TILE_W

        if round(self.player_tile) == self._enemy_tile():
            self.hit_count += 1
            reward -= 1.0

        if not self.collected and round(self.player_tile) == COLLECTIBLE_TILE:
            self.collected = True
            reward += 5.0

        self.tick += 1
        done = round(self.player_tile) >= GOAL_TILE or self.tick > 400
        if done and round(self.player_tile) >= GOAL_TILE:
            reward += 20.0

        self.reward_total += reward
        state = self._state(done)
        return state, self._frame(), reward, done, {
            "collected": self.collected, "hits": self.hit_count,
        }

    def _state(self, done: bool = False) -> GameState:
        return GameState(player=self.player, entities=self._entities(),
                          goal=self.goal, tick=self.tick, done=done,
                          info={"reward_total": self.reward_total})

    def render(self) -> np.ndarray:
        return self._frame().rgb

    def _frame(self) -> Frame:
        return Frame(rgb=np.zeros((40, N_TILES * TILE_W, 3), dtype=np.uint8), tick=self.tick)
