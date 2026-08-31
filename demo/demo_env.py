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
        h = 40
        w = N_TILES * TILE_W
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Background
        rgb[:] = [70, 120, 200]

        # Ground/platform tiles
        ground_y0 = h - 18
        ground_h = 18
        for tile_idx, solid in enumerate(GROUND):
            if solid == 1:
                x0 = tile_idx * TILE_W
                rgb[ground_y0:ground_y0 + ground_h, x0:x0 + TILE_W] = [34, 139, 34]

        # Player
        player_x = int(round(self.player_tile * TILE_W))
        player_y = ground_y0 - 12
        rgb[player_y:player_y + 12, player_x:player_x + TILE_W - 4] = [255, 215, 0]

        # Moving platform
        mp_tile = self._moving_platform_tile()
        mp_x = mp_tile * TILE_W
        mp_y = 20
        mp_w = TILE_W * 2
        mp_h = 8
        rgb[mp_y:mp_y + mp_h, mp_x:mp_x + mp_w] = [160, 82, 45]

        # Enemy
        enemy_tile = self._enemy_tile()
        enemy_x = enemy_tile * TILE_W
        enemy_y = ground_y0 - 10
        rgb[enemy_y:enemy_y + 10, enemy_x:enemy_x + TILE_W] = [220, 50, 50]

        # Collectible
        if not self.collected:
            cx = COLLECTIBLE_TILE * TILE_W + TILE_W // 2
            cy = ground_y0 - 10
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    if dx * dx + dy * dy <= 36:
                        x = cx + dx
                        y = cy + dy
                        if 0 <= x < w and 0 <= y < h:
                            rgb[y, x] = [255, 255, 0]

        # Small goal marker at the far right
        goal_x = (N_TILES - 2) * TILE_W + TILE_W // 2
        goal_y = ground_y0 - 12
        rgb[goal_y:goal_y + 12, goal_x - 4:goal_x + 4] = [255, 255, 255]

        return Frame(rgb=rgb, tick=self.tick)
