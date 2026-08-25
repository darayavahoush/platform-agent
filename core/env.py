"""
Minimal env contract so all 4 people can build/test against a stub before the
real level engine exists. Swap `MockPlatformEnv` for the real one later
without touching any of the 4 module packages.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from core.interfaces import Action, GameState, Frame, Entity


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> tuple[GameState, Frame]: ...

    @abstractmethod
    def step(self, action: Action) -> tuple[GameState, Frame, float, bool, dict]:
        """returns state, frame, reward, done, info"""
        ...

    @abstractmethod
    def render(self) -> np.ndarray: ...


class MockPlatformEnv(BaseEnv):
    """Fake env: flat pixel noise + a trivial physics stub. Enough to unblock
    everyone's I/O plumbing on day 1. Replace with real level loader (assets/levels)
    once it exists."""

    def __init__(self, width: int = 256, height: int = 144, seed: int = 0):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.player = Entity(x=10, y=100, kind="player")
        self.goal = (self.w - 10, 100)

    def reset(self):
        self.tick = 0
        self.player = Entity(x=10, y=100, kind="player")
        state = GameState(player=self.player, entities=self._entities(), goal=self.goal, tick=self.tick)
        return state, self._frame()

    def step(self, action: Action):
        dx = {Action.LEFT: -2, Action.RIGHT: 2, Action.LEFT_JUMP: -2, Action.RIGHT_JUMP: 2}.get(action, 0)
        dy = -3 if action in (Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP) else 0
        self.player.x = float(np.clip(self.player.x + dx, 0, self.w))
        self.player.y = float(np.clip(self.player.y + dy, 0, self.h))
        self.tick += 1
        done = self.player.x >= self.goal[0] or self.tick > 1000
        reward = 1.0 if done and self.player.x >= self.goal[0] else -0.01
        state = GameState(player=self.player, entities=self._entities(), goal=self.goal, tick=self.tick, done=done)
        return state, self._frame(), reward, done, {}

    def render(self) -> np.ndarray:
        return self._frame().rgb

    def _entities(self):
        return [Entity(x=80, y=110, kind="moving_platform", extra={"period": 60, "amplitude": 30}),
                Entity(x=150, y=100, kind="enemy", extra={"patrol": [(140, 100), (170, 100)]}),
                Entity(x=200, y=90, kind="collectible")]

    def _frame(self) -> Frame:
        rgb = self.rng.integers(0, 255, (self.h, self.w, 3), dtype=np.uint8)
        return Frame(rgb=rgb, tick=self.tick)
