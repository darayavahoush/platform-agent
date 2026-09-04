from __future__ import annotations

import numpy as np
from core.interfaces import Action, Entity, Frame, GameState


class BaseEnv:
    """Environment contract consumed by the integration orchestrator."""

    def reset(self) -> tuple[GameState, Frame]:
        raise NotImplementedError

    def step(self, action: Action) -> tuple[GameState, Frame, float, bool, dict]:
        raise NotImplementedError


class MockPlatformEnv(BaseEnv):
    """
    Reachable randomized platform-navigation environment for RL.

    Design goals:
    - Every main platform is reachable with the current jump physics.
    - Gaps and vertical changes require actual jumping.
    - Hazards/enemies are avoidable rather than unavoidable.
    - Moving platforms are present as optional alternate routes.
    - Goal position changes with the seed.
    - Reward emphasizes survival, landing, and goal completion rather
      than unlimited distance farming.
    """

    WIDTH = 900
    HEIGHT = 500

    PLAYER_W = 24
    PLAYER_H = 32

    GRAVITY = 0.65
    MOVE_SPEED = 4.0
    JUMP_SPEED = -11.5

    MAX_STEPS = 300

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.tick = 0
        self.done = False

        self.player = Entity(
            x=60.0, y=390.0, vx=0.0, vy=0.0, kind="player"
        )

        self.goal = (850.0, 350.0)

        self.platforms: list[Entity] = []
        self.moving_platforms: list[Entity] = []
        self.hazards: list[Entity] = []
        self.enemies: list[Entity] = []

        self.visited_platforms: set[int] = set()
        self.previous_distance = 0.0
        self.difficulty = 1.0
        self.difficulty = 1.0

    def set_difficulty(self, difficulty: float):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def set_difficulty(self, difficulty: float):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ==========================================================
    # LEVEL GENERATION
    # ==========================================================

    def _generate_level(self):
        self.platforms = []
        self.moving_platforms = []
        self.hazards = []
        self.enemies = []
        self.visited_platforms = set()

        # Main route. Horizontal gap <= 65 and vertical change <= 55,
        # comfortably within the jump envelope of the environment.
        start = Entity(
            x=20.0,
            y=430.0,
            kind="platform",
            extra={"width": 150, "height": 20},
        )
        self.platforms.append(start)

        x = 170.0
        y = 430.0

        # Five additional main platforms. Their route varies by seed.
        for i in range(5):
            max_gap = 35 + int(30 * self.difficulty)
            gap = int(self.rng.integers(35, max_gap + 1))
            width = int(self.rng.integers(100, 141))

            # Vertical timing is harder to learn than horizontal spacing,
            # so it ramps in on a slower, squared curve relative to difficulty.
            vertical_difficulty = self.difficulty ** 2
            delta_range = max(1, int(55 * vertical_difficulty))
            delta_y = int(self.rng.integers(-delta_range, delta_range + 1))
            y = int(np.clip(y + delta_y, 285, 430))

            x += gap

            # Keep the route inside the world.
            if x + width > 875:
                x = 875 - width

            platform = Entity(
                x=float(x),
                y=float(y),
                vx=0.0,
                vy=0.0,
                kind="platform",
                extra={"width": width, "height": 20},
            )
            self.platforms.append(platform)
            x += width

        # Put hazards on some platforms, but keep a clear landing corridor.
        for i, platform in enumerate(self.platforms[1:], start=1):
            width = platform.extra["width"]

            if i >= 2 and self.rng.random() < 0.40 * self.difficulty:
                hazard_x = platform.x + width * 0.55
                self.hazards.append(
                    Entity(
                        x=float(hazard_x),
                        y=float(platform.y - 15),
                        kind="hazard",
                        extra={"width": 18, "height": 15},
                    )
                )

            if i >= 2 and self.rng.random() < 0.35 * self.difficulty:
                enemy_x = platform.x + width * 0.35
                self.enemies.append(
                    Entity(
                        x=float(enemy_x),
                        y=float(platform.y - 30),
                        vx=1.0,
                        vy=0.0,
                        kind="enemy",
                        extra={
                            "width": 22,
                            "height": 30,
                            "min_x": float(platform.x + 20),
                            "max_x": float(platform.x + width - 42),
                        },
                    )
                )

        # Optional moving platforms. They are not required for the main
        # route, preventing an impossible level if timing is missed.
        for _ in range(2):
            base_x = float(self.rng.integers(250, 680))
            base_y = float(self.rng.integers(220, 340))
            self.moving_platforms.append(
                Entity(
                    x=base_x,
                    y=base_y,
                    vx=0.0,
                    vy=0.0,
                    kind="moving_platform",
                    extra={
                        "width": 90,
                        "height": 18,
                        "base_x": base_x,
                        "amplitude": float(self.rng.integers(35, 75)),
                        "speed": float(self.rng.uniform(0.018, 0.030)),
                        "phase": float(self.rng.uniform(0, 2 * np.pi)),
                    },
                )
            )

        # Goal is on the final main platform and varies within it.
        last = self.platforms[-1]
        goal_margin = 25.0
        goal_x = self.rng.uniform(
            last.x + goal_margin,
            last.x + last.extra["width"] - goal_margin,
        )
        self.goal = (float(goal_x), float(last.y - 45))

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):
        self.tick = 0
        self.done = False
        self._generate_level()

        self.player = Entity(
            x=60.0,
            y=390.0,
            vx=0.0,
            vy=0.0,
            kind="player",
        )

        self.previous_distance = self._distance_to_goal()
        return self._get_state(), self._frame()

    # ==========================================================
    # WORLD UPDATE
    # ==========================================================

    def _update_moving_platforms(self):
        for platform in self.moving_platforms:
            e = platform.extra
            platform.x = (
                e["base_x"]
                + e["amplitude"]
                * np.sin(e["speed"] * self.tick + e["phase"])
            )

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy.x += enemy.vx
            min_x = enemy.extra["min_x"]
            max_x = enemy.extra["max_x"]

            if enemy.x <= min_x:
                enemy.x = min_x
                enemy.vx = abs(enemy.vx)
            elif enemy.x >= max_x:
                enemy.x = max_x
                enemy.vx = -abs(enemy.vx)

    # ==========================================================
    # ACTION / PHYSICS
    # ==========================================================

    def _apply_action(self, action: Action):
        if action == Action.LEFT:
            self.player.vx = -self.MOVE_SPEED
        elif action == Action.RIGHT:
            self.player.vx = self.MOVE_SPEED
        elif action == Action.JUMP:
            self.player.vx *= 0.8
            self._try_jump()
        elif action == Action.NOOP:
            self.player.vx *= 0.80
        elif action == Action.LEFT_JUMP:
            self.player.vx = -self.MOVE_SPEED
            self._try_jump()
        elif action == Action.RIGHT_JUMP:
            self.player.vx = self.MOVE_SPEED
            self._try_jump()

    def _is_on_platform(self):
        px = self.player.x
        bottom = self.player.y + self.PLAYER_H

        for platform in self.platforms + self.moving_platforms:
            left = platform.x
            right = left + platform.extra["width"]
            top = platform.y

            overlap = px + self.PLAYER_W > left and px < right
            close = abs(bottom - top) < 8

            if overlap and close:
                return True

        return False

    def _try_jump(self):
        if self._is_on_platform():
            self.player.vy = self.JUMP_SPEED

    # ==========================================================
    # COLLISION
    # ==========================================================

    @staticmethod
    def _rect_overlap(ax, ay, aw, ah, bx, by, bw, bh):
        return (
            ax < bx + bw
            and ax + aw > bx
            and ay < by + bh
            and ay + ah > by
        )

    def _check_hazard_collision(self):
        for h in self.hazards:
            if self._rect_overlap(
                self.player.x, self.player.y,
                self.PLAYER_W, self.PLAYER_H,
                h.x, h.y,
                h.extra["width"], h.extra["height"],
            ):
                return True
        return False

    def _check_enemy_collision(self):
        for e in self.enemies:
            if self._rect_overlap(
                self.player.x, self.player.y,
                self.PLAYER_W, self.PLAYER_H,
                e.x, e.y,
                e.extra["width"], e.extra["height"],
            ):
                return True
        return False

    def _handle_landing(self, old_y):
        all_platforms = self.platforms + self.moving_platforms

        old_bottom = old_y + self.PLAYER_H
        new_bottom = self.player.y + self.PLAYER_H

        for index, platform in enumerate(all_platforms):
            left = platform.x
            right = left + platform.extra["width"]
            top = platform.y

            overlap = (
                self.player.x + self.PLAYER_W > left
                and self.player.x < right
            )

            # Only count a landing when the player crosses the platform
            # top from above. The previous <= check also fired while the
            # player was already standing on a platform, giving +0.75
            # repeatedly on every frame and allowing reward farming.
            crossed = old_bottom < top - 1e-6 and new_bottom >= top

            if overlap and crossed and self.player.vy >= 0:
                self.player.y = top - self.PLAYER_H
                self.player.vy = 0.0

                # Landing reward is earned once per platform visit.
                is_new_visit = index not in self.visited_platforms
                self.visited_platforms.add(index)

                return is_new_visit

        return False

    # ==========================================================
    # REWARD
    # ==========================================================

    def _distance_to_goal(self):
        dx = self.goal[0] - self.player.x
        dy = self.goal[1] - self.player.y
        return float(np.hypot(dx, dy))

    def _next_platform_index(self):
        candidates = []
        for i, p in enumerate(self.platforms):
            if p.x + p.extra["width"] > self.player.x + 5:
                candidates.append((p.x, i))
        if not candidates:
            return len(self.platforms) - 1
        return min(candidates)[1]

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, action: Action):
        if self.done:
            return (
                self._get_state(),
                self._frame(),
                0.0,
                True,
                {"reached_goal": False},
            )

        self.tick += 1
        old_distance = self._distance_to_goal()
        old_y = self.player.y

        self._apply_action(action)
        self._update_moving_platforms()
        self._update_enemies()

        self.player.vy += self.GRAVITY
        self.player.x += self.player.vx
        self.player.y += self.player.vy
        self.player.vx *= 0.92

        self.player.x = float(
            np.clip(self.player.x, 0, self.WIDTH - self.PLAYER_W)
        )

        landed = self._handle_landing(old_y)

        new_distance = self._distance_to_goal()
        progress = old_distance - new_distance

        # Navigation-focused reward shaping.
        reward = -0.03

        # Reward movement toward the goal.
        reward += float(np.clip(progress, -3.0, 3.0)) * 0.05

        # Reward discovering a new platform.
        if landed:
            reward += 2.0

        # Small penalty for making no progress.
        if abs(progress) < 0.10:
            reward -= 0.015

        # Encourage horizontal movement toward the goal.
        goal_dx_old = self.goal[0] - self.player.x
        goal_direction = 1.0 if goal_dx_old > 0 else -1.0

        if self.player.vx * goal_direction > 0.5:
            reward += 0.015

        # Penalize substantial movement away from the goal.
        if self.player.vx * goal_direction < -0.5:
            reward -= 0.02

        # Hazards / enemies / fall are terminal and strongly negative.
        if self._check_hazard_collision():
            self.done = True
            return (
                self._get_state(),
                self._frame(),
                -1.5,
                True,
                {
                    "reached_goal": False,
                    "collision": True,
                    "reason": "hazard",
                    "progress": float(progress),
                },
            )

        if self._check_enemy_collision():
            self.done = True
            return (
                self._get_state(),
                self._frame(),
                -1.5,
                True,
                {
                    "reached_goal": False,
                    "collision": True,
                    "reason": "enemy",
                    "progress": float(progress),
                },
            )

        if self.player.y > self.HEIGHT:
            self.done = True
            return (
                self._get_state(),
                self._frame(),
                -2.0,
                True,
                {
                    "reached_goal": False,
                    "fallen": True,
                    "reason": "fell",
                    "progress": float(progress),
                },
            )

        # Goal reward dominates all intermediate rewards.
        goal_distance = float(np.hypot(
            self.player.x - self.goal[0],
            self.player.y - self.goal[1],
        ))

        if goal_distance < 45.0:
            self.done = True
            return (
                self._get_state(),
                self._frame(),
                5.0,
                True,
                {
                    "reached_goal": True,
                    "success": True,
                    "progress": float(progress),
                    "distance_to_goal": goal_distance,
                },
            )

        if self.tick >= self.MAX_STEPS:
            self.done = True
            return (
                self._get_state(),
                self._frame(),
                -1.8,
                True,
                {
                    "reached_goal": False,
                    "timeout": True,
                    "progress": float(progress),
                },
            )

        return (
            self._get_state(),
            self._frame(),
            float(reward),
            False,
            {
                "reached_goal": False,
                "landed": landed,
                "progress": float(progress),
                "distance_to_goal": float(new_distance),
            },
        )

    # ==========================================================
    # STATE
    # ==========================================================

    def _entities(self):
        return (
            self.platforms
            + self.moving_platforms
            + self.hazards
            + self.enemies
        )

    def _get_state(self):
        return GameState(
            player=self.player,
            entities=self._entities(),
            goal=self.goal,
            tick=self.tick,
            done=self.done,
            info={},
        )

    # ==========================================================
    # SYNTHETIC FRAME
    # ==========================================================

    def _frame(self):
        rgb = np.zeros(
            (self.HEIGHT, self.WIDTH, 3),
            dtype=np.uint8,
        )
        rgb[:, :, :] = 30

        for p in self.platforms + self.moving_platforms:
            x1 = int(np.clip(p.x, 0, self.WIDTH - 1))
            x2 = int(np.clip(
                p.x + p.extra["width"], 0, self.WIDTH
            ))
            y = int(np.clip(p.y, 0, self.HEIGHT - 1))
            rgb[y:min(y + 10, self.HEIGHT), x1:x2, :] = 100

        for h in self.hazards:
            x1 = int(np.clip(h.x, 0, self.WIDTH - 1))
            x2 = int(np.clip(
                h.x + h.extra["width"], 0, self.WIDTH
            ))
            y1 = int(np.clip(h.y, 0, self.HEIGHT - 1))
            y2 = int(np.clip(
                h.y + h.extra["height"], 0, self.HEIGHT
            ))
            rgb[y1:y2, x1:x2, :] = 200

        for e in self.enemies:
            x1 = int(np.clip(e.x, 0, self.WIDTH - 1))
            x2 = int(np.clip(
                e.x + e.extra["width"], 0, self.WIDTH
            ))
            y1 = int(np.clip(e.y, 0, self.HEIGHT - 1))
            y2 = int(np.clip(
                e.y + e.extra["height"], 0, self.HEIGHT
            ))
            rgb[y1:y2, x1:x2, :] = 180

        px = int(np.clip(
            self.player.x, 0, self.WIDTH - 1
        ))
        py = int(np.clip(
            self.player.y, 0, self.HEIGHT - 1
        ))
        rgb[
            py:min(py + self.PLAYER_H, self.HEIGHT),
            px:min(px + self.PLAYER_W, self.WIDTH),
            :
        ] = 255

        return Frame(rgb=rgb, tick=self.tick)
