import numpy as np
from core.interfaces import GameState, Plan, PerceptionOutput


class GameStateFeaturizer:
    """
    Fixed 64-dimensional observation.

    Layout:
      0-3   player x/y/vx/vy
      4-6   goal dx/dy/distance
      7     normalized tick
      8-43  six nearest platforms/moving platforms, 6 features each
      44-55 four nearest hazards, 3 features each
      56-63 three nearest enemies, 3 features each

    Entity features:
      relative_x, relative_y, width, vertical_velocity,
      type_id, horizontal_distance
    """

    OBS_DIM = 65

    def __init__(
        self,
        position_scale=256.0,
        velocity_scale=10.0,
        distance_scale=256.0,
        tick_scale=300.0,
    ):
        self.position_scale = float(position_scale)
        self.velocity_scale = float(velocity_scale)
        self.distance_scale = float(distance_scale)
        self.tick_scale = float(tick_scale)

    @staticmethod
    def _safe(value):
        value = float(value)
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _type_id(kind):
        return {
            "platform": 1.0,
            "moving_platform": 2.0,
        }.get(kind, 0.0)

    @staticmethod
    def _normalize(value, scale):
        return np.clip(
            GameStateFeaturizer._safe(value) / scale,
            -1.0,
            1.0,
        )

    def _platform_features(self, entity, player):
        width = entity.extra.get("width", 100)
        near_edge_dx = entity.x - player.x
        far_edge_dx = (entity.x + width) - player.x
        dy = entity.y - player.y

        return [
            self._normalize(near_edge_dx, self.position_scale),
            self._normalize(far_edge_dx, self.position_scale),
            self._normalize(dy, self.position_scale),
            self._normalize(width, 160.0),
            self._type_id(entity.kind) / 2.0,
            self._normalize(min(abs(near_edge_dx), abs(far_edge_dx)), self.distance_scale),
        ]

    def _hazard_features(self, entity, player):
        dx = entity.x - player.x
        dy = entity.y - player.y

        return [
            self._normalize(dx, self.position_scale),
            self._normalize(dy, self.position_scale),
            self._normalize(abs(dx), self.distance_scale),
        ]

    def _enemy_features(self, entity, player):
        dx = entity.x - player.x
        dy = entity.y - player.y

        return [
            self._normalize(dx, self.position_scale),
            self._normalize(dy, self.position_scale),
            self._normalize(entity.vx, self.velocity_scale),
        ]

    def transform(
        self,
        state: GameState,
        plan: Plan | None = None,
        perception: PerceptionOutput | None = None,
    ):
        player = state.player

        goal_dx = state.goal[0] - player.x
        goal_dy = state.goal[1] - player.y
        goal_dist = np.hypot(goal_dx, goal_dy)

        is_grounded = 1.0 if abs(player.vy) < 0.5 else 0.0

        features = [
            self._normalize(player.x, 450.0),
            self._normalize(player.y, 300.0),
            self._normalize(player.vx, self.velocity_scale),
            self._normalize(player.vy, self.velocity_scale),
            self._normalize(goal_dx, self.position_scale),
            self._normalize(goal_dy, self.position_scale),
            self._normalize(goal_dist, self.distance_scale),
            is_grounded,
        ]

        entities = list(state.entities)

        platforms = [
            e for e in entities
            if e.kind in ("platform", "moving_platform")
        ]
        hazards = [e for e in entities if e.kind == "hazard"]
        enemies = [e for e in entities if e.kind == "enemy"]

        # Prioritize the main route: static platforms are ordered left-to-right.
        # Moving platforms are considered only after the main route.
        static_platforms = [
            e for e in platforms
            if e.kind == "platform"
        ]

        moving_platforms = [
            e for e in platforms
            if e.kind == "moving_platform"
        ]

        # Main route ordering is important for navigation.
        static_platforms.sort(key=lambda e: e.x)

        # Moving platforms remain available as secondary information.
        moving_platforms.sort(
            key=lambda e: abs(e.x - player.x)
        )

        platforms = static_platforms + moving_platforms
        hazards.sort(
            key=lambda e: abs(e.x - player.x)
        )
        enemies.sort(
            key=lambda e: abs(e.x - player.x)
        )

        # Six platform slots.
        for i in range(6):
            if i < len(platforms):
                features.extend(
                    self._platform_features(platforms[i], player)
                )
            else:
                features.extend([0.0] * 6)

        # Four hazard slots.
        for i in range(4):
            if i < len(hazards):
                features.extend(
                    self._hazard_features(hazards[i], player)
                )
            else:
                features.extend([0.0] * 3)

        # Three enemy slots.
        for i in range(3):
            if i < len(enemies):
                features.extend(
                    self._enemy_features(enemies[i], player)
                )
            else:
                features.extend([0.0] * 3)

        obs = np.asarray(features, dtype=np.float32)

        if obs.size != self.OBS_DIM:
            raise RuntimeError(
                f"Expected {self.OBS_DIM} features, got {obs.size}"
            )

        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        return obs
