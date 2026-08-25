"""A small hand-built level to exercise the real A*/MCTS planner end to end.
Not meant to be the final level format — just enough tile-grid structure
(gaps, a moving platform, a patrolling enemy, a collectible, a goal) to prove
the planning module works on something non-trivial before the real level
engine exists.
"""

TILE_W = 20
N_TILES = 40

# 1 = solid ground, 0 = gap. Single-tile gaps (index 5) are jump-able;
# the wider gap at 16-18 is only crossable by timing the moving platform.
GROUND = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
          1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
          1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

MOVING_PLATFORM = {"home_tile": 16, "period": 48, "amplitude_tiles": 2}
ENEMY = {"patrol_tiles": (21, 25)}
COLLECTIBLE_TILE = 13
GOAL_TILE = N_TILES - 2
START_TILE = 1

HAZARD_TILES = set(range(ENEMY["patrol_tiles"][0], ENEMY["patrol_tiles"][1] + 1))
# Tiles only reachable while the moving platform is passing through them —
# A*'s static graph treats these as traversable-with-cost (macro route),
# the MCTS layer handles the actual jump *timing* against them.
BRIDGE_TILES = set(range(MOVING_PLATFORM["home_tile"] - MOVING_PLATFORM["amplitude_tiles"],
                          MOVING_PLATFORM["home_tile"] + MOVING_PLATFORM["amplitude_tiles"] + 1))
