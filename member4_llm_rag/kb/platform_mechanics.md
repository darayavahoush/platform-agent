# Platform and Gap Mechanics

Vertical gaps taller than one jump arc need a chained-jump subgoal (e.g.
"chain_jumps_to_ledge") rather than a single JUMP action. Give a small
positive reward_shaping_hint for reducing vertical distance to the next
platform, not just the final goal, so the agent gets signal during long
climbs.

Horizontal gaps are best crossed with the compound RIGHT_JUMP / LEFT_JUMP
actions rather than separate RIGHT then JUMP steps.

Entities with kind="moving_platform" carry `extra.period` and
`extra.amplitude_tiles`, meaning their x position oscillates. The subgoal
should be phrased as "time_jump_to_moving_platform" so downstream planning
waits for the platform's predicted position instead of committing to the
platform's current position.
