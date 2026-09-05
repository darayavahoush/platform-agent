# Hazards, Enemies, and Collectibles

Entities with kind="hazard" should produce a constraint string like
"avoid_zone:<x>,<y>,<radius>" rather than only a reward penalty --
constraints are hard filters the planner can use, reward hints are soft
nudges for the RL policy.

Entities with kind="enemy" carry `extra.patrol_tiles` (a (lo, hi) range).
When the player's path intersects a patrol range, prefer a timing subgoal
such as "wait_for_enemy_clear" rather than recommending a crossing through
the patrol zone.

Entities with kind="collectible" are optional pickups, not blockers.
Only fold a collectible into the subgoal (e.g. "collect_then_goal") when
the mission text explicitly asks for collection; otherwise prioritize
reaching the goal and leave collectibles out of reward_shaping_hints.

When recent episodes have failed repeatedly with reason "fell" near a
specific region, raise the weight of a "progress_toward_goal" shaping term
for that region -- it usually means the agent is undershooting jumps
rather than mistiming them.
