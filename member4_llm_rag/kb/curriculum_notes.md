# Curriculum and Difficulty Notes

Early in curriculum (difficulty below ~50%), keep subgoals simple and
reward shaping conservative -- a single dominant hint like "reach_goal"
with a modest distance-based shaping term. Overly specific subgoals early
on can conflict with easier level layouts and confuse the policy.

Late in curriculum (difficulty above ~80%), guidance can be more specific
about hazard avoidance and moving-platform timing, since those elements
are consistently present in generated levels by then.

Keep reward_shaping_hints values in a bounded range (roughly -1.0 to 1.0
per hint) so they nudge rather than dominate the environment's own reward
signal from core/env.py.
