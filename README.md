# Intelligent Platform Navigation Agent

Autonomous agent for platform-game levels (gaps, moving platforms, enemies,
collectibles). Four independent workstreams, one shared contract.

## Architecture

```
        MissionGuidance (periodic, LLM/RAG)
                 |
Frame -> [Perception: CNN+LSTM] -> PerceptionOutput --\
                                                         > [Policy: PPO/DQN] -> Action -> Env
GameState -> [Planning: A*+MCTS] -> Plan --------------/
```

`core/interfaces.py` is the ONLY shared surface. Everyone codes against the
dataclasses/abstract classes there and against `core/env.py`'s `MockPlatformEnv`
stub. Nobody imports across member packages except `member4_llm_rag/orchestrator.py`,
which wires all four together.

## Team split

| Member | Package | Owns | Interface |
|---|---|---|---|
| 1 | `member1_perception/` | CNN visual perception + LSTM temporal modeling / trajectory prediction | `PerceptionModule` |
| 2 | `member2_planning/` | A* global route planning + MCTS tactical decision-making | `PlanningModule` |
| 3 | `member3_rl/` | PPO/DQN policy learning, generalization across levels | `PolicyModule` |
| 4 | `member4_llm_rag/` | LLM/SLM + RAG mission interpretation, strategic guidance, **system integration** | `MissionModule` + `orchestrator.py` |

Each package has its own README with scope, milestones, and how to test it
standalone against `MockPlatformEnv` — nobody needs to wait on anyone else to
start.

## Working independently

1. `core/interfaces.py` and `core/env.py` are frozen after this commit unless
   discussed as a group — changing a dataclass field breaks 3 other people's code.
2. Each member works only inside their own `memberN_*/` folder + tests.
3. `git branch <name>/<feature>`, PR into `main`, at least one other member reviews
   before merge (rotate reviewers).
4. `tests/test_smoke.py` must keep passing — it's the integration contract, run it
   before every push: `PYTHONPATH=. pytest tests/`.
5. Weekly sync: each person demos against `MockPlatformEnv`; swap in real level
   assets once the level engine (assets/levels) exists — TBD who owns that, raise
   it in first sync if nobody's claimed it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/
```
