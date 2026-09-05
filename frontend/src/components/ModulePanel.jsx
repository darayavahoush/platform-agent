import { Eye, Route, Cpu, MessageSquare } from 'lucide-react'
import { label, panel, pillStyle, mono } from '../theme.js'

// Mirrors PipelineFlow's data — kept as a separate detail list (owner, file,
// longer description) rather than folded into the flow diagram, which stays
// compact on purpose.
const MODULES = [
  {
    n: '01', name: 'PERCEPTION', icon: Eye, owner: 'Member 1',
    detail: 'CNN(SmallColorCNN) + LSTM tracker',
    file: 'member1_perception/perception.py',
    desc: 'Detects platforms/enemies per frame and tracks trajectories over time.',
    active: false,
  },
  {
    n: '02', name: 'PLANNING', icon: Route, owner: 'Member 2',
    detail: 'A* macro route + MCTS jump timing',
    file: 'member2_planning/planning.py',
    desc: 'Global route search over the tile graph, with tactical rollout search for jump timing.',
    active: true,
  },
  {
    n: '03', name: 'POLICY (RL)', icon: Cpu, owner: 'Member 3',
    detail: 'PPO / DQN control policy',
    file: 'member3_rl/policy.py',
    desc: 'Learned control policy, trained to generalize across level distributions.',
    active: false,
  },
  {
    n: '04', name: 'MISSION (LLM/RAG)', icon: MessageSquare, owner: 'Member 4',
    detail: 'Prompted SLM (Ollama) + RAG',
    file: 'member4_llm_rag/mission.py',
    desc: 'Turns mission text into a subgoal and reward-shaping hints, grounded in 3 knowledge docs.',
    active: false,
  },
]

export default function ModulePanel() {
  return (
    <aside style={{ borderLeft: '1px solid var(--line)', padding: '22px 20px' }}>
      <div style={{ ...label, marginBottom: 14 }}>Module Detail</div>
      {MODULES.map(m => {
        const Icon = m.icon
        return (
          <div key={m.n} style={{ ...panel, padding: '12px 13px', marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: mono, fontSize: 12, fontWeight: 500 }}>
                <Icon size={13} color={m.active ? 'var(--live)' : 'var(--text-dim)'} />
                {m.n} · {m.name}
              </span>
              <span style={pillStyle(m.active ? 'live' : 'pending')}>
                {m.active ? 'DRIVING RUN' : 'IMPLEMENTED'}
              </span>
            </div>
            <div style={{ fontFamily: mono, fontSize: 10, color: 'var(--live)', marginBottom: 5 }}>{m.detail}</div>
            <p style={{ margin: 0, fontSize: 11.5, color: 'var(--text-dim)', lineHeight: 1.5 }}>{m.desc}</p>
            <div style={{ fontFamily: mono, fontSize: 10, color: 'var(--pending)', marginTop: 8, letterSpacing: '.04em' }}>
              {m.owner} — {m.file}
            </div>
          </div>
        )
      })}
    </aside>
  )
}
