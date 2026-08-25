const MODULES = [
  {
    n: '01', name: 'PERCEPTION', status: 'pending', owner: 'Member 1',
    desc: 'CNN detection + LSTM trajectory tracking of platforms/enemies from raw frames.',
  },
  {
    n: '02', name: 'PLANNING', status: 'live', owner: 'Member 2 (this build)',
    desc: 'A* macro route + lightweight MCTS rollout search for jump timing. Driving this run.',
  },
  {
    n: '03', name: 'POLICY (RL)', status: 'pending', owner: 'Member 3',
    desc: 'PPO/DQN control policy trained to generalize across level distributions.',
  },
  {
    n: '04', name: 'MISSION (LLM/RAG)', status: 'pending', owner: 'Member 4',
    desc: 'Strategic subgoals + reward shaping from mission text, grounded via RAG.',
  },
]

function Pill({ status }) {
  const live = status === 'live'
  return (
    <span style={{
      fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '.08em', fontWeight: 700,
      padding: '2px 7px', borderRadius: 20,
      color: live ? 'var(--live)' : 'var(--text-dim)',
      background: live ? 'var(--live-dim)' : '#1A212B',
    }}>
      {live ? 'LIVE' : 'PENDING'}
    </span>
  )
}

export default function ModulePanel() {
  return (
    <aside style={{ borderLeft: '1px solid var(--line)', padding: '22px 20px' }}>
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-dim)',
        letterSpacing: '.1em', textTransform: 'uppercase', marginBottom: 14,
      }}>
        Module Status
      </div>
      {MODULES.map(m => (
        <div key={m.n} style={{
          border: '1px solid var(--line)', borderRadius: 6, padding: '12px 13px',
          marginBottom: 10, background: 'var(--panel)',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 500 }}>{m.n} · {m.name}</span>
            <Pill status={m.status} />
          </div>
          <p style={{ margin: 0, fontSize: 11.5, color: 'var(--text-dim)', lineHeight: 1.5 }}>{m.desc}</p>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--pending)', marginTop: 8, letterSpacing: '.04em' }}>
            OWNER — {m.owner}
          </div>
        </div>
      ))}
    </aside>
  )
}
