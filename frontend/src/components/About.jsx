import { X } from 'lucide-react'
import { mono, sans } from '../theme.js'

export default function About({ onClose }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(5,7,10,0.72)',
      display: 'grid', placeItems: 'center', zIndex: 50, padding: 20,
    }} onClick={onClose}>
      <div
        style={{
          maxWidth: 560, width: '100%', maxHeight: '85vh', overflowY: 'auto',
          background: 'var(--panel)', border: '1px solid var(--line)', borderRadius: 10,
          padding: '26px 28px', fontFamily: sans, color: 'var(--text)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
        }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
          <div>
            <div style={{ fontFamily: mono, fontSize: 10, color: 'var(--live)', letterSpacing: '.1em', marginBottom: 4 }}>
              ABOUT
            </div>
            <h2 style={{ margin: 0, fontSize: 19, fontWeight: 700 }}>What NAV_TRACE is</h2>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', padding: 4 }}
          >
            <X size={18} />
          </button>
        </div>

        <p style={{ fontSize: 13.5, lineHeight: 1.65, color: 'var(--text)', margin: '0 0 14px' }}>
          <strong>platform-agent</strong> is a 4-person build of an autonomous agent for a
          platform-game level — gaps, a moving platform, a patrolling enemy, and a
          collectible on the way to a goal tile. Each person owns one module behind a
          shared interface (<code style={{ fontFamily: mono, fontSize: 12 }}>core/interfaces.py</code>),
          and <code style={{ fontFamily: mono, fontSize: 12 }}>orchestrator.py</code> wires all four
          into a single tick loop.
        </p>

        <p style={{ fontSize: 13.5, lineHeight: 1.65, color: 'var(--text)', margin: '0 0 14px' }}>
          This page isn't a live agent — it's a <strong>replay viewer</strong>. Each run is generated
          offline and dumped tick-by-tick to a trace file; the dashboard just scrubs through that
          fixed recording. Use the PLANNING / RL AGENT toggle above to switch between the two
          recordings.
        </p>

        <p style={{ fontSize: 13.5, lineHeight: 1.65, color: 'var(--text)', margin: '0 0 14px' }}>
          <strong>PLANNING</strong> replays Member 2's A*/MCTS search on the flat demo tile level
          (<code style={{ fontFamily: mono, fontSize: 12 }}>demo/demo_env.py</code>). <strong>RL AGENT</strong> replays
          Member 3's trained PPO policy acting greedily in its own native training world
          (<code style={{ fontFamily: mono, fontSize: 12 }}>core/env.py</code>'s continuous-physics environment —
          gravity, jump arcs, no fixed tile grid). These are two different worlds with incompatible
          observation spaces, which is why the RL policy isn't simply dropped into the tile-level
          loop: the recording shown is one genuine episode on an unseen seed, and this checkpoint's
          own rigorous evaluation (100 unseen episodes) lands around a 31% success rate — most
          individual episodes don't reach the goal.
        </p>

        <div style={{ ...{}, borderTop: '1px solid var(--line)', margin: '16px 0', paddingTop: 14 }}>
          <div style={{ fontFamily: mono, fontSize: 10, color: 'var(--text-dim)', letterSpacing: '.08em', marginBottom: 8 }}>
            HOW TO READ THE VIEWPORT
          </div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.7, color: 'var(--text-dim)' }}>
            <li><span style={{ color: 'var(--text)' }}>White dot</span> — the agent</li>
            <li><span style={{ color: 'var(--danger)' }}>Red dot</span> — the patrolling enemy</li>
            <li><span style={{ color: 'var(--risk)' }}>Orange bar</span> — the moving platform</li>
            <li><span style={{ color: '#F2E14A' }}>Yellow diamond</span> — the collectible (disappears once picked up)</li>
            <li><span style={{ color: 'var(--live)' }}>Teal dashed line</span> — the goal tile and the agent's trail so far</li>
          </ul>
        </div>

        <div style={{ borderTop: '1px solid var(--line)', paddingTop: 14 }}>
          <div style={{ fontFamily: mono, fontSize: 10, color: 'var(--text-dim)', letterSpacing: '.08em', marginBottom: 8 }}>
            CONTROLS
          </div>
          <p style={{ margin: 0, fontSize: 13, lineHeight: 1.7, color: 'var(--text-dim)' }}>
            Space to play/pause, ←/→ to step one tick, drag the scrubber or click anywhere on the
            chart above to jump to a tick, and the speed selector controls playback rate.
          </p>
        </div>
      </div>
    </div>
  )
}
