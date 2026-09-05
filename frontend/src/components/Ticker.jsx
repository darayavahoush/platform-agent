import { mono, STAT_TONES } from '../theme.js'

function Stat({ label: text, value, tone }) {
  return (
    <div style={{ fontFamily: mono }}>
      <span style={{
        fontSize: 9.5, color: 'var(--text-dim)', letterSpacing: '.1em',
        textTransform: 'uppercase', display: 'block', marginBottom: 3,
      }}>{text}</span>
      <span style={{ fontSize: 15, fontWeight: 700, color: STAT_TONES[tone] || STAT_TONES.default }}>{value}</span>
    </div>
  )
}

export default function Ticker({ frame }) {
  // Guards the render for the one tick where trace/idx haven't settled yet
  // (e.g. an out-of-range scrub, or a trace with 0 ticks) instead of
  // crashing on frame.risk.toFixed(...).
  if (!frame) {
    return (
      <div style={{
        borderTop: '1px solid var(--line)', padding: '14px 28px',
        fontFamily: mono, fontSize: 11, color: 'var(--text-dim)', background: 'var(--panel-2)',
      }}>
        NO FRAME AT CURRENT INDEX
      </div>
    )
  }

  return (
    <div style={{
      borderTop: '1px solid var(--line)', padding: '14px 28px', display: 'flex',
      gap: 28, flexWrap: 'wrap', background: 'var(--panel-2)',
    }}>
      <Stat label="Tick" value={frame.t} />
      <Stat label="Player Tile" value={frame.player_tile} />
      <Stat label="Action" value={frame.action} tone="live" />
      <Stat label="Risk" value={frame.risk?.toFixed(2) ?? '—'} tone="risk" />
      <Stat label="Reward (Σ)" value={frame.reward_total?.toFixed(2) ?? '—'} />
      <Stat label="Collectibles" value={`${frame.collected ? 1 : 0}/1`} />
      <Stat label="Hazard Hits" value={frame.hits ?? 0} tone={(frame.hits ?? 0) > 0 ? 'danger' : 'default'} />
    </div>
  )
}
