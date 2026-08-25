function Stat({ label, value, tone }) {
  const colors = { live: 'var(--live)', risk: 'var(--risk)', danger: 'var(--danger)', default: 'var(--text)' }
  return (
    <div style={{ fontFamily: 'var(--mono)' }}>
      <span style={{
        fontSize: 9.5, color: 'var(--text-dim)', letterSpacing: '.1em',
        textTransform: 'uppercase', display: 'block', marginBottom: 3,
      }}>{label}</span>
      <span style={{ fontSize: 15, fontWeight: 700, color: colors[tone] || colors.default }}>{value}</span>
    </div>
  )
}

export default function Ticker({ frame }) {
  return (
    <div style={{
      borderTop: '1px solid var(--line)', padding: '14px 28px', display: 'flex',
      gap: 28, flexWrap: 'wrap', background: 'var(--panel-2)',
    }}>
      <Stat label="Tick" value={frame.t} />
      <Stat label="Player Tile" value={frame.player_tile} />
      <Stat label="Action" value={frame.action} tone="live" />
      <Stat label="Risk" value={frame.risk.toFixed(2)} tone="risk" />
      <Stat label="Reward (Σ)" value={frame.reward_total.toFixed(2)} />
      <Stat label="Collectibles" value={`${frame.collected ? 1 : 0}/1`} />
      <Stat label="Hazard Hits" value={frame.hits} tone={frame.hits > 0 ? 'danger' : 'default'} />
    </div>
  )
}
