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

// mode: 'planning' ticks carry tile position/risk/collected/hits from the
// tile-grid demo; 'rl' ticks carry continuous x/y from MockPlatformEnv and
// no risk score (the planner's risk model isn't part of the RL agent).
export default function Ticker({ frame, mode = 'planning' }) {
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

  const stats = mode === 'rl'
    ? [
        <Stat key="t" label="Tick" value={frame.t} />,
        <Stat key="pos" label="Player X, Y" value={`${Math.round(frame.player.x)}, ${Math.round(frame.player.y)}`} />,
        <Stat key="action" label="Action" value={frame.action} tone="live" />,
        <Stat key="reward" label="Reward" value={frame.reward?.toFixed(2) ?? '—'} />,
        <Stat key="rewardTotal" label="Reward (Σ)" value={frame.reward_total?.toFixed(2) ?? '—'} />,
        <Stat key="enemies" label="Enemies Nearby" value={frame.enemies?.length ?? 0} tone={(frame.enemies?.length ?? 0) > 0 ? 'danger' : 'default'} />,
      ]
    : [
        <Stat key="t" label="Tick" value={frame.t} />,
        <Stat key="tile" label="Player Tile" value={frame.player_tile} />,
        <Stat key="action" label="Action" value={frame.action} tone="live" />,
        <Stat key="risk" label="Risk" value={frame.risk?.toFixed(2) ?? '—'} tone="risk" />,
        <Stat key="rewardTotal" label="Reward (Σ)" value={frame.reward_total?.toFixed(2) ?? '—'} />,
        <Stat key="collected" label="Collectibles" value={`${frame.collected ? 1 : 0}/1`} />,
        <Stat key="hits" label="Hazard Hits" value={frame.hits ?? 0} tone={(frame.hits ?? 0) > 0 ? 'danger' : 'default'} />,
      ]

  return (
    <div style={{
      borderTop: '1px solid var(--line)', padding: '14px 28px', display: 'flex',
      gap: 28, flexWrap: 'wrap', background: 'var(--panel-2)',
    }}>
      {stats}
    </div>
  )
}
