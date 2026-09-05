import { AreaChart, Area, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts'
import { mono } from '../theme.js'

function CustomTooltip({ active, payload, label, showRisk }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div style={{
      background: 'var(--panel)', border: '1px solid var(--line)', borderRadius: 5,
      padding: '8px 10px', fontFamily: mono, fontSize: 11,
    }}>
      <div style={{ color: 'var(--text-dim)', marginBottom: 4 }}>TICK {label}</div>
      {showRisk && <div style={{ color: 'var(--risk)' }}>risk {payload[0]?.value?.toFixed(2)}</div>}
      <div style={{ color: 'var(--live)' }}>reward Σ {payload[showRisk ? 1 : 0]?.value?.toFixed(2)}</div>
    </div>
  )
}

function Legend({ showRisk }) {
  const items = showRisk
    ? [{ color: 'var(--risk)', text: 'Risk' }, { color: 'var(--live)', text: 'Reward (Σ)' }]
    : [{ color: 'var(--live)', text: 'Reward (Σ)' }]
  return (
    <div style={{ display: 'flex', gap: 14, marginBottom: 4 }}>
      {items.map(i => (
        <div key={i.text} style={{ display: 'flex', alignItems: 'center', gap: 5, fontFamily: mono, fontSize: 10, color: 'var(--text-dim)' }}>
          <span style={{ width: 8, height: 2, background: i.color, display: 'inline-block' }} />
          {i.text}
        </div>
      ))}
    </div>
  )
}

// mode: 'planning' (has per-tick risk from the A*/MCTS planner) or
// 'rl' (PPO ticks carry no risk score, so that line is omitted rather than
// showing a flat/undefined series).
export default function TelemetryChart({ ticks, idx, onScrub, mode = 'planning' }) {
  if (!ticks?.length) return null
  const showRisk = mode === 'planning'
  const data = ticks.map(t => ({ t: t.t, risk: t.risk, reward: t.reward_total }))
  const currentT = ticks[idx]?.t

  return (
    <div>
      <Legend showRisk={showRisk} />
      <div style={{ height: 82 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 6, right: 4, bottom: 0, left: 4 }}
            onClick={(e) => {
              if (e && e.activeLabel != null) onScrub(Number(e.activeLabel) - 1)
            }}
          >
            <defs>
              <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#F2A44A" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#F2A44A" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="rewardFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#45E0C4" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#45E0C4" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="t" hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip content={<CustomTooltip showRisk={showRisk} />} cursor={{ stroke: 'var(--line)' }} />
            {showRisk && (
              <Area type="monotone" dataKey="risk" stroke="#F2A44A" strokeWidth={1.5} fill="url(#riskFill)" isAnimationActive={false} />
            )}
            <Area type="monotone" dataKey="reward" stroke="#45E0C4" strokeWidth={1.5} fill="url(#rewardFill)" isAnimationActive={false} />
            <ReferenceLine x={currentT} stroke="#E7ECF2" strokeWidth={1} strokeDasharray="2 2" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
