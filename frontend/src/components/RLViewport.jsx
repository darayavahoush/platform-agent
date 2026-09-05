import { useEffect, useRef } from 'react'
import { mono } from '../theme.js'

const LEGEND = [
  { color: '#E7ECF2', label: 'Agent' },
  { color: '#F2545B', label: 'Enemy' },
  { color: '#F2A44A', label: 'Moving platform' },
  { color: '#9AA7B5', label: 'Static platform' },
  { color: '#C93A44', label: 'Hazard' },
  { color: '#45E0C4', label: 'Goal / trail' },
]

// Renders MockPlatformEnv's native continuous-physics world (900x500,
// real gravity + jump arcs) — a different coordinate system from the flat
// tile grid CanvasViewport draws for the Planning-only demo world.
export default function RLViewport({ world, playerSize, staticGeo, ticks, idx }) {
  const canvasRef = useRef(null)
  const W = world.width
  const H = world.height

  useEffect(() => {
    const canvas = canvasRef.current
    const frame = ticks[idx]
    if (!canvas || !frame) return
    const ctx = canvas.getContext('2d')

    ctx.clearRect(0, 0, W, H)

    // subtle grid
    ctx.strokeStyle = '#141B23'
    ctx.lineWidth = 1
    for (let x = 0; x <= W; x += 60) {
      ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); ctx.stroke()
    }

    // static platforms
    ctx.fillStyle = '#1B2530'
    for (const p of staticGeo.platforms) {
      ctx.fillRect(p.x, p.y, p.width, p.height)
      ctx.fillStyle = '#2C3A48'
      ctx.fillRect(p.x, p.y, p.width, 2)
      ctx.fillStyle = '#1B2530'
    }

    // static hazards
    ctx.fillStyle = '#C93A44'
    ctx.shadowColor = '#C93A44'; ctx.shadowBlur = 6
    for (const h of staticGeo.hazards) {
      ctx.fillRect(h.x, h.y, h.width, h.height)
    }
    ctx.shadowBlur = 0

    // moving platforms (this tick's position)
    ctx.fillStyle = '#F2A44A'
    ctx.shadowColor = '#F2A44A'; ctx.shadowBlur = 8
    for (const m of frame.moving_platforms) {
      ctx.fillRect(m.x, m.y, m.width, m.height)
    }
    ctx.shadowBlur = 0

    // enemies (this tick's position)
    ctx.fillStyle = '#F2545B'
    for (const e of frame.enemies) {
      ctx.beginPath()
      ctx.arc(e.x + e.width / 2, e.y + e.height / 2, e.width / 2, 0, Math.PI * 2)
      ctx.fill()
    }

    // goal marker
    const [gx, gy] = staticGeo.goal
    ctx.strokeStyle = '#45E0C4'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.arc(gx, gy, 16, 0, Math.PI * 2)
    ctx.stroke()
    ctx.setLineDash([])

    // trail up to current index
    ctx.strokeStyle = 'rgba(69,224,196,0.5)'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i <= idx; i++) {
      const px = ticks[i].player.x + playerSize.w / 2
      const py = ticks[i].player.y + playerSize.h / 2
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
    }
    ctx.stroke()

    // player
    ctx.shadowColor = '#45E0C4'; ctx.shadowBlur = 10
    ctx.fillStyle = '#E7ECF2'
    ctx.fillRect(frame.player.x, frame.player.y, playerSize.w, playerSize.h)
    ctx.shadowBlur = 0
  }, [idx, ticks, world, staticGeo, playerSize, W, H])

  return (
    <div style={{
      border: '1px solid var(--line)', borderRadius: 6, overflow: 'hidden',
      background: 'radial-gradient(circle at 30% 20%, rgba(69,224,196,0.04), transparent 60%), #0C1218',
    }}>
      <canvas ref={canvasRef} width={W} height={H} style={{ display: 'block', width: '100%', height: 'auto' }} />
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: '6px 16px', padding: '8px 12px',
        borderTop: '1px solid var(--line)', background: 'var(--panel)',
      }}>
        {LEGEND.map(l => (
          <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: mono, fontSize: 10, color: 'var(--text-dim)' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: l.color, display: 'inline-block' }} />
            {l.label}
          </div>
        ))}
      </div>
    </div>
  )
}
