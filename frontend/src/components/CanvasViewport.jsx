import { useEffect, useRef } from 'react'

const groundY = 150

function movingPlatformTile(level, t) {
  const mp = level.moving_platform
  const phase = (t % mp.period) / mp.period
  const tri = (4 * Math.abs(phase - 0.5) - 1) * mp.amplitude_tiles
  return Math.round(mp.home_tile + tri)
}

function enemyTile(level, t) {
  const [lo, hi] = level.enemy.patrol_tiles
  const span = hi - lo
  if (span === 0) return lo
  const cyc = t % (2 * span)
  return lo + Math.abs(cyc - span)
}

export default function CanvasViewport({ level, ticks, idx }) {
  const canvasRef = useRef(null)
  const TW = level.tile_w
  const NT = level.n_tiles
  const W = NT * TW
  const H = 200

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const frame = ticks[idx]

    ctx.clearRect(0, 0, W, H)

    // subtle grid
    ctx.strokeStyle = '#141B23'
    ctx.lineWidth = 1
    for (let x = 0; x <= W; x += TW) {
      ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); ctx.stroke()
    }

    // ground / gaps
    for (let i = 0; i < NT; i++) {
      if (level.ground[i] === 1) {
        const x = i * TW
        ctx.fillStyle = '#1B2530'
        ctx.fillRect(x, groundY, TW, H - groundY)
        ctx.fillStyle = '#25313F'
        ctx.fillRect(x, groundY, TW, 2)
      }
    }

    // moving platform
    const mpTile = movingPlatformTile(level, frame.t)
    ctx.shadowColor = '#F2A44A'; ctx.shadowBlur = 8
    ctx.fillStyle = '#F2A44A'
    ctx.fillRect(mpTile * TW + 2, groundY - 4, TW - 4, 6)
    ctx.shadowBlur = 0

    // enemy
    const enTile = enemyTile(level, frame.t)
    ctx.fillStyle = '#F2545B'
    ctx.beginPath()
    ctx.arc(enTile * TW + TW / 2, groundY - 10, 6, 0, Math.PI * 2)
    ctx.fill()

    // collectible
    if (!frame.collected) {
      ctx.save()
      ctx.translate(level.collectible_tile * TW + TW / 2, groundY - 12)
      ctx.rotate(Math.PI / 4)
      ctx.fillStyle = '#F2E14A'
      ctx.fillRect(-5, -5, 10, 10)
      ctx.restore()
    }

    // goal marker
    ctx.strokeStyle = '#45E0C4'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(level.goal_tile * TW + TW / 2, groundY - 40)
    ctx.lineTo(level.goal_tile * TW + TW / 2, groundY)
    ctx.stroke()
    ctx.setLineDash([])

    // trail up to current index
    ctx.strokeStyle = 'rgba(69,224,196,0.55)'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i <= idx; i++) {
      const px = ticks[i].player_tile * TW + TW / 2
      const py = groundY - 14
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
    }
    ctx.stroke()

    // player
    const px = frame.player_tile * TW + TW / 2
    ctx.shadowColor = '#45E0C4'; ctx.shadowBlur = 10
    ctx.fillStyle = '#E7ECF2'
    ctx.beginPath()
    ctx.arc(px, groundY - 14, 7, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0
  }, [idx, ticks, level, W])

  return (
    <div style={{
      border: '1px solid var(--line)', borderRadius: 6, overflow: 'hidden',
      background: 'radial-gradient(circle at 30% 20%, rgba(69,224,196,0.04), transparent 60%), #0C1218',
    }}>
      <canvas ref={canvasRef} width={W} height={H} style={{ display: 'block', width: '100%', height: 'auto' }} />
    </div>
  )
}
