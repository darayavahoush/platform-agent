import { useEffect, useRef, useState, useCallback } from 'react'
import { Play, Pause, Radio, Info } from 'lucide-react'
import CanvasViewport from './components/CanvasViewport.jsx'
import ModulePanel from './components/ModulePanel.jsx'
import Ticker from './components/Ticker.jsx'
import TelemetryChart from './components/TelemetryChart.jsx'
import PipelineFlow from './components/PipelineFlow.jsx'
import About from './components/About.jsx'
import { mono, sans } from './theme.js'

const SPEEDS = [
  { label: '1×', ms: 60 },
  { label: '2×', ms: 30 },
  { label: '4×', ms: 15 },
  { label: '12×', ms: 5 },
]

export default function App() {
  const [trace, setTrace] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedMs, setSpeedMs] = useState(15)
  const [showAbout, setShowAbout] = useState(false)
  const timerRef = useRef(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}trace.json`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(setTrace)
      .catch(err => {
        console.error('Failed to load trace.json', err)
        setLoadError(err.message)
      })
  }, [])

  const ticks = trace?.ticks
  const level = trace?.level

  const step = useCallback(() => {
    setIdx(i => {
      if (!ticks) return i
      if (i >= ticks.length - 1) {
        setPlaying(false)
        return i
      }
      return i + 1
    })
  }, [ticks])

  useEffect(() => {
    if (!playing) return
    timerRef.current = setInterval(step, speedMs)
    return () => clearInterval(timerRef.current)
  }, [playing, speedMs, step])

  const togglePlay = useCallback(() => {
    if (!ticks) return
    setPlaying(p => {
      if (!p && idx >= ticks.length - 1) setIdx(0)
      return !p
    })
  }, [ticks, idx])

  useEffect(() => {
    function onKey(e) {
      if (!ticks) return
      if (e.code === 'Space') { e.preventDefault(); togglePlay() }
      if (e.code === 'ArrowRight') { setPlaying(false); setIdx(i => Math.min(i + 1, ticks.length - 1)) }
      if (e.code === 'ArrowLeft') { setPlaying(false); setIdx(i => Math.max(i - 1, 0)) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [ticks, togglePlay])

  if (loadError) {
    return (
      <div style={{
        minHeight: '100vh', display: 'grid', placeItems: 'center', textAlign: 'center',
        fontFamily: mono, color: 'var(--danger)', fontSize: 12, letterSpacing: '.05em', padding: 24,
      }}>
        FAILED TO LOAD trace.json — {loadError}
        <br />
        <span style={{ color: 'var(--text-dim)' }}>Regenerate it with `python3 demo/run_demo.py` from the repo root.</span>
      </div>
    )
  }

  if (!trace || !ticks?.length) {
    return (
      <div style={{
        minHeight: '100vh', display: 'grid', placeItems: 'center',
        fontFamily: mono, color: 'var(--text-dim)', fontSize: 12, letterSpacing: '.1em',
      }}>
        LOADING TRACE…
      </div>
    )
  }

  const frame = ticks[idx]

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <div style={{
        maxWidth: 1180, margin: '0 auto',
        borderLeft: '1px solid var(--line)', borderRight: '1px solid var(--line)',
        background: 'linear-gradient(var(--line) 1px, transparent 1px) 0 0/100% 28px, var(--bg)',
      }}>
        <header style={{
          padding: '24px 28px 18px', borderBottom: '1px solid var(--line)',
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap',
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <Radio size={16} color="var(--live)" style={{ filter: 'drop-shadow(0 0 4px var(--live))' }} />
              <span style={{ fontFamily: mono, fontSize: 10.5, letterSpacing: '.16em', color: 'var(--live)' }}>
                RECORDED RUN
              </span>
            </div>
            <h1 style={{
              fontFamily: sans, fontWeight: 700, fontSize: 26, margin: '0 0 4px', letterSpacing: '-0.01em',
            }}>
              NAV_TRACE
            </h1>
            <p style={{ margin: 0, fontFamily: sans, fontSize: 13, color: 'var(--text-dim)' }}>
              4-module autonomous platformer agent · replaying Level 01 via the A*/MCTS planner
            </p>
          </div>
          <button
            onClick={() => setShowAbout(true)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              fontFamily: mono, fontSize: 11, letterSpacing: '.06em',
              background: 'var(--panel)', border: '1px solid var(--line)', color: 'var(--text)',
              padding: '8px 14px', borderRadius: 6, cursor: 'pointer', flexShrink: 0,
            }}
          >
            <Info size={13} /> ABOUT
          </button>
        </header>

        <div style={{ padding: '16px 28px', borderBottom: '1px solid var(--line)' }}>
          <PipelineFlow />
        </div>

        <div style={{ padding: '14px 28px 0', borderBottom: '1px solid var(--line)' }}>
          <TelemetryChart ticks={ticks} idx={idx} onScrub={(i) => { setPlaying(false); setIdx(Math.max(0, Math.min(i, ticks.length - 1))) }} />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px' }}>
          <main style={{ padding: '18px 28px' }}>
            <div style={{
              fontFamily: mono, fontSize: 11, color: 'var(--text-dim)', letterSpacing: '.1em',
              textTransform: 'uppercase', marginBottom: 10, display: 'flex', justifyContent: 'space-between',
            }}>
              <span>Agent Viewport</span>
              <span>Path Risk <b style={{ color: 'var(--live)' }}>{frame.risk.toFixed(2)}</b></span>
            </div>

            <CanvasViewport level={level} ticks={ticks} idx={idx} />

            <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 12 }}>
              <button
                onClick={togglePlay}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  fontFamily: mono, fontSize: 12, letterSpacing: '.06em',
                  background: 'var(--panel)', border: '1px solid var(--line)', color: 'var(--text)',
                  padding: '8px 14px', borderRadius: 5, cursor: 'pointer',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--live)'; e.currentTarget.style.color = 'var(--live)' }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--line)'; e.currentTarget.style.color = 'var(--text)' }}
              >
                {playing ? <Pause size={13} /> : <Play size={13} />}
                {playing ? 'PAUSE' : 'PLAY'}
              </button>

              <input
                type="range" min={0} max={ticks.length - 1} value={idx}
                onChange={e => { setPlaying(false); setIdx(Number(e.target.value)) }}
                style={{ flex: 1, accentColor: 'var(--live)' }}
                aria-label="Scrub through trace"
              />

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: mono, fontSize: 11, color: 'var(--text-dim)' }}>
                SPEED
                <select
                  value={speedMs} onChange={e => setSpeedMs(Number(e.target.value))}
                  style={{
                    background: 'var(--panel)', color: 'var(--text)', border: '1px solid var(--line)',
                    borderRadius: 4, fontFamily: 'inherit', fontSize: 11, padding: '4px 6px',
                  }}
                >
                  {SPEEDS.map(s => <option key={s.ms} value={s.ms}>{s.label}</option>)}
                </select>
              </div>
            </div>
          </main>

          <ModulePanel />
        </div>

        <Ticker frame={frame} />
      </div>

      {showAbout && <About onClose={() => setShowAbout(false)} />}
    </div>
  )
}
