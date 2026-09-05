// Shared style primitives for the NAV_TRACE dashboard.
// Colors/fonts are CSS custom properties in index.css; this file holds the
// *shapes* every component repeated inline (labels, cards, pills, glows).

export const mono = 'var(--mono)'
export const sans = 'var(--sans)'

export const label = {
  fontFamily: mono,
  fontSize: 11,
  color: 'var(--text-dim)',
  letterSpacing: '.1em',
  textTransform: 'uppercase',
}

export const panel = {
  border: '1px solid var(--line)',
  borderRadius: 8,
  background: 'var(--panel)',
}

const PILL_TONES = {
  live: { color: 'var(--live)', background: 'var(--live-dim)' },
  pending: { color: 'var(--text-dim)', background: '#1A212B' },
  danger: { color: 'var(--danger)', background: '#3A171A' },
}

export function pillStyle(tone = 'pending') {
  const t = PILL_TONES[tone] || PILL_TONES.pending
  return {
    fontFamily: mono,
    fontSize: 9.5,
    letterSpacing: '.08em',
    fontWeight: 700,
    padding: '2px 7px',
    borderRadius: 20,
    color: t.color,
    background: t.background,
    whiteSpace: 'nowrap',
  }
}

export const STAT_TONES = {
  live: 'var(--live)',
  risk: 'var(--risk)',
  danger: 'var(--danger)',
  default: 'var(--text)',
}

// Soft glow used on the active pipeline node / live indicators.
export const glow = (hex) => `drop-shadow(0 0 6px ${hex})`
