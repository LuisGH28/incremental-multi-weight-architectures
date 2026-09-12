export function formatPercent(value, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatElapsedSeconds(seconds) {
  return `${Math.max(0, Math.round(seconds))}s`;
}
