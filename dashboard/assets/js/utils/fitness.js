export function fitnessColor(f) {
  if (f >= 0.85) return '#22c55e';
  if (f >= 0.65) return '#f59e0b';
  return '#e05a3a';
}

export function fitnessOpacity(f) {
  return String(0.5 + f * 0.5);
}
