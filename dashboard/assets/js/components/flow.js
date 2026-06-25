export function drawFlow(ev) {
  const classes = ev.class_counts || {};
  const arch = ev.arch || {};
  const W = 760, H = 130;
  const n = ev.n_patterns || 200;
  let html = `<svg width="100%" viewBox="0 0 ${W} ${H}" style="min-width:600px">`;
  html += `<rect x="10" y="30" width="130" height="70" rx="8" fill="rgba(99,102,241,0.08)" stroke="rgba(99,102,241,0.3)" stroke-width="1"/>`;
  html += `<text x="75" y="52" text-anchor="middle" fill="#6366f1" font-size="11" font-weight="600">Batch ${(ev.session||0)+1} / 6</text>`;
  html += `<text x="75" y="68" text-anchor="middle" fill="#64748b" font-size="10">${n} patrones</text>`;
  const classArr = Object.entries(classes).slice(0, 10);
  classArr.forEach(([cls, cnt], i) => {
    const px = 10 + (i % 5) * 24;
    const py = 78 + Math.floor(i/5)*14;
    html += `<rect x="${px}" y="${py}" width="20" height="11" rx="3" fill="rgba(99,102,241,0.2)"/>`;
    html += `<text x="${px+10}" y="${py+8.5}" text-anchor="middle" fill="#6366f1" font-size="8">${cls}:${cnt}</text>`;
  });
  html += `<line x1="140" y1="65" x2="195" y2="65" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>`;
  const nHid = arch.n_hid || '?';
  html += `<rect x="200" y="20" width="160" height="90" rx="8" fill="rgba(99,102,241,0.05)" stroke="rgba(99,102,241,0.3)" stroke-width="1"/>`;
  html += `<text x="280" y="45" text-anchor="middle" fill="#6366f1" font-size="11" font-weight="600">MLP</text>`;
  html += `<text x="280" y="62" text-anchor="middle" fill="#64748b" font-size="10">64 → ${nHid} → 10</text>`;
  if (arch.use_dual) {
    html += `<text x="280" y="92" text-anchor="middle" fill="#f59e0b" font-size="9">w + fw¹ (Bullinaria)</text>`;
    html += `<text x="280" y="104" text-anchor="middle" fill="#e05a3a" font-size="9">+ fw² extensión</text>`;
  }
  html += `<line x1="360" y1="65" x2="415" y2="65" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>`;
  html += `<rect x="420" y="30" width="130" height="70" rx="8" fill="rgba(34,197,94,0.08)" stroke="rgba(34,197,94,0.3)" stroke-width="1"/>`;
  html += `<text x="485" y="52" text-anchor="middle" fill="#22c55e" font-size="11" font-weight="600">Backprop</text>`;
  html += `<text x="485" y="82" text-anchor="middle" fill="#64748b" font-size="9">4 learning rates</text>`;
  html += `<line x1="550" y1="65" x2="605" y2="65" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr)"/>`;
  html += `<rect x="610" y="30" width="130" height="70" rx="8" fill="rgba(245,158,11,0.08)" stroke="rgba(245,158,11,0.3)" stroke-width="1"/>`;
  html += `<text x="675" y="52" text-anchor="middle" fill="#f59e0b" font-size="11" font-weight="600">Fitness val</text>`;
  html += `<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#94a3b8" stroke-width="1.5"/></marker></defs>`;
  html += '</svg>';
  document.getElementById('flow-svg-wrap').innerHTML = html;
}
