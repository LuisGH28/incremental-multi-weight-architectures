function svgText(x, y, text, attrs = '') {
  return `<text x="${x}" y="${y}" ${attrs}>${text}</text>`;
}

export function renderFlow(elements, ev) {
  const classes = ev.class_counts || {};
  const arch = ev.arch || {};
  const W = 760;
  const H = 130;
  const n = ev.n_patterns || 200;
  const nHid = arch.n_hid || '?';
  const fwLines = arch.n_fw || 4;
  let html = `<svg width="100%" viewBox="0 0 ${W} ${H}" class="flow-svg">`;

  html += '<rect x="10" y="30" width="130" height="70" rx="8" class="flow-node flow-node-input"/>';
  html += svgText(75, 52, `Batch ${(ev.session || 0) + 1} / 6`, 'text-anchor="middle" class="flow-title flow-title-input"');
  html += svgText(75, 68, `${n} patrones`, 'text-anchor="middle" class="flow-muted"');

  Object.entries(classes).slice(0, 10).forEach(([cls, cnt], i) => {
    const px = 10 + (i % 5) * 24;
    const py = 78 + Math.floor(i / 5) * 14;
    html += `<rect x="${px}" y="${py}" width="20" height="11" rx="3" class="flow-class-pill"/>`;
    html += svgText(px + 10, py + 8.5, `${cls}:${cnt}`, 'text-anchor="middle" class="flow-class-text"');
  });

  html += '<line x1="140" y1="65" x2="195" y2="65" class="flow-arrow" marker-end="url(#arr)"/>';
  html += '<rect x="200" y="20" width="160" height="90" rx="8" class="flow-node flow-node-model"/>';
  html += svgText(280, 45, 'MLP', 'text-anchor="middle" class="flow-title flow-title-model"');
  html += svgText(280, 62, `64 -> ${nHid} -> 10`, 'text-anchor="middle" class="flow-muted"');
  if (arch.use_dual) {
    html += svgText(280, 86, `w + fw1 + fw2 + fw3 (${fwLines} líneas)`, 'text-anchor="middle" class="flow-fw3"');
  }

  html += '<line x1="360" y1="65" x2="415" y2="65" class="flow-arrow" marker-end="url(#arr)"/>';
  html += '<rect x="420" y="30" width="130" height="70" rx="8" class="flow-node flow-node-train"/>';
  html += svgText(485, 52, 'Backprop', 'text-anchor="middle" class="flow-title flow-title-train"');
  html += svgText(485, 82, '4 learning rates', 'text-anchor="middle" class="flow-muted"');
  html += '<line x1="550" y1="65" x2="605" y2="65" class="flow-arrow" marker-end="url(#arr)"/>';
  html += '<rect x="610" y="30" width="130" height="70" rx="8" class="flow-node flow-node-fitness"/>';
  html += svgText(675, 52, 'Fitness val', 'text-anchor="middle" class="flow-title flow-title-fitness"');
  html += '<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" class="flow-arrow-head"/></marker></defs>';
  html += '</svg>';

  elements.flowWrap.innerHTML = html;
}
