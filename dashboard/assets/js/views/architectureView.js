function hash01(seed) {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function shouldDrawConnection(layerSeed, fromIndex, toIndex, density) {
  return hash01(layerSeed + fromIndex * 101 + toIndex * 17) <= density;
}

const WEIGHT_STYLES = [
  { stroke: 'rgba(99, 102, 241, 0.35)', width: 1.2, dash: '' },
  { stroke: 'rgba(245, 158, 11, 0.25)', width: 1.5, dash: '3 3' },
  { stroke: 'rgba(224, 90, 58, 0.22)', width: 1.5, dash: '5 3 1 3' },
  { stroke: 'rgba(20, 184, 166, 0.22)', width: 1.5, dash: '2 4' },
  { stroke: 'rgba(168, 85, 247, 0.22)', width: 1.5, dash: '6 2' },
];

function weightStyle(idx) {
  const style = WEIGHT_STYLES[idx % WEIGHT_STYLES.length];
  const dash = style.dash ? ` stroke-dasharray="${style.dash}"` : '';
  return `stroke="${style.stroke}" stroke-width="${style.width}"${dash}`;
}

function escapeText(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function weightLabelsFor(genotype) {
  const labels = Array.isArray(genotype.weightLabels) && genotype.weightLabels.length
    ? genotype.weightLabels
    : genotype.weight_labels;
  const cleaned = Array.isArray(labels) ? labels.filter(label => typeof label === 'string' && label.trim()) : ['w'];
  return genotype.useDual === false ? cleaned.slice(0, 1) : cleaned;
}

function renderWeightedLines(x1, y1, x2, y2, labels) {
  return labels.map((label, idx) => (
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="arch-link" ${weightStyle(idx)}><title>${escapeText(label)}</title></line>`
  )).join('');
}

export function renderArchitecture(elements, genotype) {
  if (!genotype) return;
  const nHid = genotype.n_hid;
  const inputCount = genotype.nInputs || genotype.n_inputs || genotype.n_in || 0;
  const outputCount = genotype.nOutputs || genotype.n_outputs || genotype.n_out || 0;
  const cIH = typeof genotype.c_ih === 'number' ? genotype.c_ih : 1;
  const cHO = typeof genotype.c_ho === 'number' ? genotype.c_ho : 1;
  const weightLabels = weightLabelsFor(genotype);
  const W = 600;
  const H = 220;
  const inputX = 80;
  const hiddenX = W / 2;
  const outputX = W - 80;
  const hiddenCount = nHid;
  const maxInputShow = 8;
  const maxHiddenShow = Math.min(nHid, 10);
  const maxOutputShow = Math.min(outputCount, 10);

  const getY = (idx, total, maxShow) => {
    const showCount = Math.min(total, maxShow);
    const y = 30 + (idx * (H - 60) / (showCount - 1 || 1));
    return Math.min(H - 25, Math.max(25, y));
  };

  let html = `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" class="architecture-svg">`;

  for (let i = 0; i < Math.min(inputCount, maxInputShow); i += 1) {
    for (let j = 0; j < Math.min(hiddenCount, maxHiddenShow); j += 1) {
      if (!shouldDrawConnection(11, i, j, cIH)) continue;
      const y1 = getY(i, inputCount, maxInputShow);
      const y2 = getY(j, hiddenCount, maxHiddenShow);
      html += renderWeightedLines(inputX, y1, hiddenX, y2, weightLabels);
    }
  }

  for (let i = 0; i < Math.min(hiddenCount, maxHiddenShow); i += 1) {
    for (let j = 0; j < Math.min(outputCount, maxOutputShow); j += 1) {
      if (!shouldDrawConnection(29, i, j, cHO)) continue;
      const y1 = getY(i, hiddenCount, maxHiddenShow);
      const y2 = getY(j, outputCount, maxOutputShow);
      html += renderWeightedLines(hiddenX, y1, outputX, y2, weightLabels);
    }
  }

  for (let i = 0; i < Math.min(inputCount, maxInputShow); i += 1) {
    const y = getY(i, inputCount, maxInputShow);
    html += `<circle cx="${inputX}" cy="${y}" r="7" class="arch-node arch-node-input"><title>Neurona entrada ${i + 1} / ${inputCount}</title></circle>`;
  }

  for (let i = 0; i < Math.min(hiddenCount, maxHiddenShow); i += 1) {
    const y = getY(i, hiddenCount, maxHiddenShow);
    html += `<circle cx="${hiddenX}" cy="${y}" r="8" class="arch-node arch-node-hidden"><title>Neurona oculta ${i + 1} / ${hiddenCount}</title></circle>`;
  }

  for (let i = 0; i < Math.min(outputCount, maxOutputShow); i += 1) {
    const y = getY(i, outputCount, maxOutputShow);
    html += `<circle cx="${outputX}" cy="${y}" r="7" class="arch-node arch-node-output"><title>Neurona salida ${i + 1} / ${outputCount}</title></circle>`;
  }

  html += `<text x="${inputX}" y="${H - 12}" text-anchor="middle" class="arch-label">${inputCount} inputs</text>`;
  html += `<text x="${hiddenX}" y="${H - 12}" text-anchor="middle" class="arch-label">${nHid} ocultas</text>`;
  html += `<text x="${outputX}" y="${H - 12}" text-anchor="middle" class="arch-label">${outputCount} salidas</text>`;

  const legendWidth = Math.max(92, 34 + weightLabels.reduce((sum, label) => sum + Math.max(38, label.length * 8), 0));
  html += `<rect x="12" y="6" width="${legendWidth}" height="28" rx="6" class="arch-legend-box"/>`;
  let legendX = 20;
  weightLabels.forEach((label, idx) => {
    html += `<line x1="${legendX}" y1="18" x2="${legendX + 14}" y2="18" class="arch-legend" ${weightStyle(idx)}/>`;
    html += `<text x="${legendX + 18}" y="22" class="arch-legend-text">${escapeText(label)}</text>`;
    legendX += Math.max(38, label.length * 8);
  });
  html += `<text x="20" y="32" class="arch-density">conectividad: ${(cIH * 100).toFixed(0)}% / ${(cHO * 100).toFixed(0)}%</text>`;
  html += '</svg>';

  elements.architectureWrap.innerHTML = html;
}
