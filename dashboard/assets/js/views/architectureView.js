function hash01(seed) {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function shouldDrawConnection(layerSeed, fromIndex, toIndex, density) {
  return hash01(layerSeed + fromIndex * 101 + toIndex * 17) <= density;
}

export function renderArchitecture(elements, genotype) {
  if (!genotype) return;
  const nHid = genotype.n_hid;
  const cIH = typeof genotype.c_ih === 'number' ? genotype.c_ih : 1;
  const cHO = typeof genotype.c_ho === 'number' ? genotype.c_ho : 1;
  const dual = genotype.use_dual !== false;
  const W = 600;
  const H = 220;
  const inputX = 80;
  const hiddenX = W / 2;
  const outputX = W - 80;
  const inputCount = 64;
  const hiddenCount = nHid;
  const outputCount = 10;
  const maxInputShow = 8;
  const maxHiddenShow = Math.min(nHid, 10);
  const maxOutputShow = 10;

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
      html += `<line x1="${inputX}" y1="${y1}" x2="${hiddenX}" y2="${y2}" class="arch-link arch-link-w"/>`;
      if (dual) {
        html += `<line x1="${inputX}" y1="${y1}" x2="${hiddenX}" y2="${y2}" class="arch-link arch-link-fw1"/>`;
        html += `<line x1="${inputX}" y1="${y1}" x2="${hiddenX}" y2="${y2}" class="arch-link arch-link-fw2"/>`;
        html += `<line x1="${inputX}" y1="${y1}" x2="${hiddenX}" y2="${y2}" class="arch-link arch-link-fw3"/>`;
      }
    }
  }

  for (let i = 0; i < Math.min(hiddenCount, maxHiddenShow); i += 1) {
    for (let j = 0; j < Math.min(outputCount, maxOutputShow); j += 1) {
      if (!shouldDrawConnection(29, i, j, cHO)) continue;
      const y1 = getY(i, hiddenCount, maxHiddenShow);
      const y2 = getY(j, outputCount, maxOutputShow);
      html += `<line x1="${hiddenX}" y1="${y1}" x2="${outputX}" y2="${y2}" class="arch-link arch-link-output"/>`;
      if (dual) {
        html += `<line x1="${hiddenX}" y1="${y1}" x2="${outputX}" y2="${y2}" class="arch-link arch-link-fw1"/>`;
        html += `<line x1="${hiddenX}" y1="${y1}" x2="${outputX}" y2="${y2}" class="arch-link arch-link-fw2"/>`;
        html += `<line x1="${hiddenX}" y1="${y1}" x2="${outputX}" y2="${y2}" class="arch-link arch-link-fw3"/>`;
      }
    }
  }

  for (let i = 0; i < Math.min(inputCount, maxInputShow); i += 1) {
    const y = getY(i, inputCount, maxInputShow);
    html += `<circle cx="${inputX}" cy="${y}" r="7" class="arch-node arch-node-input"><title>Neurona entrada ${i + 1} / 64</title></circle>`;
  }

  for (let i = 0; i < Math.min(hiddenCount, maxHiddenShow); i += 1) {
    const y = getY(i, hiddenCount, maxHiddenShow);
    html += `<circle cx="${hiddenX}" cy="${y}" r="8" class="arch-node arch-node-hidden"><title>Neurona oculta ${i + 1} / ${hiddenCount}</title></circle>`;
  }

  for (let i = 0; i < Math.min(outputCount, maxOutputShow); i += 1) {
    const y = getY(i, outputCount, maxOutputShow);
    html += `<circle cx="${outputX}" cy="${y}" r="7" class="arch-node arch-node-output"><title>Neurona salida ${i + 1} / 10</title></circle>`;
  }

  html += `<text x="${inputX}" y="${H - 12}" text-anchor="middle" class="arch-label">64 inputs</text>`;
  html += `<text x="${hiddenX}" y="${H - 12}" text-anchor="middle" class="arch-label">${nHid} ocultas</text>`;
  html += `<text x="${outputX}" y="${H - 12}" text-anchor="middle" class="arch-label">10 salidas</text>`;
  html += '<rect x="12" y="6" width="218" height="28" rx="6" class="arch-legend-box"/>';
  html += '<line x1="20" y1="18" x2="34" y2="18" class="arch-legend arch-link-w"/><text x="38" y="22" class="arch-legend-text">w</text>';
  html += '<line x1="58" y1="18" x2="72" y2="18" class="arch-legend arch-link-fw1"/><text x="76" y="22" class="arch-legend-text arch-fw1-text">fw1</text>';
  html += '<line x1="106" y1="18" x2="120" y2="18" class="arch-legend arch-link-fw2"/><text x="124" y="22" class="arch-legend-text arch-fw2-text">fw2</text>';
  html += '<line x1="154" y1="18" x2="168" y2="18" class="arch-legend arch-link-fw3"/><text x="172" y="22" class="arch-legend-text arch-fw3-text">fw3</text>';
  html += `<text x="20" y="32" class="arch-density">conectividad: ${(cIH * 100).toFixed(0)}% / ${(cHO * 100).toFixed(0)}%</text>`;
  html += '</svg>';

  elements.architectureWrap.innerHTML = html;
}
