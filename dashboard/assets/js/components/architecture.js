export function drawArch(g) {
  const container = document.getElementById('arch-svg-wrap');
  const nHid = g.n_hid;
  const cIH = g.c_ih;
  const cHO = g.c_ho;
  const dual = g.use_dual !== false;

  // Dimensiones aumentadas
  const W = 600;
  const H = 220;
  
  // Posiciones X de las capas (mejor distribuidas)
  const inputX = 80;
  const hiddenX = W / 2;
  const outputX = W - 80;
  
  // Número máximo de neuronas a mostrar por capa (por legibilidad)
  const maxInputShow = 8;
  const maxHiddenShow = Math.min(nHid, 10);
  const maxOutputShow = 10;
  
  // Función para calcular posición Y de cada neurona
  const getY = (idx, total, maxShow) => {
    const showCount = Math.min(total, maxShow);
    const spacing = (H - 60) / (showCount - 1 || 1);
    const y = 30 + (idx * (H - 60) / (showCount - 1 || 1));
    return Math.min(H - 25, Math.max(25, y));
  };

  let html = `<svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" style="background: var(--color-surface); border-radius: 12px; font-family: monospace;">`;
  
  // ========== 1. DIBUJAR CONEXIONES (ordenadas, no aleatorias) ==========
  
  // Conexiones Input → Hidden (basadas en c_ih)
  const inputCount = 64;
  const hiddenCount = nHid;
  
  for (let i = 0; i < Math.min(inputCount, maxInputShow); i++) {
    for (let j = 0; j < Math.min(hiddenCount, maxHiddenShow); j++) {
      // Conexión determinística basada en c_ih
      if (Math.random() > cIH) continue; // Mantiene la lógica original pero estable por generación
      
      const y1 = getY(i, inputCount, maxInputShow);
      const y2 = getY(j, hiddenCount, maxHiddenShow);
      
      // Peso estándar (w) - más oscuro y grueso
      html += `<line x1="${inputX}" y1="${y1}" x2="${hiddenX}" y2="${y2}" 
        stroke="rgba(99,102,241,0.35)" stroke-width="1.2" stroke-linecap="round"/>`;
      
      if (dual) {
        // Fast weight 1 (fw¹) - Bullinaria original
        html += `<line x1="${inputX}" y1="${y1}" x2="${hiddenX}" y2="${y2}" 
          stroke="rgba(245,158,11,0.25)" stroke-width="1.5" stroke-dasharray="3 3" stroke-linecap="round"/>`;
        // Fast weight 2 (fw²) - extensión propuesta
        html += `<line x1="${inputX}" y1="${y1}" x2="${hiddenX}" y2="${y2}" 
          stroke="rgba(224,90,58,0.2)" stroke-width="1.5" stroke-dasharray="5 3 1 3" stroke-linecap="round"/>`;
      }
    }
  }
  
  // Conexiones Hidden → Output (basadas en c_ho)
  const outputCount = 10;
  
  for (let i = 0; i < Math.min(hiddenCount, maxHiddenShow); i++) {
    for (let j = 0; j < Math.min(outputCount, maxOutputShow); j++) {
      if (Math.random() > cHO) continue;
      
      const y1 = getY(i, hiddenCount, maxHiddenShow);
      const y2 = getY(j, outputCount, maxOutputShow);
      
      html += `<line x1="${hiddenX}" y1="${y1}" x2="${outputX}" y2="${y2}" 
        stroke="rgba(34,197,94,0.35)" stroke-width="1.2" stroke-linecap="round"/>`;
      
      if (dual) {
        html += `<line x1="${hiddenX}" y1="${y1}" x2="${outputX}" y2="${y2}" 
          stroke="rgba(245,158,11,0.25)" stroke-width="1.5" stroke-dasharray="3 3" stroke-linecap="round"/>`;
        html += `<line x1="${hiddenX}" y1="${y1}" x2="${outputX}" y2="${y2}" 
          stroke="rgba(224,90,58,0.2)" stroke-width="1.5" stroke-dasharray="5 3 1 3" stroke-linecap="round"/>`;
      }
    }
  }
  
  // ========== 2. DIBUJAR NEURONAS ==========
  
  // Capa de entrada
  for (let i = 0; i < Math.min(inputCount, maxInputShow); i++) {
    const y = getY(i, inputCount, maxInputShow);
    const isFirst = i === 0;
    const isLast = i === Math.min(inputCount, maxInputShow) - 1;
    
    html += `<circle cx="${inputX}" cy="${y}" r="7" 
      fill="#3b82f6" stroke="#fff" stroke-width="2" 
      style="cursor:help; filter:drop-shadow(0 1px 2px rgba(0,0,0,0.1))"
      title="Neurona entrada ${i+1} / 64"/>`;
  }
  
  // Capa oculta
  for (let i = 0; i < Math.min(hiddenCount, maxHiddenShow); i++) {
    const y = getY(i, hiddenCount, maxHiddenShow);
    const size = 8;
    html += `<circle cx="${hiddenX}" cy="${y}" r="${size}" 
      fill="#8b5cf6" stroke="#fff" stroke-width="2" 
      style="cursor:help; filter:drop-shadow(0 1px 2px rgba(0,0,0,0.1))"
      title="Neurona oculta ${i+1} / ${hiddenCount}"/>`;
  }
  
  // Capa de salida
  for (let i = 0; i < Math.min(outputCount, maxOutputShow); i++) {
    const y = getY(i, outputCount, maxOutputShow);
    html += `<circle cx="${outputX}" cy="${y}" r="7" 
      fill="#10b981" stroke="#fff" stroke-width="2" 
      style="cursor:help; filter:drop-shadow(0 1px 2px rgba(0,0,0,0.1))"
      title="Neurona salida ${i+1} / 10"/>`;
  }
  
  // ========== 3. ETIQUETAS DE CAPAS ==========
  
  html += `<text x="${inputX}" y="${H - 12}" text-anchor="middle" fill="#64748b" font-size="11" font-weight="600">64 inputs</text>`;
  html += `<text x="${hiddenX}" y="${H - 12}" text-anchor="middle" fill="#64748b" font-size="11" font-weight="600">${nHid} ocultas</text>`;
  html += `<text x="${outputX}" y="${H - 12}" text-anchor="middle" fill="#64748b" font-size="11" font-weight="600">10 salidas</text>`;
  
  // ========== 4. LEYENDA (dentro del SVG) ==========
  
  const legendY = 18;
  html += `<rect x="12" y="6" width="180" height="28" rx="6" fill="rgba(255,255,255,0.9)" stroke="#e2e8f0"/>`;
  
  html += `<line x1="20" y1="${legendY}" x2="34" y2="${legendY}" stroke="rgba(99,102,241,0.6)" stroke-width="2"/>`;
  html += `<text x="38" y="${legendY+4}" fill="#64748b" font-size="9">w</text>`;
  
  html += `<line x1="58" y1="${legendY}" x2="72" y2="${legendY}" stroke="rgba(245,158,11,0.7)" stroke-width="1.5" stroke-dasharray="3 2"/>`;
  html += `<text x="76" y="${legendY+4}" fill="#f59e0b" font-size="9">fw¹</text>`;
  
  html += `<line x1="100" y1="${legendY}" x2="114" y2="${legendY}" stroke="rgba(224,90,58,0.7)" stroke-width="1.5" stroke-dasharray="5 2 1 2"/>`;
  html += `<text x="118" y="${legendY+4}" fill="#e05a3a" font-size="9">fw²</text>`;
  
  html += `<text x="150" y="${legendY+4}" fill="#64748b" font-size="8">| conectividad: ${(cIH*100).toFixed(0)}% / ${(cHO*100).toFixed(0)}%</text>`;
  
  html += '</svg>';
  container.innerHTML = html;
}
