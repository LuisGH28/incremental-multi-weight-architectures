import { fitnessColor } from '../utils/fitness.js';

export function updatePopStrip(fitnessAll) {
  const strip = document.getElementById('pop-strip');
  strip.innerHTML = fitnessAll.map((f, i) =>
    `<div class="pop-cell" style="background:${fitnessColor(f)}; opacity:${0.5 + f*0.5}" title="Ind ${i}: ${(f*100).toFixed(1)}%"></div>`
  ).join('');
}
