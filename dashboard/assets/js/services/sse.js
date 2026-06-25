import { fitnessChart, sessionChart } from '../config/charts.js';
import { fitnessColor } from '../utils/fitness.js';
import { drawArch } from '../components/architecture.js';
import { drawFlow } from '../components/flow.js';
import { updateGenotype } from '../components/genotype.js';
import { updatePopStrip } from '../components/population.js';
import { logSession } from '../components/log.js';

let cfg = { n_generations: 0, pop_size: 0 };
let currentGen = 0;
let t0 = Date.now();

// SSE listener (original intacto)
const pill = document.getElementById('status-pill');

// ── Conexión SSE con reconexión automática ─────────────────────────
// Usamos el puerto actual de la página para no hardcodear localhost:8765
const SSE_URL = `${window.location.origin}/events`;
let es = null;
let _connected = false;
let _finished = false;

export function connectSSE() {
  if (_finished) return;
  es = new EventSource(SSE_URL);

  es.onopen = () => {
    _connected = true;
    pill.textContent = 'Conectado';
    pill.className = 'status-pill running';
    t0 = Date.now();
  };

  es.onerror = () => {
    if (_finished) return;
    _connected = false;
    pill.textContent = 'Reconectando…';
    pill.className = 'status-pill';
    // EventSource reconecta automáticamente; solo actualizamos el pill
  };

  es.onmessage = handleMessage;
}

function handleMessage(e) {
  let ev; try { ev = JSON.parse(e.data); } catch { return; }
  switch (ev.type) {
    case 'config': cfg = ev; document.getElementById('prog-label').textContent = `Gen 0 / ${ev.n_generations}`; break;
   case 'gen_start': 
  if (!_finished) {
    currentGen = ev.gen; 
    document.getElementById('m-gen').textContent = ev.gen; 
    const pct = cfg.n_generations ? ((ev.gen / cfg.n_generations) * 100).toFixed(1) : 0; 
    document.getElementById('gen-prog').style.width = pct + '%'; 
    document.getElementById('prog-label').textContent = `Gen ${ev.gen} / ${cfg.n_generations}`; 
    const elapsed = ((Date.now() - t0) / 1000).toFixed(0); 
    document.getElementById('prog-time').textContent = elapsed + 's transcurridos';
  }
  break;
    case 'session_start': if (ev.individual < 3) { drawFlow(ev); logSession(`[Gen ${ev.gen} | Ind ${ev.individual} | Sesión T${ev.session+1}] ${ev.n_patterns} patrones`, 'session'); if (ev.arch) drawArch({ ...ev.arch, use_dual: ev.arch.use_dual }); } break;
    case 'epoch': if (ev.individual < 3) { const acc = ((ev.correct / ev.n) * 100).toFixed(1); logSession(`  época ${ev.epoch} | correctos ${ev.correct}/${ev.n} (${acc}%) | CE=${ev.ce}`, 'epoch'); } break;
    case 'session_end': if (ev.individual < 3) logSession(`  ✓ fin sesión T${ev.session+1} en ${ev.epochs_run} épocas`, 'done'); break;
    case 'individual_done': const strip = document.getElementById('pop-strip'); if (strip.children[ev.individual]) { strip.children[ev.individual].style.background = fitnessColor(ev.fitness); strip.children[ev.individual].style.opacity = 0.5 + ev.fitness * 0.5; } break;
    case 'gen_end': document.getElementById('m-best').textContent = (ev.best_fitness*100).toFixed(2) + '%'; document.getElementById('m-top10').textContent = (ev.top10_mean*100).toFixed(2) + '%'; document.getElementById('m-mean').textContent = (ev.pop_mean*100).toFixed(2) + '%'; fitnessChart.data.labels.push(ev.gen); fitnessChart.data.datasets[0].data.push(ev.best_fitness); fitnessChart.data.datasets[1].data.push(ev.top10_mean); fitnessChart.data.datasets[2].data.push(ev.pop_mean); fitnessChart.update('none'); if (ev.fitness_all) updatePopStrip(ev.fitness_all); if (ev.best_genotype) { updateGenotype(ev.best_genotype); drawArch(ev.best_genotype); } break;
    case 'final_result': document.getElementById('final-banner').style.display = 'block'; document.getElementById('final-acc').textContent = ev.mean_acc.toFixed(2) + '% ± ' + ev.std_acc.toFixed(2) + '%'; document.getElementById('final-sub').textContent = `Accuracy final en test set · objetivo Bullinaria: ${ev.target_acc}%`; if (ev.session_accs && ev.session_accs.length) { sessionChart.data.datasets[0].data = ev.session_accs; sessionChart.update(); } pill.textContent = 'Completado'; pill.className = 'status-pill done'; 
    document.getElementById('gen-prog').style.width = '100%';
document.getElementById('prog-label').textContent = `Gen ${cfg.n_generations} / ${cfg.n_generations}`;break;
    case 'done': case 'server_done':
      _finished = true;
      pill.textContent = 'Completado';
      pill.className = 'status-pill done';
      if (es) es.close();
      break;
  }
}
