import { EventTypes, StreamActionTypes } from '../api/eventTypes.js';
import { byId } from '../utils/dom.js';
import { renderConnectionStatus } from './statusView.js';
import { renderGenerationMetric, renderFitnessMetrics } from './metricsView.js';
import { renderProgress } from './progressView.js';
import { renderFitnessChart, renderSessionChart } from './chartsView.js';
import { renderFinalResult } from './finalResultView.js';
import { logMalformedEvent, logSession } from './logView.js';
import { renderFlow } from './flowView.js';
import { renderArchitecture } from './architectureView.js';
import { renderGenotype } from './genotypeView.js';
import { renderIndividualFitness, renderPopulationStrip } from './populationView.js';

export function getDashboardElements() {
  return {
    statusPill: byId('status-pill'),
    generation: byId('m-gen'),
    bestFitness: byId('m-best'),
    top10Mean: byId('m-top10'),
    populationMean: byId('m-mean'),
    progressBar: byId('gen-prog'),
    progressLabel: byId('prog-label'),
    progressElapsed: byId('prog-time'),
    finalBanner: byId('final-banner'),
    finalAccuracy: byId('final-acc'),
    finalSubtitle: byId('final-sub'),
    architectureWrap: byId('arch-svg-wrap'),
    populationStrip: byId('pop-strip'),
    flowWrap: byId('flow-svg-wrap'),
    genotypeGrid: byId('geno-grid'),
    sessionLog: byId('session-log'),
  };
}

export function renderInitialDashboard(elements, state) {
  renderConnectionStatus(elements.statusPill, state.connection);
  renderProgress({
    bar: elements.progressBar,
    label: elements.progressLabel,
    elapsed: elements.progressElapsed,
  }, state.progress);
}

export function renderStreamAction(elements, action, state) {
  if (action.type === StreamActionTypes.MALFORMED_EVENT) {
    logMalformedEvent(elements, action.error);
  }
  renderConnectionStatus(elements.statusPill, state.connection);
}

export function renderExperimentEvent(elements, event, state) {
  switch (event.type) {
    case EventTypes.CONFIG:
      renderProgress({
        bar: elements.progressBar,
        label: elements.progressLabel,
        elapsed: elements.progressElapsed,
      }, state.progress);
      break;

    case EventTypes.GEN_START:
      renderGenerationMetric(elements, state);
      renderProgress({
        bar: elements.progressBar,
        label: elements.progressLabel,
        elapsed: elements.progressElapsed,
      }, state.progress);
      break;

    case EventTypes.SESSION_START:
      if (event.individual < 3) {
        renderFlow(elements, event);
        logSession(elements, `[Gen ${event.gen} | Ind ${event.individual} | Sesión T${event.session + 1}] ${event.n_patterns} patrones`, 'session');
        if (event.arch) renderArchitecture(elements, event.arch);
      }
      break;

    case EventTypes.EPOCH:
      if (event.individual < 3) {
        const acc = ((event.correct / event.n) * 100).toFixed(1);
        logSession(elements, `  época ${event.epoch} | correctos ${event.correct}/${event.n} (${acc}%) | CE=${event.ce}`, 'epoch');
      }
      break;

    case EventTypes.SESSION_END:
      if (event.individual < 3) {
        logSession(elements, `  fin sesión T${event.session + 1} en ${event.epochs_run} épocas`, 'done');
      }
      break;

    case EventTypes.INDIVIDUAL_DONE:
      renderIndividualFitness(elements, event.individual, event.fitness);
      break;

    case EventTypes.GEN_END:
      renderFitnessMetrics(elements, state.metrics);
      renderFitnessChart(state.charts);
      if (event.fitness_all) renderPopulationStrip(elements, state.populationFitness);
      if (state.bestGenotype) {
        renderGenotype(elements, state.bestGenotype);
        renderArchitecture(elements, state.bestGenotype);
      }
      break;

    case EventTypes.FINAL_START:
      logSession(elements, 'Evaluación final sobre test set...', 'session');
      break;

    case EventTypes.FINAL_RESULT:
      renderFinalResult({
        banner: elements.finalBanner,
        accuracy: elements.finalAccuracy,
        subtitle: elements.finalSubtitle,
      }, state.finalResult);
      renderSessionChart(state.charts);
      renderProgress({
        bar: elements.progressBar,
        label: elements.progressLabel,
        elapsed: elements.progressElapsed,
      }, state.progress);
      renderConnectionStatus(elements.statusPill, state.connection);
      break;

    case EventTypes.DONE:
    case EventTypes.SERVER_DONE:
      renderConnectionStatus(elements.statusPill, state.connection);
      break;
  }
}
