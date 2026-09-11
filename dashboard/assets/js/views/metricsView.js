import { formatPercent } from '../utils/formatting.js';

export function renderGenerationMetric(elements, state) {
  elements.generation.textContent = String(state.currentGen);
}

export function renderFitnessMetrics(elements, metrics) {
  if (metrics.bestFitness !== null) {
    elements.bestFitness.textContent = formatPercent(metrics.bestFitness);
  }
  if (metrics.top10Mean !== null) {
    elements.top10Mean.textContent = formatPercent(metrics.top10Mean);
  }
  if (metrics.populationMean !== null) {
    elements.populationMean.textContent = formatPercent(metrics.populationMean);
  }
}
