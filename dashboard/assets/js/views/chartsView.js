import { fitnessChart, sessionChart } from '../config/charts.js';

export function renderFitnessChart(charts) {
  fitnessChart.data.labels = charts.fitnessLabels;
  fitnessChart.data.datasets[0].data = charts.bestFitness;
  fitnessChart.data.datasets[1].data = charts.top10Mean;
  fitnessChart.data.datasets[2].data = charts.populationMean;
  fitnessChart.update('none');
}

function sessionLabels(nSessions) {
  return Array.from({ length: nSessions }, (_, idx) => `T${idx + 1}`);
}

export function configureSessionChart(config) {
  const nSessions = config?.nSessions || 0;
  sessionChart.data.labels = sessionLabels(nSessions);

  const reference = config?.reference;
  const referenceDataset = sessionChart.data.datasets[1];
  if (reference?.sessionAccuracies?.length) {
    referenceDataset.label = reference.name;
    referenceDataset.data = reference.sessionAccuracies;
    referenceDataset.hidden = false;
  } else {
    referenceDataset.label = 'Reference';
    referenceDataset.data = [];
    referenceDataset.hidden = true;
  }

  sessionChart.update('none');
}

export function renderSessionChart(charts, config) {
  if (config) configureSessionChart(config);
  sessionChart.data.datasets[0].data = charts.sessionAccuracies;
  sessionChart.update();
}
