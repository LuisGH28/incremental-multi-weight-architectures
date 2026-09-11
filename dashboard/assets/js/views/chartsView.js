import { fitnessChart, sessionChart } from '../config/charts.js';

export function renderFitnessChart(charts) {
  fitnessChart.data.labels = charts.fitnessLabels;
  fitnessChart.data.datasets[0].data = charts.bestFitness;
  fitnessChart.data.datasets[1].data = charts.top10Mean;
  fitnessChart.data.datasets[2].data = charts.populationMean;
  fitnessChart.update('none');
}

export function renderSessionChart(charts) {
  sessionChart.data.datasets[0].data = charts.sessionAccuracies;
  sessionChart.update();
}
