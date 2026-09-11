export function createInitialState() {
  return {
    connection: 'connecting',
    lifecycle: 'idle',
    config: {
      n_generations: 0,
      pop_size: 0,
      use_dual: true,
      max_epochs: 0,
      n_train: 0,
      n_test: 0,
      n_fw_lines: 3,
    },
    startedAt: Date.now(),
    currentGen: 0,
    metrics: {
      bestFitness: null,
      top10Mean: null,
      populationMean: null,
    },
    progress: {
      percent: 0,
      label: 'Gen 0 / -',
      elapsedLabel: '-',
    },
    charts: {
      fitnessLabels: [],
      bestFitness: [],
      top10Mean: [],
      populationMean: [],
      sessionAccuracies: [],
    },
    populationFitness: [],
    bestGenotype: null,
    currentSession: null,
    finalResult: null,
    lastError: null,
  };
}
