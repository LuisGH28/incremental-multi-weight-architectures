import { EventTypes, StreamActionTypes } from '../api/eventTypes.js';
import { formatElapsedSeconds } from '../utils/formatting.js';

function progressForGen(gen, total) {
  if (!total) return 0;
  return Number(((gen / total) * 100).toFixed(1));
}

function withConnection(state, connection) {
  if (state.lifecycle === 'completed') return state;
  return { ...state, connection };
}

export function reduceExperimentState(state, action) {
  switch (action.type) {
    case StreamActionTypes.OPEN:
      return {
        ...state,
        connection: 'connected',
        startedAt: Date.now(),
        lastError: null,
      };

    case StreamActionTypes.ERROR:
      return withConnection(state, 'reconnecting');

    case StreamActionTypes.MALFORMED_EVENT:
      return { ...state, lastError: action.error };

    case EventTypes.CONFIG:
      return {
        ...state,
        config: { ...state.config, ...action },
        progress: {
          ...state.progress,
          label: `Gen 0 / ${action.n_generations}`,
        },
      };

    case EventTypes.GEN_START: {
      const total = state.config.n_generations || action.n_gen || 0;
      return {
        ...state,
        lifecycle: 'running',
        currentGen: action.gen,
        progress: {
          percent: progressForGen(action.gen, total),
          label: `Gen ${action.gen} / ${total || '-'}`,
          elapsedLabel: `${formatElapsedSeconds((Date.now() - state.startedAt) / 1000)} transcurridos`,
        },
      };
    }

    case EventTypes.SESSION_START:
      return {
        ...state,
        currentSession: {
          gen: action.gen,
          individual: action.individual,
          session: action.session,
          n_patterns: action.n_patterns,
          class_counts: action.class_counts || {},
          arch: action.arch
            ? {
                n_inputs: state.config.n_inputs,
                n_outputs: state.config.n_outputs,
                n_fast_weight_lines: state.config.n_fast_weight_lines,
                weight_labels: state.config.weight_labels,
                ...action.arch,
              }
            : null,
        },
      };

    case EventTypes.GEN_END:
      return {
        ...state,
        metrics: {
          bestFitness: action.best_fitness,
          top10Mean: action.top10_mean,
          populationMean: action.pop_mean,
        },
        charts: {
          ...state.charts,
          fitnessLabels: [...state.charts.fitnessLabels, action.gen],
          bestFitness: [...state.charts.bestFitness, action.best_fitness],
          top10Mean: [...state.charts.top10Mean, action.top10_mean],
          populationMean: [...state.charts.populationMean, action.pop_mean],
        },
        populationFitness: action.fitness_all || state.populationFitness,
        bestGenotype: action.best_genotype
          ? {
              n_inputs: state.config.n_inputs,
              n_outputs: state.config.n_outputs,
              n_fast_weight_lines: state.config.n_fast_weight_lines,
              weight_labels: state.config.weight_labels,
              use_dual: state.config.use_dual,
              ...action.best_genotype,
            }
          : state.bestGenotype,
      };

    case EventTypes.FINAL_START:
      return { ...state, lifecycle: 'finalizing' };

    case EventTypes.FINAL_RESULT:
      return {
        ...state,
        connection: 'completed',
        lifecycle: 'completed',
        finalResult: {
          meanAcc: action.mean_acc,
          stdAcc: action.std_acc,
          targetAcc: action.target_acc,
        },
        bestGenotype: action.best_genotype
          ? {
              n_inputs: action.n_inputs || state.config.n_inputs,
              n_outputs: action.n_outputs || state.config.n_outputs,
              n_fast_weight_lines: action.n_fast_weight_lines || state.config.n_fast_weight_lines,
              weight_labels: action.weight_labels || state.config.weight_labels,
              use_dual: state.config.use_dual,
              ...action.best_genotype,
            }
          : state.bestGenotype,
        charts: {
          ...state.charts,
          sessionAccuracies: action.session_accs || [],
        },
        progress: {
          ...state.progress,
          percent: 100,
          label: `Gen ${state.config.n_generations} / ${state.config.n_generations}`,
        },
      };

    case EventTypes.DONE:
    case EventTypes.SERVER_DONE:
      return {
        ...state,
        connection: 'completed',
        lifecycle: 'completed',
      };

    default:
      return state;
  }
}
