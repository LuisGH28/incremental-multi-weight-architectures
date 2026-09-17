import { SupportedEventTypes } from './eventTypes.js';

/**
 * @typedef {object} AdapterResult
 * @property {boolean} ok
 * @property {object=} event
 * @property {string=} error
 * @property {string=} raw
 */

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasNumber(event, key) {
  return typeof event[key] === 'number' && Number.isFinite(event[key]);
}

function hasPositiveInteger(event, key) {
  return Number.isInteger(event[key]) && event[key] > 0;
}

function hasOptionalArray(event, key) {
  return event[key] === undefined || Array.isArray(event[key]);
}

function normalizeReference(reference) {
  if (reference == null) return null;
  if (!isRecord(reference)) return null;

  const sessionAccuracies = Array.isArray(reference.session_accs)
    ? reference.session_accs
    : reference.sessionAccuracies;

  if (!Array.isArray(sessionAccuracies) || !sessionAccuracies.every(Number.isFinite)) {
    return null;
  }

  return {
    name: typeof reference.name === 'string' ? reference.name : 'Reference',
    dataset: typeof reference.dataset === 'string' ? reference.dataset : null,
    targetAcc: hasNumber(reference, 'target_acc')
      ? reference.target_acc
      : Number.isFinite(reference.targetAcc) ? reference.targetAcc : null,
    sessionAccuracies,
  };
}

function adaptConfigEvent(event) {
  if (!hasNumber(event, 'n_generations')
    || !hasNumber(event, 'pop_size')
    || !hasPositiveInteger(event, 'n_inputs')
    || !hasPositiveInteger(event, 'n_outputs')
    || !hasPositiveInteger(event, 'n_sessions')
    || !Number.isInteger(event.n_fast_weight_lines)
    || event.n_fast_weight_lines < 0
    || !Array.isArray(event.weight_labels)
    || event.weight_labels.length !== event.n_fast_weight_lines + 1
    || !event.weight_labels.every(label => typeof label === 'string' && label.trim() !== '')) {
    return null;
  }

  const reference = normalizeReference(event.reference);
  if (reference && reference.sessionAccuracies.length !== event.n_sessions) {
    return null;
  }

  return {
    type: event.type,
    config: {
      dataset: typeof event.dataset === 'string' ? event.dataset : 'unknown',
      nGenerations: event.n_generations,
      popSize: event.pop_size,
      useDual: event.use_dual !== false,
      maxEpochs: hasNumber(event, 'max_epochs') ? event.max_epochs : 0,
      nTrain: hasNumber(event, 'n_train') ? event.n_train : 0,
      nTest: hasNumber(event, 'n_test') ? event.n_test : 0,
      nInputs: event.n_inputs,
      nOutputs: event.n_outputs,
      nSessions: event.n_sessions,
      nFastWeightLines: event.n_fast_weight_lines,
      weightLabels: event.weight_labels.map(label => label.trim()),
      reference,
    },
  };
}

function validateShape(event) {
  switch (event.type) {
    case 'config':
      return adaptConfigEvent(event) !== null;
    case 'gen_start':
      return hasNumber(event, 'gen');
    case 'session_start':
      return hasNumber(event, 'gen')
        && hasNumber(event, 'individual')
        && hasNumber(event, 'session')
        && hasNumber(event, 'n_patterns');
    case 'epoch':
      return hasNumber(event, 'epoch')
        && hasNumber(event, 'correct')
        && hasNumber(event, 'n');
    case 'session_end':
      return hasNumber(event, 'individual')
        && hasNumber(event, 'session')
        && hasNumber(event, 'epochs_run');
    case 'individual_done':
      return hasNumber(event, 'individual') && hasNumber(event, 'fitness');
    case 'gen_end':
      return hasNumber(event, 'gen')
        && hasNumber(event, 'best_fitness')
        && hasNumber(event, 'top10_mean')
        && hasNumber(event, 'pop_mean')
        && hasOptionalArray(event, 'fitness_all');
    case 'final_result':
      return hasNumber(event, 'mean_acc')
        && hasNumber(event, 'std_acc')
        && hasOptionalArray(event, 'session_accs');
    case 'final_start':
    case 'done':
    case 'server_done':
      return true;
    default:
      return false;
  }
}

/**
 * Converts raw Server-Sent Event message data into a known dashboard event.
 * This is intentionally defensive rather than exhaustive: the backend remains
 * the authority for scientific values, while the frontend rejects malformed
 * transport payloads before they reach state or rendering modules.
 *
 * @param {string} raw
 * @returns {AdapterResult}
 */
export function adaptSseMessage(raw) {
  if (raw === '') {
    return { ok: false, error: 'Empty SSE data event.', raw };
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, error: `Invalid JSON event payload: ${raw.slice(0, 80)}`, raw };
  }

  if (!isRecord(parsed)) {
    return { ok: false, error: 'SSE payload must be a JSON object.', raw };
  }

  if (typeof parsed.type !== 'string' || !SupportedEventTypes.has(parsed.type)) {
    return { ok: false, error: `Unsupported event type: ${String(parsed.type)}`, raw };
  }

  if (!validateShape(parsed)) {
    return { ok: false, error: `Malformed payload for event type: ${parsed.type}`, raw };
  }

  if (parsed.type === 'config') {
    return { ok: true, event: adaptConfigEvent(parsed) };
  }

  return { ok: true, event: parsed };
}
