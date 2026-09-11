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

function hasOptionalArray(event, key) {
  return event[key] === undefined || Array.isArray(event[key]);
}

function validateShape(event) {
  switch (event.type) {
    case 'config':
      return hasNumber(event, 'n_generations') && hasNumber(event, 'pop_size');
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
        && hasNumber(event, 'target_acc')
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
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, error: 'Invalid JSON event payload.', raw };
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

  return { ok: true, event: parsed };
}
