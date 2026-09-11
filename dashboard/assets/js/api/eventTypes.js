export const EventTypes = Object.freeze({
  CONFIG: 'config',
  GEN_START: 'gen_start',
  SESSION_START: 'session_start',
  EPOCH: 'epoch',
  SESSION_END: 'session_end',
  INDIVIDUAL_DONE: 'individual_done',
  GEN_END: 'gen_end',
  FINAL_START: 'final_start',
  FINAL_RESULT: 'final_result',
  DONE: 'done',
  SERVER_DONE: 'server_done',
});

export const StreamActionTypes = Object.freeze({
  OPEN: 'stream_open',
  ERROR: 'stream_error',
  MALFORMED_EVENT: 'malformed_event',
});

export const TerminalEventTypes = new Set([
  EventTypes.DONE,
  EventTypes.SERVER_DONE,
]);

export const SupportedEventTypes = new Set(Object.values(EventTypes));
