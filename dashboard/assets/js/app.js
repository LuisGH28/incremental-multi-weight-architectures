import { adaptSseMessage } from './api/eventAdapter.js';
import { createExperimentStream } from './api/experimentStream.js';
import { TerminalEventTypes, StreamActionTypes } from './api/eventTypes.js';
import { createInitialState } from './state/initialState.js';
import { reduceExperimentState } from './state/experimentReducer.js';
import {
  getDashboardElements,
  renderExperimentEvent,
  renderInitialDashboard,
  renderStreamAction,
} from './views/dashboardView.js';

const StreamActionValues = new Set(Object.values(StreamActionTypes));

export function startDashboard() {
  const elements = getDashboardElements();
  let state = createInitialState();
  let stream = null;

  function dispatch(action) {
    state = reduceExperimentState(state, action);
    if (StreamActionValues.has(action.type)) {
      renderStreamAction(elements, action, state);
      return;
    }
    renderExperimentEvent(elements, action, state);
  }

  renderInitialDashboard(elements, state);

  stream = createExperimentStream({
    onOpen: () => dispatch({ type: StreamActionTypes.OPEN }),
    onError: () => dispatch({ type: StreamActionTypes.ERROR }),
    onMessage: (raw) => {
      const result = adaptSseMessage(raw);
      if (!result.ok) {
        dispatch({ type: StreamActionTypes.MALFORMED_EVENT, error: result.error });
        return;
      }

      dispatch(result.event);
      if (TerminalEventTypes.has(result.event.type)) {
        stream.close();
      }
    },
  });

  stream.connect();
  window.addEventListener('beforeunload', () => stream.close(), { once: true });
}
