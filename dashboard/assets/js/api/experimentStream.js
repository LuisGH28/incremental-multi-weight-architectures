/**
 * EventSource facade for experiment telemetry.
 *
 * The rest of the dashboard depends on this small API instead of depending
 * directly on browser EventSource details.
 */
export function createExperimentStream({
  url = `${window.location.origin}/events`,
  onOpen,
  onError,
  onMessage,
}) {
  let source = null;
  let finished = false;

  function connect() {
    if (finished || source) return;
    source = new EventSource(url);
    source.onopen = () => onOpen?.();
    source.onerror = (event) => {
      if (!finished) onError?.(event);
    };
    source.onmessage = (event) => onMessage?.(event.data);
  }

  function close() {
    finished = true;
    if (source) {
      source.close();
      source = null;
    }
  }

  return { connect, close };
}
