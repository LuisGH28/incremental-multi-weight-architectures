export function logSession(elements, msg, cls = '') {
  const line = document.createElement('div');
  line.className = `log-line ev-${cls}`;
  line.textContent = msg;
  elements.sessionLog.appendChild(line);
  elements.sessionLog.scrollTop = elements.sessionLog.scrollHeight;
  while (elements.sessionLog.children.length > 80) {
    elements.sessionLog.removeChild(elements.sessionLog.firstChild);
  }
}

export function logMalformedEvent(elements, message) {
  logSession(elements, `Evento SSE ignorado: ${message}`, 'error');
}
