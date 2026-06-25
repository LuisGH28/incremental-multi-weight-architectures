export function logSession(msg, cls = '') {
  const log = document.getElementById('session-log');
  const line = document.createElement('div');
  line.className = `log-line ev-${cls}`;
  line.textContent = msg;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
  while (log.children.length > 80) log.removeChild(log.firstChild);
}
