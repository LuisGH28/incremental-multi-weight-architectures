export function renderConnectionStatus(pill, connection) {
  const labels = {
    connecting: 'Conectando...',
    connected: 'Conectado',
    reconnecting: 'Reconectando...',
    completed: 'Completado',
    error: 'Error',
  };

  pill.textContent = labels[connection] || labels.connecting;
  pill.className = 'status-pill';
  if (connection === 'connected') pill.classList.add('running');
  if (connection === 'completed') pill.classList.add('done');
}
