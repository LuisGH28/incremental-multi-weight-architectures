export const fitnessChart = new Chart(document.getElementById('chart-fitness'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Mejor', data: [], borderColor: '#6366f1', borderWidth: 2, pointRadius: 0, tension: 0.3 },
      { label: 'Top 10%', data: [], borderColor: '#22c55e', borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [4,2] },
      { label: 'Media', data: [], borderColor: '#64748b', borderWidth: 1, pointRadius: 0, tension: 0.3, borderDash: [2,3] },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: '#e2e8f0' } },
      y: { min: 0, max: 1, ticks: { color: '#64748b', callback: v => (v*100).toFixed(0)+'%' }, grid: { color: '#e2e8f0' } }
    }
  }
});

export const sessionChart = new Chart(document.getElementById('chart-sessions'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Test acc', data: [], borderColor: '#f59e0b', borderWidth: 2, pointRadius: 4,
        pointBackgroundColor: '#f59e0b', tension: 0.2 },
      { label: 'Reference', data: [],
        borderColor: 'rgba(99,102,241,0.4)', borderWidth: 1, pointRadius: 2, borderDash: [4,3], tension: 0.2, hidden: true }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#64748b' }, grid: { color: '#e2e8f0' } },
      y: { min: 80, max: 100, ticks: { color: '#64748b', callback: v => v+'%' }, grid: { color: '#e2e8f0' } }
    }
  }
});
