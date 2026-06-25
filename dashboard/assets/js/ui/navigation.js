export function setupNavigation() {
  document.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const sectionId = btn.getAttribute('data-section');
      if (sectionId) {
        const target = document.getElementById(sectionId);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
  const toggleBtn = document.getElementById('menuToggle');
  const navLinksDiv = document.getElementById('navLinks');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => { navLinksDiv.classList.toggle('show'); });
  }
}

export function setupLucideIcons() {
  // Inicializar Lucide Icons (reemplaza <i data-lucide="..."> por SVG)
  document.addEventListener('DOMContentLoaded', () => {
    if (typeof lucide !== 'undefined') {
      lucide.createIcons();
    }
  });
  // También llamar después de actualizar contenido dinámico (si se necesita)
  setTimeout(() => { if (typeof lucide !== 'undefined') lucide.createIcons(); }, 200);
}
