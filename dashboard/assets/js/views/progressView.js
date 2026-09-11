export function renderProgress(elements, progress) {
  elements.bar.style.width = `${progress.percent}%`;
  elements.label.textContent = progress.label;
  elements.elapsed.textContent = progress.elapsedLabel;
}
