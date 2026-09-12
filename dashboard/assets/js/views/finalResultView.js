export function renderFinalResult(elements, finalResult) {
  if (!finalResult) return;
  elements.banner.style.display = 'block';
  elements.accuracy.textContent = `${finalResult.meanAcc.toFixed(2)}% +/- ${finalResult.stdAcc.toFixed(2)}%`;
  elements.subtitle.textContent = `Accuracy final en test set · referencia Bullinaria: ${finalResult.targetAcc}%`;
}
