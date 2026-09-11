import { fitnessColor, fitnessOpacity } from '../utils/fitness.js';

export function renderPopulationStrip(elements, fitnessAll) {
  elements.populationStrip.replaceChildren(
    ...fitnessAll.map((fitness, index) => {
      const cell = document.createElement('div');
      cell.className = 'pop-cell';
      cell.style.background = fitnessColor(fitness);
      cell.style.opacity = fitnessOpacity(fitness);
      cell.title = `Ind ${index}: ${(fitness * 100).toFixed(1)}%`;
      return cell;
    }),
  );
}

export function renderIndividualFitness(elements, individual, fitness) {
  const cell = elements.populationStrip.children[individual];
  if (!cell) return;
  cell.style.background = fitnessColor(fitness);
  cell.style.opacity = fitnessOpacity(fitness);
}
