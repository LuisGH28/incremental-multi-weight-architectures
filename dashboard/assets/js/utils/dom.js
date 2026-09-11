export function byId(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing dashboard element: #${id}`);
  }
  return element;
}

export function setText(element, value) {
  element.textContent = value;
}

export function setDisplay(element, value) {
  element.style.display = value;
}
