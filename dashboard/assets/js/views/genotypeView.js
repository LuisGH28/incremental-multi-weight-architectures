function genotypeItems(g) {
  return [
    { key: 'Neuronas ocultas', val: g.n_hid },
    { key: 'Conectividad IH', val: `${(g.c_ih * 100).toFixed(1)}%` },
    { key: 'Conectividad HO', val: `${(g.c_ho * 100).toFixed(1)}%` },
    { key: 'eta input-hidden', val: g.eta_ih.toExponential(2) },
    { key: 'eta hidden bias', val: g.eta_hb.toExponential(2) },
    { key: 'eta hidden-out', val: g.eta_ho.toExponential(2) },
    { key: 'eta output bias', val: g.eta_ob.toExponential(2) },
    { key: 'oSPO', val: g.ospo.toFixed(4) },
    { key: 'lambda decay', val: g.lam.toExponential(2) },
    { key: 'Tol. t', val: g.tol_t.toFixed(3) },
    { key: 'Tol. s', val: g.tol_s.toFixed(3) },
    { key: 'delta fw1 decay', val: g.fw_decay.toFixed(4) },
    { key: 'sigma fw1 scale', val: g.fw_scale.toFixed(1) },
    { key: 'delta2 fw2 decay', val: g.fw2_decay != null ? g.fw2_decay.toFixed(4) : '-' },
    { key: 'sigma2 fw2 scale', val: g.fw2_scale != null ? g.fw2_scale.toFixed(1) : '-' },
  ];
}

export function renderGenotype(elements, genotype) {
  if (!genotype) return;
  elements.genotypeGrid.replaceChildren(
    ...genotypeItems(genotype).map((item) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'geno-item';

      const key = document.createElement('div');
      key.className = 'geno-key';
      key.textContent = item.key;

      const value = document.createElement('div');
      value.className = 'geno-val';
      value.textContent = item.val;

      wrapper.append(key, value);
      return wrapper;
    }),
  );
}
