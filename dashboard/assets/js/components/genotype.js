export function updateGenotype(g) {
  const items = [
    { key: 'Neuronas ocultas', val: g.n_hid, hi: true },
    { key: 'Conectividad IH', val: (g.c_ih*100).toFixed(1)+'%' },
    { key: 'Conectividad HO', val: (g.c_ho*100).toFixed(1)+'%' },
    { key: 'η input→hidden', val: g.eta_ih.toExponential(2), hi: true },
    { key: 'η hidden bias',  val: g.eta_hb.toExponential(2) },
    { key: 'η hidden→out',  val: g.eta_ho.toExponential(2) },
    { key: 'η output bias', val: g.eta_ob.toExponential(2) },
    { key: 'oSPO',          val: g.ospo.toFixed(4) },
    { key: 'λ (decay)',     val: g.lam.toExponential(2) },
    { key: 'Tol. t',        val: g.tol_t.toFixed(3) },
    { key: 'Tol. s',        val: g.tol_s.toFixed(3) },
    { key: 'δ fast decay',   val: g.fw_decay.toFixed(4),  hi: true },
    { key: 'σ fast scale',   val: g.fw_scale.toFixed(1),  hi: true },
    { key: 'δ₂ fw² decay',  val: g.fw2_decay != null ? g.fw2_decay.toFixed(4) : '—', hi: true },
    { key: 'σ₂ fw² scale',  val: g.fw2_scale != null ? g.fw2_scale.toFixed(1) : '—', hi: true },
  ];
  document.getElementById('geno-grid').innerHTML = items.map(item =>
    `<div class="geno-item"><div class="geno-key">${item.key}</div><div class="geno-val ${item.hi ? '' : ''}">${item.val}</div></div>`
  ).join('');
}
