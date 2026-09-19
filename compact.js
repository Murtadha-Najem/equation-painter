/* Compact engine: a picture as a few named parts, the way Yeganeh builds his.
   1. background   a smooth field: a small cosine series per channel, least squares
   2. parts        k-means on colour, connected components; each part's outline is one closed curve,
                   rho < exp(L(phi)) with L a short Fourier series, masked as e^{-e^{kappa(rho e^{-L} - 1)}};
                   parts that one such curve cannot hold are split in two
   3. shading      each part's colour is a plane over the picture, alpha + beta (x - a) + gamma (y - b)
   4. refinement   all constants tuned together by gradient descent (TensorFlow.js), outlines first, shading second
   5. details      extra small parts where the error is largest, kept only while each constant buys enough error
   6. texture      a grain in his form (a sum of thresholded cos^10 lattices turned by s^2), its frequency and
                   strength matched to the picture's fine texture statistics, not to its pixels
   Everything is deterministic: fixed seeds, fixed orders. */
(function(){
"use strict";
const Compact = {};
const PI = Math.PI;

/* ---------------- small numeric helpers ---------------- */
function blurPlane(src, w, h, s){                 // separable Gaussian on one plane
  if (s <= 0) return Float32Array.from(src);
  const r = Math.ceil(3*s), k = new Float32Array(2*r + 1); let z = 0;
  for (let i = -r; i <= r; i++){ k[i + r] = Math.exp(-i*i/(2*s*s)); z += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= z;
  const tmp = new Float32Array(w*h), out = new Float32Array(w*h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++){ let a = 0; for (let i = -r; i <= r; i++){ const xx = Math.min(w - 1, Math.max(0, x + i)); a += k[i + r]*src[y*w + xx]; } tmp[y*w + x] = a; }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++){ let a = 0; for (let i = -r; i <= r; i++){ const yy = Math.min(h - 1, Math.max(0, y + i)); a += k[i + r]*tmp[yy*w + x]; } out[y*w + x] = a; }
  return out;
}
function blurRGB(T, w, h, s){                     // T interleaved rgb -> interleaved
  const out = new Float32Array(w*h*3);
  for (let c = 0; c < 3; c++){
    const p = new Float32Array(w*h); for (let i = 0; i < w*h; i++) p[i] = T[i*3 + c];
    const b = blurPlane(p, w, h, s); for (let i = 0; i < w*h; i++) out[i*3 + c] = b[i];
  }
  return out;
}
function toLab(r, g, b){
  const f = c => c > 0.04045 ? Math.pow((c + 0.055)/1.055, 2.4) : c/12.92;
  const R = f(r), G = f(g), B = f(b);
  let X = (0.4124*R + 0.3576*G + 0.1805*B)/0.95047, Y = 0.2126*R + 0.7152*G + 0.0722*B, Z = (0.0193*R + 0.1192*G + 0.9505*B)/1.08883;
  const t = v => v > 0.008856 ? Math.cbrt(v) : 7.787*v + 16/116;
  X = t(X); Y = t(Y); Z = t(Z);
  return [116*Y - 16, 500*(X - Y), 200*(Y - Z)];
}
function mulberry(a){ return function(){ a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0)/4294967296; }; }
function kmeans(data, n, dim, k, iters, rng){     // data flat n x dim
  const cen = new Float64Array(k*dim), lab = new Int32Array(n);
  // k-means++ seeding, deterministic
  let first = Math.floor(rng()*n); for (let d = 0; d < dim; d++) cen[d] = data[first*dim + d];
  const dist = new Float64Array(n).fill(Infinity);
  for (let c = 1; c < k; c++){
    let sum = 0;
    for (let i = 0; i < n; i++){ let s = 0; for (let d = 0; d < dim; d++){ const e = data[i*dim + d] - cen[(c - 1)*dim + d]; s += e*e; } if (s < dist[i]) dist[i] = s; sum += dist[i]; }
    let r = rng()*sum, pick = 0; for (let i = 0; i < n; i++){ r -= dist[i]; if (r <= 0){ pick = i; break; } }
    for (let d = 0; d < dim; d++) cen[c*dim + d] = data[pick*dim + d];
  }
  for (let it = 0; it < iters; it++){
    for (let i = 0; i < n; i++){ let best = 0, bd = Infinity; for (let c = 0; c < k; c++){ let s = 0; for (let d = 0; d < dim; d++){ const e = data[i*dim + d] - cen[c*dim + d]; s += e*e; } if (s < bd){ bd = s; best = c; } } lab[i] = best; }
    const acc = new Float64Array(k*dim), cnt = new Float64Array(k);
    for (let i = 0; i < n; i++){ cnt[lab[i]]++; for (let d = 0; d < dim; d++) acc[lab[i]*dim + d] += data[i*dim + d]; }
    for (let c = 0; c < k; c++) if (cnt[c]) for (let d = 0; d < dim; d++) cen[c*dim + d] = acc[c*dim + d]/cnt[c];
  }
  return lab;
}
function components(mask, w, h){                  // 8-connected components of a Uint8 mask
  const lab = new Int32Array(w*h).fill(-1), comps = [];
  const stack = new Int32Array(w*h);
  for (let s = 0; s < w*h; s++){
    if (!mask[s] || lab[s] >= 0) continue;
    const id = comps.length, pix = []; let sp = 0; stack[sp++] = s; lab[s] = id; let border = 0;
    while (sp){
      const q = stack[--sp]; pix.push(q); const x = q % w, y = (q/w) | 0;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) border++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++){
        const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const r = yy*w + xx; if (mask[r] && lab[r] < 0){ lab[r] = id; stack[sp++] = r; }
      }
    }
    comps.push({pix, border});
  }
  return comps;
}
function hullArea(pts){                            // pts: [[x,y]...], monotone chain
  if (pts.length < 3) return 0;
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p){ while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--){ const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  const hull = lo.slice(0, -1).concat(up.slice(0, -1)); let a = 0;
  for (let i = 0; i < hull.length; i++){ const j = (i + 1) % hull.length; a += hull[i][0]*hull[j][1] - hull[j][0]*hull[i][1]; }
  return Math.abs(a)/2;
}
function lstsq(rows, ys, dim){                    // normal equations with a small ridge; ys: n x 3
  const A = new Float64Array(dim*dim), B = new Float64Array(dim*3);
  for (let i = 0; i < rows.length; i++){ const r = rows[i];
    for (let a = 0; a < dim; a++){ for (let b = 0; b < dim; b++) A[a*dim + b] += r[a]*r[b]; for (let c = 0; c < 3; c++) B[a*3 + c] += r[a]*ys[i][c]; } }
  for (let a = 0; a < dim; a++) A[a*dim + a] += 1e-6*(1 + A[a*dim + a]);
  // Gauss-Jordan
  const M = Array.from({length: dim}, (_, i) => Array.from(A.slice(i*dim, i*dim + dim)).concat(Array.from(B.slice(i*3, i*3 + 3))));
  for (let c = 0; c < dim; c++){
    let p = c; for (let r = c + 1; r < dim; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]]; const d = M[c][c] || 1e-12;
    for (let k = c; k < dim + 3; k++) M[c][k] /= d;
    for (let r = 0; r < dim; r++) if (r !== c){ const f = M[r][c]; if (f) for (let k = c; k < dim + 3; k++) M[r][k] -= f*M[c][k]; }
  }
  return M.map(row => row.slice(dim));            // dim x 3
}

/* ---------------- geometry of one part ---------------- */
// the outline as a Fourier series of log radius around the centroid, order chosen by a tolerance
function starInit(pix, w, X, Y, half, kmax){
  let cx = 0, cy = 0; for (const q of pix){ cx += X[q % w]; cy += Y[(q/w) | 0]; } cx /= pix.length; cy /= pix.length;
  const NB = 96, r = new Float64Array(NB).fill(-1), px = 1/half;
  for (const q of pix){ const dx = X[q % w] - cx, dy = Y[(q/w) | 0] - cy, a = Math.atan2(dy, dx), rr = Math.hypot(dx, dy) + 0.6*px;
    const b = Math.min(NB - 1, Math.floor((a + PI)/(2*PI)*NB)); if (rr > r[b]) r[b] = rr; }
  const ok = []; for (let i = 0; i < NB; i++) if (r[i] > 0) ok.push(i);
  if (!ok.length) return null;
  for (let i = 0; i < NB; i++) if (r[i] <= 0){          // fill empty directions by circular interpolation
    let lo = i, hi = i; while (r[(lo + NB) % NB] <= 0) lo--; while (r[hi % NB] <= 0) hi++;
    const a = r[(lo + NB) % NB], b = r[hi % NB], t = (i - lo)/(hi - lo); r[i] = a + (b - a)*t;
  }
  const lr = Array.from(r, v => Math.log(Math.max(v, px)));
  const ang = i => (i + 0.5)/NB*2*PI - PI;
  const coef = K => { const c = {a0: 0, ab: []}; for (let i = 0; i < NB; i++) c.a0 += lr[i]/NB;
    for (let k = 1; k <= K; k++){ let a = 0, b = 0; for (let i = 0; i < NB; i++){ a += lr[i]*Math.cos(k*ang(i)); b += lr[i]*Math.sin(k*ang(i)); } c.ab.push([2*a/NB, 2*b/NB]); } return c; };
  const area = pix.length, tol = Math.max(1, Math.sqrt(area)/30)*px;
  let K = 1, c = coef(1);
  for (K = 1; K <= kmax; K++){
    c = coef(K); let e = 0;
    for (let i = 0; i < NB; i++){ let L = c.a0; for (let k = 1; k <= K; k++) L += c.ab[k - 1][0]*Math.cos(k*ang(i)) + c.ab[k - 1][1]*Math.sin(k*ang(i)); e += (Math.exp(L) - r[i])**2; }
    if (Math.sqrt(e/NB) < tol) break;
  }
  K = Math.min(K, kmax); c = coef(K);
  return {a: cx, b: cy, a0: c.a0, ab: c.ab, K};
}
function insideStar(d, x, y){
  const dx = x - d.a, dy = y - d.b, rho = Math.hypot(dx, dy), ph = Math.atan2(dy, dx); let L = d.a0;
  for (let k = 1; k <= d.K; k++) L += d.ab[k - 1][0]*Math.cos(k*ph) + d.ab[k - 1][1]*Math.sin(k*ph);
  return rho < Math.exp(L);
}
// a part one closed star curve cannot hold is split along its main axis, at most twice
function fitParts(pix, w, X, Y, half, kmax, depth, out, minPix){
  const d = starInit(pix, w, X, Y, half, kmax); if (!d) return;
  const set = new Set(pix); let inter = 0, starN = 0;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const q of pix){ const i = q % w, j = (q/w) | 0; x0 = Math.min(x0, i); x1 = Math.max(x1, i); y0 = Math.min(y0, j); y1 = Math.max(y1, j); }
  const pad = Math.ceil((x1 - x0 + y1 - y0)*0.25);
  for (let j = Math.max(0, y0 - pad); j <= y1 + pad; j++) for (let i = Math.max(0, x0 - pad); i <= Math.min(w - 1, x1 + pad); i++){
    if (insideStar(d, X[i], Y[Math.min(Y.length - 1, j)])){ starN++; if (set.has(j*w + i)) inter++; }
  }
  const iou = inter/(pix.length + starN - inter);
  if (iou < 0.78 && depth < 2 && pix.length > 2*minPix){
    // split by the main axis through the centroid
    let cx = 0, cy = 0; for (const q of pix){ cx += q % w; cy += (q/w) | 0; } cx /= pix.length; cy /= pix.length;
    let sxx = 0, syy = 0, sxy = 0; for (const q of pix){ const dx = q % w - cx, dy = ((q/w) | 0) - cy; sxx += dx*dx; syy += dy*dy; sxy += dx*dy; }
    const th = 0.5*Math.atan2(2*sxy, sxx - syy), ux = Math.cos(th), uy = Math.sin(th);
    const A = [], B = []; for (const q of pix){ ((q % w - cx)*ux + (((q/w) | 0) - cy)*uy < 0 ? A : B).push(q); }
    for (const part of [A, B]) if (part.length >= minPix) fitParts(part, w, X, Y, half, kmax, depth + 1, out, minPix);
    return;
  }
  d.pix = pix; d.iou = iou; out.push(d);
}

/* ---------------- the model in TensorFlow.js ---------------- */
function bgBasis(X, Y, w, h, nx, ny, xr, yr){       // cos(pi i u) cos(pi j s), u and s in 0..1 across the picture frame
  const B = new Float32Array(w*h*nx*ny);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++){
    const u = (X[i] + xr)/(2*xr), s = (Y[j] + yr)/(2*yr), p = (j*w + i)*nx*ny; let q = 0;
    for (let a = 0; a < nx; a++) for (let b = 0; b < ny; b++) B[p + q++] = Math.cos(PI*a*u)*Math.cos(PI*b*s);
  }
  return B;
}

Compact.run = async function(o){
  const tf = window.tf; if (!tf) throw new Error("TensorFlow.js did not load");
  const {T, w, h, X, Y, half} = o, n = w*h, t0 = performance.now(), rng = mulberry(20260919);
  const secs = () => (performance.now() - t0)/1000, left = () => o.budget - secs();
  const tick = () => new Promise(r => setTimeout(r, 0));
  const say = (stage, frac) => o.onProgress && o.onProgress(stage, frac);
  const Tb = blurRGB(T, w, h, 1.0), Tl = blurRGB(T, w, h, 2.0);
  const NX = o.nx || 4, NY = o.ny || 3, XR = r3((w/2)/half), YR = r3((h/2)/half);

  /* 1-2. segmentation */
  say('Splitting the picture into parts', 0.02); await tick();
  const Ts = blurRGB(T, w, h, 3.0), feats = new Float32Array(n*5);
  for (let i = 0; i < n; i++){ const L = toLab(Ts[i*3], Ts[i*3 + 1], Ts[i*3 + 2]); feats[i*5] = L[0]; feats[i*5 + 1] = L[1]*1.2; feats[i*5 + 2] = L[2]*1.2;
    feats[i*5 + 3] = 18*X[i % w]; feats[i*5 + 4] = 18*Y[(i/w) | 0]; }
  const K_SEG = o.kseg || 6;
  const lab = kmeans(feats, n, 5, K_SEG, 12, rng);
  let comps = [];
  for (let c = 0; c < K_SEG; c++){
    const m = new Uint8Array(n); for (let i = 0; i < n; i++) m[i] = lab[i] === c ? 1 : 0;
    for (const cp of components(m, w, h)) if (cp.pix.length > 0.008*n) comps.push(cp);
  }
  comps.sort((a, b) => b.pix.length - a.pix.length);
  // the background is the large region that touches the frame most; it becomes the smooth field
  const top = comps.slice(0, 4); let bgc = top.reduce((a, b) => (b.border > a.border ? b : a), top[0]);
  const partsPix = comps.filter(c => c !== bgc).slice(0, o.maxParts || 9);
  const bgMask = new Uint8Array(n); if (bgc) for (const q of bgc.pix) bgMask[q] = 1;

  /* 3. background: least squares on the background pixels (all pixels if there is none) */
  const BB = bgBasis(X, Y, w, h, NX, NY, XR, YR), nb = NX*NY;
  const rows = [], ys = [];
  for (let i = 0; i < n; i += 2) if (!bgc || bgMask[i]){ rows.push(Array.from(BB.subarray(i*nb, i*nb + nb))); ys.push([Tl[i*3], Tl[i*3 + 1], Tl[i*3 + 2]]); }
  const bg0 = lstsq(rows, ys, nb);

  /* parts: outlines, painter order by hull area (containers first), shading by least squares */
  let parts = [];
  for (const cp of partsPix) fitParts(cp.pix, w, X, Y, half, o.kmax || 6, 0, parts, Math.max(12, 0.004*n));
  parts.forEach(d => d.hull = hullArea(d.pix.map(q => [q % w, (q/w) | 0])));
  parts.sort((a, b) => b.hull - a.hull);
  const shade = (d, deg) => { const r = [], y = []; for (const q of d.pix){ const xx = X[q % w] - d.a, yy = Y[(q/w) | 0] - d.b; r.push(deg ? [1, xx, yy] : [1]); y.push([Tl[q*3], Tl[q*3 + 1], Tl[q*3 + 2]]); } return lstsq(r, y, deg ? 3 : 1); };
  parts.forEach(d => { d.col = shade(d, 1); d.kappa = 30; d.deg = 1; });
  say('Fitting ' + parts.length + ' parts', 0.08); await tick();

  /* 4. the differentiable model, all parts at once: masks (P, n), then the painter's product in closed form,
        H = B prod_i (1 - M_i) + sum_j C_j M_j prod_{i>j} (1 - M_i), through a reversed cumulative sum of logs */
  const KM = Math.max(3, ...parts.map(d => d.K));
  const Xt = tf.tensor2d(Array.from({length: n}, (_, i) => X[i % w]), [1, n]), Yt = tf.tensor2d(Array.from({length: n}, (_, i) => Y[(i/w) | 0]), [1, n]);
  const BBt = tf.tensor2d(BB, [n, nb]), tgt = tf.tensor2d(Tb, [n, 3]).transpose(), tgtL = Tl;
  const bgV = tf.variable(tf.tensor2d(bg0.flat(), [nb, 3]));
  const group = list => {                                   // one trainable block of parts
    const P = list.map(d => { const r = [d.a, d.b, d.a0, Math.log(d.kappa)]; for (let k = 0; k < KM; k++){ const c = d.ab[k] || [0, 0]; r.push(c[0], c[1]); } return r; });
    const Km = list.map(d => { const r = []; for (let k = 0; k < KM; k++){ const on = k < d.K ? 1 : 0; r.push(on, on); } return r; });
    const C = list.map(d => [d.col[0], d.deg ? d.col[1] : [0, 0, 0], d.deg ? d.col[2] : [0, 0, 0]]);
    const Dm = list.map(d => [[1, 1, 1], d.deg ? [1, 1, 1] : [0, 0, 0], d.deg ? [1, 1, 1] : [0, 0, 0]]);
    return {list, P: tf.variable(tf.tensor2d(P)), Km: tf.tensor2d(Km), C: tf.variable(tf.tensor3d(C)), Dm: tf.tensor3d(Dm)};
  };
  const groups = [group(parts)];
  let gate = 0;
  const forward = () => tf.tidy(() => {
    const P = tf.concat(groups.map(g => g.P), 0), Km = tf.concat(groups.map(g => g.Km), 0);
    const C = tf.mul(tf.concat(groups.map(g => g.C), 0), tf.concat(groups.map(g => g.Dm), 0));
    const a = P.slice([0, 0], [-1, 1]), b = P.slice([0, 1], [-1, 1]), a0 = P.slice([0, 2], [-1, 1]), lk = P.slice([0, 3], [-1, 1]);
    const dx = tf.sub(Xt, a), dy = tf.sub(Yt, b);
    const rho = tf.sqrt(tf.add(tf.add(tf.square(dx), tf.square(dy)), 1e-8)), ph = tf.atan2(dy, dx);
    const coef = tf.mul(P.slice([0, 4], [-1, 2*KM]), Km);
    let L = a0;
    for (let k = 1; k <= KM; k++) L = tf.add(L, tf.add(tf.mul(tf.cos(tf.mul(ph, k)), coef.slice([0, 2*k - 2], [-1, 1])), tf.mul(tf.sin(tf.mul(ph, k)), coef.slice([0, 2*k - 1], [-1, 1]))));
    const z = tf.clipByValue(tf.mul(tf.exp(lk), tf.sub(tf.mul(rho, tf.exp(tf.neg(L))), 1)), -30, 30);
    const M = tf.minimum(tf.exp(tf.neg(tf.exp(z))), 1 - 1e-6);                       // (P, n)
    const logKeep = tf.log(tf.sub(1, M));
    const Pn = M.shape[0], U = tf.tensor2d(Array.from({length: Pn*Pn}, (_, q) => ((q % Pn) > ((q/Pn) | 0) ? 1 : 0)), [Pn, Pn]);
    const after = tf.exp(tf.matMul(U, logKeep));                                     // prod over later parts (strict upper triangle)
    const all = tf.exp(tf.sum(logKeep, 0));                                          // (n)
    const c0 = C.slice([0, 0, 0], [-1, 1, 3]).squeeze([1]), c1 = C.slice([0, 1, 0], [-1, 1, 3]).squeeze([1]), c2 = C.slice([0, 2, 0], [-1, 1, 3]).squeeze([1]);
    const wgt = tf.mul(M, after);                                                     // (P, n)
    // sum_j (c0_j + gate (c1_j dx_j + c2_j dy_j)) wgt_j, per channel
    const part = tf.add(tf.matMul(c0, wgt, true, false), tf.mul(gate, tf.add(tf.matMul(c1, tf.mul(wgt, dx), true, false), tf.matMul(c2, tf.mul(wgt, dy), true, false))));
    const bgImg = tf.matMul(bgV, BBt, true, true);                                    // (3, n)
    return tf.add(tf.mul(bgImg, all), part);                                          // (3, n)
  });
  const lossOf = () => tf.mean(tf.square(tf.sub(forward(), tgt)));
  const snapshot = () => {
    const parts2 = [];
    for (const g of groups){ const P = g.P.arraySync(), C = g.C.arraySync();
      g.list.forEach((d, i) => parts2.push({P: P[i].slice(0, 4 + 2*d.K), C: d.deg ? C[i] : [C[i][0]], K: d.K, deg: d.deg})); }
    return {bg: bgV.arraySync(), parts: parts2, gate, nx: NX, ny: NY, xr: XR, yr: YR};
  };
  async function fit(untilSec, lr, varList, stage, f0, f1, maxIters){
    const opt = tf.train.adam(lr), start = secs(); let i = 0;
    for (; i < (maxIters || 1e9) && secs() < untilSec; i++){
      if (o.shouldStop && o.shouldStop()) break;
      opt.minimize(lossOf, false, varList);
      if (i % 5 === 4){ say(stage, f0 + (f1 - f0)*Math.min(1, (secs() - start)/Math.max(0.1, untilSec - start))); await tick(); }
      if (o.onPreview && i % 25 === 24) o.onPreview(snapshot());
    }
    opt.dispose(); return i;
  }
  const pull = () => { const t = forward(), d = t.transpose().dataSync(); t.dispose(); return d; };   // interleaved rgb
  const allVars = () => [bgV].concat(...groups.map(g => [g.P, g.C]));
  const B_ = o.budget;
  gate = 0; await fit(0.30*B_, 0.01, allVars(), 'Fitting outlines', 0.1, 0.35);        // shapes with flat colours
  gate = 1; await fit(0.52*B_, 0.005, allVars(), 'Shading the parts', 0.35, 0.6);      // shading on

  /* 5. details: small parts at the largest remaining error, kept while each constant buys enough */
  const rmseL = () => { const img = pull(); const b = blurRGB(img, w, h, 2.0); let s = 0; for (let i = 0; i < n*3; i++){ const e = (b[i] - tgtL[i])*255; s += e*e; } return Math.sqrt(s/(n*3)); };
  const nconst = () => nb*3 + groups.reduce((a, g) => a + g.list.reduce((c, d) => c + 4 + 2*d.K + (d.deg ? 9 : 3), 0), 0);
  let cur = rmseL(), misses = 0, added = 0;
  const lam = o.lambda || 0.02;
  while (secs() < 0.8*B_ && misses < 4 && parts.length + added < (o.maxTotal || 40)){
    if (o.shouldStop && o.shouldStop()) break;
    const img = pull(), b = blurRGB(img, w, h, 2.0), res = new Float32Array(n);
    for (let i = 0; i < n; i++) res[i] = (Math.abs(b[i*3] - tgtL[i*3]) + Math.abs(b[i*3 + 1] - tgtL[i*3 + 1]) + Math.abs(b[i*3 + 2] - tgtL[i*3 + 2]))/3*255;
    const sorted = Float32Array.from(res).sort(), thr = Math.max(10, sorted[Math.floor(n*0.985)]);
    const mk = new Uint8Array(n); for (let i = 0; i < n; i++) mk[i] = res[i] > thr ? 1 : 0;
    const cc = components(mk, w, h); if (!cc.length) break;
    cc.sort((p, q) => q.pix.reduce((s, i) => s + res[i], 0) - p.pix.reduce((s, i) => s + res[i], 0));
    const best = cc[Math.min(misses, cc.length - 1)];             // after a rejection, try the next largest error
    if (best.pix.length < 6){ misses++; continue; }
    const d = starInit(best.pix, w, X, Y, half, 3); if (!d){ misses++; continue; }
    d.deg = 0; d.kappa = 20; d.K = Math.min(d.K, KM);
    { const r = [], y = []; for (const q of best.pix){ r.push([1]); y.push([Tl[q*3], Tl[q*3 + 1], Tl[q*3 + 2]]); } d.col = lstsq(r, y, 1); }
    const before = nconst(), g = group([d]); groups.push(g);
    await fit(Math.min(0.8*B_, secs() + Math.max(1.5, 0.03*B_)), 0.01, [g.P, g.C], 'Adding detail ' + (added + 1), 0.6, 0.8);
    const r2 = rmseL(), gain = (cur - r2)/(nconst() - before);
    if (gain < lam){ groups.pop(); g.P.dispose(); g.C.dispose(); g.Km.dispose(); g.Dm.dispose(); misses++; continue; }
    cur = r2; misses = 0; added++;
  }
  await fit(0.88*B_, 0.002, allVars(), 'Final tuning', 0.8, 0.9);

  /* 6. grain: frequency and strength from the fine texture of the parts (statistics, not pixels) */
  const snap = snapshot();
  const model = Compact.round(snap);
  let grain = null;
  if (o.grain !== false){
    say('Matching the texture', 0.92); await tick();
    const img = Compact.renderSmall(model, X, Y, w, h, null);
    const lum = a => { const L = new Float32Array(n); for (let i = 0; i < n; i++) L[i] = 0.299*a[i*3] + 0.587*a[i*3 + 1] + 0.114*a[i*3 + 2]; return L; };
    const hp = L => { const b = blurPlane(L, w, h, 1.5); for (let i = 0; i < n; i++) b[i] = L[i] - b[i]; return b; };
    // texture lives in the residual, away from edges: skip pixels near any part boundary or strong edge of the model
    const tL = lum(T), mL = lum(img), rH = new Float32Array(n); for (let i = 0; i < n; i++) rH[i] = tL[i] - mL[i];
    const rh = hp(rH), inPart = new Uint8Array(n), Am = Compact.unionMask(model, X, Y, w, h), mB = blurPlane(mL, w, h, 1.0);
    for (let j = 1; j < h - 1; j++) for (let i = 1; i < w - 1; i++){
      const q = j*w + i, gx = mB[q + 1] - mB[q - 1], gy = mB[q + w] - mB[q - w];
      inPart[q] = Am[q] > 0.97 && Math.hypot(gx, gy) < 0.02 ? 1 : 0;
    }
    const stats = a => { let s = 0, g = 0, c = 0; for (let j = 1; j < h; j++) for (let i = 1; i < w; i++){ const q = j*w + i; if (!inPart[q] || !inPart[q - 1] || !inPart[q - w]) continue; s += a[q]*a[q]; g += (a[q] - a[q - 1])**2 + (a[q] - a[q - w])**2; c++; } return c ? {sd: Math.sqrt(s/c), gr: Math.sqrt(g/(2*c)), c} : null; };
    const st = stats(rh);
    // enough smooth area, and a texture of at least about 4/255 spread through it
    if (st && st.c > 0.08*n && st.sd > 0.016){
      let bestF = 30, bestE = Infinity, G = null;
      for (let f = 14; f <= 60; f += 4){                // the frequency whose texture has the picture's grain size
        const g = Compact.grainField(f, X, Y, w, h, 24), s2 = stats(hp(g)); if (!s2 || !s2.sd) continue;
        const e = Math.abs(s2.gr/s2.sd - st.gr/st.sd); if (e < bestE){ bestE = e; bestF = f; G = g; }
      }
      if (G){
        const s2 = stats(hp(G)), amp = Math.round(Math.min(3, st.sd/Math.max(1e-4, s2.sd))*20)/20;
        let mu = 0; for (let i = 0; i < n; i++) mu += G[i]; mu = Math.round(mu/n*1000)/1000;
        if (amp > 0) grain = {f: bestF, amp, n: 24, mu};
      }
    }
  }
  model.grain = grain;
  // free the GPU memory
  [Xt, Yt, BBt, tgt, bgV].forEach(t => t.dispose()); groups.forEach(g => [g.P, g.C, g.Km, g.Dm].forEach(t => t.dispose()));
  return model;
};

/* ---------------- rounding and exact rendering ---------------- */
const r3 = v => Math.round(v*1000)/1000, r40 = v => Math.round(v*40)/40, r80 = v => Math.round(v*80)/80;
Compact.round = function(s){
  return {nx: s.nx, ny: s.ny, xr: s.xr, yr: s.yr, bg: s.bg.map(row => row.map(r80)),
    parts: s.parts.map(p => ({a: r3(p.P[0]), b: r3(p.P[1]), a0: r3(p.P[2]), kappa: Math.max(1, Math.round(Math.exp(p.P[3]))),
      ab: Array.from({length: p.K}, (_, k) => [r3(p.P[4 + 2*k]), r3(p.P[5 + 2*k])]), K: p.K, deg: p.deg,
      col: p.C.map((row, i) => row.map(i === 0 ? r40 : r40))}))};
};
Compact.grainField = function(f, X, Y, w, h, N){  // his skin grain: sum over s of thresholded cos^10 lattices turned by s^2
  const G = new Float32Array(w*h);
  for (let s = 1; s <= N; s++){
    const c2 = Math.cos(s*s), s2 = Math.sin(s*s), fr = f + s/5, p1 = 10*Math.sin(10*s), p2 = 10*Math.sin(9*s);
    for (let j = 0; j < h; j++){ const y = Y[j];
      for (let i = 0; i < w; i++){ const x = X[i];
        const a = Math.cos(fr*(x*c2 + y*s2) + p1), b = Math.cos(fr*(x*s2 - y*c2) + p2);
        const z = -200*((a*a)**5*(b*b)**5 - 0.5);
        G[j*w + i] += (z > 30 ? 0 : Math.exp(-Math.exp(z)))/N;
      }
    }
  }
  return G;
};
function maskAt(p, x, y){
  const dx = x - p.a, dy = y - p.b, rho = Math.sqrt(dx*dx + dy*dy + 1e-8), ph = Math.atan2(dy, dx); let L = p.a0;
  for (let k = 1; k <= p.K; k++) L += p.ab[k - 1][0]*Math.cos(k*ph) + p.ab[k - 1][1]*Math.sin(k*ph);
  let z = p.kappa*(rho*Math.exp(-L) - 1); if (z > 30) return 0; if (z < -30) z = -30;
  return Math.exp(-Math.exp(z));
}
Compact.unionMask = function(m, X, Y, w, h){
  const A = new Float32Array(w*h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++){ let keep = 1; for (const p of m.parts) keep *= 1 - maskAt(p, X[i], Y[j]); A[j*w + i] = 1 - keep; }
  return A;
};
/* evaluate the rounded formula on any grid; upto limits the parts (for the video); out interleaved rgb */
Compact.renderSmall = function(m, X, Y, w, h, upto){
  const out = new Float32Array(w*h*3);
  const N = upto == null ? m.parts.length : upto;
  const G = m.grain ? Compact.grainField(m.grain.f, X, Y, w, h, m.grain.n) : null;
  for (let j = 0; j < h; j++){ const y = Y[j], s = (y + m.yr)/(2*m.yr);
    for (let i = 0; i < w; i++){ const x = X[i], u = (x + m.xr)/(2*m.xr);
      const c = [0, 0, 0]; let q = 0;
      for (let a = 0; a < m.nx; a++){ const ca = Math.cos(PI*a*u); for (let b = 0; b < m.ny; b++, q++){ const v = ca*Math.cos(PI*b*s); c[0] += m.bg[q][0]*v; c[1] += m.bg[q][1]*v; c[2] += m.bg[q][2]*v; } }
      let keep = 1;
      for (let k = 0; k < N; k++){ const p = m.parts[k], M = maskAt(p, x, y); if (!M) continue; keep *= 1 - M;
        const dx = x - p.a, dy = y - p.b;
        for (let v = 0; v < 3; v++){ const col = p.col[0][v] + (p.deg ? p.col[1][v]*dx + p.col[2][v]*dy : 0); c[v] = c[v]*(1 - M) + col*M; } }
      const g = G ? 1 + m.grain.amp*(G[j*w + i] - m.grain.mu)*(1 - keep) : 1;
      const o = (j*w + i)*3; out[o] = c[0]*g; out[o + 1] = c[1]*g; out[o + 2] = c[2]*g;
    }
  }
  return out;
};
Compact.constants = m => m.nx*m.ny*3 + m.parts.reduce((a, p) => a + 4 + 2*p.K + (p.deg ? 9 : 3), 0) + (m.grain ? 2 : 0);

window.Compact = Compact;
})();
