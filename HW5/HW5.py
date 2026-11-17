"""Automatic object counting (HW05 pipeline: grayscale -> denoise -> threshold -> morphology ->
connected components), with every parameter chosen from the image itself.

Several candidate binary masks are built (brightness, colour difference, saturation, edges;
with and without illumination correction). Each one goes through the same morphology +
watershed + connected-component steps, and the result whose objects look most like real
objects (clear boundary, different from their surroundings, compact) is kept."""
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

WORK_SIDE = 800  # long side used for processing


# ---------------------------------------------------------------- helpers
def odd(k):
    k = max(1, int(round(k)))
    return k if k % 2 else k + 1


def ellipse(k):
    k = odd(k)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def resize_long(img, side=WORK_SIDE):
    s = side / max(img.shape[:2])
    if abs(s - 1) < 1e-3:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=interp), s


def to_u8(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def otsu(img):
    t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def small_op(img, k, op):
    """Large morphology / blur done on a small copy (fast), result resized back."""
    s = min(1.0, 250 / max(img.shape))
    sm = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ks = odd(max(3, k * s))
    out = op(sm, ks)
    return cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def components(mask):
    lab, n = ndimage.label(mask)
    areas = np.bincount(lab.ravel())[1:] if n else np.array([], int)
    return lab, areas


def estimate_r(mask, total):
    """Typical object radius from the areas of the (non-speck) components."""
    _, a = components(mask)
    a = a[a > max(15, 0.0003 * total)]
    if len(a) == 0:
        return np.sqrt(total / 200 / np.pi)
    a = a[a >= np.percentile(a, 30)]
    return float(np.sqrt(np.median(a) / np.pi))


def relabel_seq(labels):
    u = np.unique(labels)
    u = u[u > 0]
    lut = np.zeros(labels.max() + 1, int)
    lut[u] = np.arange(1, len(u) + 1)
    return lut[labels]


def relabel(labels, keep):
    lut = np.zeros(labels.max() + 1, int)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels]


def solidity_of(labels, n):
    sol = np.zeros(n)
    for i, sl in enumerate(ndimage.find_objects(labels)[:n]):
        if sl is None:
            continue
        m = (labels[sl] == i + 1).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        sol[i] = cv2.contourArea(c) / hull if hull > 0 else 0
    return sol


def background_color_distance(lab):
    """Lab distance from the background colours (k-means clusters that dominate the border)."""
    H, W, _ = lab.shape
    sm = cv2.resize(lab, None, fx=200 / max(H, W), fy=200 / max(H, W), interpolation=cv2.INTER_AREA)
    pix = sm.reshape(-1, 3).astype(np.float32)
    k = 4
    _, lbl, centers = cv2.kmeans(pix, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(sm.shape[:2])
    w = max(1, min(lbl.shape) // 25)
    border = np.concatenate([lbl[:w].ravel(), lbl[-w:].ravel(), lbl[:, :w].ravel(), lbl[:, -w:].ravel()])
    cnt = np.bincount(border, minlength=k)
    bg = centers[cnt >= 0.25 * cnt.max()]
    d = np.min(np.linalg.norm(lab.reshape(-1, 1, 3).astype(np.float32) - bg[None], axis=2), axis=1)
    return d.reshape(H, W)


# ---------------------------------------------------------------- candidate masks
def mask_from_feature(feat, polarity, correct, total):
    """Threshold one feature image. polarity: objects brighter ('bright') or darker ('dark')."""
    img = feat if polarity == "bright" else 255 - feat
    m0 = cv2.morphologyEx((img > otsu(img)).astype(np.uint8), cv2.MORPH_OPEN, ellipse(3))
    r0 = estimate_r(m0, total)
    if correct:
        # illumination correction: subtract a background estimated with an element larger
        # than the objects (white top-hat), then divide by the local maximum so dim objects
        # in dark corners reach the same level as bright ones
        bg = small_op(img, 4 * r0 + 15,
                      lambda x, k: cv2.GaussianBlur(cv2.morphologyEx(x, cv2.MORPH_OPEN, ellipse(k)), (0, 0), k / 2))
        corr = np.clip(img.astype(np.float32) - bg, 0, None)
        lmax = small_op(corr, 3 * r0 + 1,
                        lambda x, k: cv2.GaussianBlur(cv2.dilate(x, ellipse(k)), (0, 0), max(1, k / 3)))
        corr = corr / np.maximum(lmax, 0.4 * np.percentile(corr, 99.5) + 1e-6)
        corr = to_u8(corr)
    else:
        corr = img.copy()
    T = otsu(corr)
    return corr, T, (corr > T).astype(np.uint8)


def mask_from_edges(grad_u8, total):
    """Objects whose outline is visible even if their inside looks like the background:
    strong edges -> close small gaps -> fill closed outlines -> remove leftover edge lines."""
    T = otsu(grad_u8)
    edges = (grad_u8 > T).astype(np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ellipse(5))
    filled = ndimage.binary_fill_holes(edges).astype(np.uint8)
    inside = filled & ~edges.astype(bool)          # region enclosed by an outline
    inside = cv2.dilate(inside.astype(np.uint8), ellipse(5)) & filled
    return grad_u8, T, inside.astype(np.uint8)


# ---------------------------------------------------------------- common steps
def inscribed_radius(dist):
    """Typical object half-width = median height of the distance-transform peaks.
    Works even when objects touch (clumps do not inflate it like areas do)."""
    from skimage.feature import peak_local_max
    pk = peak_local_max(dist, min_distance=3, threshold_abs=1.5, exclude_border=False)
    if len(pk) == 0:
        return max(2.0, float(dist.max()))
    v = dist[pk[:, 0], pk[:, 1]]
    return float(np.median(v[v >= np.percentile(v, 25)]))



def merge_oversplit(labels, surf, grad, edge_T, inten=None):
    """Region merging after watershed: two touching pieces are merged back when the
    border between them is neither a 'valley' in the shape/brightness surface
    nor a real edge in the image (typical for long objects cut in the middle)."""
    n = labels.max()
    if n < 2:
        return labels
    peak = ndimage.maximum(surf, labels, np.arange(1, n + 1))
    med_in = ndimage.median(inten, labels, np.arange(1, n + 1)) if inten is not None else None
    pairs = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - dy, : labels.shape[1] - dx]
        b = labels[dy:, dx:]
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        la, lb = a[m], b[m]
        sv = np.minimum(surf[: surf.shape[0] - dy, : surf.shape[1] - dx][m], surf[dy:, dx:][m])
        gv = np.maximum(grad[: grad.shape[0] - dy, : grad.shape[1] - dx][m], grad[dy:, dx:][m])
        if inten is not None:
            sv = np.minimum(inten[: inten.shape[0] - dy, : inten.shape[1] - dx][m], inten[dy:, dx:][m]).astype(np.float32)
        lo, hi = np.minimum(la, lb), np.maximum(la, lb)
        for i, j, s_, g_ in zip(lo, hi, sv, gv):
            e = pairs.setdefault((i, j), [0, 0.0, 0.0, -1.0])
            e[0] += 1; e[1] += g_; e[2] += s_; e[3] = max(e[3], s_)
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (i, j), (cnt, gsum, ssum, smax) in sorted(pairs.items(), key=lambda kv: -kv[1][3]):
        if cnt < 3:
            continue
        ri, rj = find(i), find(j)
        if ri == rj or (gsum / cnt) >= edge_T:      # a real edge separates them -> keep split
            continue
        if med_in is not None and (ssum / cnt) < 0.85 * min(med_in[i - 1], med_in[j - 1]):
            continue                                  # darker gap between them -> keep split
        # merge only if the union is one convex blob (long object cut in two),
        # not two round blobs touching (union has a concave waist)
        members = np.nonzero((lut_now := np.array([find(x) for x in range(n + 1)])) == ri)[0].tolist() + \
                  np.nonzero(lut_now == rj)[0].tolist()
        u = np.isin(labels, members)
        ys, xs = np.nonzero(u)
        sub = u[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull > 0 and cv2.contourArea(c) / hull >= 0.9:
            parent[rj] = ri
    lut = np.array([find(x) for x in range(n + 1)])
    return lut[labels]


def segment(mask, total, keep_border=False, intensity=None, grad=None):
    """Morphology clean-up, watershed split of touching objects, size/shape filtering."""
    H, W = mask.shape
    r = estimate_r(mask, total)
    r_in = inscribed_radius(ndimage.distance_transform_edt(mask))
    k = min(0.25 * r, 0.5 * r_in)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(max(3, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(max(3, 0.12 * r)))
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    r = estimate_r(mask, total)

    # split touching objects with a marker-controlled watershed.
    # surface = shape (distance to background) + brightness inside the object,
    # so both narrow "necks" and darker gaps between touching objects become borders
    dist = ndimage.distance_transform_edt(mask)
    r_in = max(2.0, inscribed_radius(dist))
    surf = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), max(1.0, 0.2 * r_in)) / r_in
    if intensity is not None:
        it = cv2.GaussianBlur(intensity.astype(np.float32), (0, 0), max(1.0, 0.2 * r_in))
        vals = it[mask > 0]
        if vals.size:
            lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
            surf = surf + 0.6 * np.clip((it - lo) / max(hi - lo, 1e-6), 0, 1)
    surf = surf * mask
    markers, _ = ndimage.label(h_maxima(surf, 0.25))
    lab_m, _ = components(mask)
    missing = np.setdiff1d(np.unique(lab_m[mask > 0]), np.unique(lab_m[markers > 0]))
    nxt = markers.max() + 1
    for l in missing:
        ys, xs = np.nonzero(lab_m == l)
        i = np.argmax(surf[ys, xs])
        markers[ys[i], xs[i]] = nxt
        nxt += 1
    labels = watershed(-surf, markers, mask=mask.astype(bool))
    if grad is not None:
        labels = merge_oversplit(labels, surf, grad, otsu(grad), intensity)
        labels, _ = ndimage.label(labels > 0, structure=None) if False else (relabel_seq(labels), None)
    dist_s = surf

    n = labels.max()
    ids = np.arange(1, n + 1)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    on_edge = np.isin(ids, edge_ids)
    ref = areas[(areas > 0) & ~on_edge] if np.any((areas > 0) & ~on_edge) else areas[areas > 0]
    typical = np.median(ref) if len(ref) else 0
    r_typ = np.sqrt(typical / np.pi) if typical else r
    r_typ = min(r_typ, 2.5 * inscribed_radius(ndimage.distance_transform_edt(mask)))   # long objects
    min_area = max(20, 0.0001 * total, 0.2 * typical)
    from skimage.segmentation import find_boundaries
    own = (labels > 0) & ~find_boundaries(labels, mode="inner")
    dist_own = ndimage.distance_transform_edt(own)
    thick = ndimage.maximum(dist_own, labels, ids) if n else np.array([])
    sol = solidity_of(labels, n)
    ok = (areas >= min_area) & (thick >= 0.35 * r_typ) & (sol >= 0.5)
    if not keep_border:
        ok &= ~on_edge                       # only partly visible -> not counted
    labels = relabel(labels, ids[ok])
    return dict(mask=mask, dist=dist_s, markers=markers, labels=labels, r=r, min_area=min_area)


def evaluate(d, lab, grad, total):
    """Remove objects that do not stand out from their surroundings and score the result."""
    labels = d["labels"]
    n = labels.max()
    d["score"] = 0.0
    d["cc"] = np.zeros((0, 2), int)
    if n == 0:
        return d
    w = max(2, int(0.3 * d["r"]))
    H, W = labels.shape
    allobj = labels > 0
    con, bgr, zsc = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, sl in enumerate(ndimage.find_objects(labels)[:n]):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - w), min(H, sl[0].stop + w)
        x0, x1 = max(0, sl[1].start - w), min(W, sl[1].stop + w)
        m = labels[y0:y1, x0:x1] == i + 1
        mu8 = m.astype(np.uint8)
        ring = cv2.dilate(mu8, ellipse(2 * w + 1)).astype(bool) & ~allobj[y0:y1, x0:x1]
        bnd = (cv2.dilate(mu8, ellipse(5)) - cv2.erode(mu8, ellipse(5))).astype(bool)
        if ring.sum() < 5:
            ring = cv2.dilate(mu8, ellipse(2 * w + 1)).astype(bool) & ~m
        if ring.sum() < 5:
            continue
        patch = lab[y0:y1, x0:x1]
        diff = np.linalg.norm(patch[m].mean(0) - patch[ring].mean(0))         # colour/brightness ΔE
        spread = 0.5 * (np.linalg.norm(patch[ring].std(0)) + np.linalg.norm(patch[m].std(0)))
        con[i] = diff
        zsc[i] = diff / (spread + 2.0)   # difference relative to the local texture
        bgr[i] = grad[y0:y1, x0:x1][bnd].mean()                              # boundary sharpness
    g_ref = np.percentile(grad, 90) + 1e-6
    ok = (con >= max(6.0, 0.35 * np.median(con))) & (zsc >= 0.5)
    labels = relabel(labels, np.arange(1, n + 1)[ok])
    con, bgr, zsc = con[ok], bgr[ok], zsc[ok]
    areas = np.bincount(labels.ravel())[1:] if labels.max() else np.array([], int)
    n = len(areas)
    d["labels"] = labels
    if n == 0:
        return d
    d["cc"] = np.stack([np.arange(1, n + 1), areas], 1)
    sol = solidity_of(labels, n)
    med = np.median(areas)
    objectness = (np.minimum(1, con / 25) * np.minimum(1, zsc / 2.0) * np.minimum(1, bgr / g_ref)
                  * np.clip((sol - 0.5) / 0.35, 0, 1))
    weight = np.minimum(1, areas / (0.002 * total))           # specks count only partly
    consistent = np.where((areas > 0.2 * med) & (areas < 5 * med), 1.0, 0.5)
    cover = areas.sum() / total
    base = float(np.sum(objectness * weight * consistent) * (1.0 if cover < 0.7 else 0.3))
    # a good segmentation has objects of similar size and every object looks like an object;
    # fragmented results (many pieces of very different size) are penalised
    spread = float(np.std(np.log(areas))) if n > 1 else 0.0
    d["score"] = base * float(np.mean(objectness)) * float(np.exp(-max(0.0, spread - 0.35)))
    return d



# ---------------------------------------------------------------- extra algorithms
def labels_from_hough(imgs, grad_u8, r_in, total, fg=None):
    """Circle Hough transform on several images (brightness, colour difference):
    finds round objects even when they overlap or are partly hidden.
    Every candidate circle is checked (outline on real edges, inside = object),
    then overlapping duplicates are removed keeping the best supported one."""
    H, W = imgs[0].shape
    rmin, rmax = max(4, int(0.6 * r_in)), max(6, int(1.6 * r_in))
    edil = cv2.dilate((grad_u8 > otsu(grad_u8)).astype(np.uint8), ellipse(3)).astype(bool)
    allc = []
    for img in imgs:
        c = cv2.HoughCircles(cv2.GaussianBlur(img, (0, 0), 1.5), cv2.HOUGH_GRADIENT, dp=1.2,
                             minDist=max(4, int(0.5 * rmin)), param1=80,
                             param2=max(10, int(0.5 * rmin)), minRadius=rmin, maxRadius=rmax)
        if c is not None:
            allc.append(c[0])
    labels = np.zeros((H, W), np.int32)
    if not allc:
        return labels
    allc = np.concatenate(allc)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    yy, xx = np.mgrid[0:H, 0:W]
    cand = []
    for x, y, r in allc:
        xs = (x + r * np.cos(ang)).astype(int)
        ys = (y + r * np.sin(ang)).astype(int)
        inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if inb.mean() < 0.6:
            continue
        support = edil[ys[inb], xs[inb]].mean()
        if support < 0.45:
            continue
        if fg is not None:
            x0, x1, y0, y1 = int(max(0, x - r)), int(min(W, x + r + 1)), int(max(0, y - r)), int(min(H, y + r + 1))
            disk = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2 <= r * r
            if disk.sum() == 0 or fg[y0:y1, x0:x1][disk].mean() < 0.7:
                continue
        cand.append((support, x, y, r))
    if not cand:
        return labels
    med = np.median([c[3] for c in cand])
    cand = [c for c in cand if 0.75 * med <= c[3] <= 1.33 * med]
    keep = []
    for sup, x, y, r in sorted(cand, reverse=True):             # best supported first
        if all((x - a) ** 2 + (y - b) ** 2 > (0.55 * min(r, q)) ** 2 for _, a, b, q in keep):
            keep.append((sup, x, y, r))
    # paint weakest first so the best supported (usually the top-most) circles stay visible
    for k, (_, x, y, r) in enumerate(sorted(keep), 1):
        cv2.circle(labels, (int(x), int(y)), int(r), k, -1)
    return labels


def labels_from_graph(rgb_den, r, total):
    """Felzenszwalb graph segmentation on colour, background = regions covering the border."""
    from skimage.segmentation import felzenszwalb
    seg = felzenszwalb(rgb_den, scale=max(50, 3 * r * r / 10), sigma=0.8, min_size=int(max(20, 0.3 * np.pi * r * r)))
    seg = seg + 1
    areas = np.bincount(seg.ravel())
    border = np.concatenate([seg[0], seg[-1], seg[:, 0], seg[:, -1]])
    bcount = np.bincount(border, minlength=len(areas))
    perim = 2 * (seg.shape[0] + seg.shape[1])
    bg = (bcount > 0.08 * perim) | (areas > 0.25 * total)
    seg[bg[seg]] = 0
    return seg.astype(np.int32)


def filter_labels(labels, total, keep_border=False):
    """Same size / thickness / shape / border filter that segment() applies."""
    labels = relabel_seq(labels)
    n = labels.max()
    if n == 0:
        return labels
    ids = np.arange(1, n + 1)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    on_edge = np.isin(ids, edge_ids)
    ref = areas[~on_edge] if np.any(~on_edge) else areas
    typical = np.median(ref)
    from skimage.segmentation import find_boundaries
    own = (labels > 0) & ~find_boundaries(labels, mode="inner")
    thick = ndimage.maximum(ndimage.distance_transform_edt(own), labels, ids)
    r_typ = min(np.sqrt(typical / np.pi), 2.5 * max(1.0, np.median(thick)))
    sol = solidity_of(labels, n)
    ok = (areas >= max(20, 0.0001 * total, 0.2 * typical)) & (thick >= 0.35 * r_typ) & (sol >= 0.5)
    if not keep_border:
        ok &= ~on_edge
    return relabel(labels, ids[ok])




def labels_fg_split(rgb_den, fg, r, total):
    """Foreground found by a threshold method, split along colour boundaries
    (Felzenszwalb graph segmentation inside the foreground)."""
    from skimage.segmentation import felzenszwalb
    seg = felzenszwalb(rgb_den, scale=max(20, r * r / 8), sigma=0.6, min_size=int(max(10, 0.15 * np.pi * r * r))) + 1
    seg[~fg] = 0
    out = np.zeros_like(seg)
    k = 0
    for i, sl in enumerate(ndimage.find_objects(seg)):     # a segment may be several pieces inside fg
        if sl is None:
            continue
        sub, nn = ndimage.label(seg[sl] == i + 1)
        for j in range(1, nn + 1):
            k += 1
            out[sl][sub == j] = k
    return out.astype(np.int32)


# ---------------------------------------------------------------- main entry
def count_objects(rgb, force_feature=None, keep_border=False):
    """rgb: uint8 HxWx3 (or HxW). Returns the best result with its intermediate images."""
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    rgb_s, scale = resize_long(rgb)
    f = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY)                         # 1. grayscale
    total = f.size

    # 2. noise reduction (median removes speckle, bilateral smooths while keeping edges)
    noise = np.median(np.abs(f.astype(np.int16) - cv2.medianBlur(f, 3).astype(np.int16)))
    mk = 3 if noise < 3 else 5
    f_den = cv2.bilateralFilter(cv2.medianBlur(f, mk), 7, 25, 7)
    rgb_den = cv2.bilateralFilter(cv2.medianBlur(rgb_s, mk), 7, 25, 7)

    lab = cv2.cvtColor(rgb_den, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255.0                                          # L to 0..100
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.sqrt(a * a + b * b)
    gsm = cv2.GaussianBlur(lab, (0, 0), 1.5)
    grad = np.zeros(f.shape, np.float32)
    for c in range(3):
        gx = cv2.Sobel(gsm[..., c], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gsm[..., c], cv2.CV_32F, 0, 1)
        grad += gx * gx + gy * gy
    grad = np.sqrt(grad)
    grad_u8 = to_u8(grad)

    # local texture (standard deviation): smooth objects on rough ground (gravel, grass, asphalt)
    fl = f.astype(np.float32)
    kt = odd(max(7, 0.02 * max(f.shape)))
    mu = cv2.blur(fl, (kt, kt))
    tex = np.sqrt(np.maximum(cv2.blur(fl * fl, (kt, kt)) - mu * mu, 0))
    tex_u8 = to_u8(cv2.GaussianBlur(tex, (0, 0), kt / 3))
    feats = [("밝기 (밝은 object)", f_den, "bright"), ("밝기 (어두운 object)", f_den, "dark"),
             ("질감 (매끈한 object)", tex_u8, "dark")]
    if np.percentile(chroma, 99) > 12:                                  # colour photo
        feats += [("배경과의 색 차이", to_u8(background_color_distance(lab)), "bright"),
                  ("채도 (색이 진한 object)", to_u8(chroma), "bright")]

    cands = []
    for name, feat, pol in feats:
        for correct in (True, False):
            cands.append((name, feat, lambda feat=feat, pol=pol, c=correct: mask_from_feature(feat, pol, c, total),
                          "조명 보정" if correct else "보정 없음"))
    cands.append(("윤곽선 (경계 안쪽 채우기)", grad_u8, lambda: mask_from_edges(grad_u8, total), ""))
    if force_feature and force_feature not in ("원 검출 (Hough circle)", "그래프 분할 (Felzenszwalb)", "색 경계 분할 (전경 + 그래프 분할)"):
        cands = [c for c in cands if c[0] == force_feature] or cands

    runs = []
    for name, feat, make, variant in cands:
        corr, T, mask = make()
        d = segment(mask, total, keep_border, intensity=None if feat is grad_u8 else corr, grad=grad_u8)
        d.update(feature_name=name, variant=variant, feature=feat, corr=corr, T=T)
        runs.append(evaluate(d, lab, grad, total))
    # ---- other algorithms, sized from what the threshold-based runs found ----
    if not force_feature or force_feature in ("원 검출 (Hough circle)", "그래프 분할 (Felzenszwalb)", "색 경계 분할 (전경 + 그래프 분할)"):
        ref = max(runs, key=lambda d: d["score"])
        r_in = max(3.0, inscribed_radius(ndimage.distance_transform_edt(ref["mask"])))
        extra = []
        bgd = None
        for nm, ft, _ in feats:
            if nm == "배경과의 색 차이":
                bgd = ft
        # foreground = union of the masks of all reasonably good threshold-based results
        fg = np.zeros(f.shape, bool)
        for d0 in runs:
            if d0["score"] >= 0.3 * ref["score"] and d0["labels"].max():
                fg |= d0["labels"] > 0
        fg = cv2.dilate(fg.astype(np.uint8), ellipse(7)) > 0
        if not force_feature or force_feature.startswith("원"):
            extra.append(("원 검출 (Hough circle)", labels_from_hough([f_den] + ([bgd] if bgd is not None else []), grad_u8, r_in, total, fg)))
        if not force_feature or force_feature.startswith("그래프"):
            extra.append(("그래프 분할 (Felzenszwalb)", labels_from_graph(rgb_den, ref["r"], total)))
        if not force_feature or force_feature.startswith("색 경계"):
            colour_runs = [d0 for d0 in runs if d0["feature_name"] in ("채도 (색이 진한 object)", "배경과의 색 차이")
                           and d0["labels"].max()]
            if colour_runs:
                cr = max(colour_runs, key=lambda d0: d0["score"])
                fgc = ndimage.binary_fill_holes(cv2.morphologyEx(cr["mask"].astype(np.uint8), cv2.MORPH_OPEN, ellipse(3)))
                extra.append(("색 경계 분할 (전경 + 그래프 분할)", labels_fg_split(rgb_den, fgc, cr["r"], total)))
        for name, lb in extra:
            lb = filter_labels(lb, total, keep_border)
            d = dict(feature_name=name, variant="", feature=grad_u8, corr=grad_u8, T=otsu(grad_u8),
                     mask=(lb > 0).astype(np.uint8), dist=ndimage.distance_transform_edt(lb > 0).astype(np.float32),
                     markers=np.zeros_like(lb), labels=lb, r=float(np.sqrt(max(1, np.median(np.bincount(lb.ravel())[1:]) if lb.max() else 1) / np.pi)),
                     min_area=0)
            d = evaluate(d, lab, grad, total)
            if name.startswith("색 경계"):
                d["score"] *= 0.9          # refinement of another result: wins only when clearly better
            runs.append(d)
    pool = [d for d in runs if d["feature_name"] == force_feature] if force_feature else []
    best = max(pool or runs, key=lambda d: d["score"])
    scores = {}
    for d in runs:
        k = d["feature_name"] + (f" / {d['variant']}" if d["variant"] else "")
        scores[k] = round(d["score"], 2)
    best.update(f=f, f_den=f_den, rgb=rgb_s, scale=scale, scores=scores,
                feature_names=sorted({c[0] for c in cands}))
    return best


# =====================================================================
# Streamlit web page
# =====================================================================
import io
import os

import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(layout="wide", page_title="MY DIP")


def colorize(labels):
    rng = np.random.default_rng(0)
    lut = rng.integers(60, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels]


FEATURES = ["자동", "밝기 (밝은 object)", "밝기 (어두운 object)", "배경과의 색 차이", "채도 (색이 진한 object)",
            "질감 (매끈한 object)", "윤곽선 (경계 안쪽 채우기)", "원 검출 (Hough circle)", "그래프 분할 (Felzenszwalb)",
            "색 경계 분할 (전경 + 그래프 분할)"]


@st.cache_data(show_spinner=False)
def analyze(image_bytes, feature, keep_border):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    rgb = np.array(img)
    res = count_objects(rgb, None if feature == "자동" else feature, keep_border)
    return rgb.shape, res


st.write("## Homework 5 마정미 : Object Counting")
st.write(":tomato:")
st.caption("사진을 올리기만 하면 물체 밝기 방향, 물체 크기, 조명 보정, 임계값, 붙은 물체 분리를 "
           "영상에서 자동으로 정해서 object 개수를 셉니다.")

st.sidebar.write("## my sidebar")
my_upload = st.sidebar.file_uploader("Upload", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
with st.sidebar.expander("결과가 이상할 때만 (기본: 자동)"):
    feature = st.selectbox("object를 구분할 기준", FEATURES, index=0,
                           help="자동이면 여러 기준을 모두 해 보고 가장 object다운 결과를 고릅니다.")
    keep_border = st.checkbox("사진 테두리에 잘린 object도 세기", False)

default_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tomato_final.jpg")
if my_upload is not None:
    data = my_upload.getvalue()
elif os.path.exists(default_image):
    with open(default_image, "rb") as fp:
        data = fp.read()
    st.info("기본 예시 사진(tomato_final.jpg)의 결과입니다. 왼쪽 사이드바에서 직접 찍은 사진을 올려 보세요.")
else:
    st.info("왼쪽 사이드바에서 직접 찍은 사진을 올려주세요.")
    st.stop()

with st.spinner("분석 중..."):
    orig_shape, r = analyze(data, feature, keep_border)
rgb = r["rgb"]

f, labels, cc = r["f"], r["labels"], r["cc"]
n = len(cc)
px = 1.0 / (r["scale"] ** 2)                    # processing pixels -> original-image pixels
areas_orig = np.round(cc[:, 1] * px).astype(int) if n else np.array([], int)

# ---------------- summary ----------------
c1, c2, c3 = st.columns(3)
c1.metric("검출된 object 수", n)
if n:
    i_max, i_min = int(np.argmax(cc[:, 1])), int(np.argmin(cc[:, 1]))
    c2.metric("가장 큰 object (pixel)", f"{areas_orig[i_max]:,}")
    c3.metric("가장 작은 object (pixel)", f"{areas_orig[i_min]:,}")
st.caption(
    f"자동 판단 · 사용한 기준: **{r['feature_name']}{(' / ' + r['variant']) if r['variant'] else ''}** · "
    f"대표 크기(반지름) ≈ {r['r'] / r['scale']:.0f} px · 임계값 T = {r['T']:.0f} (Otsu) · "
    f"원본 {orig_shape[1]}×{orig_shape[0]} px (처리는 긴 변 {WORK_SIDE}px로 축소, pixel 수는 원본 기준으로 환산)")

# ---------------- steps ----------------
st.subheader("1. 입력 영상 → grayscale")
a, b = st.columns(2)
a.image(rgb, caption="Input", width="stretch")
b.image(f, caption="Grayscale", width="stretch")

st.subheader("2. Noise 제거 · object 강조 · 조명 보정")
a, b, c = st.columns(3)
a.image(r["f_den"], caption="Median + bilateral filter", width="stretch")
b.image(r["feature"], caption=f"선택된 기준 영상: {r['feature_name']}", width="stretch")
c.image(r["corr"], caption="조명 보정 (top-hat + local normalization)", width="stretch")
with st.expander("기준별 점수 (높은 쪽이 자동 선택됨)"):
    st.write(r["scores"])

st.subheader("3. Thresholding → binary mask")
a, b = st.columns(2)
hist = np.bincount(r["corr"].ravel(), minlength=256)
hist[0] = 0                                     # 0 (완전 배경)은 너무 커서 그래프에서 제외
a.bar_chart(hist, height=260, x_label="Pixel intensity (보정 영상, 0 제외)", y_label="Frequency")
a.caption(f"Otsu threshold T = {r['T']:.0f}")
b.image((r["mask"] > 0).astype(np.uint8) * 255, caption="Binary mask (opening · closing · hole filling)",
        width="stretch")

st.subheader("4. Connected component (붙은 object는 watershed로 분리)")
a, b, c = st.columns(3)
dist = r["dist"]
dist_v = (255 * dist / dist.max()).astype(np.uint8) if dist.max() > 0 else dist.astype(np.uint8)
dist_rgb = cv2.applyColorMap(dist_v, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
ys, xs = np.nonzero(r["markers"])
for y, x in zip(ys, xs):
    cv2.circle(dist_rgb, (int(x), int(y)), 3, (255, 60, 60), -1)
a.image(dist_rgb, caption="Distance transform + 밝기 + seeds (빨간 점)", width="stretch")
b.image(colorize(labels), caption=f"Components: {n}", width="stretch")

overlay = rgb.copy()
m = labels > 0
overlay[m] = (0.45 * overlay[m] + 0.55 * colorize(labels)[m]).astype(np.uint8)
fs = max(0.45, f.shape[1] / 1600)
for k, (idx, _) in enumerate(cc, 1):
    yy, xx = np.nonzero(labels == idx)
    pos = (int(xx.mean()) - 10, int(yy.mean()) + 8)
    cv2.putText(overlay, str(k), pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, str(k), pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
c.image(overlay, caption="번호 표시", width="stretch")

if n:
    st.subheader("5. 가장 큰 object / 가장 작은 object")
    hl = rgb.copy()
    hl[labels == cc[i_max, 0]] = [230, 50, 50]
    hl[labels == cc[i_min, 0]] = [40, 110, 230]
    a, b, c = st.columns(3)
    a.image((labels == cc[i_max, 0]).astype(np.uint8) * 255,
            caption=f"Largest: {areas_orig[i_max]:,} px", width="stretch")
    b.image((labels == cc[i_min, 0]).astype(np.uint8) * 255,
            caption=f"Smallest: {areas_orig[i_min]:,} px", width="stretch")
    c.image(hl, caption="빨강 = 가장 큼, 파랑 = 가장 작음", width="stretch")

    with st.expander("object별 pixel 수"):
        st.dataframe({"번호": np.arange(1, n + 1), "pixels (원본 기준)": areas_orig},
                     hide_index=True, width="stretch")
else:
    st.warning("object를 찾지 못했습니다. 물체와 배경의 밝기 차이가 큰 사진으로 다시 시도해 보세요.")

with st.expander("자동 처리 방법"):
    st.markdown(
        "- **Noise 제거**: median filter + bilateral filter (경계는 살리고 잡음만 줄임)\n"
        "- **여러 알고리즘 자동 선택**: thresholding(밝기·색 차이·채도·질감·윤곽선 영상, 조명 보정 유/무), "
        "Hough circle(겹친 원형 object), Felzenszwalb 그래프 분할(붙어 있는 object), "
        "색 경계 분할(색으로 찾은 영역을 색 경계로 다시 나눔, 낙엽처럼 작은 object가 빽빽할 때)로 "
        "각각 object를 찾아 보고, 주변과 대비가 뚜렷하고 모양이 볼록하며 크기가 고른 결과를 자동 선택\n"
        "- **조명 보정**: object보다 큰 구조 요소로 morphological opening/closing 한 배경을 빼고(top-hat), "
        "주변 최댓값으로 나눠 어두운 구석의 object도 같은 밝기로 맞춤\n"
        "- **Thresholding**: 보정 영상에 Otsu, object 크기에 비례한 opening/closing, hole filling\n"
        "- **분리**: distance transform(모양)과 내부 밝기를 합친 영상의 h-maxima를 seed로 watershed, "
        "그 뒤 사이에 실제 경계(edge·어두운 틈)가 없고 합친 모양이 볼록하면 다시 합침 (길쭉한 object가 잘리는 것 방지)\n"
        "- **필터링**: 너무 작은 것, 얇은 줄(그림자·테두리), 볼록하지 않은 모양, 주변과 구분이 안 되는 것"
        "(바닥 무늬), 사진 테두리에 잘린 object 제외")
