import os

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV(cv2) is required for this app. Install it with: pip install opencv-python-headless"
    ) from exc

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from scipy import ndimage
from skimage.segmentation import watershed, clear_border

st.set_page_config(layout="wide", page_title="MY DIP")


# ---------------------------------------------------------------
# Processing (HW05 pipeline)
# ---------------------------------------------------------------
def odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def ellipse(k):
    k = odd(k)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def colorize(labels):
    """label image -> RGB (background black, each object a random color)"""
    rng = np.random.default_rng(0)
    lut = rng.integers(60, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels]


def run_pipeline(f, p):
    # 2. noise reduction
    f_den = cv2.medianBlur(f, odd(p["median"])) if p["median"] > 1 else f.copy()

    # 3. thresholding
    otsu, _ = cv2.threshold(f_den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T = otsu if p["use_otsu"] else p["T"]
    mask = (f_den > T) if p["bright"] else (f_den < T)
    mask = mask.astype(np.uint8)
    if p["open"] > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(p["open"]))
    mask = ndimage.binary_fill_holes(mask)
    if p["clear_border"]:
        mask = clear_border(mask)

    # 4. connected components (optionally split touching objects with watershed)
    seeds = None
    if p["watershed"]:
        seeds = cv2.erode(mask.astype(np.uint8), ellipse(p["erode"]))
        if p["dilate"] > 1:
            seeds = cv2.dilate(seeds, ellipse(p["dilate"])) & mask.astype(np.uint8)
        markers, _ = ndimage.label(seeds)
        labels = watershed(-ndimage.distance_transform_edt(mask), markers, mask=mask)
    else:
        labels, _ = ndimage.label(mask)

    ids = np.arange(1, labels.max() + 1)
    areas = ndimage.sum(np.ones_like(labels), labels, ids).astype(int) if len(ids) else np.array([], int)
    keep = ids[areas >= p["min_area"]]
    labels = np.where(np.isin(labels, keep), labels, 0)
    cc = np.array([[i, a] for i, a in zip(ids, areas) if a >= p["min_area"]], dtype=int).reshape(-1, 2)
    return f_den, otsu, T, mask, seeds, labels, cc


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.write("## Homework 5 마정미 : Object Counting")
st.write(":tomato:")
st.caption("의료영상처리 HW05 기반 · 이미지를 올리면 grayscale 변환 → noise 제거 → thresholding → "
           "connected component 분석으로 object 개수를 셉니다.")

with st.sidebar:
    st.write("## my sidebar")
    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    st.header("⚙️ 파라미터")
    preset = st.selectbox("프리셋", ["직접 설정", "밝은 물체 · 어두운 배경 (예: 땡땡이)", "어두운 물체 · 밝은 배경 (예: 토마토)"], index=2)
    if preset.startswith("밝은"):
        d = dict(bright=True, median=5, use_otsu=True, T=128, open=5, clear_border=True,
                 watershed=False, erode=25, dilate=9, min_area=100, roi=((0, 100), (0, 100)))
    elif preset.startswith("어두운"):
        d = dict(bright=False, median=5, use_otsu=False, T=100, open=11, clear_border=False,
                 watershed=True, erode=45, dilate=15, min_area=1200, roi=((5, 83), (19, 86)))
    else:
        d = dict(bright=True, median=5, use_otsu=True, T=128, open=5, clear_border=False,
                 watershed=False, erode=25, dilate=9, min_area=50, roi=((0, 100), (0, 100)))

    max_side = st.slider("처리 해상도 (긴 변, px)", 300, 2000, 1000, 100,
                         help="큰 사진은 이 크기로 줄여서 처리합니다 (속도).")
    st.subheader("1. 관심 영역 (ROI)")
    roi_y = st.slider("세로 범위 (%)", 0, 100, d["roi"][0])
    roi_x = st.slider("가로 범위 (%)", 0, 100, d["roi"][1])
    st.subheader("2. Noise 제거")
    median = st.slider("Median filter 크기", 1, 31, d["median"], 2)
    st.subheader("3. Thresholding")
    bright = st.radio("Object 밝기", ["배경보다 밝음", "배경보다 어두움"],
                      index=0 if d["bright"] else 1) == "배경보다 밝음"
    use_otsu = st.checkbox("Otsu 자동 임계값 사용", d["use_otsu"])
    T = st.slider("임계값 T (Otsu 미사용 시)", 0, 255, d["T"], disabled=use_otsu)
    open_k = st.slider("Opening 커널 (작은 점 제거)", 1, 61, d["open"], 2)
    clear_b = st.checkbox("테두리에 닿은 object 제거", d["clear_border"])
    st.subheader("4. Connected component")
    ws = st.checkbox("붙은 object 분리 (erosion + watershed)", d["watershed"])
    erode_k = st.slider("Erosion 커널 (씨앗 만들기)", 3, 151, d["erode"], 2, disabled=not ws)
    dilate_k = st.slider("Dilation 커널 (씨앗 조각 합치기)", 1, 61, d["dilate"], 2, disabled=not ws)
    min_area = st.number_input("최소 면적 (pixel)", 0, 1_000_000, d["min_area"], 50)

default_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tomato_final.jpg")  # 같은 폴더의 직접 찍은 사진
if uploaded is not None:
    source = uploaded
elif os.path.exists(default_image):
    source = default_image
    st.caption("기본 예시 사진(tomato_final.jpg)을 표시 중입니다. 사이드바에서 다른 사진을 올릴 수 있어요. "
               "토마토 사진은 프리셋 '어두운 물체 · 밝은 배경'을 선택하세요.")
else:
    st.info("왼쪽 사이드바에서 직접 찍은 사진을 올려주세요.")
    st.stop()

# 1. grayscale load
img = Image.open(source)
img = ImageOps.exif_transpose(img)   # 휴대폰 사진이 옆으로 눕지 않게 회전 정보 반영
img = img.convert("L")
scale = min(1.0, max_side / max(img.size))
if scale < 1.0:
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
f_full = np.array(img)
H, W = f_full.shape
y0, y1 = int(H * roi_y[0] / 100), max(int(H * roi_y[1] / 100), int(H * roi_y[0] / 100) + 10)
x0, x1 = int(W * roi_x[0] / 100), max(int(W * roi_x[1] / 100), int(W * roi_x[0] / 100) + 10)
f = f_full[y0:y1, x0:x1]

params = dict(median=median, bright=bright, use_otsu=use_otsu, T=T, open=open_k,
              clear_border=clear_b, watershed=ws, erode=erode_k, dilate=dilate_k, min_area=min_area)
with st.spinner("처리 중..."):
    f_den, otsu, T_used, mask, seeds, labels, cc = run_pipeline(f, params)

# ---- result summary ----
n = len(cc)
c1, c2, c3 = st.columns(3)
c1.metric("검출된 object 수", n)
if n:
    i_max, i_min = int(np.argmax(cc[:, 1])), int(np.argmin(cc[:, 1]))
    c2.metric("가장 큰 object (pixel)", f"{cc[i_max, 1]:,}")
    c3.metric("가장 작은 object (pixel)", f"{cc[i_min, 1]:,}")
st.caption(f"처리 크기 {f.shape[1]}×{f.shape[0]} px · Otsu = {otsu:.0f} · 사용한 T = {T_used:.0f}")

# ---- step-by-step images ----
st.subheader("단계별 결과")
a, b, c = st.columns(3)
roi_box = cv2.cvtColor(f_full, cv2.COLOR_GRAY2RGB)
cv2.rectangle(roi_box, (x0, y0), (x1 - 1, y1 - 1), (255, 60, 60), max(2, W // 300))
a.image(roi_box, caption="1. Grayscale + ROI (빨간 박스)", width="stretch")
b.image(f_den, caption=f"2. Median {odd(median)}×{odd(median)}", width="stretch")
c.image((mask * 255).astype(np.uint8), caption=f"3. Binary mask (T={T_used:.0f})", width="stretch")

hist = np.bincount(f_den.ravel(), minlength=256)
st.bar_chart(hist, height=180, x_label="Pixel intensity", y_label="Frequency")

a, b, c = st.columns(3)
if seeds is not None:
    a.image(seeds * 255, caption="4-1. Watershed seeds", width="stretch")
a2 = b if seeds is not None else a
a2.image(colorize(labels), caption=f"4. Connected components ({n})", width="stretch")

# overlay with numbers
overlay = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
col = colorize(labels)
m = labels > 0
overlay[m] = (0.5 * overlay[m] + 0.5 * col[m]).astype(np.uint8)
for k, (idx, _) in enumerate(cc, 1):
    ys, xs = np.nonzero(labels == idx)
    cv2.putText(overlay, str(k), (int(xs.mean()) - 8, int(ys.mean()) + 6),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, f.shape[1] / 1600), (255, 255, 255), 2, cv2.LINE_AA)
(c if seeds is not None else b).image(overlay, caption="번호 표시", width="stretch")

# ---- 5. largest / smallest ----
if n:
    st.subheader("5. 가장 큰 object / 가장 작은 object")
    hl = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
    hl[labels == cc[i_max, 0]] = [230, 50, 50]
    hl[labels == cc[i_min, 0]] = [40, 110, 230]
    a, b, c = st.columns(3)
    a.image((labels == cc[i_max, 0]).astype(np.uint8) * 255,
            caption=f"Largest: {cc[i_max, 1]:,} px", width="stretch")
    b.image((labels == cc[i_min, 0]).astype(np.uint8) * 255,
            caption=f"Smallest: {cc[i_min, 1]:,} px", width="stretch")
    c.image(hl, caption="빨강 = 가장 큼, 파랑 = 가장 작음", width="stretch")

    with st.expander("object별 pixel 수 표"):
        st.dataframe({"번호": np.arange(1, n + 1), "label": cc[:, 0], "pixels": cc[:, 1]},
                     hide_index=True, width="stretch")
else:
    st.warning("검출된 object가 없습니다. 사이드바에서 Object 밝기나 임계값을 바꿔 보세요.")
