"""
measure_utils.py — AAA-level utility library
Handles ANY shape: rectangles, circles, ellipses, organic blobs,
fruits, seeds, glasses — anything with a clear boundary.
"""

import cv2
import numpy as np
import math


# ═══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def enhanceImage(img, apply=False):
    """
    CLAHE contrast enhancement for dark / unevenly lit images.
    Only applied when --enhance flag is set.
    """
    if not apply:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def autoCanny(img, sigma=0.33):
    """
    Compute Canny thresholds from image median pixel value.
    Works across any lighting condition — no hardcoding needed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    v     = np.median(gray)
    lower = int(max(0,   (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # Ensure there's always a useful gap between thresholds
    if upper - lower < 30:
        lower = max(0, lower - 20)
        upper = min(255, upper + 20)
    return [lower, upper]


# ═══════════════════════════════════════════════════════════════
#  CONTOUR DETECTION — works on ALL shape types
# ═══════════════════════════════════════════════════════════════

def getContours(img, cThr=[100, 100], showCanny=False,
                minArea=1000, filter=0, draw=False):
    """
    Detect contours. filter=0 → all shapes. filter=N → N-sided polygons only.
    Returns (annotated_img, sorted_contour_list).
    Each contour entry: [vertex_count, area, approx, bbox, raw_contour]
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])

    kernel   = np.ones((5, 5))
    imgDial  = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre  = cv2.erode(imgDial,  kernel, iterations=2)

    if showCanny:
        cv2.imshow('Canny', imgThre)

    contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < minArea:
            continue
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        bbox  = cv2.boundingRect(approx)

        if filter > 0:
            if len(approx) == filter:
                finalContours.append([len(approx), area, approx, bbox, c])
        else:
            finalContours.append([len(approx), area, approx, bbox, c])

    finalContours.sort(key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


# ═══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════

def reorder(myPoints):
    """
    Reorder 4 contour points: [TL, TR, BL, BR].
    Safe — extracts clean [x, y] arrays.
    """
    myPointsNew = np.zeros_like(myPoints)
    pts = myPoints.reshape((4, 2))
    add  = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    myPointsNew[0] = pts[np.argmin(add)]   # TL
    myPointsNew[1] = pts[np.argmin(diff)]  # TR
    myPointsNew[2] = pts[np.argmax(diff)]  # BL
    myPointsNew[3] = pts[np.argmax(add)]   # BR
    return myPointsNew


def warpImg(img, points, w, h, pad=20):
    """Perspective-correct the A4 paper to a flat w×h rectangle."""
    points = reorder(points)
    pts1   = np.float32(points)
    pts2   = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad,
                      pad:imgWarp.shape[1] - pad]
    return imgWarp


def findDis(pts1, pts2):
    """Euclidean distance between two (x, y) points (as lists or arrays)."""
    return math.sqrt((pts2[0] - pts1[0])**2 + (pts2[1] - pts1[1])**2)


# ═══════════════════════════════════════════════════════════════
#  SHAPE CLASSIFICATION — the key to handling ANY object
# ═══════════════════════════════════════════════════════════════

def detectShapeType(approx):
    """
    Classify shape from its polygon approximation.
    Covers rectangles, triangles, circles, ellipses, organic blobs.
    """
    n = len(approx)

    if n == 3:
        return 'triangle'

    if n == 4:
        pts  = approx.reshape(4, 2).astype(np.float32)
        side_lengths = [
            findDis(pts[i], pts[(i + 1) % 4]) for i in range(4)
        ]
        angles = []
        for i in range(4):
            p0 = pts[(i - 1) % 4]
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            v1 = p0 - p1
            v2 = p2 - p1
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(math.degrees(math.acos(np.clip(cos_a, -1, 1))))

        max_s, min_s = max(side_lengths), min(side_lengths)
        all_right    = all(abs(a - 90) < 20 for a in angles)
        is_square    = (max_s / (min_s + 1e-6)) < 1.15

        if all_right and is_square:
            return 'square'
        if all_right:
            return 'rectangle'
        return 'quadrilateral'

    if n == 5:
        return 'pentagon'
    if n == 6:
        return 'hexagon'

    # Circle vs ellipse vs organic
    area  = cv2.contourArea(approx)
    peri  = cv2.arcLength(approx, True)
    if peri == 0:
        return 'blob'

    circularity = 4 * math.pi * area / (peri * peri)

    if circularity > 0.85:
        return 'circle'
    if circularity > 0.65:
        return 'ellipse'

    return 'organic'   # fruits, seeds, leaves, glasses, etc.


def shapeColor(shape):
    """Return BGR color for each shape type — used in overlays."""
    palette = {
        'rectangle':    (255, 100,  50),
        'square':       (255, 150,  50),
        'circle':       ( 50, 200, 255),
        'ellipse':      ( 50, 150, 255),
        'triangle':     (100, 255, 100),
        'pentagon':     (200, 100, 255),
        'hexagon':      (255, 200,  50),
        'quadrilateral':(200, 200, 100),
        'organic':      ( 50, 255, 180),   # ← fruits, seeds, etc.
        'blob':         (150, 150, 150),
    }
    return palette.get(shape, (200, 200, 200))


# ═══════════════════════════════════════════════════════════════
#  MEASUREMENT DRAWING — works for ALL shape types
# ═══════════════════════════════════════════════════════════════

def drawMeasurements(img, results, mode='a4'):
    """
    Draw contours, labels, arrows, and bounding boxes for every detected object.
    Handles both rectangular and organic shapes elegantly.
    """
    overlay = img.copy()

    for i, r in enumerate(results):
        shape   = r['shape']
        color   = shapeColor(shape)
        approx  = r['contour']
        raw_cnt = r['raw_contour']
        x, y, w, h = r['bbox']
        nW = r['width_cm']
        nH = r['height_cm']

        # ── 1. Semi-transparent fill ──
        cv2.fillPoly(overlay, [raw_cnt], color)

        # ── 2. Contour outline ──
        cv2.polylines(img, [approx], True, color, 2)
        cv2.drawContours(img, [raw_cnt], -1, color, 2)

        # ── 3. Measurements — style varies by shape ──
        if shape in ('rectangle', 'square') and len(approx) == 4:
            pts = reorder(approx)
            p0 = tuple(pts[0][0])
            p1 = tuple(pts[1][0])
            p2 = tuple(pts[2][0])
            # Arrows along actual edges
            cv2.arrowedLine(img, p0, p1, (255, 50, 255), 2, tipLength=0.08)
            cv2.arrowedLine(img, p0, p2, (255, 50, 255), 2, tipLength=0.08)
            # Width label above bounding box
            _putLabelSafe(img, f'{nW}cm', (x + w // 2, y - 12), color)
            # Height label left of bounding box
            _putLabelSafe(img, f'{nH}cm', (x - 60,    y + h // 2), color)
        else:
            # ── Organic / circular shapes: use bounding box dimensions ──
            cx = x + w // 2
            cy = y + h // 2
            # Horizontal arrow
            cv2.arrowedLine(img, (x, cy), (x + w, cy), (255, 50, 255), 2, tipLength=0.06)
            # Vertical arrow
            cv2.arrowedLine(img, (cx, y), (cx, y + h), (255, 50, 255), 2, tipLength=0.06)
            _putLabelSafe(img, f'{nW}cm', (x + w // 2, y - 12), color)
            _putLabelSafe(img, f'{nH}cm', (x - 65, cy), color)

        # ── 4. Shape label chip ──
        label = f'{shape.upper()} #{i+1}'
        if mode == 'no_ref':
            label += ' (relative)'
        _putChip(img, label, (x, y - 36), color)

        # ── 5. Area label ──
        if r.get('area_cm2'):
            _putLabelSafe(img, f"{r['area_cm2']}cm²", (x + w // 2, y + h + 22), color)

    # Apply semi-transparent overlay
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    return img


def _putLabelSafe(img, text, pos, color, scale=0.65, thickness=2):
    """Draw text with a dark backing rectangle so it's always readable."""
    font   = cv2.FONT_HERSHEY_DUPLEX
    tw, th = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y   = pos
    x = max(0, min(x - tw // 2, img.shape[1] - tw - 4))
    y = max(th + 4, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + 3),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _putChip(img, text, pos, color, scale=0.5, thickness=1):
    """Draw a colored badge chip for the shape label."""
    font   = cv2.FONT_HERSHEY_SIMPLEX
    tw, th = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y   = pos
    x = max(0, min(x, img.shape[1] - tw - 12))
    y = max(th + 8, y)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 8, y + 4), color, cv2.FILLED)
    cv2.putText(img, text, (x + 3, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  NO-REFERENCE MODE — when no A4 paper is present
# ═══════════════════════════════════════════════════════════════

def detectObjectsNoA4(img, args):
    """
    When no A4 paper is found, detect objects and report relative
    measurements (pixel dimensions + shape classification).
    Objects are still detected, classified, and drawn — just labeled
    as 'relative' without a cm reference.
    """
    cThr = autoCanny(img, sigma=args.sigma)
    _, conts = getContours(img, cThr=cThr, minArea=args.obj_area,
                           filter=0, draw=False)
    results = []
    for obj in conts:
        approx  = obj[2]
        bbox    = obj[3]
        raw_cnt = obj[4]
        shape   = detectShapeType(approx)
        x, y, w, h = bbox

        results.append({
            'shape':     shape,
            'width_cm':  f'{w}px',   # no cm conversion without reference
            'height_cm': f'{h}px',
            'area_cm2':  None,
            'bbox':      bbox,
            'contour':   approx,
            'raw_contour': raw_cnt,
        })

    return results


# ═══════════════════════════════════════════════════════════════
#  MULTI-SCALE A4 DETECTION — tries harder to find the paper
# ═══════════════════════════════════════════════════════════════

def findA4Robust(img, min_area=30000):
    """
    Try multiple preprocessing strategies to find A4 paper.
    Falls back gracefully if none succeed.
    Returns approx contour or None.
    """
    strategies = [
        lambda i: autoCanny(i, sigma=0.33),
        lambda i: autoCanny(i, sigma=0.20),
        lambda i: autoCanny(i, sigma=0.50),
        lambda i: [30, 100],   # fixed low thresholds
        lambda i: [80, 200],   # fixed high thresholds
    ]
    for fn in strategies:
        cThr = fn(img)
        _, conts = getContours(img, cThr=cThr, minArea=min_area, filter=4)
        if len(conts) > 0:
            return conts[0][2]

    return None