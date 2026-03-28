"""
╔══════════════════════════════════════════════════════════════╗
║         AAA-LEVEL OBJECT MEASUREMENT SYSTEM                  ║
║  Works on ANY image — rectangles, organic shapes, anything   ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from utils.utils import (
    autoCanny, getContours, reorder, warpImg,
    findDis, detectShapeType, drawMeasurements,
    detectObjectsNoA4, enhanceImage
)

# ─────────────────────────── CONFIG ───────────────────────────
A4_W_CM  = 21.0
A4_H_CM  = 29.7
SCALE    = 3
wP       = int(210 * SCALE)
hP       = int(297 * SCALE)
PAD      = 20

# ── Webcam settings ──
CAP_BRIGHTNESS = 160
CAP_WIDTH      = 1920
CAP_HEIGHT     = 1080
# ──────────────────────────────────────────────────────────────


def process_frame(img, args):
    """
    Full pipeline for one frame.
    Returns annotated display images.
    """
    if img is None:
        return None, None, []

    img = enhanceImage(img, args.enhance)
    display = img.copy()
    results  = []

    # ── STEP 1: Try to find A4 reference paper ──
    cThr = autoCanny(img, sigma=args.sigma)
    _, conts = getContours(img, cThr=cThr, minArea=args.a4_area, filter=4)

    if len(conts) != 0:
        # ─── A4 MODE: use paper as calibration reference ───
        biggest = conts[0][2]
        imgWarp = warpImg(img, biggest, wP, hP, pad=PAD)

        eff_w = wP - 2 * PAD   # effective pixel width  = 21.0 cm
        eff_h = hP - 2 * PAD   # effective pixel height = 29.7 cm

        cThr2 = autoCanny(imgWarp, sigma=args.sigma)
        imgOut, conts2 = getContours(
            imgWarp, cThr=cThr2,
            minArea=args.obj_area, filter=0,   # filter=0 → ALL shapes, not just quads
            draw=False
        )

        for obj in conts2:
            approx   = obj[2]
            bbox     = obj[3]
            raw_cnt  = obj[4]
            shape    = detectShapeType(approx)

            # ── Bounding-box dimensions → cm ──
            x, y, w, h = bbox
            nW = round(w / eff_w * A4_W_CM, 1)
            nH = round(h / eff_h * A4_H_CM, 1)

            # ── For rectangles: use actual edge distances ──
            if shape in ('rectangle', 'square') and len(approx) == 4:
                pts = reorder(approx)
                p0, p1, p2 = pts[0][0], pts[1][0], pts[2][0]
                nW = round(findDis(p0, p1) / eff_w * A4_W_CM, 1)
                nH = round(findDis(p0, p2) / eff_h * A4_H_CM, 1)

            # ── Area in cm² ──
            area_px = cv2.contourArea(raw_cnt)
            area_cm = round(area_px / (eff_w * eff_h) * (A4_W_CM * A4_H_CM), 2)

            results.append({
                'shape': shape,
                'width_cm': nW,
                'height_cm': nH,
                'area_cm2': area_cm,
                'bbox': bbox,
                'contour': approx,
                'raw_contour': raw_cnt,
            })

        imgOut = drawMeasurements(imgOut, results, mode='a4')
        cv2.imshow('Warped A4', imgOut)

        # Draw detected A4 outline on original
        cv2.polylines(display, [biggest], True, (0, 255, 100), 3)
        cv2.putText(display, 'A4 DETECTED', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)

    else:
        # ─── NO-REFERENCE MODE: relative measurements ───
        # Uses known object sizes or relative pixel analysis
        results = detectObjectsNoA4(img, args)
        display = drawMeasurements(display, results, mode='no_ref')

        cv2.putText(display, 'NO A4 — RELATIVE MODE', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    return display, results


def run_image(path, args):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[ERROR] Cannot read: {path}")
        sys.exit(1)

    print(f"\n[INFO] Processing: {path}")
    t0 = time.time()

    display, results = process_frame(img, args)

    elapsed = (time.time() - t0) * 1000
    print(f"[INFO] Done in {elapsed:.1f}ms — {len(results)} object(s) found\n")

    for i, r in enumerate(results, 1):
        print(f"  Object {i}: {r['shape'].upper():<12} "
              f"W={r['width_cm']}cm  H={r['height_cm']}cm  "
              f"Area={r['area_cm2']}cm²")

    display_resized = cv2.resize(display, (0, 0), None, 0.6, 0.6)
    cv2.imshow('Object Measurement — Press Q to quit', display_resized)

    print("\n[INFO] Press Q or ESC to close.")
    while True:
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()


def run_webcam(args):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, CAP_BRIGHTNESS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    print("[INFO] Webcam active. Press Q to quit, S to save snapshot.")
    snap_n = 0

    while True:
        success, img = cap.read()
        if not success:
            continue

        display, results = process_frame(img, args)

        # FPS overlay
        fps_text = f"Objects: {len(results)}"
        cv2.putText(display, fps_text, (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        small = cv2.resize(display, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Object Measurement (Webcam)', small)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            break
        elif k in (ord('s'), ord('S')):
            fname = f"snapshot_{snap_n:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"[INFO] Saved {fname}")
            snap_n += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='AAA Object Measurement — works on any image',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--image', '-i',  type=str, default='Resources/download3.jpg',
                        help='Path to input image')
    parser.add_argument('--webcam', '-w', action='store_true',
                        help='Use webcam instead of image')
    parser.add_argument('--a4-area',  type=int, default=50000,
                        help='Min pixel area to consider as A4 paper (default 50000)')
    parser.add_argument('--obj-area', type=int, default=2000,
                        help='Min pixel area for objects (default 2000)')
    parser.add_argument('--sigma',    type=float, default=0.33,
                        help='Canny auto-threshold sigma (default 0.33)')
    parser.add_argument('--enhance',  action='store_true',
                        help='Apply CLAHE contrast enhancement (helps dark/uneven images)')
    args = parser.parse_args()

    if args.webcam:
        run_webcam(args)
    else:
        run_image(args.image, args)


if __name__ == '__main__':
    main()