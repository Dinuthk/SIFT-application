"""
SIFT Multi-Image Matching & Full Panorama Stitching
=====================================================
Reads ALL images from ./input_images/ and:
  1. Detects SIFT keypoints on each image
  2. Matches every pair using Lowe's ratio test
  3. Builds a stitch ORDER graph (greedy MST — most-matches first)
  4. Chains all images into a single panorama via homography + gradient blend
  5. Crops black borders from the final result

Output folder (auto-created):
  output/
    01_keypoints_<name>.png          — keypoints on each input
    02_pairwise_<A>_vs_<B>.png       — match lines for every pair
    03_combined_keypoints.png        — all inputs tiled with keypoints
    04_stitch_order.png              — visualises the chaining order used
    05_stitch_clean.png              — final panorama, no annotations
    05_stitch_annotated.png          — final panorama + chain summary panel

Usage:
    python sift_matching.py                        # default folders
    python sift_matching.py input_images output    # custom folders
"""

import os, sys, itertools
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_DIR     = "pano_photos"
OUTPUT_DIR    = "pano_results"
MATCH_RATIO   = 0.75     # Lowe's ratio test threshold
MIN_MATCH     = 8        # minimum good matches to attempt homography
MAX_DRAW      = 60       # max match lines drawn per pair figure
RANSAC_THRESH = 4.0      # RANSAC reprojection threshold (px)

SIFT_PARAMS = dict(
    nfeatures=0,
    nOctaveLayers=3,
    contrastThreshold=0.03,
    edgeThreshold=10,
    sigma=1.6,
)

# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    files = sorted(f for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in exts)
    imgs, names = [], []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            imgs.append(img)
            names.append(os.path.splitext(f)[0])
            print(f"  Loaded: {f}  ({img.shape[1]}x{img.shape[0]})")
    return imgs, names


def save_img(path, img_bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr)
    print(f"  Saved : {path}")


def save_fig(path, fig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved : {path}")


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def scale_kps(kps, s):
    return [cv2.KeyPoint(kp.pt[0]*s, kp.pt[1]*s,
                         kp.size*s, kp.angle, kp.response, kp.octave)
            for kp in kps]


def resize_for_display(img, max_w=900):
    h, w = img.shape[:2]
    s = min(1.0, max_w / w)
    return cv2.resize(img, (int(w*s), int(h*s))), s


# ── SIFT detection & matching ──────────────────────────────────────────────────

def detect_all(sift, imgs):
    all_kps, all_desc = [], []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        all_kps.append(kps)
        all_desc.append(desc)
    return all_kps, all_desc


def lowe_match(desc1, desc2):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    bf    = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in pairs if m.distance < MATCH_RATIO * n.distance]


def compute_all_matches(all_desc, names):
    results = {}
    for i, j in itertools.combinations(range(len(all_desc)), 2):
        good = lowe_match(all_desc[i], all_desc[j])
        results[(i, j)] = good
        print(f"  {names[i]:>18} <-> {names[j]:<18}  {len(good)} matches")
    return results


# ── Stage 1: keypoint images ───────────────────────────────────────────────────

def stage_keypoints(imgs, names, all_kps, out_dir):
    print("\n[Stage 1] Keypoint images ...")
    for img, name, kps in zip(imgs, names, all_kps):
        out = img.copy()
        cv2.drawKeypoints(img, kps, out,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        save_img(os.path.join(out_dir, f"01_keypoints_{name}.png"), out)


# ── Stage 2: pairwise match figures ───────────────────────────────────────────

def stage_pairwise(imgs, names, all_kps, match_results, out_dir):
    print("\n[Stage 2] Pairwise match figures ...")
    for (i, j), good in match_results.items():
        ni, nj  = names[i], names[j]
        draw_n  = min(len(good), MAX_DRAW)
        si, s   = resize_for_display(imgs[i])
        sj, _   = resize_for_display(imgs[j])
        vis = cv2.drawMatches(
            si, scale_kps(all_kps[i], s),
            sj, scale_kps(all_kps[j], s),
            good[:draw_n], None,
            matchColor=(0, 230, 118),
            singlePointColor=(80, 80, 200),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        fig, ax = plt.subplots(figsize=(14, 4.5))
        fig.patch.set_facecolor("#111")
        ax.imshow(bgr2rgb(vis))
        ax.set_title(
            f"{ni}  <->  {nj}  |  kpts: {len(all_kps[i])}/{len(all_kps[j])}"
            f"  |  good matches: {len(good)}  (showing {draw_n})",
            color="white", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        save_fig(os.path.join(out_dir, f"02_pairwise_{ni}_vs_{nj}.png"), fig)


# ── Stage 3: combined tile ─────────────────────────────────────────────────────

def stage_combined(imgs, names, all_kps, out_dir):
    print("\n[Stage 3] Combined keypoints tile ...")
    n    = len(imgs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.2))
    fig.suptitle("SIFT Keypoints — All Input Images",
                 fontsize=13, fontweight="bold", color="white")
    fig.patch.set_facecolor("#111")
    flat = axes.flatten() if n > 1 else [axes]
    idx = 0
    for idx, (img, name, kps) in enumerate(zip(imgs, names, all_kps)):
        out = img.copy()
        cv2.drawKeypoints(img, kps, out,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        flat[idx].imshow(bgr2rgb(out))
        flat[idx].set_title(f"{name}\n{len(kps)} keypoints",
                            color="white", fontsize=9)
        flat[idx].axis("off")
    for k in range(idx + 1, len(flat)):
        flat[k].set_visible(False)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, "03_combined_keypoints.png"), fig)


# ── Homography ─────────────────────────────────────────────────────────────────

def compute_homography(kp1, kp2, good_matches):
    """
    Compute homography from kp2 -> kp1 using RANSAC.
    Returns (H, mask) or (None, None) if not enough matches.
    """
    if len(good_matches) < MIN_MATCH:
        return None, None

    src = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    return H, mask


# ── Warp & gradient blend helpers ─────────────────────────────────────────────

def warp_onto_canvas(img, H, canvas_w, canvas_h):
    """
    Warp img onto a canvas of size (canvas_w x canvas_h) using homography H.
    Returns (warped_bgr, mask_uint8).
    """
    warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h))
    gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return warped, mask


def gradient_blend_into(canvas, canvas_mask, warped, wmask):
    """
    Blend warped image into canvas using a gradient weight in the overlap zone.
    - Non-overlap areas: direct copy.
    - Overlap areas: smooth linear blend based on distance from each image's edge.
    """
    overlap = cv2.bitwise_and(canvas_mask, wmask)

    # Non-overlap: just copy new pixels in
    new_only = cv2.bitwise_and(wmask, cv2.bitwise_not(canvas_mask))
    canvas[new_only > 0] = warped[new_only > 0]

    # Overlap: blend using distance transform weights
    if overlap.any():
        dist_canvas = cv2.distanceTransform(canvas_mask, cv2.DIST_L2, 5)
        dist_warped = cv2.distanceTransform(wmask,        cv2.DIST_L2, 5)

        ov_mask = overlap > 0
        w_c = dist_canvas[ov_mask]
        w_w = dist_warped[ov_mask]
        total = w_c + w_w + 1e-6
        alpha = (w_c / total)[:, np.newaxis]   # weight for existing canvas

        canvas[ov_mask] = (
            alpha       * canvas[ov_mask].astype(np.float32) +
            (1 - alpha) * warped[ov_mask].astype(np.float32)
        ).astype(np.uint8)

    canvas_mask |= wmask


def crop_black_borders(img, thresh=5):
    """Remove black border rows/cols from a stitched panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


# ── Stage 4: stitch order diagram ─────────────────────────────────────────────

def build_stitch_order(n, match_results):
    """
    Greedy MST: pick image with most total matches as hub,
    then repeatedly attach the unmerged image with the most matches
    to any already-merged image.
    Returns (hub_index, [(src, dst), ...]) edge list.
    """
    totals = [0] * n
    for (i, j), good in match_results.items():
        totals[i] += len(good)
        totals[j] += len(good)

    hub    = int(np.argmax(totals))
    merged = {hub}
    order  = []

    while len(merged) < n:
        best_count, best_edge = -1, None
        for i in merged:
            for j in range(n):
                if j in merged:
                    continue
                key   = (min(i, j), max(i, j))
                count = len(match_results.get(key, []))
                if count > best_count:
                    best_count = count
                    best_edge  = (i, j)
        if best_edge is None:
            # No connection — just append remaining images in order
            for j in range(n):
                if j not in merged:
                    last = max(merged, key=lambda x: x)
                    order.append((last, j))
                    merged.add(j)
            break
        order.append(best_edge)
        merged.add(best_edge[1])

    return hub, order


def stage_stitch_order(names, hub, order, match_results, out_dir):
    """Visualise the MST chaining order as a node-edge diagram."""
    print("\n[Stage 4] Stitch order diagram ...")
    n   = len(names)
    fig, ax = plt.subplots(figsize=(max(10, n * 2.2), 4))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_title("Stitch Chain Order  (orange = hub)",
                 color="white", fontsize=11, fontweight="bold")

    xs = np.linspace(0, n - 1, n)

    for step, (src, dst) in enumerate(order):
        key   = (min(src, dst), max(src, dst))
        count = len(match_results.get(key, []))
        x1, x2 = xs[src], xs[dst]
        ax.annotate("", xy=(x2, 1.0), xytext=(x1, 1.0),
                    arrowprops=dict(arrowstyle="->", color="#00E676", lw=2.5))
        ax.text((x1 + x2) / 2, 1.22, f"step {step+1}\n{count} matches",
                ha="center", va="bottom", color="#00E676", fontsize=8)

    for i, name in enumerate(names):
        color = "#EF9F27" if i == hub else "#4FC3F7"
        ax.plot(xs[i], 1.0, "o", markersize=36, color=color, zorder=5)
        ax.text(xs[i], 1.0, str(i+1), ha="center", va="center",
                color="#111", fontsize=11, fontweight="bold", zorder=6)
        # wrap long names
        short = name if len(name) <= 16 else name[:14] + "…"
        ax.text(xs[i], 0.52, short, ha="center", va="top",
                color="white", fontsize=7.5)

    ax.text(xs[hub], 1.56, "hub", ha="center", va="bottom",
            color="#EF9F27", fontsize=9, fontstyle="italic")

    plt.tight_layout()
    save_fig(os.path.join(out_dir, "04_stitch_order.png"), fig)


# ── Stage 5: multi-image panorama ─────────────────────────────────────────────

def stage_stitch_all(imgs, names, all_kps, all_desc, match_results, out_dir):
    print("\n[Stage 5] Multi-image panorama stitch ...")
    n = len(imgs)

    hub, order = build_stitch_order(n, match_results)
    chain_str  = " -> ".join([names[hub]] + [names[j] for _, j in order])
    print(f"  Hub   : {names[hub]}")
    print(f"  Chain : {chain_str}")

    stage_stitch_order(names, hub, order, match_results, out_dir)

    # H_global[idx] = homography that maps image idx -> canvas space
    H_global   = {hub: np.eye(3, dtype=np.float64)}
    inlier_log = {}

    for src_idx, dst_idx in order:
        key  = (min(src_idx, dst_idx), max(src_idx, dst_idx))
        good = list(match_results.get(key, []))

        # Correct match direction: queryIdx belongs to smaller-index image
        if src_idx > dst_idx:
            good = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in good]

        H_local, hmask = compute_homography(
            all_kps[src_idx], all_kps[dst_idx], good)

        if H_local is None:
            print(f"  [WARN] Cannot compute H for "
                  f"{names[src_idx]}->{names[dst_idx]}, using identity fallback.")
            H_global[dst_idx] = H_global.get(src_idx, np.eye(3)).copy()
            inlier_log[(src_idx, dst_idx)] = (len(good), 0)
            continue

        inliers = int(hmask.sum())
        inlier_log[(src_idx, dst_idx)] = (len(good), inliers)
        print(f"  {names[src_idx]:>18} -> {names[dst_idx]:<18}  "
              f"inliers: {inliers}/{len(good)}")

        H_src_to_canvas = H_global.get(src_idx, np.eye(3))
        H_global[dst_idx] = H_src_to_canvas @ np.linalg.inv(H_local)

    # ── Compute canvas bounds ─────────────────────────────────────────────
    all_corners = []
    for idx, img in enumerate(imgs):
        h, w    = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
        H       = H_global.get(idx, np.eye(3))
        all_corners.append(cv2.perspectiveTransform(corners, H))

    pts     = np.concatenate(all_corners)
    x_min   = int(np.floor(pts[:, :, 0].min()))
    y_min   = int(np.floor(pts[:, :, 1].min()))
    x_max   = int(np.ceil (pts[:, :, 0].max()))
    y_max   = int(np.ceil (pts[:, :, 1].max()))

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    tx, ty   = -x_min, -y_min
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    print(f"  Canvas : {canvas_w}x{canvas_h}  offset ({tx},{ty})")

    # ── Warp and blend in chain order ─────────────────────────────────────
    process_order  = [hub] + [j for _, j in order]
    canvas_result  = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask    = np.zeros((canvas_h, canvas_w),    dtype=np.uint8)

    for idx in process_order:
        if idx not in H_global:
            continue
        H_full        = T @ H_global[idx]
        warped, wmask = warp_onto_canvas(imgs[idx], H_full, canvas_w, canvas_h)
        gradient_blend_into(canvas_result, canvas_mask, warped, wmask)
        print(f"  Blended: {names[idx]}")

    # ── Crop & save ───────────────────────────────────────────────────────
    final = crop_black_borders(canvas_result)
    print(f"  Final  : {final.shape[1]}x{final.shape[0]}")
    save_img(os.path.join(out_dir, "05_stitch_clean.png"), final)

    # ── Annotated figure ──────────────────────────────────────────────────
    disp, _ = resize_for_display(final, max_w=1200)

    summary_lines = [f"Hub: {names[hub]}", f"Images: {n}", ""]
    for step, (si, di) in enumerate(order):
        key    = (min(si, di), max(si, di))
        n_good = len(match_results.get(key, []))
        n_inl  = inlier_log.get((si, di), (n_good, 0))[1]
        summary_lines.append(
            f"Step {step+1}: {names[si]}\n"
            f"      -> {names[di]}\n"
            f"  matches={n_good}  inliers={n_inl}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#111")
    fig.suptitle(f"SIFT Panorama — {n} images chained via MST",
                 fontsize=13, fontweight="bold", color="white")

    axes[0].imshow(bgr2rgb(disp))
    axes[0].set_title("Final panorama  (gradient blend · black borders cropped)",
                      color="white", fontsize=10)
    axes[0].axis("off")

    axes[1].set_facecolor("#1a1a1a")
    axes[1].axis("off")
    axes[1].set_title("Chain summary", color="white", fontsize=10)
    axes[1].text(0.05, 0.97, "\n".join(summary_lines),
                 transform=axes[1].transAxes, color="white",
                 fontsize=8, va="top", family="monospace")

    plt.tight_layout()
    save_fig(os.path.join(out_dir, "05_stitch_annotated.png"), fig)

    return final


# ── Match matrix summary ───────────────────────────────────────────────────────

def print_summary(names, all_kps, match_results):
    n     = len(names)
    col_w = max(len(nm) for nm in names) + 2
    print("\n── Match Matrix (good matches) ──────────────────────────")
    print(f"{'':>{col_w}}" + "".join(f"{nm:>{col_w}}" for nm in names))
    for i in range(n):
        row = f"{names[i]:>{col_w}}"
        for j in range(n):
            if i == j:
                row += f"{'---':>{col_w}}"
            else:
                key = (min(i, j), max(i, j))
                row += f"{len(match_results.get(key, [])):>{col_w}}"
        print(row)
    print()
    for i, (nm, kps) in enumerate(zip(names, all_kps)):
        print(f"  [{i+1}] {nm}: {len(kps)} keypoints")
    print("─────────────────────────────────────────────────────────")


# ── Entry ──────────────────────────────────────────────────────────────────────

def main():
    in_dir  = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR

    if not os.path.isdir(in_dir):
        print(f"[ERROR] Input folder not found: '{in_dir}'")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Input  : {in_dir}/")
    print(f"[INFO] Output : {out_dir}/")

    print("\n[Load] Reading images ...")
    imgs, names = load_images(in_dir)
    if len(imgs) < 2:
        print("[ERROR] Need at least 2 images in input_images/")
        sys.exit(1)

    print(f"\n[SIFT] Detecting keypoints ({len(imgs)} images) ...")
    sift = cv2.SIFT_create(**SIFT_PARAMS)
    all_kps, all_desc = detect_all(sift, imgs)
    for nm, kps in zip(names, all_kps):
        print(f"  {nm}: {len(kps)} keypoints")

    print("\n[Match] Computing all pairwise matches ...")
    match_results = compute_all_matches(all_desc, names)

    stage_keypoints(imgs, names, all_kps, out_dir)
    stage_pairwise(imgs, names, all_kps, match_results, out_dir)
    stage_combined(imgs, names, all_kps, out_dir)
    stage_stitch_all(imgs, names, all_kps, all_desc, match_results, out_dir)

    print_summary(names, all_kps, match_results)
    print(f"\n[DONE] All outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
