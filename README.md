# SIFT-application

This repository contains a Python-based application for **SIFT Multi-Image Matching & Full Panorama Stitching**. It automatically processes a set of images, finds matching overlapping features using the SIFT algorithm, calculates the homography, and seamlessly blends the images together into a single panorama.

## Features

- **Automatic SIFT Keypoint Detection**: Extracts SIFT features from all input images.
- **Robust Feature Matching**: Uses Lowe's Ratio Test to find reliable matches between all pairs of images.
- **Optimal Stitch Order**: Constructs a Minimum Spanning Tree (MST) based on match counts to determine the best sequence for aligning and chaining images.
- **Homography Alignment**: Uses RANSAC to align images reliably despite outliers.
- **Gradient Blending**: Seamlessly blends overlapping areas using distance transform weights to avoid visible seams.
- **Comprehensive Visual Output**: Generates annotated visualizations of internal steps, such as detected keypoints, pairwise match lines, and the logical stitch order graph.
- **Black Border Cropping**: Automatically detects and crops the black unused borders from the final stitched panorama.

## Requirements

Ensure you have Python 3 installed. You will need the following libraries:

```bash
pip install opencv-python numpy matplotlib
```

*(Note: The script uses `cv2.SIFT_create()`, so an OpenCV version that includes SIFT natively — like `opencv-python` >= 4.4.0 — is required)*

## Usage

### 1. Panorama Stitching
Place the images you want to stitch into the input directory (by default `my_photos/`), and run:

```bash
python sift_matching.py
```

You can also specify custom input and output directories as arguments:

```bash
python sift_matching.py path/to/input_folder path/to/output_folder
```

#### Output
The script generates the following artifacts in the output directory (default: `my_results/`):

- `01_keypoints_<name>.png` - Individual images with their SIFT keypoints annotated.
- `02_pairwise_<A>_vs_<B>.png` - Pairwise visual match lines between every pair of overlapping images.
- `03_combined_keypoints.png` - A grid summarizing all extracted keypoints.
- `04_stitch_order.png` - A node graph showing the MST calculated for the stitching sequence.
- `05_stitch_clean.png` - The final stitched and blended panorama.
- `05_stitch_annotated.png` - The final panorama alongside summary text of the matching and inlier counts.

### 2. Testing with Image Splitting
If you want to test the panorama stitching on an existing wide image, use `split_image.py`. It takes a wide image, splits it into 3 overlapping vertical sections, and saves them into the `pano_photos/` directory.

```bash
python split_image.py
```
*(You may need to edit the `img_path` variable inside `split_image.py` to point to a valid image)*

Once generated, you can stitch these overlapping parts back together by running:
```bash
python sift_matching.py pano_photos pano_results
```
