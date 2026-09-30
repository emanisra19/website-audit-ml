import cv2
import numpy as np
import pandas as pd
from pathlib import Path

SCREENSHOT_DIR = Path("data/visual_screenshots")
OUTPUT_FILE = Path("data/advanced_screenshot_features.csv")


def extract_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    height, width = gray.shape
    total_pixels = height * width

    # =========================================================
    # BASIC VISUAL FEATURES
    # =========================================================

    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    contrast = float(np.std(gray))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / total_pixels)

    white_pixels = np.sum(gray > 245)
    whitespace_ratio = float(white_pixels / total_pixels)

    dark_pixels = np.sum(gray < 50)
    dark_ratio = float(dark_pixels / total_pixels)

    saturation_mean = float(np.mean(hsv[:, :, 1]))
    saturation_std = float(np.std(hsv[:, :, 1]))

    # =========================================================
    # COLOR DIVERSITY
    # =========================================================

    small = cv2.resize(image, (100, 100))
    pixels = small.reshape(-1, 3).astype(np.float32)

    quantized = (pixels // 32).astype(np.int32)
    unique_colors = len(np.unique(quantized, axis=0))

    color_diversity = float(unique_colors)

    # =========================================================
    # IMAGE DETAIL / COMPLEXITY
    # =========================================================

    laplacian_variance = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    horizontal_variation = float(
        np.mean(
            np.abs(
                np.diff(gray.astype(np.float32), axis=1)
            )
        )
    )

    vertical_variation = float(
        np.mean(
            np.abs(
                np.diff(gray.astype(np.float32), axis=0)
            )
        )
    )

    # =========================================================
    # NEW: REGIONAL VISUAL FEATURES
    # =========================================================

    # Divide screenshot into horizontal bands.
    # This helps capture how visual density changes
    # from the top of the page to the bottom.

    bands = 6
    band_height = height // bands

    band_brightness = []
    band_edge_density = []
    band_whitespace = []
    band_dark_ratio = []

    for i in range(bands):

        start = i * band_height

        if i == bands - 1:
            end = height
        else:
            end = (i + 1) * band_height

        band_gray = gray[start:end]
        band_edges = edges[start:end]

        band_pixels = band_gray.size

        band_brightness.append(
            float(np.mean(band_gray))
        )

        band_edge_density.append(
            float(np.count_nonzero(band_edges) / band_pixels)
        )

        band_whitespace.append(
            float(np.sum(band_gray > 245) / band_pixels)
        )

        band_dark_ratio.append(
            float(np.sum(band_gray < 50) / band_pixels)
        )

    # =========================================================
    # REGIONAL VARIATION
    # =========================================================

    brightness_variation = float(
        np.std(band_brightness)
    )

    edge_variation = float(
        np.std(band_edge_density)
    )

    whitespace_variation = float(
        np.std(band_whitespace)
    )

    dark_ratio_variation = float(
        np.std(band_dark_ratio)
    )

    # =========================================================
    # TOP / MIDDLE / BOTTOM FEATURES
    # =========================================================

    top_end = max(1, int(height * 0.25))
    middle_start = int(height * 0.25)
    middle_end = int(height * 0.75)
    bottom_start = int(height * 0.75)

    top = gray[:top_end]
    middle = gray[middle_start:middle_end]
    bottom = gray[bottom_start:]

    top_brightness = float(np.mean(top))
    middle_brightness = float(np.mean(middle))
    bottom_brightness = float(np.mean(bottom))

    top_edges = cv2.Canny(top, 100, 200)
    middle_edges = cv2.Canny(middle, 100, 200)
    bottom_edges = cv2.Canny(bottom, 100, 200)

    top_edge_density = float(
        np.count_nonzero(top_edges) / top.size
    )

    middle_edge_density = float(
        np.count_nonzero(middle_edges) / middle.size
    )

    bottom_edge_density = float(
        np.count_nonzero(bottom_edges) / bottom.size
    )

    # =========================================================
    # TEXT / CONTENT DENSITY PROXY
    # =========================================================

    # Dark pixels can approximate text and dense content
    # against lighter backgrounds.

    content_density = float(
        np.sum(gray < 180) / total_pixels
    )

    # =========================================================
    # VISUAL BALANCE
    # =========================================================

    left = gray[:, :width // 2]
    right = gray[:, width // 2:]

    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))

    brightness_balance = float(
        abs(left_mean - right_mean)
    )

    left_edges = cv2.Canny(left, 100, 200)
    right_edges = cv2.Canny(right, 100, 200)

    left_edge_density = float(
        np.count_nonzero(left_edges) / left.size
    )

    right_edge_density = float(
        np.count_nonzero(right_edges) / right.size
    )

    edge_balance = float(
        abs(left_edge_density - right_edge_density)
    )

    # =========================================================
    # SCREENSHOT DIMENSIONS
    # =========================================================

    aspect_ratio = float(width / height)

    # =========================================================
    # RETURN FEATURES
    # =========================================================

    return {

        # Basic
        "image_width": width,
        "image_height": height,
        "aspect_ratio": aspect_ratio,

        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "contrast": contrast,

        "edge_density": edge_density,

        "whitespace_ratio": whitespace_ratio,
        "dark_ratio": dark_ratio,

        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,

        "color_diversity": color_diversity,

        "laplacian_variance": laplacian_variance,

        "horizontal_variation": horizontal_variation,
        "vertical_variation": vertical_variation,

        # Regional variation
        "brightness_variation": brightness_variation,
        "edge_variation": edge_variation,
        "whitespace_variation": whitespace_variation,
        "dark_ratio_variation": dark_ratio_variation,

        # Page regions
        "top_brightness": top_brightness,
        "middle_brightness": middle_brightness,
        "bottom_brightness": bottom_brightness,

        "top_edge_density": top_edge_density,
        "middle_edge_density": middle_edge_density,
        "bottom_edge_density": bottom_edge_density,

        # Content
        "content_density": content_density,

        # Visual balance
        "brightness_balance": brightness_balance,
        "edge_balance": edge_balance,

        # Side-specific
        "left_mean": left_mean,
        "right_mean": right_mean,

        "left_edge_density": left_edge_density,
        "right_edge_density": right_edge_density,
    }


def main():

    print("Starting advanced screenshot feature extraction...")
    print(f"Screenshot directory: {SCREENSHOT_DIR}")

    screenshots = sorted(
        SCREENSHOT_DIR.glob("*.png")
    )

    if not screenshots:
        print("No screenshots found.")
        return

    print(f"Found {len(screenshots)} screenshots.")

    rows = []

    for i, image_path in enumerate(
        screenshots,
        start=1
    ):

        print(
            f"[{i}/{len(screenshots)}] "
            f"Processing {image_path.name}"
        )

        try:

            features = extract_features(
                image_path
            )

            filename = image_path.stem

            if "_desktop" in filename:

                website_id = filename.replace(
                    "_desktop",
                    ""
                )

                device = "desktop"

            elif "_mobile" in filename:

                website_id = filename.replace(
                    "_mobile",
                    ""
                )

                device = "mobile"

            else:

                website_id = filename
                device = "unknown"

            features["website_id"] = website_id
            features["device"] = device
            features["filename"] = image_path.name

            rows.append(features)

        except Exception as e:

            print(
                f"ERROR processing "
                f"{image_path.name}: {e}"
            )

    df = pd.DataFrame(rows)

    id_columns = [
        "website_id",
        "device",
        "filename"
    ]

    other_columns = [
        col
        for col in df.columns
        if col not in id_columns
    ]

    df = df[
        id_columns + other_columns
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("========================================")
    print("Advanced screenshot extraction complete")
    print("========================================")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()