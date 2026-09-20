import json
import math
import numpy as np
import torch.nn as nn

# Activation mapping used to build the classification head
activation = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
}

def load_class_labels(classes_json_path=None, test_csv_path=None):
    """
    Load coral class mapping.
    Tries test_csv_path if provided, then classes_json_path, with fallback to default 28 classes.
    """
    if test_csv_path:
        import pandas as pd
        df = pd.read_csv(test_csv_path)
        id2label = dict(zip(df["e_label"].astype(int), df["label"].astype(str)))
        if 27 not in id2label:
            id2label[27] = "Unknown/Out-of-distribution"
        return id2label

    if classes_json_path:
        with open(classes_json_path, "r") as f:
            data = json.load(f)
            return {int(k): str(v) for k, v in data.items()}

    # Default fallback
    return {i: f"Coral_Class_{i}" for i in range(27)} | {27: "Unknown/Out-of-distribution"}

# TagLab differential encoding string
def contour_to_taglab_string(contour):
    """
    Convert a contour (N, 2) array to TagLab's differential encoding string.
    TagLab format: scale by 10, compute differences, flatten, join with spaces.
    """
    c = contour.astype(float)
    d = (c * 10).astype(int)
    d = np.diff(d, axis=0, prepend=[[0, 0]])
    d = d.reshape(-1)
    return " ".join(map(str, d))

def calculate_centroid(mask):
    """Calculate centroid using skimage moments (TagLab standard)."""
    from skimage import measure
    m = measure.moments(mask)
    if m[0, 0] == 0:
        return [0.0, 0.0]
    cx = m[0, 1] / m[0, 0]
    cy = m[1, 0] / m[0, 0]
    return [cx, cy]

def calculate_area(mask):
    """Calculate area as total positive pixel count."""
    return float(mask.sum())

def calculate_perimeter(contour):
    """Calculate perimeter as sum of Euclidean distances between contour points."""
    N = contour.shape[0]
    if N < 2:
        return 0.0
    px1, py1 = contour[0]
    pxlast, pylast = contour[N - 1]
    perim = math.sqrt((px1 - pxlast) ** 2 + (py1 - pylast) ** 2)
    for i in range(1, N):
        px2, py2 = contour[i]
        perim += math.sqrt((px1 - px2) ** 2 + (py1 - py2) ** 2)
        px1, py1 = px2, py2
    return perim

def generate_blob_name(blob_id, centroid):
    """Generate blob name in TagLab format: c-<id>-<cx>x-<cy>y"""
    return f"c-{blob_id}-{centroid[0]:.1f}x-{centroid[1]:.1f}y"

def find_contours_with_holes(mask):
    """
    Find outer contour and inner contours (holes) from a binary mask using RETR_TREE.
    """
    import cv2
    mask_uint8 = (np.array(mask) * 255).astype(np.uint8)
    contours_cv, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_cv) == 0:
        return None, []

    outer_contour = None
    inner_contours = []

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        max_area = 0
        outer_idx = 0
        for i, h in enumerate(hierarchy):
            if h[3] == -1:  # Outer boundary
                area = cv2.contourArea(contours_cv[i])
                if area > max_area:
                    max_area = area
                    outer_idx = i

        outer_contour = contours_cv[outer_idx].reshape(-1, 2)
        for i, h in enumerate(hierarchy):
            if h[3] == outer_idx:
                inner = contours_cv[i].reshape(-1, 2)
                if len(inner) > 3:
                    inner_contours.append(inner)
    else:
        outer_contour = max(contours_cv, key=cv2.contourArea).reshape(-1, 2)

    return outer_contour, inner_contours

def calculate_perimeter_with_holes(outer_contour, inner_contours):
    """Calculate total perimeter including both outer boundary and inner holes."""
    if outer_contour is None:
        return 0.0
    total = calculate_perimeter(outer_contour)
    for inner in inner_contours:
        total += calculate_perimeter(inner)
    return total

custom_colors = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255],
    [0, 255, 128], [255, 0, 128], [128, 255, 0], [0, 128, 255],
    [255, 128, 128], [128, 255, 128], [128, 128, 255], [255, 255, 128],
    [255, 128, 255], [128, 255, 255], [192, 64, 0], [64, 192, 0],
    [0, 192, 64], [192, 0, 64], [64, 0, 192], [0, 64, 192],
    [192, 192, 64], [192, 64, 192], [64, 192, 192], [255, 165, 0],
]
