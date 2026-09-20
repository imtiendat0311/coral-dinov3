#!/usr/bin/env python3
"""
Mosaic Segmentation & Classification Pipeline
Tiles large underwater orthomosaics, segments colonies via SAM,
classifies species with ConvNextVIT, and exports TagLab annotations.

Usage:
    python mosaic_pipeline.py --image mosaic.png --output ./results --checkpoint model-9.pth
"""

import os
import json
import argparse
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
try:
    import json5
except ImportError:
    import json as json5

from coral_clf.model import build_coral_classifier
from coral_clf.transforms import make_transform
from coral_clf.helper import (
    load_class_labels,
    calculate_centroid,
    calculate_area,
    generate_blob_name,
    find_contours_with_holes,
    contour_to_taglab_string,
    calculate_perimeter_with_holes,
    custom_colors,
)

Image.MAX_IMAGE_PIXELS = None

def compute_taglab_meta(mask_np_tile, ox, oy, img_w, img_h):
    """
    Computes TagLab differential contours and spatial metadata shifted to global coordinates.
    """
    outer_contour, inner_contours = find_contours_with_holes(mask_np_tile)
    if outer_contour is None:
        return None, [], None, None, None

    offset = np.array([[ox, oy]], dtype=outer_contour.dtype)
    outer_global = outer_contour + offset
    inner_global = [c + offset for c in inner_contours]

    contour_str = contour_to_taglab_string(outer_global)
    inner_c = [contour_to_taglab_string(c) for c in inner_global]

    m = measure.moments(mask_np_tile.astype(np.uint8))
    if m[0, 0] != 0:
        cx = m[0, 1] / m[0, 0] + ox
        cy = m[1, 0] / m[0, 0] + oy
        centroid = [cx, cy]
    else:
        centroid = [float(ox), float(oy)]

    area = float(mask_np_tile.sum())
    perimeter = calculate_perimeter_with_holes(outer_global, inner_global)
    return contour_str, inner_c, centroid, area, perimeter

def iter_crops(img_path, crop_w, crop_h, img_w, img_h, overlap, out_dir):
    """
    Generator that yields one tile at a time to minimize memory footprint.
    """
    tile_dir = os.path.join(out_dir, "crop_tiles")
    os.makedirs(tile_dir, exist_ok=True)
    crop_id = 0
    src = Image.open(img_path).convert("RGB")
    for y in range(0, img_h, max(1, crop_h - overlap)):
        for x in range(0, img_w, max(1, crop_w - overlap)):
            left = x
            upper = y
            right = min(x + crop_w, img_w)
            lower = min(y + crop_h, img_h)
            tile = src.crop((left, upper, right, lower))

            tile_path = os.path.join(tile_dir, f"crop_{crop_id}_x{left}_y{upper}.png")
            tile.save(tile_path)
            meta = {
                "img_path": tile_path,
                "x_offset": left,
                "y_offset": upper,
                "width": right - left,
                "height": lower - upper,
            }
            with open(os.path.join(tile_dir, f"crop_{crop_id}_x{left}_y{upper}.json"), "w") as mf:
                json.dump(meta, mf)
            crop_id += 1
            yield {"img": tile, "metadata": meta}
            del tile
    del src

def parse_args():
    parser = ArgumentParser(description="Large Orthomosaic Coral Segmentation & Classification Pipeline")
    parser.add_argument("--image", "-i", type=str, help="Path to input orthomosaic image")
    parser.add_argument("--output", "-o", type=str, default="./mosaic_output", help="Output directory")
    parser.add_argument("--config", "-c", type=str, default=None, help="Optional path to json config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to ConvNextVIT checkpoint (.pth/.safetensors)")
    parser.add_argument("--classes", type=str, default="classes.json", help="Path to classes.json or test.csv")
    parser.add_argument("--prompt", type=str, default="coral", help="Text prompt for SAM segmentation")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[3256, 3256], help="Crop width and height")
    parser.add_argument("--crop-overlap", type=int, default=0, help="Overlap between crops in pixels")
    parser.add_argument("--blackout", action="store_true", help="Black out background outside coral mask before classifying")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mock", action="store_true", help="Run with mock backbones")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config file if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json5.load(f)

    img_path = args.image or cfg.get("input_image_path")
    out_dir = args.output or cfg.get("output_directory", "./mosaic_output")
    device = torch.device(args.device or cfg.get("device", "cpu"))
    crop_w, crop_h = args.crop_size or cfg.get("crop_size", [3256, 3256])
    overlap = args.crop_overlap if args.crop_overlap is not None else cfg.get("crop_overlap", 0)
    prompt = args.prompt or cfg.get("prompt", "coral")
    ckpt_path = args.checkpoint or cfg.get("weight_path")
    is_blackout = args.blackout or cfg.get("is_black_out", False)

    if not img_path:
        raise ValueError("Please provide an input image via --image or within --config.")

    os.makedirs(out_dir, exist_ok=True)
    labels_file = args.classes if os.path.exists(args.classes) else cfg.get("test_path")
    class_labels = load_class_labels(classes_json_path=labels_file)

    # 1. Initialize Segmentation Model (SAM 3)
    print("[INFO] Initializing segmentation model (SAM 3)...")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        seg_model = build_sam3_image_model(device=device)
        processor = Sam3Processor(seg_model, device=device)
    except ImportError:
        print("[WARN] SAM3 package not installed in current environment. Tiling will proceed without segmentation.")
        processor = None

    # 2. Initialize Classifier (ConvNextVIT)
    print("[INFO] Initializing ConvNextVIT classifier...")
    cls_model, _ = build_coral_classifier(
        checkpoint_path=ckpt_path,
        device=device,
        use_mock_backbones=args.mock,
    )
    transform = make_transform(resize_size=256, is_train=False)

    # 3. Process Mosaic Tiles
    img = Image.open(img_path)
    img_w, img_h = img.size
    print(f"[INFO] Mosaic image resolution: {img_w} x {img_h}")
    del img

    anns = {"regions": [], "points": []}
    tile_mask_base = os.path.join(out_dir, "tile_masks")
    os.makedirs(tile_mask_base, exist_ok=True)

    for i, crop in enumerate(iter_crops(img_path, crop_w, crop_h, img_w, img_h, overlap, out_dir)):
        print(f"\n=== Processing tile {i} ===")
        tile_img = crop["img"]
        ox = crop["metadata"]["x_offset"]
        oy = crop["metadata"]["y_offset"]
        tile_w = crop["metadata"]["width"]
        tile_h = crop["metadata"]["height"]

        if processor is None:
            continue

        state = processor.set_image(tile_img)
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        print(f"  Found {len(masks)} candidate regions in tile {i}")

        np_tile_u8 = np.array(tile_img, dtype=np.uint8)
        tile_out_dir = os.path.join(tile_mask_base, f"tile_{i}")
        os.makedirs(tile_out_dir, exist_ok=True)

        for j, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(tile_w, int(box[2])), min(tile_h, int(box[3]))
            crop_u8 = np_tile_u8[y1:y2, x1:x2, :]
            if crop_u8.shape[0] == 0 or crop_u8.shape[1] == 0:
                continue

            mask_np_full = mask[0].cpu().numpy().astype(bool)

            # Black out background if requested
            if is_blackout:
                mask_crop_bool = mask_np_full[y1:y2, x1:x2]
                blacked = np.zeros_like(crop_u8)
                blacked[mask_crop_bool] = crop_u8[mask_crop_bool]
                cls_input = transform(blacked).unsqueeze(0).to(device)
            else:
                cls_input = transform(crop_u8).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = cls_model(cls_input, cls_input)
                pred_idx = torch.argmax(logits, dim=1).item()
                species_name = class_labels.get(pred_idx, f"Class_{pred_idx}")

            # TagLab differential contour calculation
            contour_str, inner_c, centroid, area, perimeter = compute_taglab_meta(
                mask_np_full, ox, oy, img_w, img_h
            )

            # Save TagLab region annotation
            if contour_str:
                anns["regions"].append({
                    "id": f"tile_{i}_mask_{j}",
                    "name": generate_blob_name(j, centroid),
                    "label": species_name,
                    "score": float(score.cpu().item()),
                    "contour": contour_str,
                    "inner_contours": inner_c,
                    "centroid": centroid,
                    "area": area,
                    "perimeter": perimeter,
                })

    # Save TagLab JSON
    taglab_out_file = os.path.join(out_dir, "taglab_annotations.json")
    with open(taglab_out_file, "w") as f:
        json.dump(anns, f, indent=2)
    print(f"\n[INFO] Complete! TagLab annotations saved to: {taglab_out_file}")

if __name__ == "__main__":
    main()
