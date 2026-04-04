#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def parse_prompt(prompt):
    """
    "xxx, pointillism" → content + style
    """
    if "," in prompt:
        parts = prompt.split(",")
        style = parts[-1].strip()
        content = ",".join(parts[:-1]).strip()
    else:
        style = "unknown"
        content = prompt.strip()
    return content, style


def list_ckpts(base_root, ckpt_root):
    base_root = Path(base_root)
    ckpt_root = Path(ckpt_root)

    ckpt_items = [("base", base_root)]

    step_dirs = []
    for p in ckpt_root.iterdir():
        if p.is_dir() and p.name.startswith("lora_step_"):
            step_dirs.append((p.name, p))

    step_dirs = sorted(step_dirs, key=lambda x: int(x[0].split("_")[-1]))
    ckpt_items.extend(step_dirs)

    return ckpt_items


def build_metadata(image_root, base_root, ckpt_root, prompt):
    content_prompt, style = parse_prompt(prompt)

    rows = []
    image_paths = list_images(image_root)

    for img_path in image_paths:
        ckpt_name = infer_checkpoint_from_image(img_path)

        if ckpt_name == "base":
            ckpt_path = str(base_root)
        else:
            ckpt_path = str(Path(ckpt_root) / ckpt_name)

        rows.append({
            "path": str(img_path),
            "prompt": content_prompt,
            "target_style": style,
            "checkpoint": ckpt_name,
            "ckpt_path": ckpt_path,
        })

    return pd.DataFrame(rows)


def infer_checkpoint_from_image(img_path):
    name = Path(img_path).stem.lower()

    if "base" in name:
        return "base"

    m = re.search(r"_s(\d+)", name)
    if m:
        step = int(m.group(1))
        return f"lora_step_{step}"

    raise ValueError(f"Cannot infer checkpoint from image name: {img_path}")

def build_lpips_pairs(df: pd.DataFrame):
    pairs = []

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["target_style"]].append(row)

    for style, items in grouped.items():
        def ckpt_key(x):
            c = str(x["checkpoint"]).lower()
            if c == "base":
                return -1
            if c.startswith("lora_step_"):
                return int(c.split("_")[-1])
            return 10**9

        items = sorted(items, key=ckpt_key)

        base_items = [x for x in items if str(x["checkpoint"]).lower() == "base"]
        other_items = [x for x in items if str(x["checkpoint"]).lower() != "base"]

        # base vs others
        for b in base_items:
            for o in other_items:
                pairs.append({
                    "path_a": b["path"],
                    "path_b": o["path"],
                    "group": f"{style}_base_vs_{o['checkpoint']}",
                    "target_style": style,
                })

        # non-base checkpoint vs checkpoint
        for i in range(len(other_items)):
            for j in range(i + 1, len(other_items)):
                a = other_items[i]
                b = other_items[j]
                pairs.append({
                    "path_a": a["path"],
                    "path_b": b["path"],
                    "group": f"{style}_{a['checkpoint']}_vs_{b['checkpoint']}",
                    "target_style": style,
                })

    return pd.DataFrame(pairs)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", type=str, required=True,
                        help="root folder of generated images")

    parser.add_argument("--ckpt_root", type=str, required=True,
                        help="root folder of generated ckpts")

    parser.add_argument("--base_root", type=str, required=True,
                        help="root folder of generated ckpts")

    parser.add_argument("--prompt", type=str, required=True,
                        help='e.g. "a quiet lakeside cottage at sunset, pointillism"')

    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building metadata...")
    df = build_metadata(args.image_root, args.base_root, args.ckpt_root, args.prompt)
    meta_path = os.path.join(args.out_dir, "generated_metadata.csv")
    df.to_csv(meta_path, index=False)
    print("Saved:", meta_path)

    pairs_df = build_lpips_pairs(df)
    pairs_path = os.path.join(args.out_dir, "lpips_pairs.csv")
    pairs_df.to_csv(pairs_path, index=False)

    print(f"Saved generated metadata to: {meta_path}")
    print(f"Saved LPIPS pairs to: {pairs_path}")
    print(f"Num images: {len(df)}")
    print(f"Num LPIPS pairs: {len(pairs_df)}")


if __name__ == "__main__":
    main()

