#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Default prompt settings
# ============================================================

DEFAULT_BASE_PROMPTS = {
    "TMP1": "a small house by a lake at sunset",
    "TMP2": "a decorative poster of a flower shop",
    "TMP3": "a woman in Renaissance dress standing near a garden",
}

STYLE_DISPLAY_NAMES = {
    "baroque": "Baroque",
    "cubism": "Cubism",
    "earlyrenaissance": "Early Renaissance",
    "expressionism": "Expressionism",
    "pointillism": "Pointillism",
    "popart": "Pop Art",
}

P2_PROMPT_SUFFIXES = {
    "TMP1": {
        "cubism": "in Cubism style, faceted house, angular forms, fragmented lake reflection",
        "expressionism": "in Expressionism style, expressive brushwork, emotional color contrast, animated lakeside forms",
        "earlyrenaissance": "in Early Renaissance style, balanced composition, simple house forms, gentle perspective",
        "baroque": "in Baroque style, small lakeside house, dramatic lighting, strong chiaroscuro",
        "pointillism": "in Pointillism style, dotted brushwork, broken color, shimmering lake reflection",
        "popart": "in Pop Art style, bold flat colors, clean outlines, simplified lake reflection",
    },

    "TMP2": {
        "cubism": "in Cubism style, faceted storefront, fragmented shop window, geometric floral forms",
        "expressionism": "in Expressionism style, expressive brushstrokes, emotionally charged colors, bold floral shopfront",
        "earlyrenaissance": "in Early Renaissance style, balanced floral display, clear linear perspective, muted earthy tones",
        "baroque": "in Baroque style, ornate shopfront, dramatic lighting, floral window display",
        "pointillism": "in Pointillism style, dotted brushwork, optical color mixture, luminous floral storefront",
        "popart": "in Pop Art style, bold flat colors, clean flower outlines, graphic floral storefront",
    },

    "TMP3": {
        "cubism": "in Cubism style, faceted figure, angular dress folds, fragmented garden backdrop",
        "expressionism": "in Expressionism style, expressive brushstrokes, intense color contrast, emotional garden atmosphere",
        "earlyrenaissance": "in Early Renaissance style, balanced figure, orderly composition, calm natural tones, classical garden setting",
        "baroque": "in Baroque style, dramatic side lighting, rich fabric folds, shadowed garden backdrop",
        "pointillism": "in Pointillism style, dotted brushwork, broken garden colors, luminous dress texture",
        "popart": "in Pop Art style, bold flat colors, clean figure outlines, graphic floral backdrop",
    },
}

PROMPT_TYPE_MAP = {
    "P0": "P0_semantic_only",
    "P1": "P1_style_name",
    "P2": "P2_descriptor_enhanced",
}


def canonicalize_style(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_images(folder: str):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# CLIP evaluator
# ============================================================

class ClipEvaluator:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode_images(self, image_paths, batch_size=16):
        feats = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP image encode"):
            batch_paths = image_paths[i:i + batch_size]
            images = [load_image(p) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
            feats.append(image_features.cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32):
        feats = []
        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text encode"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
            feats.append(text_features.cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def clipscore(self, image_paths, prompts, batch_size=16):
        assert len(image_paths) == len(prompts)

        scores = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIPScore"):
            batch_paths = image_paths[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            images = [load_image(p) for p in batch_paths]

            inputs = self.processor(
                text=batch_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            image_features = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            text_inputs = {
                k: v for k, v in inputs.items()
                if k in ["input_ids", "attention_mask"]
            }
            text_features = self.model.get_text_features(**text_inputs)

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            batch_scores = (image_features * text_features).sum(dim=-1)
            scores.extend(batch_scores.detach().cpu().tolist())

        return scores


# ============================================================
# Metadata construction
# ============================================================

def normalize_base_prompt_id(folder_name: str) -> str:
    """
    Convert folder names such as:
        prompt_engineering_outputs_TMP1 -> TMP1
        TMP1 -> TMP1
    """
    m = re.search(r"(TMP\d+)", folder_name)
    if m:
        return m.group(1)
    return folder_name


def discover_prompt_output_roots(input_roots):
    """
    Support two modes:

    1) User directly passes
    2) User passes parent folder
    """
    discovered = []

    for root in input_roots:
        root = os.path.abspath(root)
        root_path = Path(root)

        if not root_path.exists():
            raise FileNotFoundError(root)

        # Case 1: root itself is a prompt output folder
        if re.search(r"prompt_engineering_outputs_TMP\d+", root_path.name) or re.fullmatch(r"TMP\d+", root_path.name):
            discovered.append(str(root_path))
            continue

        # Case 2: root is parent folder
        children = sorted([
            p for p in root_path.iterdir()
            if p.is_dir() and re.search(r"prompt_engineering_outputs_TMP\d+", p.name)
        ])

        if children:
            discovered.extend([str(p) for p in children])
        else:
            # fallback: use root directly
            discovered.append(str(root_path))

    discovered = sorted(list(dict.fromkeys(discovered)))
    print("[Info] discovered prompt output roots:")
    for d in discovered:
        print("  ", d)

    return discovered


def infer_base_prompt_id_and_style(path: str, prompt_output_root: str):
    """
    Expected structure:
        prompt_output_root/
            Baroque/
                P0_semantic_only_seed42.png
                P1_style_name_seed42.png
                P2_descriptor_enhanced_seed42.png
    """
    rel = Path(path).relative_to(Path(prompt_output_root))
    parts = rel.parts

    if len(parts) < 2:
        raise ValueError(f"Cannot infer style from path: {path}")

    base_prompt_id = normalize_base_prompt_id(Path(prompt_output_root).name)
    style = parts[0]

    return base_prompt_id, style


def parse_prompt_type(filename: str):
    m = re.match(r"^(P[012])_", filename)
    if not m:
        raise ValueError(f"Cannot parse prompt type from filename: {filename}")
    return PROMPT_TYPE_MAP[m.group(1)]


def parse_seed(filename: str):
    m = re.search(r"seed(\d+)", filename)
    if m:
        return int(m.group(1))
    return -1


def build_full_prompt(base_prompt_id: str, base_prompt: str, style_raw: str, prompt_type: str):
    style_key = canonicalize_style(style_raw)
    style_name = STYLE_DISPLAY_NAMES.get(style_key, style_raw.replace("_", " "))

    if prompt_type == "P0_semantic_only":
        return base_prompt

    if prompt_type == "P1_style_name":
        return f"{base_prompt}, in {style_name} style"

    if prompt_type == "P2_descriptor_enhanced":
        suffix = P2_PROMPT_SUFFIXES.get(base_prompt_id, {}).get(style_key)

        if suffix is None:
            raise ValueError(
                f"No TMP-specific P2 suffix found for "
                f"base_prompt_id={base_prompt_id}, style={style_raw} -> {style_key}"
            )

        return f"{base_prompt}, {suffix}"

    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def build_generated_metadata(input_roots, base_prompt_map=None):
    rows = []

    if base_prompt_map is None:
        base_prompt_map = DEFAULT_BASE_PROMPTS.copy()
    else:
        tmp = DEFAULT_BASE_PROMPTS.copy()
        tmp.update(base_prompt_map)
        base_prompt_map = tmp

    prompt_output_roots = discover_prompt_output_roots(input_roots)

    for prompt_output_root in prompt_output_roots:
        image_paths = list_images(prompt_output_root)

        for path in image_paths:
            filename = Path(path).name

            # Only evaluate P0/P1/P2 prompt engineering outputs
            if not re.match(r"^P[012]_", filename):
                continue

            prompt_type = parse_prompt_type(filename)
            seed = parse_seed(filename)

            base_prompt_id, style_raw = infer_base_prompt_id_and_style(
                path=path,
                prompt_output_root=prompt_output_root,
            )

            if base_prompt_id not in base_prompt_map:
                raise ValueError(
                    f"No base prompt found for base_prompt_id={base_prompt_id}. "
                    f"Available keys: {list(base_prompt_map.keys())}. "
                    f"Please provide it through --base_prompt_map_json."
                )

            base_prompt = base_prompt_map[base_prompt_id]
            full_prompt = build_full_prompt(base_prompt_id, base_prompt, style_raw, prompt_type)

            rows.append({
                "path": path,
                "base_prompt_id": base_prompt_id,
                "target_style": style_raw,
                "target_style_key": canonicalize_style(style_raw),
                "prompt_type": prompt_type,
                "seed": seed,
                "base_prompt": base_prompt,
                "prompt": full_prompt,
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No generated P0/P1/P2 images found.")

    df = df.sort_values(
        ["base_prompt_id", "target_style_key", "prompt_type", "seed", "path"]
    ).reset_index(drop=True)

    print("[Info] metadata summary:")
    print(df.groupby(["base_prompt_id", "target_style", "prompt_type"]).size())

    return df


# ============================================================
# Style reference loading
# ============================================================

def load_style_reference_paths(style_ref_root: str):
    """
    Expected structure:
        style_ref_root/
            Baroque/
            Cubism/
            Early_Renaissance/
            Expressionism/
            Pointillism/
            Pop_Art/
    """
    style_ref_root = os.path.abspath(style_ref_root)

    subdirs = [
        d for d in os.listdir(style_ref_root)
        if os.path.isdir(os.path.join(style_ref_root, d))
    ]

    if len(subdirs) == 0:
        raise ValueError(
            "For this prompt-engineering experiment, style_ref_root should contain "
            "multiple style subfolders."
        )

    style_to_paths = {}
    for d in sorted(subdirs):
        subdir = os.path.join(style_ref_root, d)
        paths = list_images(subdir)
        if len(paths) > 0:
            style_to_paths[canonicalize_style(d)] = paths

    if not style_to_paths:
        raise ValueError(f"No style reference images found under: {style_ref_root}")

    return style_to_paths


def compute_style_centroids(clip_eval: ClipEvaluator, style_to_paths: dict, batch_size=16):
    style_centroids = {}

    for style_key, paths in style_to_paths.items():
        print(f"[Style Ref] {style_key}: {len(paths)} images")
        feats = clip_eval.encode_images(paths, batch_size=batch_size)
        centroid = feats.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, dim=-1).squeeze(0)
        style_centroids[style_key] = centroid

    style_keys = sorted(style_centroids.keys())
    centroid_mat = torch.stack([style_centroids[k] for k in style_keys], dim=0)

    return style_keys, centroid_mat


# ============================================================
# Main metrics
# ============================================================

def compute_clip_and_style_metrics(
    gen_df: pd.DataFrame,
    clip_eval: ClipEvaluator,
    style_keys,
    centroid_mat,
    out_dir: str,
    batch_size: int = 16,
):
    image_paths = gen_df["path"].tolist()

    print("\n[Metric] CLIPScore-full")
    clip_full = clip_eval.clipscore(
        image_paths,
        gen_df["prompt"].tolist(),
        batch_size=batch_size,
    )

    print("\n[Metric] CLIPScore-semantic")
    clip_semantic = clip_eval.clipscore(
        image_paths,
        gen_df["base_prompt"].tolist(),
        batch_size=batch_size,
    )

    print("\n[Metric] CLIP image features for style similarity")
    gen_feats = clip_eval.encode_images(image_paths, batch_size=batch_size)
    sims = gen_feats @ centroid_mat.T
    sims_np = sims.numpy()

    rows = []
    for i, row in gen_df.iterrows():
        target_key = row["target_style_key"]

        if target_key not in style_keys:
            raise ValueError(
                f"Target style {row['target_style']} -> {target_key} "
                f"not found in style reference folders. Available: {style_keys}"
            )

        target_idx = style_keys.index(target_key)
        target_sim = float(sims_np[i, target_idx])

        non_target = [
            float(sims_np[i, j])
            for j in range(len(style_keys))
            if j != target_idx
        ]
        non_target_mean = float(np.mean(non_target)) if non_target else np.nan
        style_margin = target_sim - non_target_mean if non_target else np.nan

        pred_idx = int(np.argmax(sims_np[i]))
        retrieval_pred_key = style_keys[pred_idx]

        out = row.to_dict()
        out["clipscore_full"] = float(clip_full[i])
        out["clipscore_semantic"] = float(clip_semantic[i])
        out["target_style_similarity"] = target_sim
        out["non_target_style_similarity_mean"] = non_target_mean
        out["style_margin"] = float(style_margin)
        out["retrieval_pred_key"] = retrieval_pred_key
        out["retrieval_correct"] = int(retrieval_pred_key == target_key)

        for j, sk in enumerate(style_keys):
            out[f"sim_to_{sk}"] = float(sims_np[i, j])

        rows.append(out)

    detail_df = pd.DataFrame(rows)
    detail_path = os.path.join(out_dir, "prompt_engineering_details.csv")
    detail_df.to_csv(detail_path, index=False)

    # Summary 1: prompt type overall
    summary_prompt = (
        detail_df
        .groupby("prompt_type")
        .agg(
            n=("path", "count"),
            clipscore_full_mean=("clipscore_full", "mean"),
            clipscore_full_std=("clipscore_full", "std"),
            clipscore_semantic_mean=("clipscore_semantic", "mean"),
            clipscore_semantic_std=("clipscore_semantic", "std"),
            style_similarity_mean=("target_style_similarity", "mean"),
            style_similarity_std=("target_style_similarity", "std"),
            style_margin_mean=("style_margin", "mean"),
            style_margin_std=("style_margin", "std"),
            retrieval_acc=("retrieval_correct", "mean"),
        )
        .reset_index()
    )
    summary_prompt.to_csv(
        os.path.join(out_dir, "summary_by_prompt_type.csv"),
        index=False,
    )

    # Summary 2: base prompt x prompt type
    summary_base_prompt = (
        detail_df
        .groupby(["base_prompt_id", "prompt_type"])
        .agg(
            n=("path", "count"),
            clipscore_full_mean=("clipscore_full", "mean"),
            clipscore_semantic_mean=("clipscore_semantic", "mean"),
            style_similarity_mean=("target_style_similarity", "mean"),
            style_margin_mean=("style_margin", "mean"),
            retrieval_acc=("retrieval_correct", "mean"),
        )
        .reset_index()
    )
    summary_base_prompt.to_csv(
        os.path.join(out_dir, "summary_by_base_prompt_and_prompt_type.csv"),
        index=False,
    )

    # Summary 3: style x prompt type
    summary_style_prompt = (
        detail_df
        .groupby(["target_style", "prompt_type"])
        .agg(
            n=("path", "count"),
            clipscore_full_mean=("clipscore_full", "mean"),
            clipscore_semantic_mean=("clipscore_semantic", "mean"),
            style_similarity_mean=("target_style_similarity", "mean"),
            style_margin_mean=("style_margin", "mean"),
            retrieval_acc=("retrieval_correct", "mean"),
        )
        .reset_index()
    )
    summary_style_prompt.to_csv(
        os.path.join(out_dir, "summary_by_style_and_prompt_type.csv"),
        index=False,
    )

    print(f"[OK] saved details to: {detail_path}")
    return detail_df


# ============================================================
# LPIPS
# ============================================================

def image_to_lpips_tensor(path: str, device: str):
    img = load_image(path)
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    x = tfm(img).unsqueeze(0).to(device)
    x = x * 2.0 - 1.0
    return x


def compute_lpips_prompt_change(detail_df: pd.DataFrame, out_dir: str, device: str):
    try:
        import lpips
    except ImportError:
        print("[WARN] lpips package not installed. Skip LPIPS.")
        return None

    print("\n[Metric] LPIPS prompt-induced change")

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    pair_rows = []

    group_cols = ["base_prompt_id", "target_style", "seed"]

    for keys, sub in detail_df.groupby(group_cols):
        base_prompt_id, style, seed = keys
        sub = sub.set_index("prompt_type")

        for a, b in [
            ("P0_semantic_only", "P1_style_name"),
            ("P0_semantic_only", "P2_descriptor_enhanced"),
            ("P1_style_name", "P2_descriptor_enhanced"),
        ]:
            if a not in sub.index or b not in sub.index:
                continue

            pa = sub.loc[a, "path"]
            pb = sub.loc[b, "path"]

            xa = image_to_lpips_tensor(pa, device)
            xb = image_to_lpips_tensor(pb, device)

            with torch.no_grad():
                val = float(loss_fn(xa, xb).item())

            pair_rows.append({
                "base_prompt_id": base_prompt_id,
                "target_style": style,
                "seed": seed,
                "pair": f"{a}_vs_{b}",
                "path_a": pa,
                "path_b": pb,
                "lpips": val,
            })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(
        os.path.join(out_dir, "lpips_prompt_change_details.csv"),
        index=False,
    )

    if len(pair_df) > 0:
        summary = (
            pair_df
            .groupby("pair")
            .agg(
                n=("lpips", "count"),
                lpips_mean=("lpips", "mean"),
                lpips_std=("lpips", "std"),
            )
            .reset_index()
        )
        summary.to_csv(
            os.path.join(out_dir, "lpips_prompt_change_summary.csv"),
            index=False,
        )

    print("[OK] saved LPIPS prompt change results.")
    return pair_df


def compute_lpips_seed_diversity(detail_df: pd.DataFrame, out_dir: str, device: str):
    try:
        import lpips
    except ImportError:
        print("[WARN] lpips package not installed. Skip LPIPS diversity.")
        return None

    print("\n[Metric] LPIPS seed diversity")

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    group_cols = ["base_prompt_id", "target_style", "prompt_type"]

    for keys, sub in detail_df.groupby(group_cols):
        base_prompt_id, style, prompt_type = keys
        if len(sub) < 2:
            continue

        records = sub.to_dict("records")
        for r1, r2 in combinations(records, 2):
            xa = image_to_lpips_tensor(r1["path"], device)
            xb = image_to_lpips_tensor(r2["path"], device)

            with torch.no_grad():
                val = float(loss_fn(xa, xb).item())

            rows.append({
                "base_prompt_id": base_prompt_id,
                "target_style": style,
                "prompt_type": prompt_type,
                "seed_a": r1["seed"],
                "seed_b": r2["seed"],
                "path_a": r1["path"],
                "path_b": r2["path"],
                "lpips": val,
            })

    div_df = pd.DataFrame(rows)
    div_df.to_csv(
        os.path.join(out_dir, "lpips_seed_diversity_details.csv"),
        index=False,
    )

    if len(div_df) > 0:
        summary = (
            div_df
            .groupby("prompt_type")
            .agg(
                n=("lpips", "count"),
                lpips_mean=("lpips", "mean"),
                lpips_std=("lpips", "std"),
            )
            .reset_index()
        )
        summary.to_csv(
            os.path.join(out_dir, "lpips_seed_diversity_summary.csv"),
            index=False,
        )

    print("[OK] saved LPIPS seed diversity results.")
    return div_df


# ============================================================
# Pretty summary
# ============================================================

def save_latex_table(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv)

    cols = [
        "prompt_type",
        "clipscore_semantic_mean",
        "clipscore_full_mean",
        "style_similarity_mean",
        "style_margin_mean",
        "retrieval_acc",
    ]
    df = df[cols].copy()

    rename = {
        "prompt_type": "Prompt Type",
        "clipscore_semantic_mean": "CLIPScore-sem. $\\uparrow$",
        "clipscore_full_mean": "CLIPScore-full $\\uparrow$",
        "style_similarity_mean": "Style Sim. $\\uparrow$",
        "style_margin_mean": "Style Margin $\\uparrow$",
        "retrieval_acc": "Retrieval Acc. $\\uparrow$",
    }
    df = df.rename(columns=rename)

    for c in df.columns:
        if c != "Prompt Type":
            df[c] = df[c].map(lambda x: f"{x:.3f}")

    latex = df.to_latex(index=False, escape=False)

    with open(out_path, "w") as f:
        f.write(latex)

    print(f"[OK] saved LaTeX table to: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_roots",
        type=str,
        nargs="+",
        required=True,
        help="Generated output folders. Can be TMP folders or their parent folder.",
    )
    parser.add_argument(
        "--style_ref_root",
        type=str,
        required=True,
        help="Root folder containing style reference subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="./clip-vit-large-patch14",
    )
    parser.add_argument(
        "--base_prompt_map_json",
        type=str,
        default=None,
        help="Optional JSON mapping from TMP folder name to base prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--skip_lpips", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    if args.base_prompt_map_json is not None:
        with open(args.base_prompt_map_json, "r") as f:
            base_prompt_map = json.load(f)
    else:
        base_prompt_map = None

    print("[Info] Build metadata")
    gen_df = build_generated_metadata(args.input_roots, base_prompt_map=base_prompt_map)
    meta_path = os.path.join(args.out_dir, "generated_metadata_auto.csv")
    gen_df.to_csv(meta_path, index=False)
    print(f"[OK] saved metadata to: {meta_path}")
    print(gen_df[["base_prompt_id", "target_style", "prompt_type", "seed", "path"]].head())

    print("\n[Info] Load style references")
    style_to_paths = load_style_reference_paths(args.style_ref_root)
    print("[Info] style refs:", {k: len(v) for k, v in style_to_paths.items()})

    print("\n[Info] Load CLIP")
    clip_eval = ClipEvaluator(args.clip_model_name, args.device)

    print("\n[Info] Compute style centroids")
    style_keys, centroid_mat = compute_style_centroids(
        clip_eval,
        style_to_paths,
        batch_size=args.batch_size,
    )

    print("\n[Info] Compute CLIP and style metrics")
    detail_df = compute_clip_and_style_metrics(
        gen_df=gen_df,
        clip_eval=clip_eval,
        style_keys=style_keys,
        centroid_mat=centroid_mat,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )

    if not args.skip_lpips:
        compute_lpips_prompt_change(detail_df, args.out_dir, args.device)
        compute_lpips_seed_diversity(detail_df, args.out_dir, args.device)

    save_latex_table(
        summary_csv=os.path.join(args.out_dir, "summary_by_prompt_type.csv"),
        out_path=os.path.join(args.out_dir, "summary_by_prompt_type.tex"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()