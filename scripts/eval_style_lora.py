#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from torchmetrics.image.kid import KernelInceptionDistance
import lpips


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_images(folder: str):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


class ClipEvaluator:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode_images(self, image_paths, batch_size=16):
        feats = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP image encode"):
            batch_paths = image_paths[i:i + batch_size]
            images = [load_image(p) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
            feats.append(image_features.cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=32):
        feats = []
        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text encode"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
            feats.append(text_features.cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def clipscore(self, image_paths, prompts, batch_size=16):
        assert len(image_paths) == len(prompts)
        all_scores = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIPScore"):
            batch_paths = image_paths[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            images = [load_image(p) for p in batch_paths]
            inputs = self.processor(
                text=batch_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
            text_features = self.model.get_text_features(**text_inputs)

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            scores = (image_features * text_features).sum(dim=-1)
            all_scores.extend(scores.detach().cpu().tolist())
        return all_scores


def load_generated_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"path", "prompt", "target_style"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"generated_metadata missing columns: {missing}")
    for p in df["path"].tolist():
        ensure_exists(p)
    if "checkpoint" not in df.columns:
        df["checkpoint"] = "unknown"
    return df


def load_style_reference_paths(style_ref_root: str):
    style_to_paths = {}

    # 判断 root 下面有没有子文件夹
    subdirs = [
        d for d in os.listdir(style_ref_root)
        if os.path.isdir(os.path.join(style_ref_root, d))
    ]

    if len(subdirs) == 0:
        paths = list_images(style_ref_root)
        if len(paths) == 0:
            raise ValueError(f"No images found in: {style_ref_root}")

        style_name = os.path.basename(style_ref_root.rstrip("/"))
        style_to_paths[style_name] = paths

    else:
        for style_name in sorted(subdirs):
            subdir = os.path.join(style_ref_root, style_name)
            paths = list_images(subdir)
            if len(paths) > 0:
                style_to_paths[style_name] = paths

        if not style_to_paths:
            raise ValueError(f"No style subfolders with images found under: {style_ref_root}")

    return style_to_paths

def compute_style_retrieval_and_classification(
    clip_eval: ClipEvaluator,
    gen_df: pd.DataFrame,
    style_to_paths: dict,
    out_dir: str,
):
    print("\n[Stage] Style retrieval / classification")

    # 1) encode style reference images
    style_ref_feats = {}
    style_centroids = {}
    ref_X = []
    ref_y = []

    for style, paths in style_to_paths.items():
        feats = clip_eval.encode_images(paths)
        style_ref_feats[style] = feats
        style_centroids[style] = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        ref_X.append(feats.numpy())
        ref_y.extend([style] * len(paths))

    ref_X = np.concatenate(ref_X, axis=0)
    ref_y = np.array(ref_y)

    # 2) train a simple classifier on CLIP image features of style refs
    clf = LogisticRegression(
        max_iter=2000,
        random_state=42,
        multi_class="auto"
    )
    clf.fit(ref_X, ref_y)

    # 3) encode generated images
    gen_paths = gen_df["path"].tolist()
    gen_feats = clip_eval.encode_images(gen_paths)
    gen_X = gen_feats.numpy()
    gen_labels = gen_df["target_style"].tolist()

    # 4) nearest centroid retrieval
    retrieval_preds = []
    retrieval_target_sims = []
    retrieval_non_target_sims = []
    retrieval_all = []

    style_names = sorted(style_centroids.keys())
    centroid_mat = torch.stack([style_centroids[s] for s in style_names], dim=0)  # [S, D]

    sims = gen_feats @ centroid_mat.T  # [N, S]
    sims_np = sims.numpy()

    for i, row in enumerate(sims_np):
        pred_idx = int(np.argmax(row))
        pred_style = style_names[pred_idx]
        retrieval_preds.append(pred_style)

        target_style = gen_labels[i]
        target_idx = style_names.index(target_style)
        target_sim = float(row[target_idx])

        non_target_vals = [float(v) for j, v in enumerate(row) if j != target_idx]
        non_target_sim = float(np.mean(non_target_vals)) if non_target_vals else float("nan")

        retrieval_target_sims.append(target_sim)
        retrieval_non_target_sims.append(non_target_sim)
        retrieval_all.append(dict(zip(style_names, row.tolist())))

    retrieval_acc = accuracy_score(gen_labels, retrieval_preds)

    # 5) classifier prediction
    clf_preds = clf.predict(gen_X)
    clf_acc = accuracy_score(gen_labels, clf_preds)

    # 6) save detailed per-image results
    detail_df = gen_df.copy()
    detail_df["retrieval_pred"] = retrieval_preds
    detail_df["retrieval_correct"] = (detail_df["retrieval_pred"] == detail_df["target_style"]).astype(int)
    detail_df["target_style_similarity"] = retrieval_target_sims
    detail_df["non_target_style_similarity_mean"] = retrieval_non_target_sims
    detail_df["clf_pred"] = clf_preds
    detail_df["clf_correct"] = (detail_df["clf_pred"] == detail_df["target_style"]).astype(int)

    for s in style_names:
        detail_df[f"sim_to_{s}"] = [d[s] for d in retrieval_all]

    detail_path = os.path.join(out_dir, "style_retrieval_classification_details.csv")
    detail_df.to_csv(detail_path, index=False)

    # 7) save summary
    summary = {
        "retrieval_accuracy": float(retrieval_acc),
        "classification_accuracy": float(clf_acc),
        "classification_report": classification_report(gen_labels, clf_preds, output_dict=True),
        "retrieval_confusion_matrix_labels": style_names,
        "retrieval_confusion_matrix": confusion_matrix(gen_labels, retrieval_preds, labels=style_names).tolist(),
        "classification_confusion_matrix_labels": style_names,
        "classification_confusion_matrix": confusion_matrix(gen_labels, clf_preds, labels=style_names).tolist(),
    }

    with open(os.path.join(out_dir, "style_retrieval_classification_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 8) grouped summary
    grouped = []
    for keys, sub in detail_df.groupby(["target_style", "checkpoint"]):
        style, ckpt = keys
        grouped.append({
            "target_style": style,
            "checkpoint": ckpt,
            "n": int(len(sub)),
            "retrieval_acc": float(sub["retrieval_correct"].mean()),
            "clf_acc": float(sub["clf_correct"].mean()),
            "target_style_similarity_mean": float(sub["target_style_similarity"].mean()),
            "non_target_style_similarity_mean": float(sub["non_target_style_similarity_mean"].mean()),
            "style_margin_mean": float((sub["target_style_similarity"] - sub["non_target_style_similarity_mean"]).mean()),
        })
    grouped_df = pd.DataFrame(grouped).sort_values(["target_style", "checkpoint"])
    grouped_df.to_csv(os.path.join(out_dir, "style_retrieval_classification_grouped.csv"), index=False)

    print(f"Saved style retrieval/classification results to: {out_dir}")


def compute_clipscore(
    clip_eval: ClipEvaluator,
    gen_df: pd.DataFrame,
    out_dir: str,
):
    print("\n[Stage] CLIPScore")

    image_paths = gen_df["path"].tolist()
    prompts = gen_df["prompt"].tolist()
    scores = clip_eval.clipscore(image_paths, prompts)

    result_df = gen_df.copy()
    result_df["clipscore"] = scores
    result_df.to_csv(os.path.join(out_dir, "clipscore_details.csv"), index=False)

    grouped = []
    for keys, sub in result_df.groupby(["target_style", "checkpoint"]):
        style, ckpt = keys
        grouped.append({
            "target_style": style,
            "checkpoint": ckpt,
            "n": int(len(sub)),
            "clipscore_mean": float(sub["clipscore"].mean()),
            "clipscore_std": float(sub["clipscore"].std(ddof=0)),
        })
    grouped_df = pd.DataFrame(grouped).sort_values(["target_style", "checkpoint"])
    grouped_df.to_csv(os.path.join(out_dir, "clipscore_grouped.csv"), index=False)

    summary = {
        "overall_mean": float(result_df["clipscore"].mean()),
        "overall_std": float(result_df["clipscore"].std(ddof=0)),
    }
    with open(os.path.join(out_dir, "clipscore_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved CLIPScore results to: {out_dir}")


def image_to_uint8_tensor(img: Image.Image, size=299):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # [0,1]
    ])
    x = tfm(img) * 255.0
    x = x.clamp(0, 255).to(torch.uint8)
    return x


def compute_kid(
    gen_df: pd.DataFrame,
    style_to_paths: dict,
    out_dir: str,
    device: str = "cuda",
    kid_subset_size: int = 50,
):
    print("\n[Stage] KID")
    results = []

    for (style, ckpt), sub in gen_df.groupby(["target_style", "checkpoint"]):
        ref_paths = list(style_to_paths.values())[0]
        gen_paths = gen_df["path"].tolist()

        kid = KernelInceptionDistance(
            subset_size=min(kid_subset_size, len(gen_paths), len(ref_paths)),
            normalize=False
        ).to(device)

        # update real
        for p in tqdm(ref_paths, desc=f"KID real {style}-{ckpt}"):
            img = load_image(p)
            x = image_to_uint8_tensor(img).unsqueeze(0).to(device)
            kid.update(x, real=True)

        # update fake
        for p in tqdm(gen_paths, desc=f"KID fake {style}-{ckpt}"):
            img = load_image(p)
            x = image_to_uint8_tensor(img).unsqueeze(0).to(device)
            kid.update(x, real=False)

        mean, std = kid.compute()
        results.append({
            "target_style": style,
            "checkpoint": ckpt,
            "n_generated": int(len(gen_paths)),
            "n_reference": int(len(ref_paths)),
            "kid_mean": float(mean.item()),
            "kid_std": float(std.item()),
        })

        # free
        del kid
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results).sort_values(["target_style", "checkpoint"])
    results_df.to_csv(os.path.join(out_dir, "kid_results.csv"), index=False)

    summary = {
        "num_groups": int(len(results_df)),
        "mean_kid_over_groups": float(results_df["kid_mean"].mean()) if len(results_df) > 0 else None,
    }
    with open(os.path.join(out_dir, "kid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved KID results to: {out_dir}")


def image_to_lpips_tensor(path: str, device: str):
    img = load_image(path)
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # [0,1]
    ])
    x = tfm(img).unsqueeze(0).to(device)
    x = x * 2.0 - 1.0  # LPIPS expects [-1,1]
    return x

def compute_single_style_fidelity(
    clip_eval,
    gen_df,
    style_to_paths,
    out_dir,
):
    print("\n[Stage] Single-style fidelity")

    assert len(style_to_paths) == 1
    style_name = list(style_to_paths.keys())[0]

    # encode reference
    ref_paths = style_to_paths[style_name]
    ref_feats = clip_eval.encode_images(ref_paths)

    ref_centroid = F.normalize(ref_feats.mean(dim=0, keepdim=True), dim=-1)

    # encode generated
    gen_paths = gen_df["path"].tolist()
    gen_feats = clip_eval.encode_images(gen_paths)

    # style similarity
    sims = (gen_feats @ ref_centroid.T).squeeze(1)  # cosine sim
    sims_np = sims.numpy()

    # intra-style consistency
    gen_feats_norm = F.normalize(gen_feats, dim=-1)
    sim_matrix = gen_feats_norm @ gen_feats_norm.T
    intra_sim = sim_matrix.mean().item()

    # variance
    var = gen_feats.var(dim=0).mean().item()

    # save
    df = gen_df.copy()
    df["style_similarity"] = sims_np
    df.to_csv(os.path.join(out_dir, "style_fidelity_details.csv"), index=False)

    summary = {
        "style_similarity_mean": float(np.mean(sims_np)),
        "style_similarity_std": float(np.std(sims_np)),
        "intra_style_similarity": float(intra_sim),
        "feature_variance": float(var),
    }

    with open(os.path.join(out_dir, "style_fidelity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved single-style fidelity results to: {out_dir}")

def compute_lpips_from_pairs(
    pairs_csv: str,
    out_dir: str,
    device: str = "cuda",
    net: str = "alex",
):
    print("\n[Stage] LPIPS")
    df = pd.read_csv(pairs_csv)
    required = {"path_a", "path_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"lpips_pairs_csv missing columns: {missing}")
    if "group" not in df.columns:
        df["group"] = "default"

    loss_fn = lpips.LPIPS(net=net).to(device).eval()

    vals = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LPIPS pairs"):
        pa = row["path_a"]
        pb = row["path_b"]
        ensure_exists(pa)
        ensure_exists(pb)

        xa = image_to_lpips_tensor(pa, device)
        xb = image_to_lpips_tensor(pb, device)

        with torch.no_grad():
            v = loss_fn(xa, xb).item()

        vals.append(v)

    df["lpips"] = vals
    df.to_csv(os.path.join(out_dir, "lpips_details.csv"), index=False)

    grouped = []
    for g, sub in df.groupby("group"):
        grouped.append({
            "group": g,
            "n": int(len(sub)),
            "lpips_mean": float(sub["lpips"].mean()),
            "lpips_std": float(sub["lpips"].std(ddof=0)),
        })
    grouped_df = pd.DataFrame(grouped).sort_values("group")
    grouped_df.to_csv(os.path.join(out_dir, "lpips_grouped.csv"), index=False)

    summary = {
        "overall_mean": float(df["lpips"].mean()),
        "overall_std": float(df["lpips"].std(ddof=0)),
    }
    with open(os.path.join(out_dir, "lpips_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved LPIPS results to: {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_metadata_csv", type=str, required=True)
    parser.add_argument("--style_ref_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--lpips_pairs_csv", type=str, default=None)
    parser.add_argument("--clip_model_name", type=str, default="/model/clip-vit-large-patch14")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_clipscore", action="store_true")
    parser.add_argument("--skip_style", action="store_true")
    parser.add_argument("--skip_kid", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--kid_subset_size", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    gen_df = load_generated_metadata(args.generated_metadata_csv)
    style_to_paths = load_style_reference_paths(args.style_ref_root)

    print("[Info] styles found:", sorted(style_to_paths.keys()))
    print("[Info] num generated images:", len(gen_df))

    num_styles = len(style_to_paths)

    clip_eval = None
    if not args.skip_clipscore or not args.skip_style:
        clip_eval = ClipEvaluator(model_name=args.clip_model_name, device=args.device)

    # CLIPScore
    if not args.skip_clipscore:
        compute_clipscore(clip_eval, gen_df, args.out_dir)

    # Style evaluation
    if not args.skip_style:
        if num_styles < 2:
            print("[Mode] Single-style fidelity")
            compute_single_style_fidelity(
                clip_eval, gen_df, style_to_paths, args.out_dir
            )
        else:
            print("[Mode] Multi-style retrieval/classification")
            compute_style_retrieval_and_classification(
                clip_eval, gen_df, style_to_paths, args.out_dir
            )

    # KID
    if not args.skip_kid:
        compute_kid(
            gen_df=gen_df,
            style_to_paths=style_to_paths,
            out_dir=args.out_dir,
            device=args.device,
            kid_subset_size=args.kid_subset_size,
        )

    # LPIPS
    if (not args.skip_lpips) and (args.lpips_pairs_csv is not None):
        compute_lpips_from_pairs(
            pairs_csv=args.lpips_pairs_csv,
            out_dir=args.out_dir,
            device=args.device,
            net="alex",
        )
    elif not args.skip_lpips:
        print("[WARN] --skip_lpips not set but no --lpips_pairs_csv provided. LPIPS skipped.")

    print("\nDone.")

if __name__ == "__main__":
    main()