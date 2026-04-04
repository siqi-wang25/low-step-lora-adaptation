#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.cuda.empty_cache()

local_rank = int(os.environ.get("LOCAL_RANK", 0))

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

sys.stdout = open("./log.txt", "w")
sys.stderr = sys.stdout

def find_images(root: Path):
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS])


def infer_caption(img_path: Path, dataset_root: Path, default_style: str) -> str:
    txt_path = img_path.with_suffix('.txt')
    if txt_path.exists():
        text = txt_path.read_text(encoding='utf-8').strip()
        if text:
            return text
    rel_parent = img_path.relative_to(dataset_root).parent
    folder_hint = ' '.join(rel_parent.parts).replace('_', ' ').replace('-', ' ').strip()
    if folder_hint and folder_hint != '.':
        return f"{folder_hint}, {default_style}" if default_style else folder_hint
    return default_style or 'painting'


def write_metadata(dataset_dir: Path, images, default_style: str, out_path: Path):
    with out_path.open('w', encoding='utf-8') as f:
        for img in images:
            cap = infer_caption(img, dataset_dir, default_style)
            record = {
                'file_name': str(img.relative_to(dataset_dir)),
                'text': cap,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def run(cmd, env=None):
    print('\n[RUN] ' + ' '.join(shlex.quote(x) for x in cmd) + '\n', flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    p = argparse.ArgumentParser(description='Prepare a local image folder and launch official Diffusers SDXL LoRA training.')
    p.add_argument('--dataset_dir', required=True, help='Folder with training images. May contain subfolders.')
    p.add_argument('--output_dir', required=True, help='Where to save LoRA checkpoints.')
    p.add_argument('--model_path', default='', help='Local SDXL base model path.')
    p.add_argument('--diffusers_repo', default='', help='Local clone of huggingface/diffusers.')
    p.add_argument('--train_script', default='', help='Optional explicit path to train_text_to_image_lora_sdxl.py')
    p.add_argument('--default_caption', default='new realism painting style', help='Fallback caption when .txt is missing.')
    p.add_argument('--resolution', type=int, default=768)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--train_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--max_train_steps', type=int, default=2000)
    p.add_argument('--checkpointing_steps', type=int, default=250)
    p.add_argument('--validation_steps', type=int, default=100)
    p.add_argument('--num_validation_images', type=int, default=4)
    p.add_argument('--validation_prompt', default='a lakeside cottage at sunset, new realism painting style')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataloader_num_workers', type=int, default=4)
    p.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], default='fp16')
    p.add_argument('--use_8bit_adam', action='store_true')
    p.add_argument('--train_text_encoder', action='store_true')
    p.add_argument('--snr_gamma', type=float, default=5.0)
    p.add_argument('--resume_from_checkpoint', default='')
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--enable_xformers_memory_efficient_attention', action='store_true')
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_script:
        train_script = Path(args.train_script)
    else:
        train_script = Path(args.diffusers_repo) / 'examples' / 'text_to_image' / 'train_text_to_image_lora_sdxl.py'

    if not train_script.exists():
        raise FileNotFoundError(
            f'Cannot find training script: {train_script}.\\n'
        )

    images = find_images(dataset_dir)
    if not images:
        raise RuntimeError(f'No images found under {dataset_dir}')

    metadata_path = dataset_dir / 'metadata.jsonl'
    write_metadata(dataset_dir, images, args.default_caption, metadata_path)
    print(f'[INFO] Found {len(images)} images')
    print(f'[INFO] Wrote metadata to {metadata_path}')
    print(f'[INFO] Training script: {train_script}')

    env = os.environ.copy()
    env.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

    cmd = [
        'accelerate', 'launch', str(train_script),
        '--pretrained_model_name_or_path', args.model_path,
        '--train_data_dir', str(dataset_dir),
        '--image_column', 'image',
        '--caption_column', 'text',
        '--resolution', str(args.resolution),
        '--random_flip',
        '--center_crop',
        '--train_batch_size', str(args.train_batch_size),
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--learning_rate', str(args.learning_rate),
        '--max_train_steps', str(args.max_train_steps),
        '--checkpointing_steps', str(args.checkpointing_steps),
        '--validation_prompt', args.validation_prompt,
        '--num_validation_images', str(args.num_validation_images),
        '--validation_epochs', '1',
        '--rank', str(args.rank),
        '--seed', str(args.seed),
        '--output_dir', str(output_dir),
        '--mixed_precision', args.mixed_precision,
        '--report_to', 'tensorboard',
        '--dataloader_num_workers', str(args.dataloader_num_workers),
        '--snr_gamma', str(args.snr_gamma),
    ]

    if args.use_8bit_adam:
        cmd.append('--use_8bit_adam')
    if args.train_text_encoder:
        cmd.append('--train_text_encoder')
    if args.gradient_checkpointing:
        cmd.append('--gradient_checkpointing')
    if args.enable_xformers_memory_efficient_attention:
        cmd.append('--enable_xformers_memory_efficient_attention')
    if args.resume_from_checkpoint:
        cmd.extend(['--resume_from_checkpoint', args.resume_from_checkpoint])

    run(cmd, env=env)

    print('\n[OK] Training finished.')
    print(f'[OK] Output dir: {output_dir}')
    print('[OK] Typical LoRA weight file path: output_dir/pytorch_lora_weights.safetensors')


if __name__ == '__main__':
    main()
