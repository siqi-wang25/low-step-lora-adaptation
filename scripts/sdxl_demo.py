import argparse
import os
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


def pick_dtype(device: str):
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main():
    parser = argparse.ArgumentParser(description="Minimal SDXL local inference demo")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to SDXL diffusers directory")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, extra fingers, bad anatomy")
    parser.add_argument("--output", type=str, default="sdxl_demo.png")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA loading")
    parser.add_argument("--lora_ckpt", type=str, default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    assert os.path.exists(args.model_path), f"Model path not found: {args.model_path}"
    dtype = pick_dtype(args.device)
    print(f"[INFO] device={args.device} dtype={dtype} model_path={args.model_path}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if args.lora:
        assert os.path.exists(args.lora_ckpt), f"LoRA path not found: {args.lora_ckpt}"
        pipe.load_lora_weights(args.lora_ckpt)
        print(f"[INFO] loaded LoRA from {args.lora_ckpt}")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    if args.device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] xformers attention enabled")
        except Exception:
            print("[INFO] xformers not available, continuing")
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output)
    print(f"[OK] saved to {args.output}")


if __name__ == "__main__":
    main()