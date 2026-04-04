# Fast Artistic Style Adaptation with Low-Step LoRA

This repository contains the code and experiments for **fast artistic style adaptation** using **low-step LoRA** on **Stable Diffusion XL (SDXL)**.

The project studies whether short LoRA training is already sufficient to produce recognizable artistic styles, and how adaptation quality changes across training checkpoints.

## Overview

We evaluate six artistic styles:

- Cubism
- Expressionism
- Early Renaissance
- Baroque
- Pointillism
- Pop Art

Instead of focusing only on the final checkpoint, we compare multiple intermediate checkpoints using both quantitative metrics and qualitative inspection.

## Main Ideas

- Low-step LoRA can already produce clear stylistic effects.
- Adaptation behavior is strongly **style-dependent** and **prompt-dependent**.
- More training is not always better.
- Intermediate checkpoints may offer a better trade-off between style strength, semantic faithfulness, and visual quality.

## Evaluation

We use the following metrics:

- **CLIPScore** for text-image alignment
- **Style Similarity** for stylistic fidelity
- **KID** for distributional similarity
- **LPIPS** for perceptual variation
- **Qualitative inspection** for visual analysis
