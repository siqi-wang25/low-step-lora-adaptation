# Prompt-Style Compatibility in Low-Step LoRA Adaptation

This repository contains the code and experiments for **fast artistic style adaptation** using **Low-Rank Adaptation (LoRA)** on **Stable Diffusion XL (SDXL)**.

The project studies how artistic style adaptation behaves under a limited training budget. Instead of only evaluating the final LoRA checkpoint, we analyze how generated images evolve across intermediate checkpoints and how prompt formulation affects the observed adaptation quality.

## Overview

We evaluate low-step LoRA adaptation across six representative artistic styles:

- Cubism
- Expressionism
- Early Renaissance
- Baroque
- Pointillism
- Pop Art

The central question is whether short LoRA training is sufficient to produce recognizable artistic styles, and how adaptation quality depends on the interaction among **style**, **prompt content**, and **training checkpoint**.

## Project Stages

### Stage 1: Low-Step LoRA Adaptation

In the first stage, we train style-specific LoRA adapters and evaluate how generation quality changes across training checkpoints.

Rather than focusing only on the final checkpoint, we compare multiple intermediate checkpoints to study when recognizable stylistic characteristics emerge and whether longer training consistently improves generation quality.

The main findings are:

- Low-step LoRA can already produce clear stylistic effects.
- Adaptation behavior is strongly style-dependent.
- More training is not always better.
- Intermediate checkpoints may offer a better trade-off between style strength, semantic faithfulness, diversity, and visual quality.

### Stage 2: Prompt-Style Compatibility

In the second stage, we study **prompt-style compatibility** in low-step LoRA adaptation.

Using SDXL as the base model, we analyze how prompt content, artistic style, and training steps jointly influence generation quality. Our results show that recognizable stylistic characteristics can emerge at early checkpoints, but adaptation quality strongly depends on whether the prompt matches the visual logic of the target style.

Compatible prompts tend to produce stronger and more stable stylistic expression, while less compatible prompts may weaken style fidelity or semantic faithfulness. We further observe that longer training is not always beneficial, since additional steps may reduce diversity and lead to more template-like generations. In contrast, prompt engineering at fixed checkpoints can improve observed style expression without additional model training.

Overall, this project suggests that low-step LoRA should be understood as a **prompt-aware adaptation problem**, where effective generation depends on style and prompt formulation rather than training duration alone.

## Evaluation

We evaluate generated images using both quantitative metrics and qualitative inspection.

The metrics include:

- **CLIPScore** for text-image alignment
- **Style Similarity** for stylistic fidelity
- **KID** for distributional similarity
- **LPIPS** for perceptual variation
- **Qualitative inspection** for visual coherence, style expression, and prompt faithfulness

## Key Takeaways

- Low-step LoRA is a practical regime for rapid artistic customization.
- Early checkpoints can already produce recognizable style transfer effects.
- Longer training may improve stylistic conformity, but can also reduce diversity and semantic flexibility.
- Prompt formulation substantially affects the observed quality of a style-adapted model.
- Prompt engineering provides a low-cost alternative to extending LoRA training.
