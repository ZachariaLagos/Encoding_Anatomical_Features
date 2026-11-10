# DeepTrace: An Evaluation System for AI-Generated Video Detection

A multimodal evaluation system — not just a classifier — for distinguishing AI-generated video from real footage, built around **measuring and closing the generalization gap** between manipulation types: face-swap deepfakes (the long-studied case) and whole-scene AI video generation (Sora, Kling, Veo — largely unaddressed by existing benchmarks).

> Team project — CS 5330 Pattern Recognition and Computer Vision, Northeastern University, 2026.
> Authors: Tavish Hookoom, **Zihao Li**, Jiajun Huang, Xijia Zeng.
> My primary contributions: training-run execution and results reporting (joint/AI-video/face-swap variants), and the difficulty-annotation protocol as one of four independent raters. Evaluation design, data pipeline, and demo were collaborative across the team — see commit history for the full breakdown.
> Original repository (full commit history): https://github.khoury.northeastern.edu/eddyjhuang/CS5330_SU26_DeepTrace_AI_Detection.git

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-1F3864?style=flat-square)](https://github.com/openai/CLIP)
[![ONNX](https://img.shields.io/badge/ONNX-Deployed-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Tests](https://img.shields.io/badge/tests-26%20passing-2E7D32?style=flat-square)](tests/)

---

## Why this is an evaluation problem, not just a classification problem

A detector trained and tested on one manipulation type will report a misleadingly high accuracy — it may have learned that manipulation type's specific artifacts rather than anything general about synthetic video. The core deliverable here is a **protocol that is designed to expose that failure before it ships**, across three axes:

1. **Cross-manipulation generalization** — train on one manipulation type, test on the other
2. **Leave-one-generator-out (LOGO)** — hold out an entire *generator* (not just clips) at test time
3. **Human-rated difficulty** — a structured, multi-rater protocol for labeling which clips would fool a naive viewer

## Dataset: DeepTrace-GV

A 246-clip benchmark combining whole-scene AI generation with real footage — built because existing deepfake benchmarks (FaceForensics++, DFDC, Celeb-DF) only cover face-swap manipulation.

<p align="center">
  <img src="assets/dataset_composition.png" alt="Dataset composition" width="560">
</p>

- Official clips via `yt-dlp` from verified generator channels; community clips only with explicit generator attribution
- Re-encoded to H.264/CRF23/720p to eliminate compression-based leakage between sources
- 12 metadata fields per clip (generator, scene type, motion level, has_faces, difficulty, split, …)
- Documented as a full **datasheet** (Gebru et al. format) covering motivation, composition, collection bias, and ethical considerations — [`data/datasheet/DATASHEET.md`](data/datasheet/DATASHEET.md)
- Published publicly: [kaggle.com/datasets/tavishh/deeptrace-gv](https://www.kaggle.com/datasets/tavishh/deeptrace-gv/)

## Automated Metrics: Cross-Manipulation Generalization Gap

Three training variants, each evaluated **in-distribution and cross-manipulation**:

<p align="center">
  <img src="assets/cross_manipulation_gap.png" alt="Cross-manipulation generalization gap" width="640">
</p>

| Variant | In-Distribution AUC | Cross-Manipulation AUC | Gap |
|---|---|---|---|
| AI-video only | 0.9708 | 0.4966 (Celeb-DF) | **0.4741** |
| Face-swap only | 0.8090 | 0.7222 (DeepTrace-GV) | **0.0868** |
| Joint training | 1.0000 | 0.8210 (Celeb-DF) | *gap closed* |

**Key finding:** a model trained only on AI-generated video performs at near-random chance (AUC 0.50) on face-swap detection — the two manipulation types are not interchangeable, and an evaluation suite that only tests in-distribution would never surface this. Joint training restores performance on both simultaneously.

## Leave-One-Generator-Out (LOGO) Evaluation

A stricter test than a held-out clip split: each generator is **entirely excluded from training**, then evaluated as a fully unseen generator.

<p align="center">
  <img src="assets/logo_evaluation.png" alt="LOGO evaluation results" width="560">
</p>

| Held-Out Generator | AUC | Accuracy |
|---|---|---|
| Sora | 1.0000 | 100.0% |
| Kling | 0.9850 | 88.5% |
| Veo | 0.9386 | 88.0% |
| **Mean** | **0.9745 ± 0.0261** | |

Veo is consistently the hardest generator to detect when held out — a finding the aggregate mean alone would hide. (Test sets are 24–26 clips per round; reported as exploratory given the small-sample size, not a statistically robust claim.)

## Reproducible Human Evaluation Protocol

Beyond automated metrics, a structured **inter-annotator-agreement protocol** rates a 60-clip stratified subset for human-perceived difficulty (would this fool a naive viewer with no AI-detection training?):

- Four team members rate independently, with no discussion before submitting
- Agreement is computed as **Cohen's kappa** across all rater pairs, interpreted on the Landis & Koch scale
- A dedicated CLI tool (`rate_difficulty.py`) presents one unrated clip at a time, saves progress after every rating, and is resumable
- Majority-vote difficulty label is merged back into the dataset manifest as a first-class field

*This protocol is designed and scripted (clip sampling, rating CLI, kappa computation); the rating pass itself (this repo snapshot) has not yet been completed, so no kappa value is reported here. I contributed as one of the four raters.*

## Deployment & Debugging Interface

All model variants exported to **ONNX** and served through an interactive Gradio workspace that exists specifically to let you inspect *why* the model made a call, not just what it decided:

- Real / AI-generated verdict with confidence score, and a configurable detection threshold
- **Frame-level evidence**: per-frame fake-probability highlights across the clip, so a false verdict can be traced back to the specific frames driving it
- End-to-end latency and FPS counter, measured live

| Metric | Value |
|---|---|
| Mean latency | 16.5 ms / clip (± 1.58ms) |
| Throughput | 303 FPS |
| Hardware | T4 GPU (CLIP) + CPU (ONNX MLP head) |

```
upload -> H.264/CRF23/720p re-encode -> 1 FPS frame extraction
       -> CLIP ViT-B/32 features -> ONNX MLP head -> mean aggregation
       -> clip verdict + per-frame evidence
```

## Engineering Rigor

- **26 passing unit tests** across the data pipeline, model modules, and evaluation metrics
- Config-driven training (`configs/config.yaml`) across all three variants and all LOGO rounds
- Full Colab training notebook retained for reproducibility: [`notebooks/DeepTrace_Training.ipynb`](notebooks/DeepTrace_Training.ipynb)

## Tech Stack

`Python` · `PyTorch` · `CLIP (ViT-B/32)` · `ONNX` · `Gradio` · `ffmpeg` · `pytest`

## What this maps to

A self-contained example of building an **evaluation system** for visual generation models — automated metric design that specifically targets generalization failure (not just accuracy), a documented and bias-audited dataset, a reproducible human-evaluation protocol with inter-rater agreement, and a deployed interface built for inspecting model behavior rather than only reporting a verdict.
