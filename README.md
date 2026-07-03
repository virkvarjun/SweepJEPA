# SweepJEPA

**Cine-Ultrasound Risk Stratification for Thyroid Nodules: A Two-Paper Research Program**

Arjun Virk · [arjunvirk.com](https://arjunvirk.com)

Two complementary studies built on the public [Stanford AIMI Thyroid Cine-clip dataset](https://aimi.stanford.edu/). Track 1 is a deployable upgrade of cine risk stratification; Track 2 learns geometry-aware sweep representations via pose-conditioned JEPA pretraining. Track 2 reuses Track 1's aggregation and decision head, so the two tracks compose into one system.

## Research Tracks

### Track 1 — Modernized Cine Risk Stratification (Paper 1)

Near-term, deployable system with lower technical risk.

```
Cine frames {f₁,…,fₙ} → US foundation encoder (ViT) → Attention-MIL → Conformal decision
                                                              ↓
                                                    biopsy / no-biopsy / defer
```

| Component | Design |
|-----------|--------|
| **Backbone** | Domain-pretrained US encoder (USFM, USF-MAE, or UltraFedFM); frozen or LoRA-adapted |
| **Aggregation** | Gated attention-based MIL; bag-level supervision; focal loss + minority oversampling |
| **Decision** | Conformal risk control with guaranteed sensitivity floor and abstention (defer) |

**Aims:** T1.1 backbone + MIL upgrade; T1.2 risk-controlled biopsy recommendation validated on ThyroidXL.

### Track 2 — Geometry-Aware Sweep Representation (Paper 2)

Representation-learning study that treats a cine-clip as a sensorless freehand 2D probe sweep.

```
Stage B: Pose recovery R_ψ  →  pseudo-volume V + pose graph
Stage C: Pose-conditioned JEPA (E_θ, P_φ, EMA target Ē_θ)
Stage D: Track 1 MIL + conformal head on frozen Stage-C encoder
```

| Stage | Description |
|-------|-------------|
| **B** | Estimate relative transforms Δᵢ→ⱼ ∈ SE(3) from image content; compose into 3D pseudo-volume |
| **C** | Predict masked spatio-temporal tubelet latents conditioned on relative probe pose |
| **D** | Fine-tune shared Track 1 head; isolate contribution of sweep geometry |

**Aims:** T2.1 sweep geometry recovery; T2.2 pose-conditioned JEPA pretraining; T2.3 downstream comparison vs Track 1 at matched compute.

## Data

| Dataset | Role |
|---------|------|
| **Stanford AIMI Thyroid Cine-clip** | Flagship: 192 biopsy-confirmed nodules, 17,412 frames |
| **ThyroidXL** | External test for domain-shift generalization |
| **OpenUS-46 / unlabeled cine corpora** | Track 2 representation pretraining |

## Project Structure

```
SweepJEPA/
├── Original Paper Code/  # Cine-CNNTrans baseline (Yamashita et al.)
├── configs/              # Experiment configs (Track 1 & 2)
├── src/
│   ├── track1/           # US encoder, attention-MIL, conformal decision
│   ├── track2/           # Pose estimator, JEPA, sweep reconstruction
│   └── shared/           # Data loading and utilities
├── scripts/              # Training and evaluation entry points
└── docs/                 # Proposal and design notes
```

## Original Paper Code (Cine-CNNTrans baseline)

Vendored from [tarakapoor/thyroid_deep_learning](https://github.com/tarakapoor/thyroid_deep_learning) (MIT). This is the MobileNet-v2 + Transformer pipeline from Yamashita et al. (*Radiology: AI* 2022), used as the Track 1 reproduction baseline. See `Original Paper Code/README.md` for setup and how to run `cnn_main.py` / `transformer_main.py`.

## Timeline

| Phase | Weeks | Track | Work |
|-------|-------|-------|------|
| 1 | 1–4 | 1 | Data assembly; backbone selection; reproduce Cine-CNNTrans baseline |
| 2 | 5–9 | 1 | MIL head; conformal layer; CV; ThyroidXL test **(Paper 1)** |
| 3 | 10–14 | 2 | Pose estimator R_ψ; sweep reconstruction |
| 4 | 15–21 | 2 | Pose-conditioned JEPA; ablations |
| 5 | 22–26 | 2 | Geometry-aware head; release volumes **(Paper 2)** |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## References

Key prior work: Yamashita et al. (Cine-CNNTrans, *Radiology: AI* 2022); I-JEPA / V-JEPA (Assran, Bardes); US foundation models (USFM, USF-MAE, UltraFedFM); ABMIL (Ilse et al., ICML 2018); conformal risk control (Angelopoulos & Bates).

See [docs/PROPOSAL.md](docs/PROPOSAL.md) for the full research proposal.
