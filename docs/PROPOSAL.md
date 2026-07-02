# Cine-Ultrasound Risk Stratification for Thyroid Nodules

Research Proposal · Arjun Virk · June 15, 2026

## Summary

Two complementary studies on the public Stanford AIMI Thyroid Cine-clip dataset:

- **Track 1** — Deployable cine risk-stratification upgrade: US foundation encoder, attention-MIL, conformal biopsy-referral with abstention.
- **Track 2** — Treats a cine-clip as a sensorless freehand sweep; recovers 3D geometry and learns nodule representations via pose-conditioned latent prediction (JEPA). Reuses Track 1's head.

## Background

US is the primary modality for thyroid nodule evaluation, but ACR TI-RADS retains a false-positive biopsy rate of ~49–56%. Yamashita et al. showed cine acquisitions carry more diagnostic signal than static frames (AUC 0.88 vs 0.72; TI-RADS specificity 79.4% vs 26.9% when revising biopsy recommendations).

Two unused properties motivate this work:

1. Cine frames are **spatially ordered slices** from a continuous probe sweep; treating them as an unordered bag discards geometry.
2. **Label broadcasting** to every frame injects noise; most frames lack diagnostic features.

## Specific Aims

### Track 1

- **T1.1** Replace ImageNet CNN with US foundation encoder + attention-MIL.
- **T1.2** Risk-controlled biopsy / no-biopsy / defer with sensitivity floor; validate on ThyroidXL.

### Track 2

- **T2.1** Recover sweep geometry without tracker (R_ψ → SE(3) poses → pseudo-volume).
- **T2.2** Pose-conditioned JEPA pretraining on spatio-temporal tubelets.
- **T2.3** Fine-tune Track 1 head; compare vs Track 1 at matched compute.

## Architecture

### Track 1 Pipeline

Per-frame ViT features → gated attention-MIL pooling (weights αᵢ) → conformal layer → clinical action.

### Track 2 Stages

**Stage B — 3D sweep reconstruction**

R_ψ regresses Δᵢ→ⱼ ∈ SE(3) between frames. Composing transforms yields pseudo-volume V and pose graph. Pretrained on tracked sweeps; refined with cycle and trajectory-smoothness losses.

**Stage C — Pose-conditioned JEPA**

```
L_JEPA = (1/|M|) Σ_{j∈M} || P_φ(E_θ(x), Δ_{x→y}, m_j) − sg[Ē_θ(y_j)] ||₁
```

Predictor P_φ is conditioned on relative pose Δ from Stage B. EMA target encoder Ē_θ; loss in feature space (no pixel reconstruction).

**Stage D — Downstream stratification**

Frozen Stage-C encoder → Track 1 MIL + conformal head. Isolates geometry contribution.

## Design Rationale

| Choice | Rationale |
|--------|-----------|
| Latent prediction over pixel reconstruction | Speckle is stochastic; JEPA ignores unpredictable noise |
| Pose-conditioned predictor | Natural "action" in freehand sweep is relative probe pose (V-JEPA 2 mechanism) |
| US foundation encoder | Label-efficient on 192-nodule cohort; documented error source in prior work |
| Attention-MIL | Bag-level supervision; interpretable frame weights |
| Conformal decision | Distribution-free sensitivity guarantee; explicit abstention with 17 positives |

## Experimental Plan

**Track 1 ablations:** backbone swap; MIL vs average pooling vs set transformer; conformal vs weighted-Youden. Metrics: malignancy AUC (DeLong CI), ThyroidXL external test, sensitivity floor, specificity, NNB, abstention rate.

**Track 2 ablations:** pose recovery quality; pose-conditioned JEPA vs pose-free vs set transformer; latent vs pixel reconstruction; true order vs shuffled frames; label efficiency curves.

**Pre-registered success criterion (Track 2):** Pose-conditioned pretraining must improve downstream AUC or decision-level specificity over pose-free and set-transformer baselines at matched compute. Null results reported as evidence on sweep geometry recoverability at this scale.

## Data

- **Stanford AIMI:** 192 nodules (175 benign, 17 malignant), 17,412 frames, segmentations, TI-RADS, histopathology.
- **ThyroidXL:** ~11.5K images / 4,093 patients, pathology-validated external benchmark.
- **Unlabeled US corpora:** OpenUS-46 and additional cine for Track 2 pretraining.
