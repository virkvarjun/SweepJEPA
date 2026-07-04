# Build Status & Missing Inputs

Scaffolding for both tracks is **implemented and tested end-to-end against
synthetic data** (`PYTHONPATH=. pytest` → all green; scripts run with `--fast`).
This file tracks everything that still needs a **real input from you** (data,
weights, credentials) and the implementation refinements deliberately deferred.

Legend: ✅ done · 🟡 works on synthetic, needs real input to be "real" · ⏳ deferred refinement

---

## 1. Real inputs you need to provide

### Datasets (gated / large — code has synthetic fallbacks)
| Dataset | Role | How to get it | Where it plugs in |
|---|---|---|---|
| 🟡 **Stanford AIMI Thyroid Cine-clip** | Labeled train/CV (Paper 1) | Registration + DUA at aimi.stanford.edu | `data/stanford_aimi_thyroid/` + `manifest.csv` (cols: `nodule_id,patient_id,label,ti_rads,frame_dir`); set `data.synthetic: false` in `configs/track1.yaml`. Loader: `ThyroidCineDataset.from_manifest`. |
| 🟡 **ThyroidXL** (MICCAI 2025) | External per-image test | Challenge access request | `data/thyroidxl/`; wire a real static-image dataset into `build_external_static_dataset` (currently synthetic). Certification pipeline already handles the large positive pool. |
| 🟡 **TUS-REC2024** | Train R_psi (open!) | Zenodo (`zenodo_get <record_id>`); `scripts/download_datasets.sh tusrec` | `data/tusrec2024/*.npz` with `frames (N,H,W|N,3,H,W)` + `poses (N,4,4)`. Loader: `TUSRECSweepDataset`; auto-detected by `load_tusrec_or_synthetic`. **Start here — it's the only open set.** |
| 🟡 **OpenUS-46 / unlabeled cine** | JEPA pretraining corpus | Open download | Feed clips into `build_clip_tubelets` in `pretrain_jepa.py` (currently synthetic clips). |

### Model weights (gated — code builds correct arch + random-inits offline)
| Weights | Used by | Notes |
|---|---|---|
| 🟡 **USFM / USF-MAE / UltraFedFM** ViT-B/16 | `USEncoder` backbone | Set `encoder.weights_path` (or env `USFM_WEIGHTS` / `USF_MAE_WEIGHTS` / `ULTRAFEDFM_WEIGHTS`). `USEncoder.load_info.source` reports whether real weights loaded vs `random_init`. Until then all AUCs are ~chance by construction. |
| 🟡 **SALT / frozen teacher** | JEPA `teacher_type="frozen"` ablation | Currently the frozen teacher is a random-init clone. For the true US-JEPA-style frozen-teacher arm, load a pretrained encoder into `target_encoder`. |

### Credentials / services
| Item | Used by | Notes |
|---|---|---|
| 🟡 **Weights & Biases API key** | `train_track1.py` logging | `logging.wandb: true` in config + `wandb login`. Import is guarded; off by default. (You said you'll wire this later.) |

---

## 2. Milestone status

| Milestone | Status | Deliverables |
|---|---|---|
| **M1** — backbones + LoRA + loaders | ✅ | `USEncoder` (timm ViT + CNN, frozen/lora/full), synthetic data, patient-level splits, cine loader, smoke test. |
| **M2** — Track 1 CV + conformal + eval (Paper 1) | ✅ | focal loss, RCPS Clopper–Pearson decision, DeLong AUC CI + decision metrics, mask-aware MIL variants, CV pipeline, `train_track1.py` / `eval.py` / `run_ablations.py`. External ThyroidXL certification demonstrated. |
| **M3** — R_psi + reconstruction | ✅ | `PoseEstimator` (rot6d/axis-angle), SE(3)/cycle/smoothness losses, TUS-REC loader, pose graph + tubelet partition + pseudo-volume, `train_pose.py`. |
| **M4** — pose-conditioned JEPA + ablations | ✅ | Transformer JEPA with tubelet masking, EMA/frozen teacher, pose-conditioned & latent/input toggles, `pretrain_jepa.py`. |
| **M5** — geometry-aware head + comparison (Paper 2) | ✅ | Stage C→D plumbing (`stage_c.py`), geometry-aware embeddings feed the unchanged Track 1 head, `run_track2_comparison.py` with the pre-registered check + honest null reporting. |

---

## 3. Deferred implementation refinements (⏳ — work on synthetic, sharpen for real runs)

- ⏳ **Full spatio-temporal tubelets.** `partition_temporal_tubelets` groups *pooled* per-frame features over time only. True V-JEPA tubelets need spatial patch tokens (US-FM last-layer patch grid) × time. Requires exposing patch tokens from `USEncoder` (currently returns pooled `(B,N,D)`).
- ⏳ **Variable-length JEPA batching.** `stack_tubelets` requires equal tubelet counts; real clips vary in length. Add padding + a padding-aware attention/loss mask (mirror `BagBatch`).
- ⏳ **Real pseudo-volume voxelization.** `build_pseudo_volume` returns frame-plane corners/extent (releasable pose graph). A resampled 3D voxel grid (scan-conversion of frame planes) is the fuller Stage B artifact for release.
- ⏳ **R_psi → Stage C wiring on real data.** `build_clip_tubelets` accepts a trained `PoseEstimator` but defaults to synthetic smooth poses. On real cine, pass the R_psi trained by `train_pose.py`.
- ⏳ **Matched-compute accounting.** Comparison matches the *backbone*; Track 2 adds JEPA pretraining compute. For the paper, log FLOPs/params per arm so "matched compute" is quantified, and add the US-JEPA (image-level, sweep-discarding) baseline arm explicitly.
- ⏳ **Label-efficiency sweep.** `run_track2_comparison.py` has the arm hooks; add the fraction-of-labels loop (subsample patients) for the label-efficiency curve.
- ⏳ **peft LoRA backend.** Built-in minimal LoRA is used; `peft` is an optional alternative (commented in `requirements.txt`).
- ⏳ **Docker GPU.** `Dockerfile` is CPU/CPU-index; add CUDA torch index + GPU base for training at scale.
- ⏳ **Environment.** Local dev is Python **3.9** (system) in `.venv --system-site-packages` reusing torch 2.8; numpy pinned `<2`. Dockerfile targets 3.10. Real training wants GPU + the pinned US-FM repo deps.

---

## 4. How to go from synthetic → real (quick recipe)

1. `scripts/download_datasets.sh tusrec` → train R_psi: `python scripts/train_pose.py`.
2. Obtain a US-FM checkpoint → set `encoder.weights_path`; confirm `load_info.source` ≠ `random_init` via `scripts/smoke_test_backbones.py`.
3. Stanford AIMI DUA → drop frames + `manifest.csv`, set `data.synthetic: false` → `python scripts/train_track1.py` (Paper 1).
4. ThyroidXL → wire real static loader → `python scripts/eval.py` (external certification).
5. OpenUS-46 → `python scripts/pretrain_jepa.py` → `python scripts/run_track2_comparison.py` (Paper 2).
