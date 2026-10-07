# AutoDL Run Order

This is a clean-room behavioral reproduction. Run it from the repository root. It does not read author code or model packages.

## 0. Copy the directory

Copy `first_place_cleanroom/` and the repository package root to `/root/autodl-tmp/rogii-reproduction/`, or clone/copy the entire workspace there. The examples assume:

```text
/root/autodl-tmp/rogii-reproduction/first_place_cleanroom
/root/autodl-tmp/rogii-wellbore-geology-prediction/train
/root/autodl-tmp/rogii-wellbore-geology-prediction/test
```

## 1. Environment and representation gate

Run these in a shell, not in a Python notebook cell:

```bash
cd /root/autodl-tmp/rogii-reproduction
pip install -r first_place_cleanroom/requirements.txt
python -m first_place_cleanroom.audit \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --version 0719_V1 --model-forward
python -m first_place_cleanroom.contract_audit \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --model-forward
python -m first_place_cleanroom.simulation_audit \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --wells 128
```

Expected invariants: input `(16,400,345)`, output `(1,400,345)`, finite forward,
perfect loss below shifted loss, a passing public-contract audit, and a passing
real-versus-simulated TVT-increment distribution audit.

## 2. Build fold-clean PF and XY caches

```bash
python -m first_place_cleanroom.prepare \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --output-root /root/autodl-tmp/first_place_cleanroom/cache \
  --component all --split both --workers 8
```

This uses 500 particles, 128 seeds, 64-row updates and FFBSi for every well. `folds.csv` uses 20 geo clusters (KMeans seed 0) and stratified five-fold assignment (seed 7), reconstructed from mean well XY because the original mapping file is unavailable. Train XY caches are materialized separately for every outer fold; both training and validation queries use only that outer fold's training-well support bank, and a training query is excluded from its own support.

Then verify the generated folds and cache contract:

```bash
python -m first_place_cleanroom.contract_audit \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --cache-root /root/autodl-tmp/first_place_cleanroom/cache
```

## 3. One unchanged-semantics program check

This check reduces wells and epochs only. It preserves the 400x345 grid, full model,
version-specific training batch size, BN running statistics, BF16, loss, PF and XY channels.

```bash
python -m first_place_cleanroom.train \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --cache-root /root/autodl-tmp/first_place_cleanroom/cache \
  --output-root /root/autodl-tmp/first_place_cleanroom/program_check \
  --version 0803_V2 --fold 1 --seed 0 --epochs 2 \
  --limit-train 32 --limit-valid 16
```

Do not interpret its RMSE as reference CV. Its purpose is only forward/backward/checkpoint validation.

## 4. Train the complete model bank

The public system contains seven versions, five folds and three seeds: 105 members. Run one version/fold/seed per command, or use the resumable orchestrator:

```bash
python -m first_place_cleanroom.run_all \
  --stage train \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --work-root /root/autodl-tmp/first_place_cleanroom
```

The orchestrator skips an existing `best.pt`. Training requires BF16 CUDA and refuses CPU fallback. Batch size remains 16.

## 5. Inference and fixed routing

Only after all 105 checkpoints exist:

```bash
python -m first_place_cleanroom.run_all \
  --stage infer \
  --data-root /root/autodl-tmp/rogii-wellbore-geology-prediction \
  --work-root /root/autodl-tmp/first_place_cleanroom \
  --submission /root/autodl-tmp/first_place_cleanroom/submission.csv
```

Inference averages 15 members per version, applies the disclosed Savitzky-Golay settings,
uses fixed published ensemble weights, and switches to the no-XY ensemble when any of
the four disclosed quality thresholds fail.

## Metric gate

The reported reference OOF is 4.627. This implementation has not earned that metric locally. If one full fold remains far above the reported per-model range (roughly 4.8-5.5), stop and audit coordinate direction, calibration/normalization and inferred augmentation/PF profile values before training the remaining folds.
