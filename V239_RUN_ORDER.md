# V239 AutoDL Run Order

Upload `ninth_place_wholewell_unet_v239.py` to `/root/autodl-tmp/`.

## 1. Representation audit

This loads 48 wells and does not train a model.

```python
import os

os.environ["V239_MODE"] = "audit"
os.environ["V239_VALID_FOLD"] = "1"
%run /root/autodl-tmp/ninth_place_wholewell_unet_v239.py
```

Required guards:

- `input_shape == [9, 384, 768]`
- `mean_future_axis_coverage >= 0.995`
- `p05_future_axis_coverage >= 0.98`

Do not train if either coverage guard fails.

## 2. Honest smoke training

This trains on 96 wells from folds 2-5 and validates on 24 unseen fold-1
wells. It is a capacity/mechanics gateway, not a final CV estimate.

```python
import os

os.environ["V239_MODE"] = "smoke"
os.environ["V239_VALID_FOLD"] = "1"
os.environ["V239_TRAIN_WELLS"] = "96"
os.environ["V239_VALID_WELLS"] = "24"
os.environ["V239_EPOCHS"] = "4"
os.environ["V239_ACCUMULATION"] = "4"
%run /root/autodl-tmp/ninth_place_wholewell_unet_v239.py
```

The first epoch may be weak. Continue only if validation RMSE improves across
epochs and reaches below 12 ft by epoch 4. A flat or worsening curve indicates
a representation/training bug and should be diagnosed before scaling.

## 3. Complete one-fold baseline

Run only after the smoke mechanics pass.

```python
import os

os.environ["V239_MODE"] = "full"
os.environ["V239_VALID_FOLD"] = "1"
os.environ["V239_EPOCHS"] = "18"
os.environ["V239_ACCUMULATION"] = "4"
%run /root/autodl-tmp/ninth_place_wholewell_unet_v239.py
```

This stage is the stock whole-well baseline. Path-relief, thickness-warp,
entropy-weighted TTA, additional folds, and seed averaging are intentionally
excluded until the stock model is measured.

