# Reproduction Report

**Reproduction level:** behavioral reproduction

**Exact/source-derived components:** 400x345 alignment geometry; public channel tuples for seven versions; ConvNeXt-small dimensions and depths; BN replacement; decoder width configuration; exponential target; CE/Huber/GR weights; BF16 batch-16 AdamW/cosine/EMA training; 500-particle, 128-seed PF shape; published XY WLS parameters; five folds and three seeds; SG decoding; final weights and XY hard gate.

**Inferred components:** visible-GR blend coefficient; fixed physical channel normalization; exact custom decoder wiring; Huber delta; TVT_bias_v2 window conversion; PF profile transition/jump values; FFBSi trajectory sample count; singular XY fallback; and parts of the internal synthetic-GR mixture whose logged parameters do not disclose control flow.

**Unavailable author artifacts:** original `train_png_typewell_map.csv` coordinate mapping and undisclosed exact implementation details. They are not left as absent runtime modules: a 20-cluster geo-stratified reconstruction from mean well XY and explicit isolated approximations are used.

**Intentional deviations:** none to disclosed model shape, batch normalization, precision, batch size, loss weights, PF population, fold/seed count, ensemble weights, or router thresholds. Program checks may reduce well count and epochs only and are never labeled CV.

**Validation performed:** syntax/import; real-well panel construction; finite-value and shape assertions; target normalization; perfect-vs-shifted loss; model forward and backward; full 128x500 one-well PF with 64-row updates; FFBSi normalization; outer-fold-specific XY generation and loading; all seven channel tuples under forced Z-shift; PF/XY remapping into simulated coordinates; geo fold balance; and real-vs-simulated target-distribution audit.

**Reference metric vs observed metric:** reference final OOF 4.627. No clean-room GPU fold has yet been trained in this local CPU environment, so no observed comparable CV is claimed.

**Next blocking requirement:** run the unchanged-semantics AutoDL program check, then one complete fold. If its per-model RMSE remains far outside the reported 4.8-5.5 range, audit inferred representation details before launching the remaining 104 members.
