# SourceInverse Project Status

Last updated: 2026-06-06

## Purpose

This file is the shared project memory for the SourceInverse workspace. After each working conversation, append a concise record of:

- what the user asked
- what Codex inspected or changed
- important reasoning and decisions
- validation results
- unresolved problems
- next steps

The goal is to preserve context across sessions without relying only on chat history.

## Project Goal

Use a physics-informed neural network (PINN) to infer an air-pollution source from station concentration observations, wind data, and station coordinates. The workflow estimates:

- source location
- time-dependent source strength `Q(t)`
- concentration plume field
- source uncertainty / confidence landscape

The output should be physically plausible, not only low-loss. In particular, generated concentration fields and `diffusion.gif` animations must be checked for reasonable plume direction, shape, timing, and station peak behavior.

## Current Code Architecture

- `pinn_source/pinn_source_pinn.py`: main entry point; calls `pipeline.run(...)`.
- `pinn_source/config.py`: global paths, training settings, loss weights, source/Q/plume parameters, diagnostics settings.
- `pinn_source/pipeline.py`: data loading, preprocessing, training loop, diagnostics, quality report, source landscape, plotting and animation orchestration.
- `pinn_source/models/pinn.py`: PINN model, source parameters, diffusion/source-strength parameters, `plume_net`, background network, and `Q(t)` implementations.
- `pinn_source/field.py`: concentration field construction using source-aligned coordinates, `source_gate`, plume strength, and `Q(t)`.
- `pinn_source/q_parameterization.py`: `neural`, `piecewise`, and `event_neural` source strength parameterization plus CSV export.
- `pinn_source/source_landscape.py`: fixed-source loss scan and source probability/confidence outputs.
- `pinn_source/viz.py`: station/source plots, station time-series plots, and `diffusion.gif` animation generation.

## Current Key Parameters

As of the latest inspected state:

- `FIELD_MODE = "no_background"`
- `TRAIN_ON_RESIDUAL = True`
- `BASELINE_MODE = "median"`
- `ENABLE_EVENT_WINDOW_CROP = True`
- `Q_MODE = "piecewise"`
- `Q_SEGMENT_LENGTH = 3`
- `Q_SMOOTH_WEIGHT = 0.01`
- `Q_L2_WEIGHT = 0.001`
- `PLUME_MAX = None`
- `SOURCE_POSITION_PAD_M = 0.0`
- `WIND_DIR_IS_FROM = True`
- `WIND_SCALE = 10.0`
- `D_MIN_PHYS = 500.0`
- `USE_SOURCE_LANDSCAPE_CONFIDENCE = True`

## Completed Changes So Far

- Added best checkpoint restore: after early stopping or final training, the model restores the epoch with the best `raw_loss`.
- Added `q_parameterization.py`.
- Added `Q_MODE = "event_neural"` experiment, then reverted default behavior after poor plume results.
- Current default uses `Q_MODE = "piecewise"` with 3-step segments.
- Removed bounded plume default by setting `PLUME_MAX = None` after bounded plume caused block-like saturation.
- Tightened source candidate domain with `SOURCE_POSITION_PAD_M = 0.0` to prevent source estimates from sticking to an expanded outer boundary.
- Added automatic quality report output: `result_quality_report.json`.
- Added/kept diagnostics:
  - `q_time_series.csv`
  - `q_segments.csv`
  - `station_peak_diagnostics.csv`
  - `training_diagnostics.csv`
  - `source_loss_landscape.csv`
  - `source_probability_map.csv`

## Latest Known Validation Results

Latest inspected output directory:

- `result/source_uncertainty`

Known latest quality report:

- source x/y: approximately `(-963.96 m, 1203.03 m)`
- source lat/lon: approximately `(30.7278458, 121.2831448)`
- raw RMSE: approximately `3.5878`
- training source and source-landscape best point are close: about `34.8 m`
- quality status: not reasonable
- warning: one or more high-value station peaks are badly missed

Known latest `q_segments.csv` issue:

- `2026-01-19 18:00:00` to `2026-01-19 20:00:00` has a large isolated Q spike around `0.1593`.
- Neighboring segments are much lower, around `0.0181` and `0.0280`.

Known latest station peak issue:

- `上石化园区卫四路站` is badly missed.
- Observed peak time: `2026-01-19 21:00:00`.
- Predicted peak time: `2026-01-20 09:00:00`.
- Peak timing error: about `12 h`.
- Predicted value at observed fit peak is near zero relative to observed peak.

Known latest internal plume issue:

- `training_diagnostics.csv` showed very large `plume_max` values, exceeding `100` near the end of training.
- This suggests the model may be using compensation between `plume_net`, `gate`, and `Q(t)` rather than learning a physically stable plume field.

## Unresolved Problems

- Concentration field and plume animation are still physically wrong.
- Total RMSE can look acceptable while high-value station peaks and plume shape remain unacceptable.
- `piecewise Q` still creates hard source-strength jumps.
- `plume_net` has too much freedom and can grow very large internally.
- Current diagnostics identify bad outputs but do not yet prevent the model from producing them.

## Current Working Hypothesis

The main problem is no longer only source location. The source estimate is currently consistent with the loss landscape, but the concentration field is unstable because:

- `Q(t)` can create sharp artificial time jumps.
- `plume_net` is underconstrained.
- `gate * plume * Q` allows compensation: one component can become very large while another suppresses it.
- A low numerical loss can still produce an implausible plume animation.

The likely next direction is to reduce plume-field freedom and use a more physical plume backbone or stronger shape constraints, while replacing hard piecewise `Q(t)` with a smoother continuous parameterization.

## Next-Step Plan

Before making further model changes:

1. Inspect the latest output files first:
   - `result_quality_report.json`
   - `station_peak_diagnostics.csv`
   - `training_diagnostics.csv`
   - `q_segments.csv`
   - `q_time_series.csv`
   - `diffusion.gif`
2. Judge result quality from both diagnostics and visualization, not from loss alone.
3. Identify whether the failure is primarily:
   - source location
   - wind direction convention
   - Q timing
   - plume shape
   - baseline/residual preprocessing
   - visualization scaling
4. Then implement one controlled change at a time and rerun at least a smoke test plus diagnostic inspection.

## Conversation Log

### 2026-06-06 Event Source Position Optimization

User asked Codex to continue analyzing the newest log/output files and optimize again.

Files inspected:

- attached run log
- `result/source_uncertainty/result_quality_report.json`
- `result/source_uncertainty/station_peak_diagnostics.csv`
- `result/source_uncertainty/q_time_series.csv`
- `result/source_uncertainty/training_diagnostics.csv`
- `result/source_uncertainty/source_confidence_landscape.json`

Findings:

- The previous peak-amplitude/source-interior version improved the first event but sacrificed the second:
  - `上石化园区卫四路站` recovered strongly (`pred_at_obs_peak_fit_ratio≈0.80`).
  - `上石化边界卫六路站` partially recovered (`≈0.58`).
  - `二工区边界新联站` collapsed to near zero.
- This confirmed that one fixed source location is not expressive enough for the two observed event structures.
- The model was alternating between fitting the 01-19 event and the 01-20 event depending on loss pressure.

Implemented changes:

- Added event-level source positions:
  - `SOURCE_POSITION_MODE = "event"`
  - detected event blocks each get learnable source coordinates
  - fallback to the old single-source behavior remains available with `SOURCE_POSITION_MODE = "single"`
- Updated `PINN` with:
  - `configure_event_sources(...)`
  - `source_xy(t)`
  - event-specific source segment parameters
- Updated `field.py` so source-aligned coordinates, gate, and plume features use the source corresponding to each sample time.
- Updated `pipeline.py` so PDE source term, residual source weighting, high-downwind loss, axis loss, source-local loss, source-interior penalty, diagnostics, and collocation resampling all support event source positions.
- Added `event_source_positions.csv` output.
- Added event-source details to `result_quality_report.json`.
- Skipped single-source confidence landscape when event source positions are enabled, because the old landscape is no longer semantically valid for multiple event sources.

Short validation:

- `py_compile` passed for `config.py`, `models/pinn.py`, `field.py`, `pipeline.py`, and `q_parameterization.py`.
- 120-epoch event-source smoke test passed.
- 500-epoch event-source short run showed the new structure can fit both major peaks:
  - `上石化园区卫四路站` observed peak ratio about `0.995`.
  - `二工区边界新联站` observed peak ratio about `1.09`.
  - However, RMSE remained high because non-target stations were overpredicted.

Additional optimization:

- Added low-station false-positive suppression:
  - `LOSS_W_LOW_FALSE_POSITIVE = 1.5`
  - `LOW_FALSE_POSITIVE_RATIO = 0.35`
  - `LOW_FALSE_POSITIVE_MARGIN = 0.03`
  - `STAGE1_LOW_FALSE_POSITIVE_MULT = 2.0`
  - `BEST_SCORE_LOW_FALSE_POSITIVE_WEIGHT = 1.0`
- This loss penalizes stations that should remain low during anomalous timestamps when the model raises them with the plume.
- Added diagnostics:
  - `low_false_positive_loss`
  - `low_false_positive_event_count`
  - `low_false_positive_mult`

Validation after false-positive loss:

- `py_compile` passed.
- 120-epoch smoke test passed in `result/smoke_event_sources_low_fp_120ep/run_smoke`.
- New diagnostics and `event_source_positions.csv` are written.

Next required validation:

- Run full default training.
- Inspect:
  - `event_source_positions.csv`
  - `station_peak_diagnostics.csv`
  - `q_time_series.csv`
  - `training_diagnostics.csv`
  - `result_quality_report.json`
  - `diffusion.gif`
- Success criteria:
  - Both `上石化园区卫四路站` and `二工区边界新联站` retain peak ratios near 1.
  - `上石化边界卫六路站` remains reasonably fitted.
  - low/non-target stations such as 抚佳 and 亚南 no longer develop large false peaks.
  - event source positions stay away from boundaries and make physical sense.
  - animation shows two event plumes instead of one fixed-source compromise.

### 2026-06-06 Analysis and Optimization After Smooth-Time Full Run

User asked Codex to analyze the new full-run log and output files, then optimize the code.

Files inspected:

- attached run log
- `result/source_uncertainty/result_quality_report.json`
- `result/source_uncertainty/station_peak_diagnostics.csv`
- `result/source_uncertainty/q_time_series.csv`
- `result/source_uncertainty/training_diagnostics.csv`
- `result/source_uncertainty/source_confidence_landscape.json`
- `result/source_uncertainty/diffusion.gif`

Findings from the smooth-time + plume-penalty full run:

- The anti-compensation plume penalty helped: latest `plume_max` dropped from the previous `~118` failure mode to about `41`.
- The result is still not acceptable: raw RMSE is about `4.07`, and quality report still flags high-value peak misses.
- Source estimate and loss landscape are still internally consistent, with source-landscape distance about `45 m`.
- `q_time_series.csv` no longer has a hard isolated 18:00-20:00 spike, but Q became almost monotonic increasing toward the 01-20 event.
- This means the model still favors the second event (`二工区边界新联站`) and sacrifices the first event (`上石化园区卫四路站` / `上石化边界卫六路站`).
- Station diagnostics after the full run:
  - `二工区边界新联站` remained fitted reasonably (`pred_at_obs_peak_fit_ratio≈0.87`).
  - `上石化园区卫四路站` was still essentially missed at its observed peak (`pred_at_obs_peak_fit_ratio≈0.00016`).
  - `上石化边界卫六路站` dropped to about `0.31` at observed peak.
- A 500-epoch short test after adding direct peak-amplitude loss showed a new side effect: the source moved to the northern boundary, so source-boundary control must also participate in training and checkpoint selection.

Implemented optimization:

- Added direct high-peak amplitude fitting:
  - `LOSS_W_PEAK_AMPLITUDE = 3.0`
  - `PEAK_AMPLITUDE_RATIO = 0.5`
  - `STAGE1_PEAK_AMPLITUDE_MULT = 4.0`
- The new loss directly compares predicted and observed amplitudes for high stations at each anomalous timestamp, instead of only enforcing ranking/shape.
- Changed best checkpoint selection from raw-loss-only to a quality score:
  - `quality_score = raw_loss + BEST_SCORE_DATA_WEIGHT * data_loss + BEST_SCORE_PEAK_WEIGHT * peak_amplitude_loss + BEST_SCORE_SOURCE_BOUNDARY_WEIGHT * source_interior_loss`
  - `BEST_SCORE_DATA_WEIGHT = 0.5`
  - `BEST_SCORE_PEAK_WEIGHT = 2.0`
  - `BEST_SCORE_SOURCE_BOUNDARY_WEIGHT = 2.0`
- Added soft source-domain interior penalty:
  - `SOURCE_INTERIOR_MARGIN_M = 250.0`
  - `LOSS_W_SOURCE_INTERIOR = 2.0`
- Added diagnostics columns:
  - `peak_amplitude_loss`
  - `peak_amplitude_event_count`
  - `peak_amplitude_mult`
  - `quality_score`
  - `source_interior_loss`

Validation:

- `py_compile` passed for `config.py`, `pipeline.py`, `models/pinn.py`, and `q_parameterization.py`.
- 120-epoch smoke test passed in `result/smoke_peak_amp_source_interior_120ep/run_smoke`.
- Diagnostics confirmed new columns are written.
- The smoke test is not expected to be a good fit; it only validates the new training path.

Next required validation:

- Run full default training again.
- Inspect whether:
  - `上石化园区卫四路站` at `2026-01-19 21:00` improves substantially.
  - `上石化边界卫六路站` at `2026-01-19 20:00` recovers.
  - `二工区边界新联站` remains acceptable.
  - source no longer sticks to the source-domain boundary.
  - `q_time_series.csv` has two-event structure rather than a monotonic ramp.
  - `diffusion.gif` no longer shows source-near blob dominance.

### 2026-06-06 Code Change: Smooth Q and Plume Compensation Penalty

User asked Codex to directly modify the code after the latest run analysis.

Files changed:

- `pinn_source/config.py`
- `pinn_source/models/pinn.py`
- `pinn_source/q_parameterization.py`
- `pinn_source/pipeline.py`

Implemented changes:

- Changed default `Q_MODE` from hard `piecewise` segments to `smooth_time`.
- Added `smooth_time` Q parameterization:
  - one learnable `logQ` node per unique training timestamp
  - linear interpolation between timestamp nodes
  - second-difference curvature regularization through `Q_SMOOTH_WEIGHT`
- Increased default `Q_SMOOTH_WEIGHT` to `0.2` for the new smooth-time curvature penalty.
- Added soft plume regularization:
  - `PLUME_L2_WEIGHT = 5e-4`
  - `PLUME_EXCESS_WEIGHT = 0.01`
  - `PLUME_SOFT_MAX = 30.0`
- Added plume regularization terms to training loss and adaptive-loss path.
- Evaluated plume penalty on both observation points and PDE collocation points, so hidden high-value plume patches in the field are discouraged.
- Added diagnostics columns:
  - `plume_l2_loss`
  - `plume_excess_loss`
  - `Q_mean_observation`
- Extended `result_quality_report.json` with field component summaries:
  - plume mean/max
  - Q mean/max
  - gate mean/max
  - source-term mean/max
- Added quality warning when learned plume factor exceeds `2 * PLUME_SOFT_MAX`.

Validation:

- `py_compile` passed for:
  - `pinn_source/config.py`
  - `pinn_source/models/pinn.py`
  - `pinn_source/q_parameterization.py`
  - `pinn_source/pipeline.py`
- 1 epoch smoke test passed in `result/smoke_smooth_time_plume_penalty/run_smoke`.
- 80 epoch short run passed in `result/smoke_smooth_time_plume_penalty_80ep/run_smoke`.
- Short run is not expected to fit well, but it confirmed the new anti-compensation mechanism is active:
  - plume max stayed around `5.5`, not tens or hundreds
  - Q time series remained continuous and did not reproduce the old hard 18:00-20:00 isolated spike
  - quality report still correctly warns that the short run is not a good final result

Next required validation:

- Run the full default training after this change.
- Inspect `q_time_series.csv`, `station_peak_diagnostics.csv`, `training_diagnostics.csv`, `result_quality_report.json`, and `diffusion.gif`.
- Pay special attention to:
  - whether `上石化园区卫四路站` at `2026-01-19 21:00` recovers
  - whether `二工区边界新联站` remains fitted
  - whether `plume_max` stays controlled
  - whether the animation no longer shows broad source-near saturated patches or narrow vertical streaks

### 2026-06-06 Analysis of Latest Full Run

User asked Codex to inspect the latest log and output files and identify what still needs to be changed.

Files inspected:

- `project_status.md`
- attached pasted run log
- `result/source_uncertainty/result_quality_report.json`
- `result/source_uncertainty/station_peak_diagnostics.csv`
- `result/source_uncertainty/q_segments.csv`
- `result/source_uncertainty/q_time_series.csv`
- `result/source_uncertainty/training_diagnostics.csv`
- `result/source_uncertainty/source_confidence_landscape.json`
- `result/source_uncertainty/diffusion.gif`, sampled into a contact sheet for visual inspection
- input Excel data for event-window wind and concentration values

Key findings:

- The latest run is still not reasonable even though raw RMSE is about `3.59`.
- Source estimate is internally consistent with the source landscape: training source and landscape best differ by only about `35 m`.
- The main failure is field/Q/plume behavior rather than source-boundary drift.
- `station_peak_diagnostics.csv` shows `上石化园区卫四路站` is badly missed: observed peak at `2026-01-19 21:00`, predicted peak at `2026-01-20 09:00`, with near-zero prediction at the observed peak.
- `q_segments.csv` shows an isolated Q spike at `2026-01-19 18:00` to `20:00` (`Q≈0.159`) followed by a sharp drop at `21:00` to `23:00` (`Q≈0.028`), which directly conflicts with the observed `21:00` 卫四路 peak.
- Training log shows data loss plateaus around `1.21` after early training while PDE loss continues decreasing. Raw loss improvement is therefore mostly PDE-driven, not better peak fitting.
- Internal plume magnitude grows throughout training: `plume_max` rises from about `28` at epoch 500 to about `118` at epoch 5000, while Q mean falls strongly. This indicates compensation between `plume_net`, `gate`, and `Q(t)`.
- The animation shows a broad saturated-looking source-near high-value patch plus a narrow vertical streak, not a clean physically plausible wind-driven plume.
- The observed data likely contains two event structures: `2026-01-19 12:00` to `21:00` mainly 卫四路/卫六路 high, and `2026-01-20 01:00` to `09:00` mainly 新联站 high.

Working conclusion:

- Do not keep tuning only `Q_SEGMENT_LENGTH`.
- The next fix should reduce compensation freedom in `gate * plume_net * Q(t)`.
- Priority changes should target smoother Q, better event handling, and stronger physical plume shape constraints.

### 2026-06-06

User requested creation of this project memory file:

> 在这个项目中生成一个project_status.md，我们每次对话后把我们的对话和你思考的记录保存在这个md文件中，用于保存上下文

Action taken:

- Created `project_status.md` at the project root.
- Seeded it with current project goal, architecture, known modifications, latest inspected validation results, unresolved problems, current hypothesis, and next-step plan.

Codex reasoning:

- The file should be compact but useful as a durable handoff document.
- It should preserve both factual project state and reasoning assumptions.
- Future conversations should append new entries under `Conversation Log` instead of rewriting history, unless the user explicitly asks to reorganize it.

### 2026-06-06 Conservative Rollback After Field Collapse

User feedback:

- The latest modified version is worse than the earlier baseline.
- The concentration field appears to be fitting input observations directly.
- The animation no longer shows a physically plausible plume from the inferred source point.
- Performance regression is unacceptable; if needed, changes should be rolled back.

Codex reasoning:

- The event-source and extra peak/false-positive loss direction added too much freedom.
- It can fit station peaks while damaging the physical field shape, especially by letting source position, plume shape, gate, and Q compensate for each other.
- The immediate priority is not another tuning layer. The priority is to restore a conservative default path that cannot silently use the rejected mechanisms.

Code actions:

- Restored default source-position mode to single source:
  - `SOURCE_POSITION_MODE = "single"`
  - `LOSS_W_SOURCE_INTERIOR = 0.0`
- Restored default Q behavior to the earlier piecewise baseline:
  - `Q_MODE = "piecewise"`
  - `Q_SEGMENT_LENGTH = 3`
  - `Q_SMOOTH_WEIGHT = 0.01`
- Disabled the experimental plume penalties by default:
  - `PLUME_L2_WEIGHT = 0.0`
  - `PLUME_EXCESS_WEIGHT = 0.0`
- Disabled direct peak-amplitude and low-station false-positive losses by default:
  - `LOSS_W_PEAK_AMPLITUDE = 0.0`
  - `LOSS_W_LOW_FALSE_POSITIVE = 0.0`
- Disabled extra best-checkpoint scoring terms by default, so checkpoint selection is again raw-loss based:
  - `BEST_SCORE_DATA_WEIGHT = 0.0`
  - `BEST_SCORE_PEAK_WEIGHT = 0.0`
  - `BEST_SCORE_LOW_FALSE_POSITIVE_WEIGHT = 0.0`
  - `BEST_SCORE_SOURCE_BOUNDARY_WEIGHT = 0.0`
- Fixed single-source compatibility in `PINN.source_xy(t)`: when a time batch is provided, single-source mode now returns source coordinates expanded to the same batch shape.

Validation:

- `py_compile` passed for the modified source files.
- A 1 epoch smoke run passed at `result/smoke_safe_rollback/run_smoke`.
- Smoke log confirmed:
  - `Q mode: piecewise`
  - `Source position mode: single`
  - restored best checkpoint uses `quality_score = raw_loss`

Important note:

- This is a safe default rollback, not a destructive git reset. Experimental code paths still exist for later controlled ablation, but they are disabled in the default run.
- Next full validation should compare this conservative default against the last acceptable pre-collapse output before adding any new physics or loss terms.

### 2026-06-06 Parameter Explanation: Peak and Low False Positive Losses

User asked what these config parameters do:

- `LOSS_W_PEAK_AMPLITUDE`
- `LOW_FALSE_POSITIVE_RATIO`
- `LOSS_W_LOW_FALSE_POSITIVE`
- `LOW_FALSE_POSITIVE_MARGIN`

Current interpretation:

- These are observation-fitting auxiliary losses, not core PDE physics losses.
- `LOSS_W_PEAK_AMPLITUDE` directly penalizes under/over prediction at stations that are high within an anomalous timestamp.
- `LOSS_W_LOW_FALSE_POSITIVE` penalizes predicted concentration at stations that are observed low while another station is high at the same timestamp.
- `LOW_FALSE_POSITIVE_RATIO` defines which stations count as observed-low relative to the timestamp maximum.
- `LOW_FALSE_POSITIVE_MARGIN` allows a small tolerated excess above the observed-low value before applying the false-positive penalty.
- These terms are currently disabled by default because previous runs showed that adding observation-fitting freedom can improve station metrics while damaging plume physical shape.

### 2026-06-06 Removed Peak/Low False Positive Auxiliary Losses

User instruction:

- Delete the loss-related changes for peak-amplitude fitting and low false-positive suppression.
- Remember the project preference: avoid adding new loss terms during future optimization unless clearly necessary.
- Current training already has many loss terms with overlapping behavior, and adding more loss terms made the system harder to reason about.

Code actions:

- Removed these config parameters:
  - `LOSS_W_PEAK_AMPLITUDE`
  - `PEAK_AMPLITUDE_RATIO`
  - `LOSS_W_LOW_FALSE_POSITIVE`
  - `LOW_FALSE_POSITIVE_RATIO`
  - `LOW_FALSE_POSITIVE_MARGIN`
  - `STAGE1_PEAK_AMPLITUDE_MULT`
  - `STAGE1_LOW_FALSE_POSITIVE_MULT`
  - `BEST_SCORE_PEAK_WEIGHT`
  - `BEST_SCORE_LOW_FALSE_POSITIVE_WEIGHT`
- Removed the corresponding imports from `pipeline.py`.
- Removed peak-amplitude and low false-positive loss calculation blocks from the training loop.
- Removed these terms from `raw_loss`, adaptive-loss composition, checkpoint quality score, debug logging, diagnostics CSV rows, and final result JSON.

Validation:

- Source search confirmed no remaining references under `pinn_source/` for:
  - `PEAK_AMPLITUDE`
  - `LOW_FALSE_POSITIVE`
  - `peak_amplitude`
  - `low_false_positive`
  - `low_fp`
  - `peak_amp`
- `py_compile` passed for the modified source files.
- A 1 epoch smoke run passed at `result/smoke_remove_peak_lowfp_losses/run_smoke`.

Future optimization rule:

- Prefer changing physical parameterization, source/Q constraints, data preprocessing, diagnostics, or rollback/ablation strategy before adding any new loss term.

### 2026-06-06 Redundant Code and File Cleanup

User instruction:

- Clean redundant code and files.
- Keep the codebase simpler after the failed optimization branch.

Code cleanup:

- Removed rejected event-source code paths:
  - `SOURCE_POSITION_MODE`
  - event source segment configuration
  - event source position export
  - event-source branches in `PINN.source_xy`
- Removed rejected Q experiments:
  - `smooth_time`
  - `event_neural`
  - related model methods and export logic
- Removed disabled loss-like experimental code:
  - source-interior soft loss
  - plume L2/excess penalties
  - extra best-checkpoint quality score terms
- Simplified best checkpoint selection back to raw loss only.
- Kept useful diagnostics:
  - `training_diagnostics.csv`
  - `station_peak_diagnostics.csv`
  - `result_quality_report.json`
  - source confidence landscape
  - `q_time_series.csv` and `q_segments.csv`

File cleanup:

- Deleted old helper/debug files:
  - `pinn_source/_patch.ps1`
  - `pinn_source/_debug_cols.py`
- Deleted temporary smoke result directories under `result/smoke_*`.
- Deleted project Python cache directories under:
  - `pinn_source/__pycache__`
  - `pinn_source/models/__pycache__`
  - `data/__pycache__`
- Did not delete formal output under `result/source_uncertainty`.
- Did not touch data files or virtual environments.

Validation:

- `py_compile` passed for the main source files.
- A 1 epoch smoke run passed after cleanup.
- The temporary smoke validation output was deleted after verification.
- Final smoke log confirmed:
  - `Q mode: piecewise`
  - `Source position mode: single`
  - best checkpoint restore reports raw loss only

### 2026-06-06 Result Output Ignore and Cleanup

User instruction:

- Some useless files under `result/` can be removed or added to git ignore.

Actions:

- Added generated outputs to `.gitignore`:
  - `result/`
  - `diffusion.gif`
- Removed untracked generated result directories:
  - `result/source_uncertainty`
  - `result/7shsh多高值`
- Kept tracked historical result folders `result/1` through `result/6` untouched.

Notes:

- `.gitignore` prevents future generated results from appearing as untracked files.
- Files already tracked by Git, such as `result/1` through `result/6` and root `diffusion.gif`, are not automatically ignored. Removing them from version control would need a separate tracked-file cleanup.

### 2026-06-07 Latest Result Analysis: Nonphysical Field and Plume Mutation

User feedback:

- The concentration field has abrupt changes.
- Plume morphology is abnormal.
- The result appears to fit observations rather than reconstruct a physically constrained plume.

Evidence inspected:

- Attached training log.
- `result/source_uncertainty/result_quality_report.json`
- `result/source_uncertainty/q_segments.csv`
- `result/source_uncertainty/station_peak_diagnostics.csv`
- `result/source_uncertainty/training_diagnostics.csv`
- `result/source_uncertainty/diffusion.gif`, sampled into a contact sheet.

Key findings:

- Source location is not the main failure: trained source and source-landscape best differ by about `28 m`.
- Final quality report marks result unreasonable:
  - `plume_max = 120.6`
  - `q_mean = 0.0509`
  - warnings include excessive plume factor and badly missed high-value peaks.
- Training shows compensation:
  - `plume_max` rises from about `27.9` at epoch 500 to `120.6` at epoch 5000.
  - `Q_mean_observation` falls from about `0.4285` to `0.0509`.
  - data loss stalls near `1.21`, while PDE loss falls from `50.1` to `0.0878`.
- `q_segments.csv` still has a sharp isolated Q spike at `2026-01-19 18:00` to `20:00`, followed by a drop at `21:00` to `23:00`.
- Station diagnostics show severe peak failures:
  - `上石化园区卫四路站` observed peak at `2026-01-19 21:00`, predicted peak at `2026-01-20 09:00`, prediction at observed peak is nearly zero.
  - `二工区边界新联站` is fitted much better, so the model sacrifices one event structure for another.
- GIF contact sheet shows broad saturated patches and thin streak-like structures rather than a plume smoothly emitted from the source.

Interpretation:

- The current field representation is too flexible:
  - `concentration = gate(source, wind) * (source_bias + plume_net(along, cross, t)) * Q(t)`.
  - `plume_net` can create arbitrary spatial-temporal shapes, while `Q(t)` and `gate` compensate.
  - The PDE residual can become small without forcing visually plausible plume morphology.
- The existing objective is still observation-heavy:
  - high observation weights are large.
  - stage 1 weakens PDE and strengthens data/top/multi-high fitting.
  - top-station and multi-high losses are still observation-fitting terms with overlapping purpose.

Recommended next direction:

- Do not add more losses.
- First run a conservative ablation that reduces observation-fitting pressure and removes staged PDE weakening.
- Then replace or bypass the free `plume_net` with a constrained analytic plume kernel so the field shape is physically parameterized rather than freely learned.

### 2026-06-07 Implemented Physical-First Analytic Plume Mode

User instruction:

- Apply the proposed changes to address abrupt concentration-field changes and abnormal plume morphology.

Code changes:

- Set default field mode to a constrained analytic plume:
  - `FIELD_MODE = "analytic_plume"`
- Added `analytic_plume_kernel(...)` in `field.py`.
  - The kernel is source-centered, wind-aligned, downwind-decaying, crosswind-Gaussian, and clamped to `[0, 1]`.
  - In this mode, `plume_net` no longer generates the main concentration field.
  - `source_term = (1 + source_bias) * analytic_plume_kernel * Q(t)`.
- Reduced observation-fitting pressure:
  - `DATA_HIGH_WEIGHT = 1.0`
  - `DATA_TIME_PEAK_WEIGHT = 1.0`
  - `LOSS_W_TOP_STATION = 0.0`
  - `LOSS_W_MULTI_HIGH = 0.0`
- Removed staged weak-PDE training behavior by making stage 1 neutral:
  - `STAGE1_EPOCHS = 0`
  - `STAGE1_PDE_FACTOR = 1.0`
  - `STAGE1_DATA_MULT = 1.0`
  - `STAGE1_TOP_STATION_MULT = 1.0`
  - `STAGE1_MULTI_HIGH_MULT = 1.0`
  - `STAGE1_HIGH_DOWNWIND_MULT = 1.0`
- Smoothed Q more strongly:
  - `Q_SEGMENT_LENGTH = 6`
  - `Q_SMOOTH_WEIGHT = 0.05`
  - `Q_L2_WEIGHT = 0.005`

Validation:

- `py_compile` passed for modified source files.
- 1 epoch smoke test passed in `result/smoke_analytic_plume/run_smoke`.
- 120 epoch smoke test passed in `result/smoke_analytic_plume_120ep/run_smoke`.
- The diagnostic plume factor stayed bounded:
  - 1 epoch: `plume_max ~= 0.71`
  - 120 epoch: `plume_max ~= 0.48`
- This directly removes the previous failure mode where `plume_max` grew above `120` while `Q` collapsed.

Remaining validation:

- A full training run is still required.
- The full run must check whether the physical plume shape improves without losing too much station-peak fit.
- Priority diagnostics after the full run:
  - `diffusion.gif`
  - `training_diagnostics.csv`
  - `station_peak_diagnostics.csv`
  - `q_segments.csv`
  - `result_quality_report.json`

### 2026-06-07 Dynamic Analytic Plume and Q-Collapse Fix

User feedback:

- The plume shape is now more plausible, but it does not look dynamically transported.
- The inferred source position is clearly wrong.
- Many high-value stations are badly underfit.
- User asked why the previous high-value fitting loss did not work.

Clarification:

- The direct peak-amplitude loss was deleted earlier at user request.
- Current defaults also have `LOSS_W_TOP_STATION = 0.0` and `LOSS_W_MULTI_HIGH = 0.0`.
- Therefore no separate high-value fitting loss was active in this run.

New evidence:

- Latest full analytic-plume run collapsed Q:
  - `q_mean ~= 0.0095`
  - `source_term_max ~= 0.0135`
  - `fit_raw_rmse ~= 5.49`
- Source was pushed to the source-domain boundary:
  - `x_m ~= 2965`
  - `y_m ~= 2115`
- Training source and landscape-best source were about `4094 m` apart.
- The PDE loss improved by driving emission strength toward zero, while data loss stayed bad.

Cause:

- In analytic-plume mode, using the same `Q(t)` both for field amplitude and for the Gaussian PDE source term creates a scale conflict.
- The Gaussian PDE source is too strong relative to the normalized analytic concentration field, so optimization reduces `Q(t)` to satisfy PDE residual.
- Once Q collapses, data fitting has little leverage and source position can drift to a boundary.

Code changes:

- Disabled Gaussian PDE source for analytic plume default:
  - `PDE_SOURCE_MODE = "none"`
- Prevented zero-emission collapse:
  - `Q_MIN = 0.2`
  - `Q_MAX = 5.0`
- Restored stronger high-value data weighting without adding a new loss:
  - `DATA_HIGH_WEIGHT = 2.0`
  - `DATA_TIME_PEAK_WEIGHT = 4.0`
- Shrunk source search domain away from station bounding-box boundary:
  - `SOURCE_POSITION_PAD_M = -300.0`
  - Added a safety check that raises if negative padding inverts the source domain.
- Added dynamic transport memory to analytic plume:
  - `ANALYTIC_PLUME_LAG_STEPS = 7`
  - `ANALYTIC_PLUME_MAX_AGE = 0.25`
  - `ANALYTIC_PLUME_AGE_DECAY = 0.18`
  - `ANALYTIC_PLUME_ALONG_SPREAD = 0.10`
  - `ANALYTIC_PLUME_CROSS_SPREAD = 0.35`
  - `analytic_plume_kernel(...)` now sums lagged wind-aligned puffs instead of using a purely steady instantaneous plume.

Validation:

- `py_compile` passed.
- 120 epoch smoke test after Q/PDE/source-domain fix passed:
  - Q stayed around `1.0`.
  - source was no longer on the boundary.
  - `pred_raw_max` increased from about `1.56` in the failed full run to about `12.0` in the short run.
- 120 epoch smoke test after dynamic-lag plume passed:
  - Q stayed around `0.92`.
  - source remained inside the domain.
  - `pred_raw_max` reached about `16.3`.
  - plume remained bounded, with `plume_max ~= 0.71`.

Remaining validation:

- Run full training and inspect whether the source remains stable after epoch 120.
- Confirm GIF shows delayed/downwind transport rather than a static steady-state patch.
- Check whether high station peaks recover without reintroducing extra loss terms.

### 2026-06-07 Restored Continuous Neural Q(t)

User request:

- Restore `Q` as a continuous time-varying function `Q(t)`.
- Combine this with the current physical-first analytic plume direction.

Reasoning:

- The current analytic-plume branch removed the worst free-plume compensation mode.
- The remaining hard piecewise `Q` can still create abrupt source-strength jumps that show up as nonphysical field changes.
- This change should not add another observation-fitting loss. It reuses the existing `Q_SMOOTH_WEIGHT` and `Q_L2_WEIGHT` regularization path.

Code changes:

- Changed default `Q_MODE` from `piecewise` to `neural`.
- Kept the bounded source-strength range:
  - `Q_MIN = 0.2`
  - `Q_MAX = 5.0`
- Adjusted continuous-Q regularization defaults:
  - `Q_SMOOTH_WEIGHT = 0.02`
  - `Q_L2_WEIGHT = 0.001`
- Added `PINN.configure_neural_q(...)` so the model stores the observed time grid for regularizing continuous `q_net(t)`.
- Extended `PINN.q_regularization()`:
  - for neural `Q(t)`, smoothness is computed from first and second differences of `q_net(t)` over the training time grid
  - L2 regularization is computed on the neural time modulation, not on the global `logQ`
- Updated Q export:
  - `q_time_series.csv` now includes `q_mode`
  - continuous mode uses `segment_id = -1`
  - `q_segments.csv` becomes a one-row summary for continuous `Q(t)` instead of pretending there are piecewise segments
- Updated training logs to print:
  - `Q mode: neural, continuous_time_nodes=...`
  - `q_smooth` and `q_l2` whenever the Q regularization weights are enabled

Validation:

- `py_compile` passed for:
  - `pinn_source/config.py`
  - `pinn_source/models/pinn.py`
  - `pinn_source/q_parameterization.py`
  - `pinn_source/pipeline.py`
- A 1 epoch smoke run passed using `.venv` at:
  - `result/smoke_continuous_q/run_smoke`
- Smoke log confirmed:
  - `Q mode: neural, continuous_time_nodes=24`
  - source checkpoint restore and diagnostics completed
- Export check confirmed:
  - `q_time_series.csv` contains `q_mode=neural` and `segment_id=-1`
  - `q_segments.csv` contains one neural summary row

Next validation:

- Run full default training.
- Inspect:
  - `q_time_series.csv` for two-event continuous structure without hard jumps
  - `diffusion.gif` for smoother plume evolution
  - `station_peak_diagnostics.csv` for high-station recovery
  - `result_quality_report.json` for source stability and field-component ranges

### 2026-06-07 Historical Puff Transport for Moving Plume

User feedback:

- The plume still looked obviously stationary after switching to continuous `Q(t)`.

Evidence inspected:

- Attached full-run log.
- Latest `result/source_uncertainty/result_quality_report.json`.
- Latest `training_diagnostics.csv`, `q_time_series.csv`, `station_peak_diagnostics.csv`.
- Generated contact sheets from short smoke GIFs.

Findings:

- `Q(t)` was continuous but nearly slow/monotonic, so it did not create event-like moving pulses.
- The analytic plume kernel still behaved like a steady-state plume:
  - every lag used the current wind direction/speed
  - all lagged puffs were centered using the same current wind-aligned coordinate system
  - the field was multiplied by current `Q(t)` instead of using emission-time `Q(t-age)`
- Normalized wind displacement over the lag window was visually too small, so puffs remained near the source and looked static.

Code changes:

- Added transport-history buffers to `PINN`:
  - `transport_times`
  - `transport_u`
  - `transport_v`
  - `configure_transport_history(...)`
- `pipeline.py` now registers the observed wind time series on the model after Q configuration.
- Reworked `analytic_plume_kernel(...)`:
  - samples historical wind at `t-age`
  - samples source strength at emission time `Q(t-age)`
  - advects each historical puff from the source by historical wind displacement
  - sums finite-lifetime puffs instead of drawing one steady current-wind plume
- Added/updated analytic plume parameters:
  - `ANALYTIC_PLUME_LAG_STEPS = 9`
  - `ANALYTIC_PLUME_MAX_AGE = 0.45`
  - `ANALYTIC_PLUME_MIN_AGE = 0.05`
  - `ANALYTIC_PLUME_AGE_DECAY = 0.18`
  - `ANALYTIC_PLUME_ALONG_SPREAD = 0.04`
  - `ANALYTIC_PLUME_CROSS_SPREAD = 0.15`
  - `ANALYTIC_PLUME_TRANSPORT_SCALE = 12.0`
  - `ANALYTIC_PLUME_SOURCE_CORE_WEIGHT = 0.0`
- Changed default Q mode to a continuous control-point function:
  - `Q_MODE = "smooth_time"`
  - one learnable logQ control point per observed timestamp
  - linear interpolation gives continuous `Q(t)`
  - first/second difference regularization keeps it smooth without hard segments

Validation:

- `py_compile` passed for:
  - `pinn_source/config.py`
  - `pinn_source/models/pinn.py`
  - `pinn_source/q_parameterization.py`
  - `pinn_source/field.py`
  - `pinn_source/pipeline.py`
- 1 epoch smoke passed:
  - `result/smoke_smooth_time_transport/run_smoke`
- 120 epoch smoke passed:
  - `result/smoke_smooth_time_transport_120ep/run_smoke`
  - confirmed `Q mode: smooth_time, continuous_time_nodes=24`
- Moving-puff GIF smoke passed:
  - `result/smoke_moving_puffs_gif/run_smoke`
  - contact sheet showed a visibly transported downwind tail instead of only a static source-centered blob
  - frame-difference mean increased to about `1.84`

Remaining validation:

- Run full default training and inspect:
  - whether `smooth_time` learns two event-like Q pulses
  - whether `diffusion.gif` keeps the transported-tail behavior after full optimization
  - whether source estimate stays consistent with source landscape
  - whether peak recovery improves without adding new observation-fitting losses

### 2026-06-07 Performance Fix A and Confidence-Landscape Review

User request:

- Apply performance方案 A first.
- Inspect the latest full-run result after the historical moving-puff change.
- The overall plume and source point looked acceptable, but the confidence interval looked wrong.
- User also set a workflow preference: future code optimization should first present plans and alternatives, then wait for confirmation before edits.

Performance change implemented:

- Removed duplicate moving-puff field evaluation in the training loop.
- `field.py` now exposes `concentration_from_components(...)`.
- `pipeline.py` now computes `field_components(...)` once for:
  - observation forward pass
  - PDE collocation forward pass
  - final diagnostics
  - debug source-center diagnostics
- This keeps the model/loss unchanged and avoids recomputing `analytic_plume_kernel(...)` immediately after `predict_concentration(...)`.

Validation:

- `py_compile` passed for `field.py` and `pipeline.py`.
- 5 epoch smoke passed at `result/smoke_perf_a/run_smoke`.
- 1 epoch debug smoke passed at `result/smoke_perf_a_timing/run_smoke`.

Latest full-run findings:

- Full run used `Q mode: smooth_time, continuous_time_nodes=24`.
- Training loop was slow after moving-puff transport:
  - epoch 500 timing: `data_forward ~= 0.029s`, `pde ~= 0.165s`, `backward ~= 0.206s`, `epoch_total ~= 0.406s`
  - later epochs were often around `0.5s` each
  - source landscape took about `112s`
- Result quality report still flags:
  - raw RMSE high
  - high-value station peaks missed
  - training source and loss-landscape best source far apart
- Training source:
  - approximately `(1515 m, -20 m)`
- Source-landscape best:
  - approximately `(-1829 m, 1803 m)`
  - distance from training source about `3809 m`

Confidence-interval issue:

- `source_confidence_landscape.png` shows the best loss-landscape source near the northwest search boundary.
- `sites_source_confidence.png` overlays that probability region while marking the trained source with the red star.
- This makes the figure visually confusing: the contours describe the landscape-best region, not uncertainty around the trained source.
- Since the best landscape point is on/near the domain boundary and far from the trained source, the current contours are truncated boundary low-loss regions, not a reliable closed confidence interval for the estimated source.

Candidate fixes to confirm before implementation:

- Option A: if landscape best is far from the trained source or lies near a boundary, label the output as an inconsistent source landscape and do not draw confidence contours on `sites_source_confidence.png`.
- Option B: run a local confidence landscape around the trained source only, and report it separately from the global source-domain scan.
- Option C: keep the global scan, but plot both the trained source and landscape-best source with different markers, and make the JSON/report explicitly state that the confidence region belongs to the landscape best.
- Recommended: Option C plus a boundary/inconsistency warning; optionally add Option B for local uncertainty around the trained source.

Implemented after user confirmation:

- Applied recommended Option C plus warnings.
- `source_confidence_landscape.json` now includes:
  - an `interpretation` string explaining that probability contours describe the scanned global loss landscape, not necessarily uncertainty around the trained source
  - `trained_source`
  - `trained_to_landscape_best_distance_m`
  - `landscape_best_boundary_margin_m`
  - `warnings`
- `source_confidence_landscape.png` now plots:
  - trained source as a red star
  - global landscape-best source as an orange X
  - warning in the title when the landscape is inconsistent or boundary-truncated
- `sites_source_confidence.png` now also plots both trained source and landscape-best source.
- `result_quality_report.json` now embeds source-landscape interpretation, distance, boundary margin, and warnings.
- A no-rescan preview was generated from existing `source_loss_landscape.csv` at:
  - `result/confidence_preview/source_confidence_landscape.png`
  - `result/confidence_preview/sites_source_confidence.png`
- Preview confirmed:
  - trained source and landscape best are clearly distinct
  - warning triggers for about `3809 m` source distance
  - warning triggers for `0 m` landscape-best boundary margin

### 2026-06-07 Reduced Moving-Puff Lag Steps for Speed

User request:

- Training was still too slow.
- User confirmed方案 1: reduce moving-puff lag steps.

Implemented:

- Changed `ANALYTIC_PLUME_LAG_STEPS` from `9` to `5`.
- This keeps the historical moving-puff plume structure but reduces the number of puff terms that PDE autograd differentiates through.
- No loss, PDE sampling, or source/Q parameterization changes were made.

Reasoning:

- Micro-benchmark showed PDE second-derivative autograd dominates runtime.
- With `N_COLLOCATION=4000`, approximate measured PDE graph cost was:
  - `lag=9`: total about `0.51s`
  - `lag=5`: total about `0.23s`
- Reducing lag steps is the lowest-risk speed improvement because it preserves the same model form with coarser transport memory.

Validation:

- `py_compile` passed for:
  - `pinn_source/config.py`
  - `pinn_source/field.py`
  - `pinn_source/pipeline.py`
- Current-setting micro-benchmark:
  - `lag_steps=5`
  - `N=4000`
  - `forward ~= 0.010s`
  - `deriv ~= 0.091s`
  - `backward ~= 0.132s`
  - total about `0.234s`

### 2026-06-07 Switched Source Confidence to Local Default

User feedback:

- Training was still slow.
- Confidence interval was still semantically wrong.
- User confirmed the proposed fix: use local confidence around the trained source and stop treating the global source-domain scan as the default confidence interval.

Implemented:

- Changed default source-confidence mode:
  - `SOURCE_LANDSCAPE_MODE = "local"`
- Global `source_domain` scan remains available by setting `SOURCE_LANDSCAPE_MODE = "source_domain"`, but it is no longer the default.
- Updated `source_landscape.py` so output semantics depend on scan mode:
  - local mode interpretation: probability contours describe local source uncertainty around the trained source with other learned parameters fixed
  - source-domain mode interpretation: probability contours describe the global scanned loss landscape, not necessarily uncertainty around the trained source
- Updated warnings:
  - local mode warns when the local best point is close to the local scan boundary and suggests increasing `SOURCE_LANDSCAPE_RADIUS_M`
  - source-domain mode keeps the previous global inconsistency/boundary warnings
- Updated `result_quality_report.json` embedding to include `scan_mode`.

Validation:

- `py_compile` passed for:
  - `pinn_source/config.py`
  - `pinn_source/source_landscape.py`
  - `pinn_source/pipeline.py`
  - `pinn_source/viz.py`
- 1 epoch smoke with `run_id=None` triggered local confidence scan at:
  - `result/smoke_local_landscape`
- Local scan size and speed:
  - grid `11 x 11 = 121` candidates
  - source landscape elapsed time about `6.8s`
  - previous source-domain scan used `1872` candidates and took about `115s`
- Smoke JSON confirmed:
  - `scan_mode = "local"`
  - local confidence interpretation is present
  - local boundary warning triggers when the local best lies at the edge of the 500 m radius scan

### 2026-06-29 Conversation Logging and Greeting Convention

User requested two ongoing collaboration conventions:

- At the start of every future conversation segment, Codex should say `Bonjour!`.
- After each working conversation in this project, Codex should save the conversation summary, reasoning, inspected/changed files, validation, unresolved issues, and next steps into `project_status.md`.

Operational note:

- Treat `project_status.md` as the project-local conversation memory and keep appends concise, evidence-based, and useful for resuming work.

### 2026-06-29 Fix Recent Leak Batch Extract Script Path

User reported that running:

- `.venv_clean\Scripts\python.exe scripts/run_recent_leak_source_inversions.py`

failed on the first selected leak with:

- `FileNotFoundError: scripts\extract_monitor_data.py`

Cause:

- `scripts/run_recent_leak_source_inversions.py` still hard-coded `EXTRACT_SCRIPT = SCRIPT_DIR / "extract_monitor_data.py"`.
- The current SHSH JS extraction script is `scripts/extract_monitor_data_shsh_js.py`.

Implemented:

- Added `resolve_extract_script(output_folder)` in `scripts/run_recent_leak_source_inversions.py`.
- For `OUTPUT_FOLDER = "shsh_js"`, the batch runner now prefers `scripts/extract_monitor_data_shsh_js.py`.
- Kept fallbacks to `scripts/extract_monitor_data.py` and `data/extract_monitor_data.py`.
- Improved the missing-script error to show all searched paths.

Validation:

- `.venv_clean\Scripts\python.exe -m py_compile scripts\run_recent_leak_source_inversions.py` passed.
- Import/path check with `.venv_clean` resolved:
  - `C:\Document\phd\SourceInverse\SourceInverse\scripts\extract_monitor_data_shsh_js.py`
  - `exists=True`

Next step:

- Re-run the batch command. If it proceeds past extraction, the next likely issues to inspect are extraction log contents, PINN run logs, and result quality reports for each leak.

### 2026-06-29 Review Latest Five Source-Inversion Results for PPT

User asked Codex to inspect the five most recent source-inversion result folders under `result/` and prepare a simple presentation-style summary.

Inspected latest five result folders by directory modification time:

- `result/20260629_192914_一氧化氮(NO)`
- `result/20260629_202633_硫化氢(H₂S)`
- `result/20260629_201032_一氧化氮(NO)`
- `result/20260629_200235_二氧化硫(SO₂)`
- `result/20260629_194630_硫化氢(H₂S)`

Files inspected in each folder:

- `result_quality_report.json`
- `station_peak_diagnostics.csv`
- `q_time_series.csv`

Main findings:

- All five runs completed and produced source coordinates, local confidence landscape outputs, Q time series, station peak diagnostics, and GIF/plot artifacts.
- All five quality reports had `is_reasonable = false`.
- In all five runs, local landscape distance was `0.0 m`, meaning the local fixed-parameter source scan was consistent with the trained source; this does not prove the source estimate is final.
- Main failure mode: the model often fits one dominant station peak very well but badly misses another high-value station peak in the same event window.
- Several runs had Q saturation at upper/lower bounds and plume maxima above the warning threshold.

Presentation guidance:

- Describe the outputs as preliminary diagnostic source-inversion results, not final regulatory conclusions.
- Emphasize that the current recurrent-PDE model can produce plausible single-source candidates and diagnostics, but the latest batch indicates multi-peak/multi-station events are not yet robustly explained by one source.
- Use maps/GIFs plus the station peak diagnostics to show both the inferred candidate source and the uncertainty/limitations.

### 2026-06-29 Expanded Per-Run PPT Interpretation

User asked for more detail for each of the latest five source-inversion results.

Additional inspection:

- Re-read `result_quality_report.json`, `station_peak_diagnostics.csv`, and `q_time_series.csv` for each latest result.
- Extracted event time windows, inferred source lat/lon, RMSE, warnings, Q statistics, plume maxima, local landscape consistency, and high-station peak fit behavior.

Key per-run interpretation:

- `20260629_192914_一氧化氮(NO)`: 2026-04-15 05:00-18:00. Candidate source near `(30.717272, 121.285623)`. Model fit the strongest `二工区南部园区站（抚佳）` peak almost exactly but missed `上石化边界卫二路站` and other elevated stations. RMSE high and plume factor excessive.
- `20260629_202633_硫化氢(H₂S)`: 2026-04-14 08:00-20:00. Candidate source near `(30.712014, 121.297750)`. Model fit `上石化边界卫六路站` peak well but missed `二工区东北园区站(亚南)`. RMSE high and plume factor just above warning threshold.
- `20260629_201032_一氧化氮(NO)`: 2026-04-14 08:00-20:00. Candidate source near `(30.712641, 121.296744)`. Model fit `上石化边界卫六路站` peak but missed the similarly large `二工区东北园区站(亚南)` peak. This is the clearest example that one-source explanation is insufficient or the current model is over-allocating the event to one plume branch.
- `20260629_200235_二氧化硫(SO₂)`: 2026-04-15 04:00-16:00. Candidate source near `(30.725555, 121.283385)`. Q and plume values were comparatively small, but the main `上石化边界卫二路站` SO2 peak was not recovered. This result should be presented as low-confidence.
- `20260629_194630_硫化氢(H₂S)`: 2026-04-15 04:00-16:00. Candidate source near `(30.717456, 121.286195)`. Model fit `二工区南部园区站（抚佳）` well but missed `上石化边界卫二路站`; similar single-branch fit pattern as the NO result on 2026-04-15.

PPT conclusion:

- The latest batch should be described as preliminary source-inversion diagnostics.
- The candidate source points are internally stable under local landscape scans, but station-peak diagnostics show unresolved multi-peak/multi-station structure.
- For sharing, clearly separate "candidate source indication" from "model quality/limitations".

### 2026-06-30 PPT Design for Workspace Method Changes

User asked Codex to inspect current workspace changes and propose 1-2 PPT pages comparing the old method with the new method, including what changed and what optimization effect it should provide.

Inspected:

- `git status --short`
- `git diff --stat`
- diffs for `pinn_source/config.py`, `pinn_source/field.py`, `pinn_source/pipeline.py`, `pinn_source/source_landscape.py`, `pinn_source/viz.py`, and `pinn_source/README.md`
- existing project status notes about avoiding additional loss terms

Observed workspace changes:

- Main code change is a method-level refactor from the previous analytic/free-plume PINN path toward `FIELD_MODE = "recurrent_pde"`.
- `config.py` was simplified from many loss/collocation/adaptive parameters to a smaller set centered on recurrent plume controls: `RECURRENT_GRID_NX`, `RECURRENT_GRID_NY`, `RECURRENT_SUBSTEPS`, `RECURRENT_SOURCE_SCALE`, `RECURRENT_DECAY`, and `RECURRENT_INITIAL_RELEASE_FRACTION`.
- `field.py` replaces source-gate / analytic plume / historical puff logic with a gridded recurrent plume solver:
  - source release on grid
  - wind advection
  - diffusion
  - decay
  - bilinear sampling back to station points
- `pipeline.py` removes PDE collocation sampling, axis/source-local/top-station/multi-high/high-downwind auxiliary losses, staged PDE/data balancing, and adaptive loss weighting. Training now mainly uses data loss plus Q smooth/L2 regularization, while physical constraints are encoded in the forward plume simulation.
- `adaptive_loss.py` is deleted.
- `source_landscape.py` removes geometry-score terms from confidence landscape and scans data fit only, reducing semantic mixing between confidence and hand-designed geometry penalties.
- `viz.py` now always renders predicted concentration, not gate-only diagnostic visualization.
- `README.md` now describes recurrent PDE plume as the default model path.

PPT recommendation:

- Page 1: "Method upgrade: from loss-constrained PINN to physics-forward plume simulation", with old vs new pipeline comparison.
- Page 2: "Expected optimization effect and current diagnostic interpretation", with benefits, evidence from latest batch, and limitations.

Important caveat:

- Latest five result runs still have `is_reasonable = false`; present the new method as a structural improvement toward physical plausibility and simpler objectives, not as a fully solved final model.

### 2026-06-30 Generated Two-Slide Method-Update PPT

User asked Codex to directly generate the 1-2 page PPT described above.

Created:

- `outputs/source_inverse_method_update.pptx`

Slide structure:

- Slide 1: `溯源模型方法升级`
  - compares the old PINN/free-plume/multi-loss method with the new recurrent-PDE gridded plume method
  - emphasizes the method shift from adding auxiliary losses to encoding physics in the forward model
- Slide 2: `修改内容与优化效果`
  - summarizes three main code/method changes:
    - recurrent-PDE plume generation
    - simplified training objective
    - clearer diagnostics and local source landscape semantics
  - includes expected effects and current diagnostic caveat that latest batch results are still preliminary

Validation:

- Rendered preview images for both slides and inspected visually.
- Fixed initial text crowding on slide 1 and tightened slide 2 caveat text.
- Imported the final PPTX with artifact-tool and confirmed:
  - slide count: 2
  - output file exists at `outputs/source_inverse_method_update.pptx`

Notes:

- The deck uses a restrained white/black/gray style with orange highlights.
- It is intended as a concise method-change section for a technical PPT, not a full source-inversion results report.

### 2026-06-30 Clarified Main Motivation for Method Upgrade

User clarified the intended explanation for the method upgrade:

- The main concern with the previous method was that plume inference was not temporally continuous enough.
- Because the plume was generated more like independent time-slice fitting, its evolution did not fully match the physical expectation that pollutant mass should be released, transported by wind, diffused, and decayed continuously through time.

Recommended framing:

- Present the recurrent-PDE upgrade first as a fix for temporal continuity and physical plume evolution.
- Present reduced loss complexity and reduced component compensation as secondary benefits.

### 2026-06-30 Speaker Notes for Group Meeting Slides 2-3

User provided `C:/Document/phd/SourceInverse/项目汇报/0630/0630组会汇报.pptx` and asked for speaker notes for slides 2 and 3.

Inspected with artifact-tool:

- Slide 2 title: `工作介绍`
  - Covers source-position confidence interval, wind-field perturbation sensitivity analysis, and diagnostic/quality report analysis.
- Slide 3 title: `详细进展`
  - Focuses on local source-position confidence interval: local grid scan around trained source, loss contour/probability conversion, 50/80/95 confidence regions, and why local scan avoids misleading global boundary low-loss areas.

Delivered:

- A natural Chinese speaking script for slide 2 and slide 3, emphasizing motivation, method, and interpretation rather than reading slide text verbatim.

### 2026-06-30 Regenerated Speaker Notes for Updated Slides 2-3

User clarified that the PPT file had changed and asked to regenerate speaker notes for the updated version of `C:/Document/phd/SourceInverse/项目汇报/0630/0630组会汇报.pptx`.

Re-inspected updated deck:

- slide count changed to 18
- Slide 2 title: `工作介绍——溯源模型方法升级`
  - focuses on method-upgrade motivation: previous plume inference lacked explicit temporal continuous propagation
  - compares old `PINN + 自由羽流网络 + 多辅助 loss` with new temporally continuous PDE gridded plume recurrence
- Slide 3 title: `工作介绍——修改内容与优化效果`
  - table comparing modification direction, concrete changes, and expected effects
  - rows cover plume expression, loss function structure, source-confidence interpretation, visualization output, and diagnostic report

Rendered slide previews:

- `outputs/ppt_slide_previews/slide-2.png`
- `outputs/ppt_slide_previews/slide-3.png`

Delivered:

- Regenerated Chinese speaker notes tailored to the updated slides 2 and 3.

### 2026-07-02 Commit Message Draft for Workspace Changes

User asked Codex to write a GitHub commit message for the current workspace changes.

Inspected:

- `git status --short`
- `git diff --stat`
- diffs for recurrent-PDE model files and related docs/visualization/source-landscape changes

Current change summary:

- Reworked source inversion from the previous analytic/free-plume path to `FIELD_MODE = "recurrent_pde"`.
- Added gridded recurrent plume evolution with source release, advection, diffusion, decay, and station sampling.
- Simplified the training objective to data fitting plus Q smooth/L2 regularization.
- Removed adaptive loss and many auxiliary geometry/observation-shaping losses.
- Simplified source landscape semantics and concentration visualization.
- Updated README and project conversation/status notes.

Delivered:

- A concise commit subject and multi-line commit body suitable for GitHub.
