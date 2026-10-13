# Source Inversion

## Run And Select An Algorithm

From the repository root:

```bash
python pinn_source/pinn_source_pinn.py
python scripts/run_recent_leak_source_inversions.py
```

Edit input paths and settings in `pinn_source/config.py`. The single selector
`INVERSION_MODE` accepts `joint_adam` (established baseline) or `ia_vpdi`.
The current local choice may differ from the deployment default. Both modes use
`models.pinn.PINN` as the parameter container; there is no `MODEL_NAME` switch.
`PINN_INVERSION_MODE` overrides both config and `run(inversion_mode=...)`.

Joint Adam optimizes position, diffusion and Q together. IA-VPDI profiles
nonnegative low-dimensional hat Q coefficients with SciPy NNLS, searches source
and diffusion candidates, and refines the profiled objective. IA-VPDI enables
constrained wind-aligned anisotropic diffusion internally. Its Q settings replace
`Q_MODE`; its search budget replaces joint Adam epochs.

All valid observations participate in IA-VPDI fitting and source selection.
Retired crossfit, anchor folds and station/time withholding are not supported.
Independent external-case evaluation is a separate task. The recurrent mode
uses explicit physical transport, not a classical neural PDE-residual PINN.

## Configuration

| Setting / environment override | Purpose |
| --- | --- |
| `SITE_PATH`, `CONC_PATH`, `WIND_PATH` | Station coordinates, concentrations and wind Excel inputs |
| `DEVICE` / `PINN_DEVICE` | auto, cpu or cuda |
| `INVERSION_MODE` / `PINN_INVERSION_MODE` | joint_adam or ia_vpdi |
| `IA_VP_Q_BASIS_COUNT` | 0 selects 3-5 hats from event length |
| `IA_VP_Q_KNOTS` | Optional fixed normalized-time knots |
| `IA_VP_Q_LAYOUT_STRATEGY` / `PINN_IA_VP_Q_LAYOUT_STRATEGY` | uniform; opt-in local_templates |
| `IA_VP_Q_UPPER_BOUND` / `PINN_IA_VP_Q_UPPER_BOUND` | Nonnegative coefficient cap in model units; infinity by default |
| `IA_VP_SEARCH_STEP_M` / `PINN_IA_VP_SEARCH_STEP_M` | Coarse candidate spacing |
| `IA_VP_TOP_K` / `PINN_IA_VP_TOP_K` | Refinement starts |
| `IA_VP_OUTER_STEPS` / `PINN_IA_VP_OUTER_STEPS` | Maximum refinement steps |
| `IA_VP_START_STRATEGY` / `PINN_IA_VP_START_STRATEGY` | score or spatial_diverse |
| `IA_VP_RESPONSE_BATCH_SIZE` / `PINN_IA_VP_RESPONSE_BATCH_SIZE` | Q response batch size (8; 1 restores serial) |
| `IA_VP_JACOBIAN_BATCH_SIZE` / `PINN_IA_VP_JACOBIAN_BATCH_SIZE` | Diagnostic VJP batch size (16; 1 restores serial) |
| `IA_VP_TIMING_ENABLED` / `PINN_IA_VP_TIMING` | Synchronized component timing; 0 disables |
| `RECURRENT_TRANSPORT_SOLVER` / `PINN_TRANSPORT_SOLVER` | grid or opt-in gaussian_parcels |
| `SIGMA_SRC_M` / `PINN_SIGMA_SRC_M` | Optional source width in metres |
| `RECURRENT_SOURCE_MODE` / `PINN_RECURRENT_SOURCE_MODE` | gaussian, cell_gaussian or point |
| `RECURRENT_DIFFUSION_SOLVER` / `PINN_RECURRENT_DIFFUSION_SOLVER` | gaussian, tensor_semigroup or scalar-only implicit |
| `RECURRENT_SOURCE_DECAY_INTEGRATOR` / `PINN_SOURCE_DECAY_INTEGRATOR` | legacy or exponential_linear (grid only) |
| `RECURRENT_PARCEL_SUBDIVISIONS` / `PINN_PARCEL_SUBDIVISIONS` | Parcel release-time quadrature (64) |
| `TRAIN_ON_RESIDUAL` / `PINN_TRAIN_ON_RESIDUAL` | Background-subtracted fitting; 0 fits raw data |
| `BASELINE_MODE` / `PINN_BASELINE_MODE` | median, q25 or q40 background estimate |

Grid controls include `RECURRENT_GRID_NX/NY`, adaptive substeps and their cap.
Wind uses meteorological direction-from when `WIND_DIR_IS_FROM=True`; smoothing
operates on vectors. Velocity is m/s, diffusivity m²/s and decay 1/hour before
normalization. `WIND_SCALE` is an effective-transport factor, not a unit conversion.
Zero concentration is valid; missing required station/wind values are excluded.

## Transport Choices And Limits

- Grid uses linear bilinear advection with absorbing zero-exterior boundaries.
  Gaussian diffusion is the legacy default. Scalar implicit diffusion and the
  wind-aligned positive tensor semigroup are optional alternatives.
- Tensor semigroup plans preserve small-step tensor covariance and positivity,
  with omitted Poisson probability at most 1e-10 per plan. Boundary loss remains
  explicit. Stencil selection is piecewise differentiable; extreme anisotropy
  requiring more than 128 cells fails explicitly.
- Exponential-linear source/decay weights integrate linear Q exactly with decay
  alone. They do not make endpoint spatial transport exact. Release knots must
  be included in the integration schedule.
- Full-space Gaussian parcels integrate interval wind/tensor histories for each
  released parcel. Release quadrature is separate from raster resolution/CFL.
  Fitting, information, GIF and TXT fields share this operator. It requires a
  single fixed finite-width Gaussian source, not a point/moving source or initial
  raster field. Grid diffusion/source-decay switches do not apply to parcels.
- Optional local temporal layouts depend only on clocks, not concentrations.
  Every candidate/layout re-estimates Q. Selected-layout information excludes
  discrete layout uncertainty; neither search nor refinement guarantees a
  global optimum.

The unused nonlinear limited-remap/pulse-scalar experiment has been removed:
it cannot support additive response matrices. Q-linear transport tests remain.

## Outputs And Interpretation

Each case retains copied inputs, `result_quality_report.json` and
`diagnostics.xlsx`. Optional plots include `station_timeseries.png`,
`sites_source.png`, `source_confidence.png` and `diffusion.gif`.

IA-VPDI report data are under `result_quality_report.json["ia_vpdi"]`.
Candidate scores, refinements, trajectories and stability tables are workbook
sheets, not duplicate standalone IA CSV/JSON files. Historical archives are
unchanged; offline tools can read their older standalone report.

`溯源输出/` contains source longitude/latitude, one field TXT per whole hour
(longitude, latitude, concentration), and `源强.txt` (timestamp, Q). Required
copies also go to `<result_root>/溯源输出/<case>/`. Background addition follows
`ADD_BASELINE_TO_VIZ`. Q is model-unit amplitude, not calibrated kg/h.

Local information removes active-Q and diffusion compensation directions.
`structural_status` describes local rank/conditioning, not global uniqueness.
`precision_assessment` compares the local ellipse with a diagnostic reference
scale. Zero residual noise, active Q caps, boundary solutions and model mismatch
limit interpretation. `sampled_profile_stability` reuses evaluated candidates;
1/5/10% near-score sets are not calibrated confidence regions. The legacy source
landscape holds other parameters fixed and is not a profiled IA landscape.

Low RMSE, a small ellipse or a plausible map location does not establish true
source accuracy. Unbounded Q may compensate for weak/wrong-location response.
Sensor recommendations are optional information-only diagnostics, not validated
deployment gains. The background error floor is a diagnostic, not a new loss.

## Observation Provenance

JJJ extraction writes an optional `concentration_quality.json` with original
strings, numeric values, status codes, units, source rows and input hashes.
Codes have unknown semantics; this does not filter measurements or certify them.
Pipeline archives it only if its concentration hash matches. Missing/stale reports
do not block training. Hash checks detect stale pairing, not malicious tampering.

## Tests, Comparisons And Deployment

```bash
python -m unittest discover -s tests
python -m unittest discover -s deployment/tests
python scripts/compare_historical_inversion_modes.py --output result/NEW_COMPARISON
python scripts/benchmark_inversion_runtime.py --output result/NEW_BENCHMARK.json
```

Historical comparison uses three predefined cases and different production
objectives/Q parameterizations: it is not an optimizer-only ablation.
`--audit-baseline OLD_RUN` replays archives; `--summarize-only` summarizes them.
The benchmark measures serial/batched equivalence and timings; `--smoke` adds
short reduced-budget flows. It does not prove full-budget speedup or localization.

Timings are nested: do not sum every component. Kernel plans are shared only
within fresh response calls, never reused across changed parameters/graphs.
Forward counts are basis-response equivalents, not comparable physical work
across different solvers.

Older protected Docker images require rebuilding for new algorithms/dependencies.
Current images use `SOURCE_INVERSION_INVERSION_MODE=joint_adam` by deployment
default; change it to `ia_vpdi` in the runtime env and recreate the service.
The runtime override takes precedence over compiled config.
See `deployment/README.md` and `proto/API_PROTOCOL.md`.

Core modules: `pipeline.py` (workflow), `field.py`/`gaussian_parcels.py`
(transport), `variable_projection.py` (basis/responses/NNLS), `ia_vpdi.py`
(search/information), `source_output.py` (TXT), `viz.py` (plots).
