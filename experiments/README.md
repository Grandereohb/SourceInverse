# Paper experiment workflow

This directory stores frozen experiment designs. Generated datasets and run
outputs should remain outside Git; only compact design files and reproducibility
metadata belong here.

The paper workflow has four stages:

1. Generate and audit synthetic scenarios with
   `generate_synthetic_matrix.py`, `generate_synthetic_particle_scenario.py`,
   `audit_synthetic_scenario.py`, and `audit_synthetic_matrix.py`.
2. Run truth-blind candidate searches with `run_inversion_matrix.py`,
   `run_gaussian_surface_matrix.py`, `run_discrete_adr_surface_matrix.py`, and
   `run_simple_baseline_matrix.py`.
3. Join source truth only after optimization with
   `aggregate_multistart_matrix_truth.py`,
   `evaluate_gaussian_candidate_surfaces_truth.py`, and
   `evaluate_simple_baseline_matrix_truth.py`.
4. Build the selected-versus-oracle failure decomposition with
   `compare_cross_method_failure_mechanisms.py`. Diagnostic sensitivity scripts
   are secondary analyses and must not be used to tune the frozen test set.

Legacy combined baseline runners and pre-freeze event summaries were removed;
their functionality is covered by the staged workflow above.

Despite its historical filename, `evaluate_gaussian_candidate_surfaces_truth.py`
now evaluates any saved candidate surface that follows the common result
contract, including discrete ADR. Use `--dataset-root` when a frozen experiment
directory has been relocated, and use `--resource-accounting
shared_per_surface` when multiple fitted source-strength profiles reuse the same
transport evaluations.
