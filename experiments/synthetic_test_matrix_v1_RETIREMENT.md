# Retirement of `synthetic_test_matrix_v1.json`

Status: **retired without generation or outcome evaluation**.

The 12-scenario design was originally reserved as an OOD-layout protocol using
the Zhenhua six-station geometry. It is no longer an independent test because:

1. the same Zhenhua layout and wind record were subsequently included in the
   48-case and 36-case development extensions;
2. its seeds overlap the later particle development matrix; and
3. 12 Gaussian-truth events are insufficient for the final reliability-audit
   claims described in the manuscript protocol.

No generated `test_puff_*` dataset, batch manifest, inversion output, or
truth-known evaluation was found during the 2026-08-27 audit. The original JSON
is retained unchanged as provenance and must not be run or reported as an
independent test.
