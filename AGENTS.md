# SourceInverse Working Rules

- Start user-facing messages with Bonjour and record concise progress in the
  local, global `project_status.md` without rewriting historical entries.
- Before algorithm optimization, briefly explain goals and tradeoffs. The user
  authorized autonomous selection of the recommended optimization route on
  2026-10-12; do not request confirmation for each algorithm change. Preserve
  existing user edits and data, and validate changes before retaining them.
- Keep `joint_adam` as the production default; IA-VPDI remains opt-in and uses
  all valid observations for variable-projection inversion.
- Do not reintroduce spatiotemporal crossfit, anchor-preserving folds, station
  withholding or time-block withholding as an inversion/source-selection path.
  This direction was explicitly retired by the user on 2026-10-12. Independent
  external-case evaluation is distinct from changing the fitting objective.
- Prefer physical structure, parameterization, variable elimination and
  identifiable diagnostics over additional empirical losses. Distinguish fit
  quality from source-location accuracy and diagnostics from measured gains.
- Do not commit or push without an explicit current request.
- Use Microsoft YaHei (微软雅黑) for all editable text in future Chinese manuscript outputs, including headings, body text, tables and captions (user instruction, 2026-11-02).
- Freeze NLS at pre-optimization commit `756337627e9561c75ec555f9961f0f6685837beb`
  (user instruction, 2026-10-28). Do not modify its solver, forward adapters,
  starts, constraints or evaluation budget, and do not add NLS optimization runs.
  Keep prior experimental NLS controls archived, not as a replacement baseline.
  Work only on IA; put any IA-specific forward extensions in `ia_*` modules
  so the frozen common/NLS path remains unchanged. Never weaken NLS to create gains.
