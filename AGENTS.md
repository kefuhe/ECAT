# ECAT release integration instructions

ECAT combines three independently installable components: `ecat-viz`,
`csi_cutde_mpiparallel` and `eqtools`, plus selected public documentation and scripts.
Read README.md, Install.md and docs/developer/architecture.md before related edits.

- Inspect actual code, documentation and callers; preserve unrelated user work.
  When separate development checkouts are configured, confirm component source
  and release destination before editing. Do not treat synchronization as a
  whole-directory mirror or copy unreviewed development content.
- Preserve the dependency direction: eqtools uses CSI/ecat_viz, CSI uses
  ecat_viz, and ecat_viz does not depend on CSI/eqtools. Objects, mesh, physical
  responses and state belong to CSI; workflow/domain analysis belongs to
  eqtools; generic plotting belongs to ecat_viz.
- Read the instructions of each component involved when available. Keep one
  implementation/state owner; do not fix package defects through case patches.
- Preserve equations, units, coordinate frames, signs, parameter/matrix layouts,
  patch indices, cache validity and MPI semantics. Validate affected contracts
  and numerical results; unexplained numerical drift is not acceptable.
- Keep the public manual in docs/ and follow its existing directory model and
  navigation. Align examples, installation commands and dependency declarations
  with the components actually shipped. Component versions and ECAT citation
  versions have different meanings.
- Synchronize only the user-authorized release scope. Preserve existing
  exclusions for tests, internal documents, unselected scripts, hidden local
  files, caches and generated data. Public files must contain no machine-local
  absolute paths or private case material.
- Run focused validation in the configured environment and check affected
  installation/packaging or documentation links as needed. Report what was
  actually verified, remaining limits and public/maintainer documentation impact.
  Run `git diff --check`; do not automatically push or publish.
