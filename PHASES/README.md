# Phase Split For Memory Debug

This folder splits the training step into the same four phases shown in:

`phase_peak_alloc_mib(fwd/bwd/opt/zero)=...`

Files:

- `01_fwd.md`: forward phase
- `02_bwd.md`: backward phase
- `03_opt.md`: optimizer phase
- `04_zero.md`: gradient zeroing phase

All snippets are copied from current source and keep original call flow.
