# Invalid as an E-003 arm (kept as evidence)

This run was started before the pretrained-initialization fix: `build_model` created
Whisper-base from random weights, so the model only learned the output prior and dev WER
stayed at ~99 % (best 99.04 % at step 3000). Do **not** use it for the architecture screen.

Kept because it is a clean checkpoint/throughput probe on the real mixture:

- 4,000 steps in 3,394.8 s (56.6 min), 0 non-finite steps, last CTC loss 6.08;
- peak MPS memory 8.446 GB;
- `ckpt_last.pt` written every 100 steps and loaded back at step 300 (verified);
- 4 greedy dev evaluations (1000/2000/3000/4000), ~4 min each.

The valid arms restart from `data/raw/whisper-base-encoder.pt` and start in
`runs/E-003/base+<n>/`.
