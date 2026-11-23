# Voxtral Mini 3B — excluded from the E-018 bake-off (resource cost)

The model is pinned (Apache-2.0, revision `3060fe34b35ba5d44202ce9ff3c097642914f8f3`,
9.36 GB of safetensors) and loads in Transformers, but it is not runnable on the
16 GB M5 for a 57.6 min set: one 2.62 s clip took **304 s** on MPS (RTFx
**0.009**), which projects to **>100 h** for the gold train set. CPU was 130 s
for the same clip. This is a resource exclusion, not an accuracy result.

Evidence: `smoke-1clip.jsonl` (correct transcript "La mia squadra ha perso il
derby.") and `smoke-1clip.meta.json`. The model remains a documented candidate
for a rented GPU; it was not used to select a teacher.
