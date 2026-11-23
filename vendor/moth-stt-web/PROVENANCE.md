# Vendored Moth web runtime

Source: `semantique-team/moth-stt`, commit
`0e15442ecb85dffcb4edd68e9d9ece18af6f35df` (2026-11-14), Apache-2.0.
That commit is the `MOTHLMB2` reader and sits on top of
`b0fa11c` (`budget_bytes`); the two local commits are the reason the
archived repository cannot be used as a runtime dependency (E-016/P4).

Files are byte-identical to the source commit:

| File | sha256 |
|---|---|
| `README.md` | `118b02bd4884e4d8594caf25190ef08dd6835c3c1864ed969ba10c55adfea6a8` |
| `decode.js` | `40bd00ca736da1914215d976fb4351a5277b3079b0fb73ebcc547f72dd59952c` |
| `index.html` | `9aea33d040b647b498842c6292debde9bbbeb5c8e7902314e6c3d973d3a7680d` |
| `lm.js` | `3d4ca348c9c99fd8bdfaa343120112b5d7163e7b004e8a3835a7f0bf4a72860a` |
| `mel.js` | `6d6851a0fd7ccaab21bb7bd4a1a17800091d9fafd70732022eaa354f4e381bbc` |
| `mic-worklet.js` | `e9a5b0a9a1ecb8326aaad0b8d379e2fce4dfbaeb42edd8fe44f91bcbac8b213c` |
| `mic.js` | `f241a9782c1d2decdbb93e8f3983176d1a81c70f7263bc79194f0c396525550a` |
| `ort.js` | `8c2677a78aef41951ddedc6fd2377aa97941a24cb30f6999069a24384f43cdbf` |
| `package.json` | `e43f20c70d272a15a5aabad44e7c36ef84409c278199da152b15a71e7f12ebdd` |
| `package-lock.json` | `327543732950ec6045783c6c9455798d9403de99a958ee873d5586e853236fe2` |
| `server.py` | `6a782324816c1f8270c363a476480070e98347853d0fadd72c9015504b8d698d` |
| `vad.js` | `7f3fe3cdfa7c371aed6070f44f48b33b474767927a95e9d7f89d3d927bec2472` |

The runtime still needs its npm dependencies: run
`npm ci --prefix vendor/moth-stt-web` once. `node_modules/` is gitignored.
The upstream `tests/` and `acceptance/` suites are **not** vendored: they
depend on Moth's private runs, fixtures and sibling checkouts. The MOTH-50
measurements are in `runs/E-016/`.

Serve a MOTH-50 release with the wrapper (it creates a temporary root so
`/web/index.html` can load `/release/`):

```bash
npm ci --prefix vendor/moth-stt-web
python tools/serve_web.py --release runs/E-023/release-0.9.7
# open http://127.0.0.1:8765/web/index.html?threads=4
```

Do not edit these files to make a measurement pass. If a runtime change is
needed, make it in this copy, record the diff here with a new hash and date,
and re-run the E-016 checks.
