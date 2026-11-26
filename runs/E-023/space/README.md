---
title: MOTH-50 — italiano con termini inglesi, nel browser
emoji: 🦋
colorFrom: gray
colorTo: gray
sdk: static
pinned: false
thumbnail: https://huggingface.co/spaces/semantique-team/moth-50-demo/resolve/main/semantique-thumbnail.png
short_description: Italiano + inglese, 49 MB, sul tuo dispositivo
models:
  - semantique-team/moth-50
license: apache-2.0
---

# MOTH-50 — live demo

Speech-to-text per **italiano con parole inglesi dentro**, eseguito **interamente nel
browser**.

Nessun audio viene caricato: il package da 49,2 MB (40,0 MB di grafo INT8 ONNX, un LM
trigram da 7,8 MB, il VAD da 1,3 MB) viene scaricato una volta e poi gira sulla tua
macchina con ONNX Runtime Web / WebAssembly. L'audio non lascia la scheda e la pagina
continua a funzionare se la connessione cade dopo il caricamento.

Questa Space serve lo stesso artefatto pubblicato su
[`semantique-team/moth-50`](https://huggingface.co/semantique-team/moth-50), byte per
byte, e verifica ogni sha256 al caricamento prima di eseguirlo.

**0.9.8-e023-r1** è l'acoustic E-023 R1 seed 0 con il LM E-034c. Il T-600 release guard
passa: **12,30% WER / 78,14% ETA** contro **13,70 / 76,13** di E-005 + E-034c. S1-dev
è **13,38% / 75,15%**; FLEURS, report-only, è **IT 11,83%** e **EN 28,20%**. Il mix
esterno CS-FLEURS-READ è **34,81% WER** (176 utterance, un solo speaker, letto una
volta). La versione 0.9.8 differisce da 0.9.7 solo nella NOTICE: grafo, LM, VAD,
vocabolario e mel filters sono gli stessi byte (D37).

## What it is good at, honestly

- Frasi italiane con termini, acronimi e brand inglesi: *"il deploy è andato bene,
  facciamo un quick sync dopo la call"*.
- Gergo tecnico italiano, acronimi, verbi italianizzati (*deployato*, *pushare*,
  *loggare*).
- Non emette nulla su silenzio e rumore stazionario: un VAD Silero fa da guardia.

## What it is not

- Niente punteggiatura, niente maiuscole: output minuscolo, numeri a parole.
- **L'inglese puro è debole** (28,20% WER su FLEURS-EN): è un modello code-switching,
  non un modello bilingue.
- **I termini inglesi restano il punto debole**: sul T-600 l'ETA è 79,27% sulle frasi
  con switch e 85,42% sulle parole inglesi di controllo. Non c'è un glossario utente.
- La copertura è S1, S2 e S3. S3 non è stata addestrata, ma è stata valutata più volte:
  non è una voce mai vista. Il parlato è letto e recitato, non spontaneo.
- Massimo 30 secondi per registrazione. Parla una frase alla volta: la pagina spezza
  sui silenzi, ma le pause aiutano.

Numeri, limiti e provenienza completa:
[`semantique-team/moth-50`](https://huggingface.co/semantique-team/moth-50).
Codice: `github.com/semantique-team/moth-50`. Apache-2.0; l'encoder iniziale è
`openai/whisper-base` (MIT).
