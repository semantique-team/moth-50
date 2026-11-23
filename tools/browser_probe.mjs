#!/usr/bin/env node
/**
 * Drive the vendored browser runtime for the E-016 checks.
 *
 * The page must be served by `tools/serve_web.py` (or any root that serves
 * `/web/`, `/release/` and the mounted audio). Examples:
 *
 *   python tools/serve_web.py --release runs/E-023/release-0.9.7 \
 *     --mount dataset=../moth-cs-dataset --mount clips=/tmp/clips --port 8780
 *   node tools/browser_probe.mjs rtfx --base-url http://127.0.0.1:8780 \
 *     --threads 1 --clips /clips/clip-5s.wav,/clips/clip-10s.wav,/clips/clip-20s.wav \
 *     --out runs/E-016/rtfx.json
 *   node tools/browser_probe.mjs s1dev --base-url http://127.0.0.1:8780 \
 *     --threads 4 --manifest /dataset/manifests/R-S1-dev.jsonl \
 *     --audio-base /dataset/ --out runs/E-016/s1-dev-browser.json
 *
 * Playwright is installed by `npm ci --prefix vendor/moth-stt-web`. If its
 * bundled Chromium is not in the user cache, run `npx playwright install
 * chromium` in that directory or set PLAYWRIGHT_CHROMIUM to an executable.
 */

import fs from "node:fs";
import path from "node:path";

import { chromium } from "../vendor/moth-stt-web/node_modules/playwright/index.mjs";

function usage(message) {
  if (message) console.error(message);
  console.error(
    "usage: browser_probe.mjs rtfx --base-url URL --clips PATH[,PATH] [--threads N] [--out FILE]\n" +
      "       browser_probe.mjs s1dev --base-url URL --manifest URL --audio-base URL [--threads N] [--segment false] [--out FILE]",
  );
  process.exit(2);
}

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) usage(`unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) usage(`missing value for --${key}`);
    options[key] = value;
    index += 1;
  }
  return options;
}

const [mode, ...rest] = process.argv.slice(2);
if (!["rtfx", "s1dev"].includes(mode)) usage();
const options = parseArgs(rest);
if (!options["base-url"]) usage("--base-url is required");
const threads = options.threads === undefined ? null : Number(options.threads);
if (threads !== null && !Number.isInteger(threads)) usage("--threads must be an integer");
const segment = options.segment === undefined ? null : options.segment !== "false";
if (options.segment !== undefined && !["true", "false"].includes(options.segment)) {
  usage("--segment must be true or false");
}
const outPath = options.out ? path.resolve(options.out) : null;

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM || undefined,
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
page.on("console", (message) => {
  if (message.type() === "error") errors.push(`console: ${message.text()}`);
});
await page.goto(options["base-url"], { waitUntil: "load" });

const common = { threads, segment };

let payload;
if (mode === "rtfx") {
  if (!options.clips) usage("--clips is required for rtfx");
  const clips = options.clips.split(",").filter(Boolean);
  payload = await page.evaluate(
    async ({ clips, threads }) => {
      const { createRunner, decodeTo16k } = await import("/web/ort.js");
      const runner = await createRunner({ baseUrl: "/release/", threads });
      const results = [];
      for (const clip of clips) {
        const pcm = await decodeTo16k(clip);
        const single = await runner.transcribe(pcm);
        const benchmark = await runner.benchmark(pcm, { warmups: 2, runs: 5 });
        results.push({
          clip,
          audio_seconds: pcm.length / 16000,
          single: { total_ms: single.timings.total_ms, rtfx: single.rtfx, text: single.text },
          benchmark,
        });
      }
      return {
        environment: {
          crossOriginIsolated: globalThis.crossOriginIsolated,
          sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
          numThreads: runner.numThreads,
          simd: runner.simd,
          loadMs: runner.loadMs,
          version: runner.config.version,
        },
        results,
      };
    },
    { clips, threads },
  );
} else {
  if (!options.manifest || !options["audio-base"]) {
    usage("--manifest and --audio-base are required for s1dev");
  }
  const init = await page.evaluate(
    async ({ manifestUrl, audioBase, threads }) => {
      const { createRunner, decodeTo16k } = await import("/web/ort.js");
      const runner = await createRunner({ baseUrl: "/release/", threads });
      const text = await (await fetch(manifestUrl)).text();
      const rows = text
        .split("\n")
        .filter((line) => line.trim())
        .map((line) => JSON.parse(line));
      window.__probe = { runner, decodeTo16k, rows, audioBase, hyps: [] };
      return {
        environment: {
          crossOriginIsolated: globalThis.crossOriginIsolated,
          sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
          numThreads: runner.numThreads,
          simd: runner.simd,
          loadMs: runner.loadMs,
          version: runner.config.version,
        },
        n: rows.length,
      };
    },
    { manifestUrl: options.manifest, audioBase: options["audio-base"], threads },
  );
  console.error(`loaded ${init.n} rows`);
  const chunk = 40;
  for (let start = 0; start < init.n; start += chunk) {
    const end = Math.min(start + chunk, init.n);
    await page.evaluate(
      async ({ start, end, segment }) => {
        const { runner, decodeTo16k, rows, audioBase, hyps } = window.__probe;
        for (let index = start; index < end; index += 1) {
          const row = rows[index];
          const pcm = await decodeTo16k(audioBase + row.audio);
          const out = await runner.transcribe(pcm, segment === null ? {} : { segment });
          hyps.push({ id: row.id, hyp: out.text });
        }
      },
      { start, end, segment },
    );
    console.error(`progress ${end}/${init.n}`);
  }
  const { hyps } = await page.evaluate(() => ({ hyps: window.__probe.hyps }));
  payload = { ...init, hyps };
}

const result = {
  tool: "tools/browser_probe.mjs",
  mode,
  base_url: options["base-url"],
  ...common,
  browser: { version: browser.version() },
  errors,
  ...payload,
};
if (outPath) {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(result, null, 2)}\n`);
  console.error(`wrote ${outPath}`);
} else {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}
await browser.close();
