<div align="center">

# pi-compact-plus

**A context-compaction extension for the [pi coding agent](https://github.com/earendil-works/pi).**

Takes over what pi does when the conversation outgrows the model's window — and rebuilds it around
your choices: which model writes the summary, how much detail it keeps, and what it is allowed to forget.

![license](https://img.shields.io/badge/license-MIT-green) ![pi](https://img.shields.io/badge/pi-0.87.x-blue) ![language](https://img.shields.io/badge/language-TypeScript-3178c6)

</div>

---

## Install

1. Download **[pi-compact-plus.ts](./pi-compact-plus.ts)** into your pi extensions folder:

   | OS | Folder |
   |---|---|
   | Windows | `%USERPROFILE%\.pi\agent\extensions\` |
   | Linux / macOS | `~/.pi/agent/extensions/` |

2. Start pi (or run `/reload`).
3. Run `/compact-plus` and add the models that should write your summaries, best one first.

That's it. No build step, no extra keys — the extension compacts through pi's own model machinery
(OpenRouter, Google, local llama.cpp servers — anything pi can reach). It creates its own config
file on first use and reads it live, so every change in its menus applies immediately.

---

## What happens when your context fills

1. **Old tool results collapse.** The biggest dead weight in a long session is raw tool output.
   Once the context crosses a threshold you set, the oldest tool results are replaced with short
   stubs that keep the command, the size, and the last lines of the output:
   `[bash "bun test" - output elided: 203 lines / 13,238 chars - tail: "Command exited with code 1"]`
   Crucially, this happens **only on the wire** — the request the model receives is slimmed down,
   while your session file and your screen keep the originals. Nothing on disk is ever modified.
2. **A watcher triggers compaction early.** While pi is idle, the context is checked every few
   seconds; near the danger line the extension compacts on its own, before providers start
   rejecting. Plain `/compact` goes through this extension too.
3. **The summarizer gets a prepared region, not a dump.** Every user turn numbered in sequence,
   replies and thinking separated, tool calls written out, old results as stubs, and the previous
   summary included so numbering and history continue.
4. **Sizes come from your content.** The summary aim is a percentage of the material actually being
   folded (default 10%), with per-model floor and ceiling — not a magic constant.
5. **Models rotate.** The first model on your list that produces an *accepted* summary wins. A model
   that errors, times out, returns something too thin, or paraphrases when it was told to copy —
   is rejected, and the next one is asked. Only if all fail does pi's built-in compaction take over.
6. **Results are gated.** Section shape, turn numbering, length range, and — for verbatim tool-call
   preservation — the actual stub lines must appear in the summary.
7. **Your disk keeps everything.** The session file is append-only: compaction changes what the
   model *carries*, never what you *keep*.

---

## The summary is a ledger, not a paragraph

A theme-based summary forgets *sequence*. This template produces a different shape:

- **Current State** — the newest truth; it wins over anything contradictory below it.
- **Turn Ledger** — one numbered entry per user turn: what was asked, done, broken, fixed, decided.
  Older turns tighten but are never dropped; changed decisions are tagged, never deleted.
- **Files & Data · Key Decisions · Next Steps · Critical Context** — paths, reasoning, the exact
  next action, and every number and error message verbatim.

## Four kinds of content × four levels of detail — all editable

| | verbatim | detailed | summary | brief |
|---|---|---|---|---|
| **Tool calls** | every stub, word for word | command, size, tail, per call | one line per call | only the important ones |
| **User prompts** | quoted in full | substance + quoted instructions | intent + parameters | intent only |
| **Assistant replies** | kept in full | substance + quoted decisions | what was concluded | outcome only |
| **Assistant thinking** | kept in full | reasoning per decision | a line or two | only the turns that changed course |

Every one of the sixteen sentences is **yours to edit** in the *Category templates* screen, and the
whole master template is editable too (with `{TOKEN}` placeholders you leave to the extension).
Each slot has three layers: the **extension default** (built-in), your **user default** (a
checkpoint you save), and the **active** wording — with read-only previews and confirmed resets
for every one.

## Per-model settings

Thinking level · timeout · summary aim (% of the folded region) · draft mode · stub and argument
caps · the four preserve levels · chain mode · a no-think tag for local models · sampling flags —
plus a **preview** that assembles the exact description that model would receive. Your own eyes
before anything is sent.

## Chaining compactions

- **fold** — carry the previous summary forward, continue its numbering (compact, slightly lossy).
- **attach** — reproduce the previous summary *verbatim* in its own section; nothing is ever lost
  between compactions, and the size limits make room automatically.
- **skip** — drop it, for maximum compression.

## Percentages, not magic numbers

Reserve, keep-recent, elision thresholds — all percentages of the current window with clamps, so
switching from a 95k local model to a 1M online model adapts everything on the spot. A generation
cap on chat requests (default 65,536 tokens) keeps requests inside permissions providers actually
honor — the overflow class that used to force emergency compactions is gone.

---

## Commands

`/compact-plus` with no argument opens the full settings board (with position memory: every edit
returns you exactly where you were). With arguments:

| Command | What it does |
|---|---|
| `status` | what is on, the next model, current context numbers |
| `on` / `off` | arm or disarm the extension |
| `models` / `models add|remove <ref>` | manage the compaction list |
| `move <ref> <up\|down\|n>` | reorder it |
| `options <ref> …` | any per-model setting from the command line |
| `elision …` | the stubbing engine: on/off, start %, tail, protect, savings |
| `retries <n> [sec]` | retry policy for flaky endpoints |
| `marker <text\|clear>` | the default no-think tag for local models |
| `instruction [text\|reset]` | the summary template |
| `preview [ref]` | the exact request the summarizer would get — nothing is sent |
| `log` | open the log |

## The log

Every compaction writes one line per event to `~/.pi/agent/pi-compact-plus.log`, in your local
time: context numbers, preprocessing, each attempt with its sizes, token counts as the provider
reported them, and the reason behind every rejection. When something looks wrong, the log tells
the whole story in order.

## Configuration

`~/.pi/agent/pi-compact-plus.json`, created on first use and updated by the screens — but plain
JSON if you prefer: `enabled`, the ordered `models` with their options, `retries`,
`retryDelaySeconds`, `log`, `directRequest`, `shapeGate`, `chatMaxTokensCap`, `scaling`
(percents + clamps), `autoCompact`, `elision`, `noThinkMarker`, `additionalInstruction`
(the active template), `templateUserDefaults`, `categorySentences`, and `categoryUserDefaults`.

---

## Requirements

A recent pi (developed against 0.87.x). Runs on Windows, Linux, and macOS; single file, no build,
no dependencies beyond what pi already ships.

## License

MIT — see [LICENSE](./LICENSE).
