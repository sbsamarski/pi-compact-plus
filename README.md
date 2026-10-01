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

1. **Old tool results collapse (elision).** Before each request, the oldest tool results are
   replaced with short stubs — **only on the wire**; your session file and your screen keep the
   originals. There is a whole section about this below, because it is the workhorse.
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

## Tool-call elision: keep the context low, much longer

This is the quiet workhorse of the extension, and it deserves its own section. In a long session
the single heaviest thing in your context is usually not your conversation — it is the raw output
of tool calls made hours ago: file listings, test runs, whole documents you read once and never
opened again. Left alone, that dead weight pushes your context toward the compaction trigger while
adding nothing to the model's understanding of the present.

Elision fixes this continuously and invisibly. Once your context crosses the starting percentage,
every request gets its old tool results replaced by stubs that keep the essential three things —
the command that ran, the size of the output, and the last few hundred characters:

```
[Tool result]: [read "src/main.ts" - output elided: 464 lines / 22,459 chars - tail: "..."]
```

The effect compounds. Instead of watching the context climb steadily toward the trigger line, you
watch it get pulled back down every time dead weight accumulates — the same session keeps running
for hours longer before a compaction is even needed, your history survives intact for much longer,
and the model works from a smaller, sharper context the whole time. On a real 1M-token session this
alone was the difference between compacting every couple of hours and barely compacting at all.

And again, the safety rule that makes all of this comfortable: **stubs live only on the wire.**
Your screen always shows the full originals, your session file is never modified, and if you exit
and resume, everything is still there. Run `/compact-plus preview` any time to see the stubbed
request exactly as the model would receive it.

The settings, all reachable in the board's elision area and through `/compact-plus elision`:

| Setting | What it does |
|---|---|
| **Start (%)** | the context-filling percentage where stubbing begins (default 20% of the window). Below it, nothing is touched. |
| **Stub tail (chars)** | how much of each result's ending survives inside the stub (default 300 characters — usually where the outcome lives). |
| **Protect (tokens)** | a protected window counted back from the newest message: tool results inside it are never stubbed, so the recent work stays fully intact. |
| **Min batch results** | stubbing waits until at least this many results qualify (default 4) — no busywork for one lonely old test run. |
| **Min batch savings** | a batch is only taken if it actually saves at least this many tokens (default 2,000). |
| **Stop gap (tokens)** | stubbing refuses to push the wire size closer than this to the reserve line — it can never over-trim toward the danger zone. |

A stub, once written, stays stubbed: the sweep does not churn. As the session grows and new tool
results age past the protected window, later sweeps pick them up. The result is a steady state:
the context hovers well below where it would naturally be, and compaction happens when *you* have
accumulated enough genuine history to be worth summarizing — not because old logs crowded it out.

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

Every model on your compaction list gets its own profile. Enter on a selected model in the board
to open its screen. Each setting, in plain words:

#### Thinking level

Can be off, minimal, low, medium, high, xhigh, or max, wherever a model supports it. Thinking makes
a model more careful but slower, and it eats into the space the summary needs — so the extension
quietly reserves extra room for it: a little at low, more at max.

#### Timeout

How many minutes one attempt may take — any number you like. When time runs out, that model is
skipped and the next one is asked. Fast online models are fine with ten minutes; a slow local
machine writing a long summary may honestly need twenty or more.

#### Summary aim (% of the folded region)

How long the summary should be, as a share of the material being folded *after* the stubbing. A
300k-token region at 10% aims for a 30k summary; a 30k region aims for 3k — the summary scales with
the work, not with the window. The **floor** (4,096) and **ceiling** (32,768) keep tiny and huge
regions sensible.

#### Draft mode

Off, mini, or full. With a full draft, the model first writes a hidden analysis of the material in
the same request, then writes the final summary with that plan in mind — and the analysis itself is
thrown away. It costs a little extra time, but long summaries come out noticeably better organized.
Online models default to full; local ones to off.

#### Tool stubs in input

Whether this model sees old tool results in their slimmed-down stub form (on) or in full (off). Off
only makes sense for a huge-window model with room to spare.

#### Tool-call arg cap

When a tool call carries a whole file inside its arguments, that file can eat the summarizer's
attention. This trims each argument value to a set number of characters (default 500), keeping the
beginning plus a marker. Set it to "full" for no trimming.

#### The four preserve levels

Tool calls, user prompts, assistant replies, assistant thinking — each with its own
verbatim / detailed / summary / brief choice. Because this is per model, a slow small model can do
brief summaries while a fast big one keeps everything word for word, or anything in between.

#### Chain mode

When this model compacts, there is usually an older summary already in the session. This decides
what happens to it: **fold** carries it forward (compact, slightly lossy), **attach** copies it
word for word into the new summary so nothing is ever lost, **skip** drops it. All three are
described in their own section below.

#### No-think tag

Some local models switch their thinking off when they see a special marker in the prompt. When
thinking is off, the extension appends that marker so the model truly stays quiet.

#### Sampling flags

Temperature and friends, applied only to the compaction call — your normal chats are never
touched. Each flag can be given a value or left at "ignore" (the server's default applies).

#### Preview

Shows the complete instructions this exact model would receive, with all of the above filled in.
Read-only — a look before you compact.

## Chaining compactions

- **fold** — carry the previous summary forward, continue its numbering (compact, slightly lossy).
- **attach** — reproduce the previous summary *verbatim* in its own section; nothing is ever lost
  between compactions, and the size limits make room automatically.
- **skip** — drop it, for maximum compression.

## Percentages, not magic numbers

Pi's built-in compaction numbers are absolute token counts — a reserve of 16,384 and a keep-recent
of 20,000 — which were sized for small windows and quietly become wrong everywhere else. This
extension computes every threshold from the current model's context window, as a percentage with
minimum and maximum clamps:

| Threshold | Default | Clamps | What it controls |
|---|---|---|---|
| **Reserve** | 18% of the window | 12,288 – 80,000 tokens | the room kept free at the top; the auto-compaction trigger fires when the context reaches *window − reserve*. |
| **Keep recent** | 20% of the window | 16,384 – 100,000 tokens | the newest exchange kept verbatim outside the summary, so resuming feels continuous. |
| **Elision start** | 20% of the window | — | where stubbing begins (see the elision section). |
| **Elision stop gap** | 4,000 tokens | — | the safety distance from the reserve line that stubbing will not cross. |

The clamps matter on small windows. On a 130k local model, 18% and 20% land at 23,400 and 26,000
tokens — sensible numbers, barely touched by the clamps. On a 1M online model they would be 180,000
and 200,000, where the clamps bring them down to 80,000 and 100,000 — still generous, no longer
absurd. Switch models mid-session — pause, change, continue — and every value recomputes from the
new window immediately, with no reload. If the window is unknown, pi's classic 16,384 / 20,000 are
used as safe fallbacks.

One more number belongs here: the **generation cap** on chat requests (default 65,536 tokens, set
it to anything, 0 disables). Pi asks providers for an output permission of *window minus its
estimate minus a small margin* — on a 1M-window model that can mean requesting nearly a million
output tokens, and the provider rejects the whole request the moment its own count is slightly
higher than pi's estimate. That exact failure forced the emergency compactions this extension was
born from. With the cap, chat requests always fit until the real context reaches *window − cap* —
the auto-compact trigger (window − reserve) sits comfortably below that line, so compaction wins
the race by design.

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
