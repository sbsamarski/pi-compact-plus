# pi-compact-plus

A context-compaction extension for the [pi coding agent](https://github.com/earendil-works/pi-coding-agent).

Pi is a terminal coding assistant, and like every coding assistant it eventually runs into a wall: the conversation grows until it no longer fits into the model's context window. When that happens, something has to shrink the history down — that something is called *compaction*. Pi already has a built-in compaction, but after living with it through some very long, very real working sessions, I found it had sharp edges: it could overflow the model's limits, it could hand the summary to a model that wasn't suited for it, it once came back in Chinese, and it always threw away far too much of the reasoning and tool work that led to the current state of things.

**pi-compact-plus** is my answer to all of that. It is a single-file extension that takes over compaction completely and rebuilds it around a few simple ideas: the summary should be written by the model you choose, in the shape you define, with the detail level you decide per kind of content; old tool noise should be quietly collapsed instead of eating the window; every threshold should scale with the size of the window rather than being a hardcoded number; and nothing that lands on disk should ever be lost.

If you use pi for long-running projects — the kind where a session spans days and hundreds of thousands of tokens — this extension is built for you.

---

## What it does, in one pass

Here is the whole story of what happens when your context fills up, in the order it happens.

**1. Old tool results quietly collapse (elision).** As you work, the biggest dead weight in the context is usually not your conversation — it is the raw output of tool calls made long ago: file listings, test runs, file contents you read once and never needed again. This extension watches the size of the conversation, and once it crosses a threshold you control (a percentage of the model's window, by default twenty percent), it begins replacing the *oldest* tool results with short stubs. A stub looks like this:

```
[Tool result]: [bash "bun test" - output elided: 203 lines / 13,238 chars - tail: "Command exited with code 1" - re-run to see the full output]
```

It keeps the command, the size, and the last few hundred characters of the output — everything a summarizer (or you) needs to know what happened — and throws away the bulk. Here is the part that matters most: **this replacement happens only on the wire, in the request sent to the model.** Your session file on disk is never modified. The screen keeps showing the full originals. If you exit pi and resume tomorrow, everything is still there, untouched. The stubs exist only in the moment a request is being built. That is why the feature is safe: it is a lens, not an edit.

**2. A watcher keeps an eye on the window.** Every few seconds, while pi is idle, the extension checks how full the context is. When it approaches the danger zone — computed from the model's window and your *reserve* percentage — it triggers a compaction on its own, before the model starts rejecting requests. You can also trigger one yourself at any moment, either with the menu or simply by running `/compact`, because this extension intercepts pi's own compaction flow and answers it.

**3. The region to summarize is prepared with care.** When a compaction starts, the extension assembles the exact text the summarizer will read. It is not a raw dump. It is a numbered transcript: every user turn gets a number that continues from the previous summary's numbering, every assistant message is split into its visible reply and its thinking, every tool call is written out with its command, and every old tool result appears as a stub, exactly as it would on the live wire. If a previous summary exists, it is included as its own labeled block, so the new summary can build on it instead of starting from zero.

**4. The sizes are computed from your content, not from guesses.** The extension decides how long the summary should be by taking a percentage of the material that is actually being folded — by default ten percent of the region after elision, with a floor and a ceiling you can set per model. From that aim it derives the minimum it will accept and the maximum it will allow, plus a generation permission that leaves room for the model's thinking. If a model is thinking-heavy, extra room is reserved automatically.

**5. A summarizer is chosen — and if it fails, the next one is asked.** You maintain an ordered list of compaction models. The extension walks down that list. The first model that produces an accepted summary wins; the others are never asked. If a model errors out, times out, returns something too thin, or (with tool-call preservation set to verbatim) paraphrases the tool calls instead of copying them, that attempt is rejected and the next model gets its chance. Only if every candidate fails does pi's own built-in compaction take over, as a last resort.

**6. The result is checked before it is accepted.** The summary must have the expected sections, its turn numbering must be present and increasing, its length must fall within the accepted range, and — if you asked for tool calls to be preserved word for word — the actual stub lines must appear in it. A summary that fails any of these is thrown away and the next model is tried. Nothing half-baked ever reaches your session.

**7. The transcript is replaced, and your disk keeps everything.** The accepted summary enters the session as a compaction entry. Pi then replays only the summary plus a small recent tail. The older messages are no longer *loaded* — but they are never deleted from the session file on disk. The file is append-only; every word of every session remains readable in it forever. Compaction, in this extension, changes what the model carries, not what you keep.

---

## Why the summary is a ledger, not a paragraph

Most summarization prompts ask for a few thematic sections and call it a day. The problem is that a theme-based summary has no memory of *sequence*: it knows what the project is about, but not what happened in which order, not which decision superseded which earlier decision, and not what the exact next step was when the context ran out.

This extension's template produces a different shape, built from real research into how long-session memory degrades:

- **Current State** — the newest truth, declared up front. If anything later in the summary contradicts it, this section wins. It is the antidote to the contradictory-bullet problem that plagues long summaries.
- **Turn Ledger** — one numbered entry per user turn, in order, each with what the user asked (quoted at the level you choose), what the assistant did, which tools ran, what broke and how it was fixed, and what was decided against which alternative. Older turns may tighten, but they are never dropped. When a later turn changes an earlier one, the earlier line stays and gets a tag — "(superseded by Turn N)" or "(amended by Turn N)". History is annotated, never deleted.
- **Files & Data** — every file that was read, created, or edited, with its full path and why it matters.
- **Key Decisions** — each choice and the reasoning behind it, cross-referenced to the turn it came from.
- **Next Steps** — the exact next action with its parameters, so the session can resume without re-deriving anything.
- **Critical Context** — every number, version, model name, path, and quota verbatim, with error messages quoted exactly.

The template also explicitly cancels the few lines of pi's own summarization prompt that fight this shape — its brevity demands, and its "you may remove what is no longer relevant" — because those instructions are exactly how history evaporates.

---

## Four kinds of content, four levels of detail — all of them yours

Not everything in a session deserves the same treatment. Your user prompts are precious; the assistant's thinking may or may not be; tool calls are either the heart of the matter or pure noise, depending on the project. So every piece of content in the summary is governed by its own setting, with four levels:

- **verbatim** — keep it exactly as written, word for word.
- **detailed** — keep the substance and quote the decisive sentences.
- **summary** — a faithful short version of each item.
- **brief** — the essence only, one line at most.

There are four content categories: **tool calls**, **user prompts**, **assistant replies**, and **assistant thinking**. Each category has its own sentence for each of the four levels — sixteen sentences in total — and every one of them is editable. You are not stuck with my wording. In the settings you will find a *Category templates* screen: pick a category, pick a level, and you can rewrite that exact instruction to say whatever you want it to say.

To keep all this manageable, the wording lives in three layers:

- The **extension default** is what ships with the extension. It can never be lost or broken.
- Your **user default** is a checkpoint you save yourself — your preferred wording for a slot, kept safe so you can always come back to it.
- The **active** wording is what is actually used right now.

Every screen tells you honestly which layer you are looking at. Resetting to either default always asks for confirmation first, and the previews show you the exact text before you commit to anything.

---

## The main template is yours too

Above the category sentences sits the master template — the actual instruction text the summarizer reads, with its output format, its rules, and its overrides. You can edit the whole thing. The only parts you should leave alone are the placeholder tokens — things like `{TOOL CALLS SUMMARY}` or `{ASSISTANT THINKING SUMMARY}` — because those are the slots where the extension inserts the per-model, per-level sentences you configured above. If you delete one by accident, the extension notices and re-adds the missing rule automatically at the end, with a note, so nothing silently disappears.

The template menu gives you read-only previews of all three versions of the text — the active one, your user default, and the extension default — plus the save and reset actions for each, and every reset asks before it overwrites.

---

## Every model gets its own personality

Different models need different handling, and this extension embraces that. For each model in your compaction list you can set:

- its **thinking level** (off, minimal, low, medium, high, xhigh, max — where the model supports it),
- its **timeout**, in any number of minutes,
- its **summary aim** as a percentage of the folded region, plus the floor and ceiling that clamp it,
- whether it writes a **draft** first (a hidden analysis inside the same request, stripped from the final text — a trick that organizes long summaries at no extra round-trip),
- whether tool results arrive as **stubs** in its input, and the character cap for long tool-call arguments,
- the four **preserve levels** (the category settings above are per model, so a fast cheap model can summarize briefly while your best model handles everything verbatim),
- its **chain mode** — what happens to the previous summary (more on that below),
- an optional **no-think tag** for local models that are switched by a marker in the prompt,
- and a full set of **sampling flags** (temperature, top-p, top-k and friends) for local llama.cpp-style servers.

A **preview** on each model's screen assembles the complete description that exact model would receive — its own settings substituted into the template — so you can see precisely what will be sent before anything is sent.

---

## Chaining compactions: fold, attach, or skip

When a session outlives several compactions, the question becomes: what happens to the previous summary? This extension gives you three answers, per model:

- **fold** (the default) — the new summary is told to carry the previous summary's content forward and continue its turn numbering. Compact, but each round trip loses a little.
- **attach** — the new summary must reproduce the previous summary *verbatim*, word for word, in its own section at the end. Nothing is ever lost between compactions: every summary carries the whole chain inside it. The size limits are raised automatically to make room, and a gate rejects any summary that fails to include the previous one intact.
- **skip** — the previous summary is dropped entirely, for maximum compression.

If your sessions are long and the history matters, **attach** is the mode that guarantees the earliest context is still there, many compactions later.

---

## Percentages instead of magic numbers

Pi's built-in settings for compaction are absolute token counts, which works tolerably for one window size and falls apart for another. This extension computes everything from percentages of the current model's context window, clamped between minimum and maximum values you control:

- **reserve** — how much room to keep free at the top of the window; the auto-compaction trigger fires when the context reaches window minus reserve.
- **keep recent** — how much of the recent tail stays outside the summary entirely, so the newest exchange is always verbatim.
- **elision start** — the context-filling percentage at which stubbing begins.
- **elision stop gap** — how far from the reserve line stubbing refuses to go, to keep the batch safe.

Change the model, change the window, and every one of these adapts on the spot — no reload needed, because the configuration is read live on every event.

There is also a **generation cap**. Pi asks for a maximum output size equal to window minus its estimate minus a small margin, and on big-window models that can mean asking for nearly a million output tokens — which providers reject the moment the estimate is slightly optimistic. That exact failure, on huge sessions, is what forced my first emergency compactions. The cap (default 65,536 tokens on chat requests) keeps every request inside a permission the provider will actually honor.

---

## The screens

Everything is configurable through a proper interface — no hand-editing required (though the JSON file is there if you prefer it):

- **The board** (`/compact-plus` with no arguments): the master list. Enable or disable everything, order the model list, tune retries, the generation cap, the percent-based scaling, the auto-compact watcher, elision, the summary template, the category templates, and read the live computed values.
- **Compaction models**: Enter on a selected model opens its full options screen; Enter on an unselected model adds it to the end of the list; Space and the number keys reorder or remove. Typing filters the list.
- **Model options**: every per-model setting listed above, each with a plain-language description underneath.
- **Category templates**: the four categories, each opening its four editable level sentences, with read-only previews of both default layers and a confirmation-guarded reset for every sentence.
- **Summary template**: the master template submenu — edit, previews of all three layers, save and reset, all confirmed.
- **Position memory**: every edit returns you exactly where you were — the same row, the same submenu, the same screen — so tuning a dozen settings in a row feels like walking down a corridor, not climbing stairs.

---

## The commands

`/compact-plus` with no argument opens the board. With arguments it speaks plain commands:

- `status` — what is on, which model would compact next, the current context numbers.
- `on` / `off` — arm or disarm the extension.
- `models` — show the compaction list; `models add <ref>` and `models remove <ref>` manage it.
- `move <ref> <up|down|n>` — reorder the list.
- `options <ref> …` — set any per-model option from the command line: thinking, timeout, aim, draft, arg cap, the four preserve levels, the chain mode, the no-think tag.
- `elision [on|off] [start …] [tail …] [protect …] [savings …]` — the stubbing engine.
- `retries <n> [sec]` — how many times a flaky endpoint is retried, and the base wait.
- `marker <text|clear>` — the no-think tag default for local models.
- `instruction [text | reset]` — the summary template, from the command line.
- `preview [ref]` — build the exact request the summarizer would receive for that model and open it in the editor. Nothing is sent. This is the best way to see the stubs, the numbering, and the substituted instructions with your own eyes.
- `log` — open the log file.

Compaction itself needs no command: the watcher fires automatically, and plain `/compact` goes through this extension too.

---

## The log

If logging is on (it is by default), every compaction writes one line per event to `~/.pi/agent/pi-compact-plus.log` — in your machine's local time. You will see the start of each run with the context numbers, the preprocessing (how many results were stubbed), every attempt with its computed sizes, the outcome, the exact token counts as the provider reported them, the draft stripping, and any gate that rejected a summary with the reason why. When something goes wrong, the log tells the whole story in order.

---

## Installation

1. Put [pi-compact-plus.ts](./pi-compact-plus.ts) into your pi extensions directory — on Windows that is `%USERPROFILE%\.pi\agent\extensions\`, on Linux and macOS `~/.pi/agent/extensions/`. Pi loads TypeScript files from that folder directly; nothing needs to be built or compiled.
2. Start pi (or run `/reload` if it is already running).
3. Run `/compact-plus` and walk through the board once: add the models you want summarizing, in the order you prefer, and set their thinking levels.

That is the whole install. The extension creates its own configuration file (`~/.pi/agent/pi-compact-plus.json`) on first use and reads it live from then on — every change in the menus takes effect immediately; only code changes need a reload.

**Requirements:** a recent pi (developed and tested against 0.87.x). Models are whatever pi already has configured — OpenRouter, Google, local llama.cpp servers, anything pi can reach — because the extension compacts through pi's own model machinery. API keys come from pi's own authentication, not from this extension.

---

## The configuration file, field by field

For those who like to look under the hood, `~/.pi/agent/pi-compact-plus.json` holds:

- `enabled` — the master switch.
- `models` — the compaction list, in order, with each model's full option set. The first entry is the first model asked.
- `retries`, `retryDelaySeconds` — the retry policy for transient provider errors.
- `log` — whether the log file is written.
- `directRequest` — how the summarizer is called (direct streaming through pi's model layer; leave it on).
- `shapeGate` — whether summaries are checked for the expected sections before being accepted.
- `chatMaxTokensCap` — the generation cap on chat requests; 0 disables it.
- `scaling` — the percent-based reserve and keep-recent settings with their clamps.
- `autoCompact` — whether the idle watcher triggers compaction on its own.
- `elision` — the stubbing engine: whether it is on, the starting percentage, the stub tail size, the savings and stop-gap thresholds, the protect window, and the minimum results before stubbing bothers.
- `noThinkMarker` — the default no-think tag for local models.
- `additionalInstruction` — the active summary template (null = follow the user default).
- `templateUserDefaults` — your saved template checkpoint.
- `categorySentences` — the active wording for each of the sixteen category slots.
- `categoryUserDefaults` — your saved checkpoints for those slots.

You never have to touch this file — the screens keep it up to date — but it is plain JSON on purpose, so that backing it up or diffing it is trivial.

---

## A note on honesty

This extension was built in the open, against real failures — the overflow that asked for 943,000 output tokens, the summary that came back in Chinese, the models that paraphrased when they were told to copy. Every gate and guard in it exists because something actually went wrong first. It has been running daily on heavy sessions; the test suite that travels with my development setup covers the serializer, the caps, the scaling, and the summary-aim math.

That said, it is one person's extension for one person's workflow, generalized as best I could. If something behaves oddly on your setup, the log file will usually say exactly why — and the previews will show you exactly what would be sent, before it is sent.

## License

MIT. See [LICENSE](./LICENSE).
