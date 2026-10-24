# pi-compact-plus: why it is built this way

The explanation that used to sit between the lines of code. Moved out on 2026-10-24 so the source can be read as source; each entry is the original comment, unedited, filed under the name of the code it documents.

## ensureWatcherTimer #2

pi-compact-plus — the full compaction pipeline for pi: elision in the live session, then a
self-built summarization request (no pi prompt underneath), producing the turn-ledger summary.
The whole pipeline, in order:
1. ELISION (live, rule-based, no LLM): tool results older than the protected recent window
become informative stubs in every request (a `context` handler; stored history untouched).
Batches fire only when they clear the token floor AND the results-count floor AND can bring
the request below the stop gap under pi's compaction trigger. This delays compaction,
sometimes indefinitely.
2. COMPACTION (pi fires session_before_compact at its usual trigger; elision has kept the wire
small): the extension builds the ENTIRE summarization request itself:
- the region serialized with DETERMINISTIC TURN NUMBERS ([User — Turn 7]: ...) using pi's
own turn-boundary rule; the turn counter continues across compactions via the
CompactionEntry details (details.lastTurnNumber);
- tool results as informative stubs (the exit line survives; pi's 2,000-char head-only
clip never touches anything important), tool-call argument values capped;
- the previous summary LABELLED (what it covers, that its numbering is final, carry
forward, continue numbering);
- ONE instruction source: the generated turn-ledger template (no pi format block, no
override sentences — nothing to blend with).
The call goes through the model registry (streamSimple — request-time auth), maxTokens set
directly, no prompt-cache writes.
After the call: rough-work stripping, size gate, SHAPE gate (sections + turn-label sanity),
pointer line; the turn counter stored in details.
FALLBACKS: direct request fails → pi's compact() with the extension instructions (the old
path) → fails → next model in the rotation → all failed → pi's own compaction.
3. PER-MODEL OPTIONS: thinking, timeout, aim/min/max shares of pi's reserveTokens
(off/mini/full), summarizer input (stubs/raw), arg cap, and four preservation options
(stubs/user/replies/thinking: verbatim | detailed | summary | brief) that generate the
template's RULES wording. Every description states its default.
pi's two compaction numbers (settings.json "compaction": reserveTokens, keepRecentTokens) are
editable on the board (free number entry; /reload applies them); the summary sizes are shares of
reserveTokens and the elision protection mirrors keepRecentTokens.
Settings menu:  /compact-plus
Attempt log: ~/.pi/agent/pi-compact-plus.log
Commands: /compact-plus status | on | off | models | move | options | elision | retries |
marker | instruction | log | preview [ref]   (no argument opens the menu)

## ensureWatcherTimer #2

Everything the ELISION stage does. Extension-wide (not per model): it fires in the live session
before the summarizer is ever asked, for every model alike.
Percent-based scaling of the two anchor values (2026-10-01 - THE UNLINK). pi's compaction
settings are absolute token counts tuned for one window size; wrong for any other. The extension
instead computes keepRecent and reserveTokens from the ACTIVE model's context window at every
use, clamped to absolute min/max, so a mid-session model switch re-derives everything on the next
request. Nothing here touches pi's settings.json and nothing needs /reload.
The two sizes the operator sets, both in TOKENS (2026-10-21, was a percent of the window with
min/max limits). Percentages were unreadable on a 170k model and wrong on any other: the same
82% meant 139,400 tokens here and 52,480 on a 64k model, and the four limit rows existed only
to catch what the percent did on windows nobody was testing.
Both are amounts of the window left FREE, which is how pi words its own compaction settings
too, and it is the form that survives a model change: 'leave 30,600 tokens free' says the same
thing on a 170k model and on a 64k one, while 'fire at 139,400' means something on this model
only. The compaction line is the window minus the reserve.

## ensureWatcherTimer #2

WHO PULLS THE TRIGGER (2026-10-06). true (the default) = pi's own check that runs before every
request to the model compares the context with the line and starts the compaction there; that is
the only place a compaction can start while the agent is working, and the compaction itself is
still written by this extension, because the session_before_compact hook answers every call.
false = this extension's idle watcher is the only trigger, which cannot fire mid-run and is why
a long working session used to overflow instead of compacting.
The elision block: stubbing the oldest tool results out of the LIVE request before the context gets
anywhere near the trigger line, so a long working session keeps its conclusions instead of losing them
to a compaction. On or off, the share of the window at which it starts, and how many of the most recent
calls are protected - see ElisionConfig, applyElision, and the elision rows on the settings board.
This field sat inside the comment below it from 2026-10-18 until 2026-10-19. The paragraph describing
the removed piLineWritten flag lost its closing marker to the patch that deleted the flag itself, so the
comment ran on and swallowed this declaration. Types are stripped rather than checked, so nothing
complained: the cost was that the Config interface stopped telling the truth about a field every
elision path uses, and no bracket scanner can see that, because it throws comments away first.

## ensureWatcherTimer #2

pi's OWN built-in compaction switch (settings.json "compaction": {"enabled": ...}).
READ ONLY, in every sense: this extension reads the flag to warn you when two compactors would both be
live, and it never writes that file. The writer that used to run here on a per-second clock was deleted
on 2026-10-18, and the log's last trigger_clock record - one second after a manual restore of the very
same block, and from the build that was still loaded - is the reason it had to go.
The hard rule still stands, and at most ONE compactor may be on. What has changed is who enforces it:
not us, never again us. You hold that line in pi's own settings, and a change made in the file needs a
restart, because pi reads settings.json at startup and then holds what it read - which is how a flag
switched off at 23:29:55 was still reported as on by the pi that started at 23:29:36.
pi's compaction block, as written in its own settings.json, with pi's own defaults filled in for
anything absent. The flag alone used to be read; the two numbers are needed since this extension
builds pi's own preparation object for a compaction it lands without stopping the turn (#43), and
the cut point depends on `keepRecentTokens`. Read at call time, like the flag, because pi holds what
it read at startup and an edit in the file is exactly what the operator is testing.
pi's own compaction module (`dist/core/compaction/compaction.js`), which holds
`prepareCompaction` - the function that decides what a summary covers and where the verbatim tail
starts. It is NOT in the package's root exports, so it is reached by resolving the same specifier
the top of this file already imports and following it to that file. Nothing here copies pi's
arithmetic: if the module or the function cannot be found, the caller falls back to pi's own path.

## ensureWatcherTimer #2

The six numbers a template may ask for by name, as text, ready to drop in wherever the name sits.
These tokens were listed in the editor's tag row and described as "substituted wherever they appear"
(see INLINE_LENGTH_TOKENS and the note above applyTemplate), and no code ever substituted one of
them. So a template that writes its own length sentence - the built-in default does, and User
templates #1 and #3 are copies of it - sent the summarizer a pair of curly braces in place of its
target, and the block carrying the real numbers was withheld as well, because the code read those
same unfilled tokens as proof that the sentence had been written out on purpose. GLM wrote 9,487
tokens against an aim of 22,100 on 2026-10-18 for exactly that reason, which is the complaint
logged on 2026-10-17 as well ("why does the online GLM model write such a short compaction of
around 10000 tokens even though it is told to write 24000 tokens?") - it was never told.
A number that cannot be computed comes out as a plain sentence saying so rather than NaN or
braces, because a sentence at least tells the model what to do instead; min 0 means "no floor"
by design (see computeSizes) and is worded as that rather than as a failure.

## ensureWatcherTimer #2

THE SUMMARY TEMPLATE (the user's 2026-10-01 design): one editable source text with {PLACEHOLDER}
tokens; the per-model options substitute the tokens at request time. The four content categories
each have four level sentences (PRESERVE_SENTENCES, 4 x 4); the stub-example rule depends on the
input mode; the previous-summary rule and section depend on the chain mode. The user may reword
ANYTHING around the tokens - only the {TOKENS} themselves must survive editing; if one is
deleted, its rule is auto-added at the end with a visible note (nothing silently drops).
Shape (from the 2026-09-25/28 compaction research): TWO LAYERS instead of pi's flat thematic
sections - CURRENT STATE is the newest truth and wins on any conflict; the TURN LEDGER preserves
ORDER (one numbered entry per user turn, supersession by explicit tag, never deletion). The
Overrides block cancels the pi prompt lines that fight this shape.

## ensureWatcherTimer #2

The events that are bookkeeping rather than news. They stay in the file - they are how a strange
compaction gets diagnosed afterwards - but they are not what the operator opens the log to read, and
in a long session they pushed the result lines off the screen entirely (2026-10-21, on request:
"sometimes I cannot even see the compaction info because the log is filled with some silly useless
lines"). 'v' in the log view shows them again.
Deliberately NOT here: start, attempt, ok, fail, direct_fail, accepted_with_warnings, give_up, done,
every *_failed, retry, skip, elision_error, auto_compaction_held. Skip stays because "model X was not in"
the model list" is the answer to a question the operator asks; auto_compaction_held stays because when
pi asks early and nothing happens, that line is the only place saying why.
What each log event is CALLED on screen (2026-10-21). Kept next to LOG_CHATTER, which decides
whether a row is shown at all: this decides what a shown row says. The two lists answer different
questions, so an event can be visible with a plain name (compaction_landed -> COMPACTED) or
hidden with one (auto_compaction_held).

## ensureWatcherTimer #2

The exact number of tokens this server would read for this text, or null when it cannot be trusted.
- First POST /v1/messages/count_tokens (Anthropic-shaped): llama.cpp has it, Strata has it, and it
renders the chat template before counting, so it answers "what will the model read". Measured here:
65 ms for 170,000 characters and 56,495 tokens where the heuristic said 55,250.
- Then POST /tokenize, which every llama.cpp-derived server has had always and which ik_llama.cpp
implements without the route above. It counts the raw text with no template, which is precisely
what {SEGMENT SIZE} means: the material, nothing else.
An answer is accepted only when it is finite, positive and within a factor of two of the character
estimate. A server that returns 0, or counts something else entirely, must not be allowed to drive
what the model is told to aim at: a wrong count is worse than an honest guess, so anything odd means
the estimate is used and the record says which ruler was in force.

## ensureWatcherTimer #2

The size window one summarisation attempt is governed by, all in tokens.
THE LIMITS (absolute tokens since 2026-10-05; before that they were percentages of the model's
WINDOW, which is what broke the 2026-10-05 session: 2% of a 1,048,575-token window is an
acceptance floor of 20,972 tokens, which no model on earth answers, so all ~100 attempts came back
"too thin" and every one of them re-uploaded the whole context). min and max are now absolute
tokens; the AIM (what the instructions request) is still summaryPercent% of the window, clamped
into [min, max]:
- max    = min(summaryMaxTokens, the region itself) — a summary longer than the source is
pointless; the generation request is capped a little above max (GEN_MARGIN) so a
summary landing exactly on max is not killed by pi's length-stop;
- min    = min(summaryMinTokens, max) — a thinner summary is thrown away.
Then trimmed down where reality demands it: an overflow compaction (pi refuses a summary longer
than the history it replaces) and the room actually left in the session window after compaction.

## ensureWatcherTimer #2

The tolerance a finished summary is judged against, and its ONLY job (2026-10-20, the operator's
design, replacing a rule that used this same number for two things at once). A summary of up to
Summary max + GEN_MARGIN is accepted; anything longer is thrown away and the next model is asked.
The ask itself is Summary max exactly, so this is not room the model is invited to write into - it
is slack for the two ways an honest reply lands over the line: an online provider has been seen to
come back 300-400 tokens over an instructed ceiling, and our token count is never the server's
token count, because the two tokenizer files disagree. A local llama-family server treats the
ceiling as a hard stop, so on those this slack almost never gets used, which is the point rather
than an accident.
It used to also be the slack on what the ENGINE IS TOLD to generate, which is what made the number
meaningless to the operator: with max at 30,000 the log reported 30,512 back at them as if it were a
measurement, and before that the scratch and thinking allowances were ADDED on top of both, so a
model with a 30,000 aim was told to generate 34,608 and ran off the window. Nothing may push the
instruction past Summary max, because a local engine asked for more than its window holds does not
fail politely - it takes the session down as a crash. If the ask and the tolerance ever need to
differ again, that is a second constant with its own name, not a second meaning smuggled into this
one. Raised to 1000 once, during the emergency of 2026-10-19, and put back to 512 the next day.

## ensureWatcherTimer #2

How big the delivered summary really is. The provider's own output count is the truth (reasoning
tokens are a subset of `output` in pi's Usage, so they are subtracted); without it the text is
estimated slightly above pi's chars/4, because bullet summaries with numbers and paths tokenize
denser than prose.
The engine is the only counter that matters (the operator's rule, 2026-10-21). Where the server
tells us how many tokens it wrote, that number is the answer. Where it tells us nothing, the word
count is the next best thing: counting words is free, and one number per word beats one number per
character, because the ratio between words and tokens is stable for a given tokenizer. So the ratio
is LEARNED per model from the pairs we do get - every time an engine reports tokens for a text whose
words we counted, that model's tokens-per-word is remembered for the next time it stays silent.

## ensureWatcherTimer #2

ELISION - the rule-based first stage of context management (the harness paper's headline finding:
stage cheap elision before LLM summarization). In every live request, tool results outside a
protected recent window are replaced by informative stubs. The session file on disk keeps every
original; the tool CALLS stay visible; the model re-runs or re-reads when it needs an old output.
Implementation: a `context` handler - it fires before EVERY LLM call with the full transcript,
and the handler's returned message list is what actually gets sent. Stored history is never
touched, so pi's compaction machinery (and the summarizer, which preprocesses its own input)
always sees the raw originals.
Trigger and batching: the context is measured on the incoming (unstubbed) messages; when it
reaches `softPercent` of the model's window, every result outside the protected window becomes
eligible, oldest first. A NEW batch of stubs is applied only when it saves at least
`minSavingsTokens` - prompt caches charge a re-prefill whenever a prefix changes, so small
changes wait and one batch amortizes across many results. Between batches the applied set is
stable, so consecutive requests keep the same shape and the cache stays warm. A batch also needs
at least `minResultsToStub` results waiting - the per-fire prefill cost is roughly fixed (the
whole tail after the break point re-prefills), so batching across many results is what makes each
fire pay off. Batches also stop firing once the live request comes within `stopGapTokens` of pi's
compaction trigger - a batch there would be wasted work, since compaction resets everything
moments later.

## ensureWatcherTimer #2

What the SERVER holds is what matters: the last real usage report this session got back
(prompt tokens including the cached prefix, plus what the model generated). pi's own number is
an estimate over the RAW transcript, so on an elided session it reads ~100k higher than the
request that actually goes out - which is how a 60 % trigger line came to arm at a completely
different size than the one on the status bar.
THE context number for this extension, in one place: what the model server currently holds.
Order of preference, first non-zero wins:
wire   - the prompt tokens the server reported for the newest request after the last compaction.
Elisions are in that figure, so it is the real fill of the model's context.
pi     - pi's own estimate from the same usage figure.
sum    - what the NEXT request would carry, for a model that answers without usage figures, right
after a compaction, or straight after a restart. Taken from pi itself: the entries pi would
assemble (newest summary + the kept tail + everything added since), NOT the whole archive.
`all` is the whole branch next to it - what is on disk rather than what is on the wire, which on an
elided session is about twice as big - and `justCompacted` says the newest of those entries is the
summary itself: the one state where no measure exists yet. `how` says which was used, so a zero can
never be mistaken for an empty context.
Tokens one session entry adds to the context. A summary rides on the entry's own text (pi keeps it in
`summary`, not in a message); pass countSummary=false to leave summaries out of a total.

## ensureWatcherTimer #2

Eligible = tool results before the protected window, oldest first, not yet applied, big
enough for a stub to pay for itself (the stub itself costs ~60-100 tokens).
THE BREAK POINT (2026-10-05: this is what made every batch re-upload the WHOLE context). The
prompt cache stays valid up to the first message whose content changes, so a batch that stubs the
OLDEST results changes the beginning of the context and forces a full re-prefill: all night on a
700k session every batch cost a ~700k-token prefill instead of the tail it saves. Batches now
fill in NEWEST-FIRST (the block just behind the protected window) and never touch anything at or
before the oldest stub already on the wire, so a batch re-prefills only the tail after it.
PASS 1 - where is the cache break point already? It sits at the OLDEST stub on the wire. The
server's prompt cache is valid only up to the first message whose content differs from the last
request, so a batch may only change things AFTER it. (The first version of this fix looked for
the frontier in the same loop that collected the candidates, so it was found at index 53 while
indexes 2..52 had already been collected - and the next batch broke the cache at index 2 again.)

## ensureWatcherTimer #2

Batch check - a batch fires only when ALL three hold:
1. enough results waiting (minResultsToStub) - the per-fire prefill cost is roughly fixed
(the whole tail after the break point re-prefills), so batching is what pays;
2. enough savings (minSavingsTokens) - the wire must shrink meaningfully;
3. the batch actually brings the live request below the stop line (a GAP below pi's
compaction trigger, window - reserveTokens) - a batch that leaves the wire at the
compaction doorstep is wasted churn: compaction resets everything moments later and
stubs everything unconditionally anyway. When a context leap jumps past the stop line,
a big enough batch can still rescue the request WITHOUT compaction (the paper's 0%
managed-overflow case).
Of everything eligible (all of it AFTER the break point), stub the block closest to the END of the
context, up to maxResultsPerBatch of them; older ones stay raw on purpose, see above.

## ensureWatcherTimer #2

Put a ceiling on the model's own thinking, in tokens, for THIS request - the one lever that works
on a server whose chat template we do not control (2026-10-20, taken from the servers themselves).
Thinking gets decided twice, and the prompt only reaches the second decision: the server picks the
size of the thinking channel from the REQUEST PARAMETERS before generating anything, while the
no-think tag is text the model reads later, by which time the budget is already spent. So say it in
the request. Every recognised spelling goes, because a server ignores the keys it does not
implement and this one request has to serve three engines: reasoning_budget_tokens is Strata's cap
(its run-config applies the server-wide value only when the request sends none of its own, so
naming ours wins), and thinking plus chat_template_kwargs is what ik_llama.cpp's server reads.
Never send 0 to Strata: its documentation says 0 means no cap rather than no thinking, which would
hand the whole request back to the thinking channel - the exact shape of last night's failure. A
small non-zero budget still lets a model close its own think block cleanly. The total asked for does
not change with this: the summary is what gains the room thinking gives up.

## ensureWatcherTimer #2

The fork's own region serializer - the thing pi's serializeConversation does, but with
DETERMINISTIC TURN NUMBERS and stubs/caps applied in the same pass. Turn boundaries follow pi's
own rule (user message, bash execution, custom message, branch/compaction summary). Labels:
[User - Turn 7]: ...        [User bash - Turn 7]: ...     [Note - Turn 7]: ...
[Assistant thinking]: ...   [Assistant]: ...
[Tool call bash("bun test")]:      [Tool result]: ...
A result longer than 2,000 characters becomes an informative stub (inputStubs on) - pi's
head-only clip then has nothing important left to cut. Argument VALUES are capped (argCap).

## ensureWatcherTimer #2

How much of the TURN LEDGER really carries a user prompt (2026-10-18), and how many prompts appear
anywhere in the summary (2026-10-19 - see userPromptsGiven for why that second figure exists).
`turns` and `withPrompt` look at the ledger: the Turn entries and, for each, whether its sub-lines hold a
line labelled with the word user with real text after the colon. The label is whatever the template chose
("user:", "user asked:", "USER PROMPT:") so the tell is the word rather than the wording, and a label with
nothing after it does not count, because an empty user line is how a model satisfies the shape without
carrying the prompt it was told to quote.
`linesAnywhere` counts the same labelled lines across the whole summary, ledger or no ledger. It exists
because the 00:50 summary on 2026-10-18 carried no Turn labels whatsoever - the shape gate said so out loud
in its warning - and a check written only on turn labels reads "zero turns, nothing to complain about" in
exactly the case where the model threw the structure away along with the prompts.
Counts, not a verdict, so the caller can tell a thin ledger from an empty one. Falls back to the whole
summary when there is no ledger section, so a differently shaped template still gets measured instead of a
silent zero.

## ensureWatcherTimer #2

How many prompts the summarizer was HANDED (2026-10-19). serializeRegion marks each one - "[User - Turn
12]: ...", and for a /bash entry or a custom note "[User bash - Turn 12]" / "[Note - Turn 12]". Those are
all things the user gave the session, so they all belong in the denominator. "[Earlier summary - Turn N]"
is deliberately not counted: that is the previous summary rather than a prompt, and counting it would make
a chained summary look better supplied than the model was.
Only prompts with at least four letters count, mirroring the summary-side rule above. A prompt whose text
was trimmed by preprocessing still counts as long as any of its text survives the trim - the four-letter
rule is what decides, not whether the material was shortened, and a prompt emptied to nothing was never
there to quote.

## ensureWatcherTimer #2

The engine's instruction ceiling, and a hard one (2026-10-19, the user's rule after a crash):
Summary max, exactly, is what the engine is told to generate, and that figure is the model's TOTAL
allowance for this summary - thinking included, because a thinking level is a model setting rather
than extra room, and it is capped inside the request (see thinkingBudgetFor) so it spends from
inside the allowance instead of quietly enlarging it. The same number reaches the template as
{MAX TOKENS}, so what the model is told and where the server will stop it are the same figure.
The margin above max is not part of this ask at all; it is only the tolerance the finished answer
is judged against, and it is applied at the size checks, not here.
History, kept because these lines are where the design went wrong twice: the allowances used to be
ADDED on top of max (a 30,000 aim became a 34,608 request and ran off the window), and then the
ask was max + margin, which made the ceiling in the log look like a measurement of the reply.

## ensureWatcherTimer #2

A generation that ran out of room is not automatically worthless, and treating it that way is
what stopped every compaction tonight (2026-10-19, live): the local model's hidden reasoning
ate most of the request budget, the visible summary stopped at the cap, and the attempt was
thrown away the instant stopReason arrived - before a single character of it was looked at. The
user's reading was right that the number made no sense: 30,512 is not a measurement of what
the model wrote, it is our own acceptance ceiling (Summary max + GEN_MARGIN) reported back at
them, while the request had actually been capped higher and the model had filled it. Judge what
arrived instead, after it has arrived: a half-finished sentence at the end is a loss, a session
that cannot compact is a bigger one. So this run keeps its honest numbers, and the gate below
(which is the ordinary size and shape checks, run on the text as it would be after a clean stop)
decides.

## ensureWatcherTimer #2

What "timed out" ought to mean (2026-10-19, item (a) of the 19:15 report). The plain wall-clock
ceiling above cannot tell a slow model from a dead request: it cut a generation that was still
producing at exactly ten minutes, and the online model that took over then finished the same
summary and billed 29.2k output tokens. Two clocks say what anybody meant. SILENCE is the one the
operator sets, and every character that arrives restarts it, so a model that keeps writing keeps
its request. The BACKSTOP is the one nobody sets, because a stream that produces forever is still
a stream that will never finish. Whichever expires first aborts, and the record says which it was:
"it went silent" and "it was still writing when the ceiling arrived" are different faults with
different cures, and "timed out" hid the difference.

## ensureWatcherTimer #2

The scratchpad share of the output budget rides on the cap here. It must NOT look at the built
instructions: those are assembled below this line, and reading them here sat in their temporal
dead zone - every compaction in this session threw "Cannot access 'instructions' before
initialization" and pi kept re-triggering on a context it could not compact. The allowance is
therefore unconditional now (the template asks for scratch work anyway) and is clamped by what
this model is permitted to emit, which is the only hard ceiling there is.
The fallback through pi's own compact() carries the same total allowance as a direct request:
Summary max and nothing beyond it. The thinking and scratch allowances are no longer added on
top (2026-10-20, the operator's design) - thinking is capped in the request body, so it spends
from inside the allowance, and a candidate that needs more than the allowance for both is a
candidate to move off rather than one to run past the window. dCeilingFallback is only the
tolerance the answer is judged against, never a licence to ask for more.

## ensureWatcherTimer #2

THE USER-PROMPT CHECK, measured against the material rather than the summary (2026-10-19).
Every model on this user's list is set to preserveUser = verbatim, so a ledger without the
user's own words is not a style - it is the loss of everything they asked for, and it was
passing: the summary that landed at 23:02 on 2026-10-18 carried one user line where the run
six hours earlier carried twelve, and the one at 00:50 quoted three out of some forty, said
"no Turn labels in the ledger" in its own shape warning, and was accepted anyway. Counting
turns inside the summary cannot see that case - with no labels there are no turns to be sad
about - so the denominator is the segment we sent: every prompt in it is marked
"[User - Turn N]" by serializeRegion, and every prompt quoted in the summary is a line whose
label says user. Both sides count only text of at least four letters, so a session run on
"ok" and "go on" cannot convict an honest summary.
Nothing quoted is a rejection. Quoting under a quarter of what was handed over, in a segment
large enough for a quarter to be more than a rounding artefact, is a rejection too: the next
model in the rotation costs one more upload, and a session that has lost its instructions
costs the session. Anything short of that but still short is a warning, because a thin ledger
is usually a model that ran out of room rather than one that ignored the instruction.

## ensureWatcherTimer #2

Which key is this keystroke? 2026-10-21, and this is the fix for "no key works in the settings".
This function used to compare bytes: Escape was the single character 0x1b, Up was the four
characters ESC [ A, Ctrl+A was the byte 0x01. That stopped being true when pi turned on the Kitty
keyboard protocol (pi 1.3.0, and Windows Terminal accepts it): a keystroke now arrives as an escape
sequence, a held key sends release and repeat events between the presses, and a lone Escape is
encoded as well. Every hand-decoded comparison in the file went quiet in the same moment, which is
exactly what the operator reported - Ctrl+ shortcuts dead, Escape not leaving a screen, holding an
arrow registering only the first press.
The encoding is pi's business now, so pi decodes it: matchesKey() knows both the old sequences and
the new ones. A key release is answered with "" so the screens ignore it, otherwise a held arrow
would step twice per repeat, once for the press and once for the release.

## ensureWatcherTimer #2

The internal defaults for a model that was never selected: it is only ever asked with these if
it is put in the Compaction models list and the user never opens its submenu.
The options a model gets when it JOINS the try order (2026-10-21, defect #53).
Every new model started with every sampling flag on "ignore". For an online model that is right - the
provider's server already knows how that model should be driven. For a local Qwen it is wrong in a way
the operator pays for on the next compaction: the server default is a chat-tuned temperature with no
top_k and no min_p, which is how a local model produces a wandering summary. Qwen publishes instruct
values, so a model whose name says Qwen joins with those, and every other model joins exactly as before.
A model already in the config is never touched - including one whose flags the operator set back to
ignore on purpose. This is a starting point, shown on the Sampling flags page and changed there.

## ensureWatcherTimer #2

Live reserveTokens from pi's own compaction settings (falls back to pi's documented default).
WHERE AUTOMATIC COMPACTION FIRES, in tokens, for one model window (2026-10-21).
The operator sets a trigger point in tokens. That number alone is unsafe, because a model with a
smaller window runs out of room long before it: a 139,400 target on a 64,000-token model would
simply never fire. So the point is the EARLIER of two numbers - the target, and the point this
model has to compact at anyway.
pi's ask point is NOT part of this arithmetic any more (2026-10-22, the operator's decision). It
used to cap the line, because a pi ask held below this extension's point costs a full pass over
the context on every turn in between. The timer is the trigger now, so there is no ask to hold;
the alarm is set to ring above this line, and the gate in the hook is left for the drift case
(pi's reserveTokens edited while pi runs, a different model loaded), where it says so.
The reserve used when the operator set no number: a tenth of the window, never less than a
normal summary plus a tail. Derived on every call, so it follows the model in use.

## ensureWatcherTimer #2

WHERE automatic compaction fires, for any window: the window minus the reserve the operator set.
This is the extension's own decision, and since 2026-10-22 the operator's own timer is what acts
on it (option A, their choice): the timer measures the context from the usage figures the server
already reported - which costs nothing to look at - and starts a compaction at this line, at a
safe point, while the session is running. pi's threshold is the ALARM: it lives above this line,
rings only if the timer failed, and this extension still answers it with its own summary.
pi's own reserveTokens therefore no longer moves this line. It used to: the point was capped at
pi's ask point so that an ask could never be held. The cap was honest arithmetic - a held ask costs
one full pass over the context per turn - but it meant a smaller-context model compacted at pi's
number instead of the one on the board, and the operator rejected that trade (2026-10-22). With
the timer as the trigger there is nothing to hold: the line is the number on the board.

## ensureWatcherTimer #2

The Compaction models screen: one list holding the selected models first (with their position
badge) and every other model pi can reach below them, alphabetically.
Keys:
digits 1-9   on a SELECTED model: move it to that position (0 = the end); on an UNSELECTED
model: filter the list (names contain digits)
space        on a SELECTED model: remove it from the order (it drops to its alphabetical
place among the unselected; its other settings are kept); on an UNSELECTED
model: a space in the filter
enter        SELECTED model: open its options submenu; UNSELECTED model: add it to the END
of the order with default options
type         filter the list
Esc          back to the board

## ensureWatcherTimer #2

One selected model's options. Every row toggles through its values on Enter/Space:
1. thinking (off..max, adjusted to what the model accepts)
2. draft block (off / mini / full - a hidden analysis before the summary, stripped afterwards)
3. timeout (2 min .. 60 min, unlimited)
4. summary aim: percent of the ctx window (the same base as the min/max limits)
5. summary floor (tokens)
6. summary ceiling (tokens)
7. no-think tag (the marker or empty)
Esc goes back; changes save the moment they are made.

## ensureWatcherTimer #2

The user's template library (2026-10-07 request): any number of saved templates, named by the
extension. Enter opens one in pi's editor - the normal editing box, where Enter saves, Shift+Enter is
a newline and Esc leaves the text unchanged - with the template's name and the insertable tags shown
above it. The keys on the header line work on the highlighted row. The ACTIVE template is a copy
assigned from here (Ctrl+A); it is not edited in place, which is why the Templates menu shows it as a
preview.
The template library's dialogs, run by the board loop. They CANNOT run inside the library screen: a
modal dialog (editor, confirm) opened while another custom screen is mounted leaves the terminal with
no focused input - the 2026-10-07 report was an editor that took the keystrokes, gave nothing back, and
needed a force close. So the library only asks, and the work happens here, over the plain chat, with the
board closed; the return value reopens the board on the library list. "" = the built-in default.

## ensureWatcherTimer #2

The "Compaction settings" submenu (2026-10-06): the two compaction switches, the retry knobs, the
local no-think tag and - behind one more row - the per-model compaction options.
THE RULE THE TWO SWITCHES OBEY: exactly ONE compactor is ever on. Turning this extension's
compaction on writes pi's own compaction OFF; turning pi's on writes this extension's OFF. Both off
is allowed - then nothing compacts and the session just grows. Elision is independent of both and
lives in its own submenu. pi's switch is a value in pi's settings.json that pi reads at startup, so
changing it does nothing until the user reloads; every change here says so.

## ensureWatcherTimer #2

A text screen of this extension's own (2026-10-06). pi's extension UI context carries only select,
confirm, input and notify - there is NO editor in it - so every editor and preview this extension
opened through it silently did nothing: picking a row closed the menu and left the user back at the
prompt with nothing shown. The previews and the editors are therefore drawn by this extension itself,
in the same kind of overlay the settings board uses.
Keys: Up/Down move (in a one-line field they do nothing), Left/Right move the caret, printable keys
type, Backspace deletes before the caret, Enter starts a new line (in a one-line field it saves),
Ctrl+S saves, Escape closes without saving.
How many screen rows a read-only view shows at once - the same window the editable view uses.

## ensureWatcherTimer #2

Set when this extension's own watcher fires while the loop is mid-turn. pi's compact() begins by
aborting what is running and pi never resumes an interrupted turn on its own, so a mid-turn trigger
would otherwise end the session's work until the user types something (2026-10-18: that is exactly
what it did, on a 55-minute turn, with nothing left to defer to). Armed here, spent by the
session_compact handler below once the summary is really in the journal, and expired by age so a
cancelled compaction cannot resume a turn much later.
Tell pi-stamp-plus that THIS compaction was cancelled on purpose (2026-10-21, on request).
pi reports every cancelled compaction the same way: `session_compact_failed` with `aborted: true`
and no error message, so a hold from the gate above and an Escape key pressed during a summary look
identical to anything listening. pi-stamp-plus stamps that as `Compaction . ABORTED`, which reads
like a fault when it is a Tuesday. The two extensions cannot read each other's mind, so the moment
of cancelling is written down next to the session file; pi-stamp-plus drops the stamp for that note
and for nothing else. This extension never reads the file back, and a session file pi did not tell
us about means no note and no change.

## ensureWatcherTimer #2

THE GATE: should a compaction that PI started be held back for now?
pi's threshold and this extension's trigger point are two different numbers on purpose. pi's is set
low (its settings.json `compaction.reserveTokens`) so that pi asks early - pi's request-time check is
the one trigger that reliably fires while a session is mid-turn, which no timer of ours can promise.
This extension's line is `window - extReserve(ctx)`, the percentage on the settings board, and it is
the number that decides when a compaction is actually worth writing.
Between the two, the answer is a cancel: nothing is summarised, nothing is cut, pi asks again on the
next request, and the same request gets a real summary once our line is reached.
A HOLD IS NOT FREE, and the operator found the price on 2026-10-21: "autocompaction now initially
triggers full context re-read before it starts creating the summary. this wasn' happening before.
Correct, and this rule is the mechanism. Cancelling pi's ask lets the session carry on, so every turn
in that gap goes to the model with the context UNCOMPACTED - a pass over the whole of it, minutes on
a local model, before the summary is finally written. With pi's reserveTokens at 50,000 and this line
at window-30,600 the gap was 19,400 tokens of re-reading, which is exactly what was reported.
So the two numbers have to agree. When they do not, the log line, the notice and the settings board
all name the reserveTokens value that ends it. The gate stays for what it is genuinely for: a
compaction that arrives materially early, which once the numbers are aligned means a mis-set
number or a session on a different window - not the everyday path.
Returns a sentence for the log while it should be held, and null when the compaction may proceed.
Pure numbers by design - no files, no pi objects - so the rule is testable with a rig.

## ensureWatcherTimer #2

WHAT THE AGENT LOOP IS DOING RIGHT NOW, kept current from pi's own events.
WHY THIS EXISTS (2026-10-14, twice in one session). The watcher used to start a compaction whenever the
size crossed the line, at any moment, because 2026-10-07 had removed the idle gate on the theory that a
long working turn must not be allowed to overflow. What that actually bought was this: ctx.compact() took
pi's abort signal with it, so the request in flight died with "This operation was aborted" - a bash tool
cut off 0.5 s into running, a turn whose only output was an abort - and the turn never restarted, so the
session sat there while the summary ran and after it. Then the summary, asked to compress an already
compacted conversation, answered "Nothing to compact (session too small)" and read as a second error.
The trigger also mistook the on-disk history for the live context and believed it was 234,591 tokens in
a 20,300-token conversation (see measureContext), which is what made it knock at all.
WHAT THAT BOUGHT, AND WHY IT IS NOW HALF-REVERSED (2026-10-15, written down because the pendulum has swung
twice). Started only at a settled point, the trigger fired 10,000 tokens LATE in a session that had already
crossed its line, and the summary it finally ran at 178,039 tokens was too late to cover the newest material:
the ledger it wrote got as far as the seventeenth turn and stopped mid-sentence, so the eighteenth prompt was
neither summarised nor kept, which is what the user read as "all the newer prompts are missing". Meanwhile
pi's own between-rounds check - the one non-aborting mid-run trigger there is - was silent the whole time.
Why is still open: the guess written into this file the same evening, that pi's measure sits 20,000-30,000
tokens under the real fill because a projection holds no system prompt and no tool schemas, was taken out by
its own regression test hours later, because the newest usage figure already counts the whole request it
answered - system prompt, tool schemas and all - and pi only adds a size guess on top of it, so its measure
is at least as big as ours. The padding was never written to pi's settings; what is left of that theory is a
log field, piMeasureDelta, and the answer will come from those numbers rather than from this paragraph.
What is not in doubt is that a settled-only rule left the session walking from the line to the wall by itself.
The rule now is: a compaction may start WHILE THE LOOP WORKS, at a round boundary. Measuring happens at any
moment; starting happens when nothing is answering, no tool is running, nothing is queued for the next round,
and every tool call the newest answer asked for has its result back - that is the boundary between rounds,
and it recurs many times in a long turn. It is still not free: ctx.compact() is pi's manual path and pi's own
source begins it with `await this.abort()`, so a start mid-run interrupts the turn, and threshold compaction
does not resume it. That is the price, and it is now accepted deliberately, because the alternative measured
out worse: one interrupted turn against a session that reaches 200,000 tokens and is refused, and a refused
session cannot be compacted at all. On top of that, within MIDRUN_DANGER_TOKENS of the window the niceties go
out of the window with them and the compaction starts in front of running tools rather than not at all.
Elision, the capped max_tokens and the engine's own fit_max_tokens clamp guard the quiet part of a turn
exactly as before.

## ensureWatcherTimer #2

Can a compaction start right now without disturbing anything? Returns the reason it cannot, so the log
says which door was shut. Everything here is an API pi really exports (types.d.ts: isIdle(), signal,
hasPendingMessages()) or a fact this file observes itself; the 2026-10-07 gates read ctx.compaction.*,
which is not a member of ExtensionContext at all, so those three checks were dead code and silently
never gated anything.
How long after an answer lands, or a tool finishes, the mid-run gate still waits before it will start a
compaction. The race it buys off is the one the user was bitten by on 2026-10-14 at 14:53:34: an answer
arrives, its tool calls have not begun to execute yet, and a start in that gap aborts a tool 0.5 s into
running it. One second is longer than pi needs to commit to the next round or to its tools.

## ensureWatcherTimer #2

WHAT PI'S OWN BETWEEN-ROUNDS CHECK MEASURES, AS A SIGNED DIFFERENCE AGAINST THE WIRE (2026-10-15, written
down because the first theory about it was wrong and the log is what will settle it). pi arms its mid-run
trigger from `settings.compaction`, and its `_exceedsCompactionThreshold` asks
`estimateProjectedContextTokens`, which is the newest usage figure in the projection plus a chars-per-3.5
size guess over the entries that came after it. The obvious guess was that a projection holds no system
prompt, no tool schema and none of the guidance this file appends, so pi's figure would sit some 20,000
tokens under the fill the server really holds, and that a line written into pi's settings ought therefore to
be padded by the shortfall. The regression harness says the arithmetic runs the other way: the newest usage
figure already counts the whole request it answered - system prompt, tool schemas and all - and pi only adds
to it, so its measure is at least as big as ours, and padding it would have made pi's line EARLIER than
ours, not later. So this number is recorded and nothing more is done with it: it is the measurement that
will show, scenario by scenario, why pi's check crosses or does not cross while this file's watchdog holds.
Positive means pi sees more than the wire; negative means it sees less; a zero means there was no usage
figure after the newest summary to go on, which is itself a reason a check cannot fire.

## ensureWatcherTimer #2

Keep pi's own compaction settings in step with this extension's trigger (2026-10-06). pi decides by
`contextTokens > window - reserveTokens`, and its contextTokens is the usage figure the server
reported - the same elided fill this extension measures - so writing our reserve into it makes pi's
line exactly our line. Nothing else changes hands: pi's stock summarizer can never answer, because
the session_before_compact hook of this extension replies to every compaction request with its own
summary, or cancels.
FULLY OFF (2026-10-06): the compaction switch is off AND the elision switch is off. In that state this
extension must behave as if it were not installed at all - it must not change a request, not answer or
cancel a compaction, not start a timer, and not write pi's settings.json - so that pi, or another
compaction extension working in a completely different way, runs exactly as designed.
The command /compact-plus still exists: a command nobody calls does nothing.
pi's own compaction flag, reported to the user once per process. Every line of this extension
READS pi's settings and not one writes them: the user switches pi's compaction on or off in pi's
own file, and this extension's automatic trigger works with pi's flag either way, because pi's
compact() and /compact ignore it - only pi's own threshold and overflow checks consult it.
(2026-10-18: the coupling that armed pi's flag on every tick is deleted, after the user found
pi's compaction enabled again without ever having enabled it, twice in as many days.)

## ensureWatcherTimer #2

THE TRIGGER IS OUR OWN TIMER (2026-10-22, option A, the operator's choice: "if you can have your
own trigger and we don't need pi's trigger then great"). Until today this block stood the timer
down whenever pi's compaction.enabled was on, because pi's threshold was made the trigger and a
second scheduler would have raced it. The race was the wrong thing to fear: what the stand-down
bought was a line that belonged to pi's reserveTokens, so a smaller-context model compacted at
pi's number. Now the timer decides WHEN, from a measurement that costs nothing (the usage
figures the server already wrote into the session), and pi's threshold - with its reserveTokens
set far smaller than our reserve - is the alarm that rings only if this timer failed. The hook
still answers pi's ask with our own summary, and pi's own summariser is still the last resort.
What has NOT changed: the safe point below (no compaction in the middle of a write-up or a
pending tool call, exactly like pi waiting for the current step), the once-a-minute floor, the
refusal to fire twice on an unmeasured guess, and `urgent`, which drops the safe point when the
request already in flight is within MIDRUN_DANGER_TOKENS of the whole window.

## ensureWatcherTimer #2

pi retires a captured ctx the moment the session behind it is replaced or the extension reloads, and
it announces the fact by THROWING from every property of that dead ctx: "This extension ctx is stale
after session replacement or reload...". Every timer in this extension holds a ctx, or the api, or
both, across that line - and on 2026-10-20 the watchdog tick reached for a session id through a ctx
the settings menu had already retired. The throw happened inside a setInterval callback with no one
above it to catch anything, so it surfaced as an uncaughtException and pi exited.
A timer that finds its context dead therefore stops itself, drops what it was holding and waits:
the next agent_start leaves a live ctx behind, and the first live request starts the clock again.
A timer error of any other kind is still swallowed, because a timer callback has nowhere else for an
error to go - but now it is swallowed INTO THE LOG, where it can be read, instead of into the exit
of the program or into silence.

## ensureWatcherTimer #2

WHAT PI DID WITH A COMPACTION AFTER THIS EXTENSION'S OWN HOOK WAS DONE (2026-10-14). The hook sees a
compaction it takes part in; these two see every one, including the ones pi raises on its own side -
and until now those were invisible: a refusal for want of anything to summarise appeared on screen as an
ERROR with nothing in the log to explain it, and a genuine failure from that path was never counted by
the circuit breaker, so the trigger could knock on the same door once a minute for the rest of the day.
WHERE THE AGENT LOOP IS (2026-10-14). These handlers exist for one reason: to be able to say, at any
instant, whether starting a compaction would interrupt something. pi emits them all (types.d.ts); the
counters are kept defensively (a start without its end, or an end without its start, cannot drift the
count below zero or leave the loop stuck as running) and every one of them also refreshes the watcher's
context, so the trigger works before the first request of a session has ever gone out.

## ensureWatcherTimer #2

LAND THE SUMMARY WITHOUT STOPPING THE TURN (2026-10-22, #43), when `landWithoutAbort` is on.
Today this extension starts its compactions with `ctx.compact()`, which is pi's MANUAL path: pi
stops the running loop, writes the summary, and waits for the operator to type something. That is
the pair of messages the operator reported - "operation aborted" and the ABORTED stamp - and it is
not a pi limitation: pi's own threshold compaction happens inside the request pipeline and never
stops anything, because a compaction that is merely APPENDED to the session is picked up by the
loop's next request on its own (pi rebuilds `context.messages` from `buildSessionProjection()` in
`prepareRequest`, agent-session.js:434-446).
So: build pi's own preparation, hand it to the same `summarizeWithRotation` the event handler
uses (identical model rotation, shape gate, retries and log rows), and append the result with the
same six arguments pi passes at agent-session.js:2268. What is NOT reproduced is pi's
`session_compact` event, which an extension cannot emit: the on-screen transcript may keep the
pre-compaction view until a reload, and pi-stamp-plus writes no stamp for it. Both are recorded
here in the log instead, and the switch stays off until that has been watched on a real run.
The danger line never takes this path: inside MIDRUN_DANGER_TOKENS a request is about to be
refused, and a refused session cannot be compacted at all, so an interrupted turn is the lesser
failure and pi's interrupting path stays in charge.
