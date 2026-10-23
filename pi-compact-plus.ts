/**
 * pi-compact-plus — the full compaction pipeline for pi: elision in the live session, then a
 * self-built summarization request (no pi prompt underneath), producing the turn-ledger summary.
 *
 * The whole pipeline, in order:
 *
 *   1. ELISION (live, rule-based, no LLM): tool results older than the protected recent window
 *      become informative stubs in every request (a `context` handler; stored history untouched).
 *      Batches fire only when they clear the token floor AND the results-count floor AND can bring
 *      the request below the stop gap under pi's compaction trigger. This delays compaction,
 *      sometimes indefinitely.
 *   2. COMPACTION (pi fires session_before_compact at its usual trigger; elision has kept the wire
 *      small): the extension builds the ENTIRE summarization request itself:
 *        - the region serialized with DETERMINISTIC TURN NUMBERS ([User — Turn 7]: ...) using pi's
 *          own turn-boundary rule; the turn counter continues across compactions via the
 *          CompactionEntry details (details.lastTurnNumber);
 *        - tool results as informative stubs (the exit line survives; pi's 2,000-char head-only
 *          clip never touches anything important), tool-call argument values capped;
 *        - the previous summary LABELLED (what it covers, that its numbering is final, carry
 *          forward, continue numbering);
 *        - ONE instruction source: the generated turn-ledger template (no pi format block, no
 *          override sentences — nothing to blend with).
 *      The call goes through the model registry (streamSimple — request-time auth), maxTokens set
 *      directly, no prompt-cache writes.
 *      After the call: rough-work stripping, size gate, SHAPE gate (sections + turn-label sanity),
 *      pointer line; the turn counter stored in details.
 *      FALLBACKS: direct request fails → pi's compact() with the extension instructions (the old
 *      path) → fails → next model in the rotation → all failed → pi's own compaction.
 *   3. PER-MODEL OPTIONS: thinking, timeout, aim/min/max shares of pi's reserveTokens
 *      (off/mini/full), summarizer input (stubs/raw), arg cap, and four preservation options
 *      (stubs/user/replies/thinking: verbatim | detailed | summary | brief) that generate the
 *      template's RULES wording. Every description states its default.
 *
 * pi's two compaction numbers (settings.json "compaction": reserveTokens, keepRecentTokens) are
 * editable on the board (free number entry; /reload applies them); the summary sizes are shares of
 * reserveTokens and the elision protection mirrors keepRecentTokens.
 *
 * Settings menu:  /compact-plus
 * Attempt log: ~/.pi/agent/pi-compact-plus.log
 * Commands: /compact-plus status | on | off | models | move | options | elision | retries |
 *   marker | instruction | log | preview [ref]   (no argument opens the menu)
 */

import { compact, convertToLlm, estimateTokens, serializeConversation } from "@earendil-works/pi-coding-agent";
import { streamSimple as piStreamSimple } from "@earendil-works/pi-ai/compat";
import { getSettingsListTheme } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Theme } from "@earendil-works/pi-coding-agent";
import type { SettingItem } from "@earendil-works/pi-tui";
import { SettingsList, truncateToWidth, isKeyRelease, matchesKey } from "@earendil-works/pi-tui";
import { spawnSync } from "node:child_process";
import type { Component, Focusable, KeybindingsManager, TUI } from "@earendil-works/pi-tui";

/** The context object pi hands to handlers and commands. Typed loosely on purpose: this file is
 *  checked on its own, without the package type declarations on the module path. */
type Ctx = any;
import { appendFileSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

const AGENT_DIR = join(homedir(), ".pi", "agent");
/** Test isolation: point COMPACTION_PLUS_CONFIG at a scratch config and the extension reads that
 *  instead of the live one - unit tests can then never touch the user's real settings. */
const CONFIG_PATH = process.env.COMPACTION_PLUS_CONFIG?.trim()
	? resolve(process.env.COMPACTION_PLUS_CONFIG.trim())
	: join(AGENT_DIR, "pi-compact-plus.json");
// The log and the state dir live next to the config: the tests (COMPACTION_PLUS_CONFIG pointed at a
// scratch dir) then never write to the user's real log (the 2026-10-04 bug: test-run elision events
// with fake-context numbers polluted the live log and looked like real sessions failing).
const LOG_PATH = join(CONFIG_PATH, "..", "pi-compact-plus.log");
const LOG_MAX_BYTES = 1_000_000;

/** Pseudo model reference meaning "the model pi is using right now". */
const SESSION_REF = "session";

type Level = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
type Kind = "local" | "online";
/** How much of a given content type the summary must keep. Each value maps to its own instruction
 *  sentence (RULE sentences in buildTemplate), so the template wording always matches the option. */
type Preserve = "verbatim" | "detailed" | "summary" | "brief";
/** A character cap: a number, or "full" = uncapped. */
type CharCap = number | "full";

/** Everything the ELISION stage does. Extension-wide (not per model): it fires in the live session
 *  before the summarizer is ever asked, for every model alike. */
/**
 * Percent-based scaling of the two anchor values (2026-10-01 - THE UNLINK). pi's compaction
 * settings are absolute token counts tuned for one window size; wrong for any other. The extension
 * instead computes keepRecent and reserveTokens from the ACTIVE model's context window at every
 * use, clamped to absolute min/max, so a mid-session model switch re-derives everything on the next
 * request. Nothing here touches pi's settings.json and nothing needs /reload.
 */
/**
/**
 * The two sizes the operator sets, both in TOKENS (2026-10-21, was a percent of the window with
 * min/max limits). Percentages were unreadable on a 170k model and wrong on any other: the same
 * 82% meant 139,400 tokens here and 52,480 on a 64k model, and the four limit rows existed only
 * to catch what the percent did on windows nobody was testing.
 *
 * Both are amounts of the window left FREE, which is how pi words its own compaction settings
 * too, and it is the form that survives a model change: 'leave 30,600 tokens free' says the same
 * thing on a 170k model and on a 64k one, while 'fire at 139,400' means something on this model
 * only. The compaction line is the window minus the reserve.
 */
type ScalingConfig = {
	/** How much of the window stays free above the compaction line: automatic compaction fires once
		*  the context passes `window - reserveTokens`. 0 = no number chosen by the operator: keep
		*  a tenth of the window free, never less than 16,384 tokens (see autoReserveFor). */
	reserveTokens: number;
	/** How much of the window stays free for the verbatim tail: what compaction keeps word for word
	 *  and what elision refuses to stub. pi's keepRecentTokens is the same idea. */
	preserveTokens: number;
};

const SCALING_DEFAULTS: ScalingConfig = {
	reserveTokens: 0,
	preserveTokens: 20_000,
};

/** The window an old percentage config is translated against when no model is loaded yet. */
const LEGACY_WINDOW = 170_000;

/**
/**
 * Read the scaling block, translating the older shapes into token counts so a config written
 * before 2026-10-21 keeps the line it was really using - a number that moves when the file is
 * merely read again would compact a live session early. A stored `startPercent: 82` on a
 * 170,000-token window is a 30,600-token reserve, which is what the board shows; the min/max
 * limits that used to surround the percentages are applied during that translation and then
 * disappear, which is all they were ever for. A file with token counts in it is read as it is.
 */
export function migrateScaling(raw: any, base: ScalingConfig, window = LEGACY_WINDOW): ScalingConfig {
	const sc = raw?.scaling ?? {};
	const num = (v: any): number | undefined => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? Math.round(v) : undefined);
	const reserve = num(sc.reserveTokens);
	const preserve = num(sc.preserveTokens);
	// A file with no scaling block at all is a NEW file, not an old one: it gets the defaults below,
	// not a translation of percentages it never had.
	const legacyKeys = ["startPercent", "reservePercent", "reserveMin", "reserveMax", "keepRecentPercent", "keepRecentMin", "keepRecentMax"];
	if (reserve === undefined && preserve === undefined && !legacyKeys.some((k) => sc[k] !== undefined)) return base;
	const win = num(sc.window) && (sc.window as number) > 0 ? Math.round(sc.window) : window;
	// The one shape written on 2026-10-21 morning: a trigger POINT instead of a reserve. Same line,
	// other direction, and it has to come out at the same line or a session would compact early.
	const point = num(sc.triggerTokens);
	const free = reserve ?? (point !== undefined && point > 0 ? Math.max(0, win - point) : undefined);
	if (free !== undefined || preserve !== undefined) {
		return { reserveTokens: free ?? base.reserveTokens, preserveTokens: preserve ?? base.preserveTokens };
	}
	const sp = num(sc.startPercent) ?? (num(sc.reservePercent) !== undefined ? 100 - (num(sc.reservePercent) as number) : 82);
	const rLo = Math.min(num(sc.reserveMin) ?? 12_288, num(sc.reserveMax) ?? 80_000);
	const rHi = Math.max(num(sc.reserveMin) ?? 12_288, num(sc.reserveMax) ?? 80_000);
	const keepLo = Math.min(num(sc.keepRecentMin) ?? 16_384, num(sc.keepRecentMax) ?? 100_000);
	const keepHi = Math.max(num(sc.keepRecentMin) ?? 16_384, num(sc.keepRecentMax) ?? 100_000);
	return {
		reserveTokens: clamp(Math.round((win * (100 - sp)) / 100), rLo, rHi),
		preserveTokens: clamp(Math.round((win * (num(sc.keepRecentPercent) ?? 20)) / 100), keepLo, keepHi),
	};
}

type ElisionConfig = {
	/** false = never stub anything in live requests. */
	enabled: boolean;
	/** Start stubbing when the live context reaches this percent of the session model's window. */
	softPercent: number;
	/** The newest this many tokens of context are never touched (the working model keeps its
	 *  recent tool results in full). */
	protectRecentTokens: number;
	/** The tail excerpt a stub carries, in characters (the exit/status line is always included).
	 *  "full" = the whole result text (a stub that saves nothing - elision becomes pointless). */
	stubTailChars: CharCap;
	/** A new stub batch is applied only when it saves at least this many tokens (prompt caches
	 *  charge a re-prefill per change; batching amortizes that cost). */
	minSavingsTokens: number;
	/** Elision stops firing new batches when the live request size comes within this many tokens
	 *  of pi's compaction trigger (window - reserveTokens) - a batch there would be wasted, since
	 *  compaction resets the context moments later anyway. */
	stopGapTokens: number;
	/** How many results ONE batch may stub, newest eligible first (see the break-point note in
 *  applyElision). Older eligible results wait for the next batch. */
	maxResultsPerBatch: number;
	/** The gap between two ELISION BATCHES in seconds (2026-10-05). Every batch moves the prompt-cache
	 *  break point, so everything after the first newly-stubbed result has to be re-prefilled on the
	 *  next request: on a big window the batches fired one after another and each one cost a fresh
	 *  upload of most of the context. One batch per gap, however many results pile up in between. */
	minBatchGapSec: number;
	/** A new batch also needs at least this many results waiting - the per-fire prefill cost is
	 *  roughly fixed (the whole tail after the break point re-prefills), so more results per fire
	 *  means better amortization. Big results each clearing minSavingsTokens would otherwise fire
	 *  one per result. */
	minResultsToStub: number;
};

/**
 * Everything one KIND of model (local or online) does differently, plus per-model overrides of any
 * of these fields in `overrides`.
 */
/**
 * The options of ONE model. Every model in the Compaction models list carries its own values; models
 * not in the list are only ever asked with the internal defaults (DEFAULT_OPTIONS below).
 */
type ModelOptions = {
	/** Thinking level for this model's summarisation request. */
	thinking: Level;
	/** How long to keep the request when NOTHING is arriving. Every character that comes in restarts
	 *  it, so this is a silence limit rather than a length limit (2026-10-19). 0 = unlimited. */
	timeoutMs: number;
	/** THE SUMMARY AIM: this percent of the model's ctx window (2026-10-03: everything - aim, min,
	 *  max - is a window percent; the numbers the settings show are the numbers enforced). */
	summaryPercent: number;
	/** THE LIMITS (2026-10-03): percent of the model's WINDOW, used DIRECTLY as the acceptance
	 *  bounds - a summary below min or above max is thrown away. The aim is clamped inside them,
	 *  so at a 95k window 23% means 21,850 tokens, and the same settings scale on a 1M model. */
	summaryMinTokens: number;
	summaryMaxTokens: number;
	/** LEGACY ONLY: configs written before 2026-10-05 stored the two limits as PERCENTS of the
	 *  model's window. Read once by optionsFor and converted with that model's window; nothing else
	 *  reads them any more - the code has worked in absolute tokens since. */
	legacyMinPct?: number;
	legacyMaxPct?: number;
	/** Hidden analysis written before the summary in the same call, stripped before storing:
	 *  "full" = a chronological per-turn analysis; "mini" = at most ten lines; "off" = none. */
	/** The summarizer's input has every tool result replaced by an informative stub (smaller,
	 *  faster, and the exit/tail message survives pi's 2,000-char clip). false = pi's raw input. */
	inputStubs: boolean;
	/** Cap for tool-call argument VALUES in the summarizer's input (whole files inside
	 *  write/edit arguments otherwise arrive uncapped). "full" = uncapped. */
	argCap: CharCap;
	/** How much of each content type the summary keeps; each maps to its own instruction sentence. */
	preserveStubs: Preserve;
	preserveUser: Preserve;
	preserveReplies: Preserve;
	preserveThinking: Preserve;
	/** How the previous compaction's summary rides into this one: fold / attach / skip. */
	chainMode: ChainMode;
	noThinkMarker: string;
	sampling: SamplingFlags;
};

type Config = {
	/** false = do nothing, pi compacts with the session model as usual. */
	enabled: boolean;
	/**
	 * The Compaction models list: insertion-ordered map of model ref -> its options. The order of the
	 * keys is the try order. JSON keeps insertion order, so the file is hand-editable too.
	 */
	models: Record<string, ModelOptions>;
	/** Extra tries per model while the endpoint reports itself busy. 0 = move on immediately. */
	retries: number;
	/** Seconds to wait before each retry. */
	retryDelaySeconds: number;
	/** Write the log file. */
	log: boolean;
	/** Append a "full raw transcript" line (the session file path) to extension-written summaries,
	 *  so the next model can recover anything the summary dropped by reading the file. */
	transcriptPointer: boolean;
	/** Put one receipt line at the top of extension-written summaries saying when the compaction ran, what the
	 *  context held, how much material the summariser actually read (and how much of it was elided
	 *  first), and how big the summary came out. pi's own on-screen line can only ever show the whole
	 *  pre-compaction context, which is why "compacted from 163,000 tokens" looks nothing like the
	 *  49,000 tokens the summariser was given. */
	/** ON = a compaction this extension starts while a turn is running is landed by appending the summary
	 *  straight to the session, instead of going through pi's manual compaction entry point - which stops
	 *  the turn, prints 'operation aborted' and waits for the operator to type a message. OFF until it has
	 *  been watched on a real long run (2026-10-22, #43); file-only for now, no menu row. */
	landWithoutAbort: boolean;
	/** ON = the extension builds the whole summarization request itself (turn-numbered input,
	 *  labelled previous summary, ONE instruction source) and calls the model through the registry;
	 *  a direct failure moves to the NEXT candidate, and pi's compact() runs only after every
	 *  candidate failed. OFF = the old layered path (pi's prompt + the extension instructions on
	 *  top, with pi's compact() building every request). */
	directRequest: boolean;
	/** Validate the summary's shape after the size gate: all six sections present, turn labels in
	 *  order, supersession/amended tags referencing turns that exist. Malformed = rejected, the
	 *  next model tries. */
	shapeGate: boolean;
	/** Cap on the CHAT request's generation permission (max_tokens / max_completion_tokens). pi sizes
	 *  it as window − estimate − 4,096, which balloons on big-window models and dies on estimate
	 *  undercounts; the cap makes every chat request fit until real ≈ window − cap. 0 = no cap. */
	chatMaxTokensCap: number;
	/** Percent-based keepRecent/reserveTokens derived from the active model's window (the unlink). */
	scaling: ScalingConfig;
	/** The extension's own proactive trigger: while the agent is IDLE and the context estimate
	 *  crosses window - reserve, run pi's manual compaction (which this fork's rotation answers).
	 *  pi's static trigger sits above the request death line on big windows, so without this the
	 *  first compaction would only ever happen via overflow recovery (which deletes entries). */
	autoCompact: boolean;
	/** Set when an auto-compaction was cancelled by the user or failed twice: the proactive trigger
	 *  stays armed off until a MANUAL compaction runs (2026-10-05 - a failing auto-compaction used to
	 *  re-fire every minute, ~100 attempts in one session, each one uploading the whole context). */
	/** Parked means PARKED FOR THIS SESSION (2026-10-17): the id of the session that stopped the trigger,
	 *  or false when armed. A legacy boolean true parks everywhere, so an old config file still means what
	 *  it meant; a new stop stores the id, so starting or resuming another session finds the trigger armed
	 *  and going back to the stopped one finds it parked where you left it. */
	autoCompactPaused: boolean | string;
	/**
	 * WHO PULLS THE TRIGGER (2026-10-06). true (the default) = pi's own check that runs before every
	 * request to the model compares the context with the line and starts the compaction there; that is
	 * the only place a compaction can start while the agent is working, and the compaction itself is
	 * still written by this extension, because the session_before_compact hook answers every call.
	 * false = this extension's idle watcher is the only trigger, which cannot fire mid-run and is why
	 * a long working session used to overflow instead of compacting.
	 */
	/**
	 * The elision block: stubbing the oldest tool results out of the LIVE request before the context gets
	 * anywhere near the trigger line, so a long working session keeps its conclusions instead of losing them
	 * to a compaction. On or off, the share of the window at which it starts, and how many of the most recent
	 * calls are protected - see ElisionConfig, applyElision, and the elision rows on the settings board.
	 *
	 * This field sat inside the comment below it from 2026-10-18 until 2026-10-19. The paragraph describing
	 * the removed piLineWritten flag lost its closing marker to the patch that deleted the flag itself, so the
	 * comment ran on and swallowed this declaration. Types are stripped rather than checked, so nothing
	 * complained: the cost was that the Config interface stopped telling the truth about a field every
	 * elision path uses, and no bracket scanner can see that, because it throws comments away first.
	 */
	elision: ElisionConfig;
	/** The default no-think tag for local models (each model's own value wins when set). */
	noThinkMarker: string;
	/**
	 * One extra instruction for every model, added after pi's own compaction prompt. null = the
	 * built-in default template (DEFAULT_TEMPLATE_TEXT, with {PLACEHOLDER} tokens) is sent; a
	 * stored text IS the template - its {TOKENS} substitute per model at request time.
	 */
	additionalInstruction: string | null;
	/** User-customized category sentences: kind ("stubs"|"user"|"replies"|"thinking") x level
	 *  (verbatim|detailed|summary|brief) -> the ACTIVE wording (what substitutes into the
	 *  template's {TOKENS}). A missing/empty entry falls back to the built-in extension default.
	 *  The USER default (categoryUserDefaults) is a checkpoint, not a substitution source. */
	categorySentences: { [kind: string]: { [level: string]: string } } | null;
	/** The USER DEFAULT per category x level - a checkpoint the user saves a sentence to (Ctrl+S).
	 *  Ctrl+U restores the ACTIVE sentence from it; it never substitutes on its own. */
	categoryUserDefaults: { [kind: string]: { [level: string]: string } } | null;
	/** The USER DEFAULT for the main summary template - a checkpoint saved from the submenu. When
	 *  the active template is empty (null), it follows this; "" pins the extension default. */
	templateUserDefaults: string | null;
	/** Which library entry is the active one, by NAME (2026-10-20, the operator's design). The text copy in
	 *  templateUserDefaults above is what actually gets sent, and it stays in step with this name: choosing
	 *  a template writes both, editing that template refreshes the copy, deleting it clears the name. Before
	 *  this there was only the text, so editing the active template detached it from its library row and the
	 *  list showed nothing as active - a choice that looked lost. null or a name that no longer exists falls
	 *  back to the text comparison, so a config written before this field still reads correctly. */
	activeTemplateName: string | null;
	/** A whole template that an older config kept in the "additional instruction" slot. Kept for
	 *  reference, never sent: the template is the single text in templateUserDefaults. */
	templatePreviousCopy: string | null;
	/** The user's template library (2026-10-07 request): any number of saved templates, names assigned
	 *  automatically ("User template #15"). The ACTIVE template is a COPY of one of these (or the
	 *  extension default), assigned from the list with Ctrl+A - which is why the Templates menu offers the
	 *  active template as a preview only: its text is edited where it lives, in this list. */
	userTemplates: { name: string; text: string }[];
};

/** The default no-think tag written into a local model's request when its thinking is off. */
const DEFAULT_NO_THINK_TAG = "<|think_off|>";


/** Timeout periods in the exact order they toggle. 0 = unlimited. */
const TIMEOUT_STEPS = [120_000, 180_000, 300_000, 420_000, 600_000, 900_000, 1_200_000, 1_800_000, 2_700_000, 3_600_000, 0];

const THINKING_CYCLE: Level[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
const PRESERVE_CYCLE: Preserve[] = ["verbatim", "detailed", "summary", "brief"];
/** How the PREVIOUS compaction's summary rides into the next one:
 *  "fold"   - the model integrates it and continues its turn numbering (compact, lossy),
 *  "attach" - the new summary must copy it VERBATIM into a PREVIOUS SUMMARY section (nothing lost),
 *  "skip"   - drop it (maximum compression; history before the previous compaction ends there). */
type ChainMode = "fold" | "attach" | "skip";
function isChainMode(v: unknown): v is ChainMode {
	return v === "fold" || v === "attach" || v === "skip";
}
/** Character-cap steps for stub tails and tool-call argument caps; "full" = uncapped. */
const CHAR_CAP_STEPS: CharCap[] = [0, 50, 100, 150, 200, 300, 500, 750, 1000, "full"];
/** The old preset ladders were removed 2026-10-01: every elision number is a FREE-ENTRY field and
 *  sanitizeElision now accepts any value in the field's range, so typed values survive the round
 *  trip exactly as entered. */

const ELISION_DEFAULTS: ElisionConfig = {
	enabled: true,
	softPercent: 70,
	protectRecentTokens: 20_000,
	stubTailChars: 200,
	minSavingsTokens: 2_000,
	stopGapTokens: 4_000,
	minResultsToStub: 5,
	minBatchGapSec: 300,
	maxResultsPerBatch: 40,
};

function isCharCap(v: unknown): v is CharCap {
	return v === "full" || (typeof v === "number" && Number.isFinite(v) && v >= 0);
}

function isPreserve(v: unknown): v is Preserve {
	return typeof v === "string" && (PRESERVE_CYCLE as string[]).includes(v);
}

function charCapLabel(v: CharCap): string {
	return v === "full" ? "full" : String(v);
}

function parseCharCap(text: string, fallback: CharCap): CharCap {
	const t = String(text).trim().toLowerCase();
	if (t === "full") return "full";
	const n = Number(t);
	return Number.isFinite(n) && n >= 0 ? n : fallback;
}

/** One step forward in the CHAR_CAP_STEPS cycle (wraps). */
function nextCharCap(current: CharCap): CharCap {
	const i = CHAR_CAP_STEPS.findIndex((s) => s === current);
	return CHAR_CAP_STEPS[(i + 1) % CHAR_CAP_STEPS.length] ?? CHAR_CAP_STEPS[0];
}

function sanitizeElision(raw: any): ElisionConfig {
	return {
		enabled: typeof raw?.enabled === "boolean" ? raw.enabled : ELISION_DEFAULTS.enabled,
		// Free-entry values must survive the round trip: accept any sane number, the ladders above
		// are only the board's cycle suggestions. (Bug fixed 2026-10-01: softPercent 10 and any
		// protectRecentTokens outside the ladder were silently reset to the defaults on every read.)
		softPercent: typeof raw?.softPercent === "number" && Number.isFinite(raw.softPercent) && raw.softPercent >= 1 && raw.softPercent <= 99 ? Math.round(raw.softPercent) : ELISION_DEFAULTS.softPercent,
		protectRecentTokens: typeof raw?.protectRecentTokens === "number" && Number.isFinite(raw.protectRecentTokens) && raw.protectRecentTokens >= 0 ? Math.round(raw.protectRecentTokens) : ELISION_DEFAULTS.protectRecentTokens,
		stubTailChars: isCharCap(raw?.stubTailChars) ? raw.stubTailChars : ELISION_DEFAULTS.stubTailChars,
		minSavingsTokens: typeof raw?.minSavingsTokens === "number" && Number.isFinite(raw.minSavingsTokens) && raw.minSavingsTokens >= 0 ? Math.round(raw.minSavingsTokens) : ELISION_DEFAULTS.minSavingsTokens,
		stopGapTokens: typeof raw?.stopGapTokens === "number" && Number.isFinite(raw.stopGapTokens) && raw.stopGapTokens >= 0 ? Math.round(raw.stopGapTokens) : ELISION_DEFAULTS.stopGapTokens,
		minResultsToStub: typeof raw?.minResultsToStub === "number" && Number.isInteger(raw.minResultsToStub) && raw.minResultsToStub >= 1 ? raw.minResultsToStub : ELISION_DEFAULTS.minResultsToStub,
		minBatchGapSec: typeof raw?.minBatchGapSec === "number" && Number.isFinite(raw.minBatchGapSec) && raw.minBatchGapSec >= 0 ? Math.round(raw.minBatchGapSec) : ELISION_DEFAULTS.minBatchGapSec,
		maxResultsPerBatch: typeof raw?.maxResultsPerBatch === "number" && Number.isFinite(raw.maxResultsPerBatch) && raw.maxResultsPerBatch >= 1 ? Math.round(raw.maxResultsPerBatch) : ELISION_DEFAULTS.maxResultsPerBatch,
	};
}

const DEFAULT_CONFIG: Config = {
	enabled: false,
	models: {},
	retries: 2,
	retryDelaySeconds: 5,
	log: true,
	transcriptPointer: true,
	// Off since 2026-10-22 at the operator's request: the same facts are printed on screen when the
	// compaction finishes, so the line at the top of the summary was read only by the next model. Turn
	// it on again from the settings menu if you want the summary to carry the receipt for itself.
	landWithoutAbort: false,
	directRequest: true,
	shapeGate: true,
	chatMaxTokensCap: 65_536,
	scaling: { ...SCALING_DEFAULTS },
	autoCompact: true,
	autoCompactPaused: false,
	elision: ELISION_DEFAULTS,
	noThinkMarker: DEFAULT_NO_THINK_TAG,
	additionalInstruction: null,
	categorySentences: null,
	categoryUserDefaults: null,
	templateUserDefaults: null,
	activeTemplateName: null,
	userTemplates: [],
	templatePreviousCopy: null,
};

/** pi's default reserveTokens (docs/compaction.md), used only to turn old absolute token caps into
 *  shares and to show approximate token figures in the menu. The real value is read from
 *  preparation.settings at compaction time. */
export const DEFAULT_RESERVE_TOKENS = 16384;

/** pi's OWN settings.json compaction.reserveTokens, READ for display and nothing else (2026-10-19).
 *  It used to say "written by the board's stock-fallback row"; that row is gone, this extension has never
 *  written pi's settings.json since the coupling came out on 2026-10-18, and the claim stayed in the comment
 *  long enough to send tonight's diagnosis looking for a writer that did not exist. The number belongs to
 *  pi's built-in compaction - the fallback that runs when this extension is disabled or every candidate has
 *  failed - and we never apply it to our own work. */
function readPiReserveTokens(): number {
	try {
		const raw = JSON.parse(readFileSync(join(AGENT_DIR, "settings.json"), "utf8"));
		const v = raw?.compaction?.reserveTokens;
		return typeof v === "number" && Number.isFinite(v) && v >= 0 ? Math.round(v) : DEFAULT_RESERVE_TOKENS;
	} catch {
		return DEFAULT_RESERVE_TOKENS;
	}
}

/** pi's OWN built-in compaction switch (settings.json "compaction": {"enabled": ...}).
 *
 *  READ ONLY, in every sense: this extension reads the flag to warn you when two compactors would both be
 *  live, and it never writes that file. The writer that used to run here on a per-second clock was deleted
 *  on 2026-10-18, and the log's last trigger_clock record - one second after a manual restore of the very
 *  same block, and from the build that was still loaded - is the reason it had to go.
 *
 *  The hard rule still stands, and at most ONE compactor may be on. What has changed is who enforces it:
 *  not us, never again us. You hold that line in pi's own settings, and a change made in the file needs a
 *  restart, because pi reads settings.json at startup and then holds what it read - which is how a flag
 *  switched off at 23:29:55 was still reported as on by the pi that started at 23:29:36. */
/** pi's compaction block, as written in its own settings.json, with pi's own defaults filled in for
 *  anything absent. The flag alone used to be read; the two numbers are needed since this extension
 *  builds pi's own preparation object for a compaction it lands without stopping the turn (#43), and
 *  the cut point depends on `keepRecentTokens`. Read at call time, like the flag, because pi holds what
 *  it read at startup and an edit in the file is exactly what the operator is testing. */
/** pi's own compaction module (`dist/core/compaction/compaction.js`), which holds
 *  `prepareCompaction` - the function that decides what a summary covers and where the verbatim tail
 *  starts. It is NOT in the package's root exports, so it is reached by resolving the same specifier
 *  the top of this file already imports and following it to that file. Nothing here copies pi's
 *  arithmetic: if the module or the function cannot be found, the caller falls back to pi's own path. */
let piCompactionMod: Promise<any> | undefined;
function piCompactionModLoad(): Promise<any> {
	if (!piCompactionMod) {
		piCompactionMod = (async () => {
			// pi loads extensions from its own bundle, where the bare package name does not necessarily
			// resolve from this file's folder (2026-10-22: "Cannot find module"). Try the specifier first,
			// because that is the correct answer when it works, then the install folder the running pi was
			// started from, then the npm global folder on Windows. One of them is always right, and if none
			// is, the caller falls back to pi's own path with a message instead of failing.
			const file = "core/compaction/compaction.js";
			const tried: string[] = [];
			try {
				tried.push(new URL("./" + file, import.meta.resolve("@earendil-works/pi-coding-agent")).href);
			} catch {
				/* the specifier is not resolvable from here; the paths below are the fallback */
			}
			const appdata = process.env.APPDATA ?? process.env.LOCALAPPDATA ?? "";
			const bs = String.fromCharCode(92);
			const fwd = (s: string) => s.split(bs).join("/");
			if (appdata) tried.push("file:///" + fwd(join(appdata, "npm", "node_modules", "@earendil-works", "pi-coding-agent", "dist", file)));
			let lastErr: any;
			for (const href of tried) {
				try {
					return await import(href);
				} catch (err) {
					lastErr = err;
				}
			}
			throw lastErr ?? new Error("pi's compaction module was not found in " + tried.length + " places");
		})().catch((err) => {
			// A failed lookup must not be remembered as a permanent answer: pi can be upgraded under us.
			piCompactionMod = undefined;
			throw err;
		});
	}
	return piCompactionMod;
}

export function readPiCompactionSettings(): { enabled: boolean; reserveTokens: number; keepRecentTokens: number } {
	const off = { enabled: false, reserveTokens: DEFAULT_RESERVE_TOKENS, keepRecentTokens: DEFAULT_KEEP_RECENT_TOKENS };
	try {
		const raw = JSON.parse(readFileSync(join(AGENT_DIR, "settings.json"), "utf8"));
		const c = raw?.compaction ?? {};
		const num = (v: unknown, dflt: number) => (Number(v) > 0 ? Math.round(Number(v)) : dflt);
		return {
			enabled: c.enabled === true,
			reserveTokens: num(c.reserveTokens, DEFAULT_RESERVE_TOKENS),
			keepRecentTokens: num(c.keepRecentTokens, DEFAULT_KEEP_RECENT_TOKENS),
		};
	} catch {
		return off;
	}
}

/** Whether pi would compact by itself. Kept as its own name because half the file asks exactly that. */
export function readPiCompactionEnabled(): boolean {
	return readPiCompactionSettings().enabled;
}

/** pi's default keepRecentTokens, used when the preparation carries no settings of its own. */
const DEFAULT_KEEP_RECENT_TOKENS = 20000;

// ---------------------------------------------------------------- built-in instructions

/**
 * The generated part of the instructions. pi's own prompt (SUMMARIZATION_PROMPT in
 * dist/core/compaction/compaction.js) is compiled into the build and ends with "Keep each section
 * concise", which is exactly what made short summaries: on 2026-09-22 a Gemini run asked for 2,074
 * words delivered 1,668 with two headings left nearly empty. The text below names that line and
 * says what replaces it, then pins the size.
 */
/** The computed length fragment, substituted into the template's {LENGTH INSTRUCTIONS} token. */
/** A stored "additional instruction" that is really a whole template (it carries the placeholders). */
function looksLikeTemplate(text: string): boolean {
	return text.includes('{SEGMENT SIZE}') || text.includes('OUTPUT FORMAT');
}

function lengthFragment(sizes: Sizes, inputTokens: number): string {
	// Every number here goes through tok(): this text rides verbatim into the summarization request,
	// and on 2026-10-05 the migrated limits reached it as NaN, so the summarizer was told to write a
	// summary of "NaN tokens" and every attempt came back at 3-4k tokens.
	const min = tok(sizes.min, -1);
	const max = tok(sizes.max, -1);
	const target = tok(sizes.target, -1);
	const region = tok(inputTokens);
	if (min < 0 || max < 0 || target < 0) {
		return `The size limits for this summary could not be computed (min/aim/max are not numbers) - this compaction is misconfigured. Write a complete, dense ledger of the whole segment; do not stop early.`;
	}
	if (min > 0) {
		return `The segment below is about ${fmt(region)} tokens. Aim for about ${fmt(target)} tokens (around ${fmt(wordsFor(target))} words). A summary under ${fmt(min)} tokens (around ${fmt(wordsFor(min))} words) or over ${fmt(max)} tokens (around ${fmt(wordsFor(max))} words) is rejected outright. If you approach the ${fmt(max)}-tok ceiling, tighten the least important details and land on a finished item, never mid-item.`;
	}
	return `The segment below is about ${fmt(region)} tokens. Aim for about ${fmt(target)} tokens (roughly ${fmt(wordsFor(target))} words); there is no hard minimum, but a very thin summary leaves the next request blind, so write freely and do not stop early. Your text is cut off at ${fmt(max)} tok.`;}

/** The six numbers a template may ask for by name, as text, ready to drop in wherever the name sits.
 *  These tokens were listed in the editor's tag row and described as "substituted wherever they appear"
 *  (see INLINE_LENGTH_TOKENS and the note above applyTemplate), and no code ever substituted one of
 *  them. So a template that writes its own length sentence - the built-in default does, and User
 *  templates #1 and #3 are copies of it - sent the summarizer a pair of curly braces in place of its
 *  target, and the block carrying the real numbers was withheld as well, because the code read those
 *  same unfilled tokens as proof that the sentence had been written out on purpose. GLM wrote 9,487
 *  tokens against an aim of 22,100 on 2026-10-18 for exactly that reason, which is the complaint
 *  logged on 2026-10-17 as well ("why does the online GLM model write such a short compaction of
 *  around 10000 tokens even though it is told to write 24000 tokens?") - it was never told.
 *  A number that cannot be computed comes out as a plain sentence saying so rather than NaN or
 *  braces, because a sentence at least tells the model what to do instead; min 0 means "no floor"
 *  by design (see computeSizes) and is worded as that rather than as a failure. */

function lengthValues(sizes: Sizes | undefined): Array<[string, string]> {
	const cant = "could not be computed for this compaction - write a complete ledger and do not stop early";
	if (!sizes) return INLINE_LENGTH_TOKENS.map((t): [string, string] => [t, cant]);
	const min = tok(sizes.min, -1);
	const max = tok(sizes.max, -1);
	const target = tok(sizes.target, -1);
	const num = (v: number): string => (v > 0 ? fmt(v) : v === 0 ? "no floor (any complete summary passes the acceptance check)" : cant);
	const words = (v: number): string => (v > 0 ? fmt(wordsFor(v)) : v === 0 ? "no floor" : cant);
	return [
		["{AIM TOKENS}", num(target)],
		["{AIM WORDS}", words(target)],
		["{MIN TOKENS}", num(min)],
		["{MIN WORDS}", words(min)],
		["{MAX TOKENS}", num(max)],
		["{MAX WORDS}", words(max)],
	];
}
/**
 * Instruction sentences per preservation level, per content type. The template's RULES section is
 * generated from these, so the wording a model reads always matches the options in the menu.
 */
export const PRESERVE_SENTENCES: Record<Preserve, Record<"stubs" | "user" | "replies" | "thinking", string>> = {
	// Directive style: the CATEGORY and LEVEL in caps first (scannable), then what to do with the
	// text, then the explicit DON'Ts. Models obey concrete, countable demands ("every ... one line
	// per call, in input order") far better than adjectives ("detailed").
	verbatim: {
		stubs: `TOOL CALLS: copy EVERY tool-call stub from the input VERBATIM, EXACTLY as written, word for word, one per line, in input order, into that turn's tool-call sub-line of the ledger entry (whatever that sub-line is called in the OUTPUT FORMAT above). Do NOT modify text, do NOT skip any call. Stubs are already compact - copying them is the requirement; changing them in any way is a failure.`,
		user: `USER PROMPTS: quote every user prompt VERBATIM, in FULL, word for word, inside quotation marks, in that turn's user-prompt sub-line of the ledger entry. Do NOT change the text in any way, do NOT omit any sentence.`,
		replies: `ASSISTANT REPLIES: preserve every assistant reply VERBATIM, in FULL, exactly as written, word for word, under an [Assistant] marker in that turn. Do NOT change the text in any way, do NOT omit any sentence.`,
		thinking: `ASSISTANT THINKING: preserve the assistant thinking blocks VERBATIM, in FULL (marked [Assistant thinking]). Expensive - only while the size limits allow.`,
	},
	detailed: {
		stubs: `TOOL CALLS: DETAILED summary of EVERY tool call. Keep the full command or path, the output size, and the stub's exit/tail message. No call may be dropped.`,
		user: `USER PROMPTS: DETAILED summary. preserve every user prompt's substance and QUOTE its instruction sentences word for word. No prompt may be dropped.`,
		replies: `ASSISTANT REPLIES: DETAILED summary. Preserve the substance of every assistant reply and QUOTE its decisive sentences word for word. No reply may be dropped.`,
		thinking: `ASSISTANT THINKING: one to three lines DETAILED summary per turn with the reasoning that led to each decision.`,
	},
	summary: {
		stubs: `TOOL CALLS: one line summary of EVERY tool call - the command or path and its outcome. Every call gets its line; do not drop calls.`,
		user: `USER PROMPTS: one to two line summary per prompt - the intent and the parameters, faithfully.`,
		replies: `ASSISTANT REPLIES: one summary per reply, what was concluded and done - no filler.`,
		thinking: `ASSISTANT THINKING: one or two lines summary per turn.`,
	},
	brief: {
		stubs: `TOOL CALLS: brief summary of only the calls that were important to the assistant's process. Routine reads and listings may be skipped - mention once that they were skipped.`,
		user: `USER PROMPTS: brief one line summary of each - the intent only.`,
		replies: `ASSISTANT REPLIES: one line each - the outcome only.`,
		thinking: `ASSISTANT THINKING: a short phrase only where the thinking changed the course.`,
	},
};

/** The four levels' exact sentences for one content type - what the model reads at each level.
 *  Shown in the row descriptions so switching a level is a visible text change, not a guess. */
function ladderText(kind: "stubs" | "user" | "replies" | "thinking", current: Preserve): string {
	return (["brief", "summary", "detailed", "verbatim"] as Preserve[])
		.map((lv) => `${lv}${lv === current ? " (CURRENT)" : ""}: "${PRESERVE_SENTENCES[lv][kind]}"`)
		.join(" | ");
}

/**
 * The draft-block instruction: a hidden analysis written before the summary, inside the SAME call,
 * stripped before storing (stripRoughWork). Anthropic's Claude Code compaction prompt runs the same
 * trick in one call (an analysis scratchpad before the summary, removed afterwards); it organizes
 * the summary and costs no extra request. It costs OUTPUT tokens, so it is a per-model switch:
 * online models default to full, local ones to off (a full draft roughly doubles a 10-20
 * tokens/second summary).
 */


/**
 * THE SUMMARY TEMPLATE (the user's 2026-10-01 design): one editable source text with {PLACEHOLDER}
 * tokens; the per-model options substitute the tokens at request time. The four content categories
 * each have four level sentences (PRESERVE_SENTENCES, 4 x 4); the stub-example rule depends on the
 * input mode; the previous-summary rule and section depend on the chain mode. The user may reword
 * ANYTHING around the tokens - only the {TOKENS} themselves must survive editing; if one is
 * deleted, its rule is auto-added at the end with a visible note (nothing silently drops).
 *
 * Shape (from the 2026-09-25/28 compaction research): TWO LAYERS instead of pi's flat thematic
 * sections - CURRENT STATE is the newest truth and wins on any conflict; the TURN LEDGER preserves
 * ORDER (one numbered entry per user turn, supersession by explicit tag, never deletion). The
 * Overrides block cancels the pi prompt lines that fight this shape.
 */
export const DEFAULT_TEMPLATE_TEXT = [
	`This is a summary request. The conversation segment below is about {SEGMENT SIZE} tok and is the memory being replaced - the moment you stop writing, the original messages are deleted and the next model continues the work from your summary alone.`,
	`{LENGTH INSTRUCTIONS}`,
	`Do not stop early and do not save tokens: covering the whole segment is the one thing that matters.`,
	`Override: the line "Keep each section concise" in the instructions above does not apply to this job. It is written for a summary that sits beside the full transcript; here the transcript is deleted and your text is the only memory the next request has. Completeness beats brevity. Long bullet lists are good. Repeating an exact value in multiple turns is fine. Splitting one vague sentence into five specific ones is the point. Vagueness and omission are the only failures.`,
	``,
	``,
	`OUTPUT FORMAT - the format block above is replaced by this one. Use only the sections below, in this order, no preamble, no closing remarks.`,
	``,
	`## CURRENT STATE (the newest truth - where anything below conflicts with this section, THIS section wins)`,
	`- Goal now: what is being built or fixed, why, plus every constraint and preference.`,
	`- Where things stand: done / in progress / blocked, with file paths.`,
	`- Next step: the exact next action, with its parameters.`,
	``,
	`## TURN LEDGER (Chronological; Newer turns get more detail, oldest may tighten to one line; Every user turn gets one entry, in ascending order, numbered with the word Turn and its number - use "Turn 12", never "12." alone; Shorten older turns but never drop them; Drop a sub-line when there is nothing to say in it)`,
	`Turn 12`,
	`    user: the user's words, per the USER PROMPTS rule in Rules below.`,
	`    assistant thinking: what the assistant was thinking, per ASSISTANT THINKING SUMMARY rules in Rules below.`,
	`    assistant replies: what the assistant replied, per ASSISTANT REPLIES SUMMARY rules in Rules below.`,
	`    tools: per the TOOL CALLS rule in Rules below; when that rule says VERBATIM, copy each stub in full.`,
	`    errors & fixes: what broke and how it was fixed.`,
	`    decided: the choice made over the alternative, and why.`,
	``,
	``,
	`## FILES & DATA`,
	`- every file read, created or edited: full path, what it is, why it matters.`,
	``,
	`## KEY DECISIONS`,
	`- each choice and why it was picked over the alternatives, cross-referenced like "(Turn 12)".`,
	``,
	`## NEXT STEPS`,
	`- the exact next action, with needed parameters.`,
	``,
	`## CRITICAL CONTEXT`,
	`- every number, version, model name, setting key, path and quota verbatim; error messages quoted exactly.`,
	`{PREVIOUS SUMMARY SECTION}`,
	``,
	`Rules:`,
	`{TOOL CALLS SUMMARY}`,
	`{USER PROMPT SUMMARY}`,
	`{ASSISTANT REPLIES SUMMARY}`,
	`{ASSISTANT THINKING SUMMARY}`,
	`- One concrete fact per bullet; never drop a file path, a number, or an error message to save space - when a heading looks thin, list the concrete items you saw instead of summarising them.`,
	`- If a later turn changes an earlier one, KEEP the earlier line and tag it: "(superseded by Turn N)" for a full reversal, "(amended by Turn N)" for a partial change. Never delete history.`,
	`- Several user prompts can arrive while the assistant is still working - each is its own Turn, even a short one. In a chain of refinements, the newest wording is the current instruction; earlier ones stay, tagged against the turn that changed them.`,
	`{STUB EXAMPLE RULE}`,
	`- No wall-clock times exist in the input; order is expressed ONLY by the turn numbers.`,
	`{PREVIOUS SUMMARY RULE}`,
	``,
	`Overrides - lines in the instructions above do not apply here:`,
	`- The format block and "Keep each section concise": the transcript is deleted the moment you stop writing and your text is the only memory the next request has - completeness beats brevity; long lists are good; vagueness and omission are failures.`,
	`- "If something is no longer relevant, you may remove it" and "UPDATE the Progress section": history is tagged, never deleted; CURRENT STATE is the progress section.`
].join("\n");

/** The previous-summary section, attached mode only. */
const PREVIOUS_SECTION_TEXT = [
	`## PREVIOUS SUMMARY (verbatim - the earlier session, attached unchanged)`,
	`<previous-summary>`,
	`(copy the previous summary's body here word for word, from the <previous-summary> block in the input; write "(none)" when there is no previous summary)`,
	`</previous-summary>`,
].join("\n");

/** One preserve sentence: the ACTIVE wording wins; otherwise the built-in default. */
export function sentenceFor(kind: "stubs" | "user" | "replies" | "thinking", level: Preserve): string {
	const custom = loadConfig().categorySentences?.[kind]?.[level];
	const t = typeof custom === "string" ? custom.trim() : "";
	// No leading bullet (2026-10-17): these sentences sit inside a template as a clause of a paragraph, and
	// the store may carry one typed by hand in the editor. The list that re-adds lost rules writes its own
	// bullet when it has to, so a bullet kept here arrives twice.
	return (t || PRESERVE_SENTENCES[level][kind]).replace(/^[-*]\s+/, "");
}

/** Which layer a sentence IS - extension default first (it wins ties), then user default, then
 *  custom. Labels everywhere use this so a slot holding the extension text never reads "custom". */
export function sentenceLayer(kind: "stubs" | "user" | "replies" | "thinking", level: Preserve, text: string): "ext" | "user" | "custom" {
	const t = (text ?? "").trim();
	if (!t || t === PRESERVE_SENTENCES[level][kind].trim()) return "ext";
	const usr = loadConfig().categoryUserDefaults?.[kind]?.[level]?.trim();
	if (usr && t === usr) return "user";
	return "custom";
}

/** The user default's own layer: same as extension, custom, or none. */
export function userDefaultLayer(kind: "stubs" | "user" | "replies" | "thinking", level: Preserve): "ext" | "custom" | "none" {
	const u = loadConfig().categoryUserDefaults?.[kind]?.[level]?.trim();
	if (!u) return "none";
	return u === PRESERVE_SENTENCES[level][kind].trim() ? "ext" : "custom";
}

/** A human tag for the ACTIVE sentence's layer. */
export function layerTag(kind: "stubs" | "user" | "replies" | "thinking", level: Preserve): string {
	const l = sentenceLayer(kind, level, sentenceFor(kind, level));
	return l === "ext" ? "extension default" : l === "user" ? "user default" : "custom";
}

/** The marker a template uses to declare that it carries every rule in its own prose, in its own place:
 *  see applyTemplate. It expands to nothing; its whole job is to be present or absent. */
export const RULES_WRITTEN_OUT = "{RULES WRITTEN OUT}";
/** Rule tokens the LAST rendered template declared written out but left no trace of - the record that takes
 *  the place of silently re-adding them, so a self-sufficient template can be checked instead of trusted. */
export const templateAbsentRules: string[] = [];

/** Substitute the {TOKENS} in a template source for one model's options. Tokens that the source
 *  lost are auto-added at the end under a visible note - a user edit can never silently drop a
 *  preserve rule, and a hand-written source still yields a complete instruction. Notes between
 *  <!-- and --> are cut first, so they can carry wording that is not live yet. */
export function applyTemplate(src: string, o: ModelOptions, inputStubs: boolean, extra?: { sizes?: Sizes; inputTokens?: number; segmentTokens?: number }): string {
	let out = stripTemplateNotes(src);
	const missing: string[] = [];
	// A TEMPLATE THAT DECLARES ITSELF COMPLETE (2026-10-17). The safety net above is right for a template that
	// uses the rule tokens and wants their wording kept: delete a token and its rule comes back at the bottom,
	// which is what makes those rules feel uneditable - the text is put back no matter what the author writes.
	// A template that would rather carry every rule in its own prose says so with this one marker, and then
	// nothing is stapled on afterwards. What it says is verifiable, so it is verified: `templateAbsentRules`
	// lists the rule tokens the declaration left no trace of, and the summary request logs that list. So the
	// declaration trades a safety net for a visible record - dropping a rule from a declaring template stops
	// being corrected and becomes something the log admits out loud. Applies to the rule tokens only: the
	// numbers are still filled wherever they sit, because those are values rather than instructions.
	const writtenOut = out.includes(RULES_WRITTEN_OUT);
	const absent = templateAbsentRules;
	absent.length = 0;
	// The marker itself never reaches the model: it is a declaration, not an instruction.
	out = out.split(RULES_WRITTEN_OUT).join("").trim();
	const sub = (token: string, value: string): void => {
		if (out.includes(token)) out = out.split(token).join(value);
		else if (!value.trim()) return;
		// Declared complete: a rule counts as written out when its text is there, token or no token. Absent
		// altogether is a different thing, and that is what gets recorded rather than silently re-added.
		else if (writtenOut) { if (!out.includes(value)) absent.push(token); }
		else missing.push(value);
	};
	// THE CATEGORY SENTENCES, AS PLAIN LINES (2026-10-21, defect #51). These used to arrive as
	// `- <sentence>`, which put a bullet in front of every category block no matter what the template
	// author wanted - and they can want a bullet, so they can write one: the tag now fills with the
	// sentence alone, and the surrounding line belongs to the template.
	sub("{TOOL CALLS SUMMARY}", sentenceFor("stubs", o.preserveStubs));
	sub("{USER PROMPT SUMMARY}", sentenceFor("user", o.preserveUser));
	sub("{ASSISTANT REPLIES SUMMARY}", sentenceFor("replies", o.preserveReplies));
	sub("{ASSISTANT THINKING SUMMARY}", sentenceFor("thinking", o.preserveThinking));
	sub(
		"{STUB EXAMPLE RULE}",
		inputStubs
			? `- Tool results arrive as stubs like "[bash "bun test" - output elided: 203 lines / 13,238 chars - tail: "Command exited with code 1" - re-run to see the full output]". A stub is the complete record of that call: copy it into the ledger; when a detail is missing, trust its re-run/re-read hint - never guess what was in the output.`
			: `- Long tool outputs arrive truncated to their first 2,000 characters: keep the path or the command and the decisive lines you saw; if a needed detail was cut, write "[clipped - re-read <path> / re-run <command>]" instead of guessing.`,
	);
	sub(
		"{PREVIOUS SUMMARY RULE}",
		o.chainMode === "attach"
			? `- The PREVIOUS SUMMARY section at the end is the earlier session's ONLY copy: reproduce the previous summary's body between the markers VERBATIM, word for word - do not rephrase it, do not shorten it, do not weave it into the ledger. The ledger covers the NEW turns only; its numbering continues from the previous summary's last turn.`
			: o.chainMode === "fold"
				? `- A previous summary, when present, is the ledger of the EARLIER turns: carry it into the ledger, continue its numbering, never restart at Turn 1. Its "Full raw transcript" line is bookkeeping - do not copy it.`
				: "",
	);
	sub("{PREVIOUS SUMMARY SECTION}", o.chainMode === "attach" ? PREVIOUS_SECTION_TEXT : "");
	// The computed fragments: the segment size, the length numbers, and nothing else
	// (there is no draft block any more: it was removed 2026-10-17).
	// {SEGMENT SIZE} describes THE SEGMENT BELOW, so it has to be the segment (2026-10-17). The whole-session
	// fill is larger - it carries pi's kept-recent slice, the fixed request overhead and these notes - and a
	// model told the segment is bigger than it is will pad toward a figure that does not exist. Falls back to
	// the fill only where a caller has no segment of its own to name.
	sub("{SEGMENT SIZE}", fmt(extra?.segmentTokens ?? extra?.inputTokens ?? 0));
		// THE NUMBERS, WHEREVER THEY SIT (see lengthValues - advertised, never filled, and two compactions
		// lost their target to it). The hand-written tell further down is measured HERE, on the text before
		// the fill, because after the fill the tokens are gone by definition, and a template that wrote its
		// own length sentence would then also receive the generated block and be told the same thing twice.
		const handWroteLength = INLINE_LENGTH_TOKENS.some((t) => out.includes(t));
		for (const [token, value] of lengthValues(extra?.sizes)) out = out.split(token).join(value);
	// THE TWO BLOCKS WHOSE WORDS THE USER MAY WANT TO WRITE HIMSELF (2026-10-15: "I don't have access to the
	// wording of those sections"). Both can still be asked for as a whole block, exactly as before, OR written
	// out in the template in the user's own words. Writing them out has to be detectable, or the safety net
	// below would staple the block on again at the end and the model would be told twice: the length block's
	// inline tell is any one of its number tokens appearing in the text
	// the scratchpad is opened with (measured on the text that is actually going to be sent - note-stripping
	// runs first, so a commented-out copy of that wording cannot count as the live version of a rule).
	{
		const v = extra?.sizes ? lengthFragment(extra.sizes, extra.inputTokens ?? 0) : "";
		if (out.includes("{LENGTH INSTRUCTIONS}")) out = out.split("{LENGTH INSTRUCTIONS}").join(v);
		else if (v.trim() && handWroteLength) { /* the numbers sit in the template's own sentence and were filled above */ }
		else if (v.trim() && writtenOut) absent.push("{LENGTH INSTRUCTIONS}");
		else if (v.trim()) missing.push(v);
	}
	// 2026-10-06: a summary that collapses the ledger sub-lines into one prose line per turn is the
	// failure mode users report as "everything was heavily summarized and the tool calls are gone". The
	// template shows the shape; this says out loud that the shape is mandatory, whatever it looks like.
	if (!writtenOut) out += `\n\nSHAPE OF THE LEDGER (mandatory): copy the shape the OUTPUT FORMAT shows. Every turn entry gets its own sub-lines, one line each, with the labels exactly as the OUTPUT FORMAT writes them - the user prompt, the assistant thinking, the assistant replies, the tool calls, the errors and fixes, what was decided. Never merge two sub-lines into one sentence and never drop a sub-line the input has content for; "nothing to say" is only true when the input really has none. A turn entry written as one long line is a failed summary.`;
	if (!writtenOut && missing.length) out += `\n\nRules the template lost (auto-added here; restore their {PLACEHOLDER} tokens to place them properly):\n${missing.join("\n")}`;
	return out;
}

/** The number tokens an inline copy of the length block is built from - see applyTemplate. Exported so the
 *  template editor can list them and a test can assert they all resolve. */
export const INLINE_LENGTH_TOKENS = ["{AIM TOKENS}", "{AIM WORDS}", "{MIN TOKENS}", "{MIN WORDS}", "{MAX TOKENS}", "{MAX WORDS}"];

/**
 * NOTES THE SUMMARIZER NEVER READS (2026-10-15). The template is the user's working document as well as the
 * request that goes out, so anything between <!-- and --> is cut out at the front of substitution - before
 * the tokens are filled and before the inline-block tells are measured, so a commented-out instruction is
 * inert in every sense. That is where the wording being edited back into place can wait unread, and where a
 * note to the next reader about why a line exists can live beside it instead of in a separate file. Runs on
 * every template, the extension's own included, which uses none of these markers.
 */
export function stripTemplateNotes(src: string): string {
	return String(src ?? "")
		.replace(/<!--[\s\S]*?-->/g, "")
		.replace(/\n{3,}/g, "\n\n")
		.trim();
}

/** The template for one model - compatibility wrapper: the default source, substituted. */
export function buildTemplate(
	p: { preserveStubs: Preserve; preserveUser: Preserve; preserveReplies: Preserve; preserveThinking: Preserve },
	inputStubs: boolean,
	chain: ChainMode = "fold",
): string {
	return applyTemplate(DEFAULT_TEMPLATE_TEXT, { ...DEFAULT_OPTIONS, ...p, chainMode: chain } as ModelOptions, inputStubs);
}

/** What the menu editor shows and resets to: the placeholder-bearing SOURCE (not a substituted
 *  render - the user edits the structure, the options fill the tokens). */
export const DEFAULT_ADDITIONAL_INSTRUCTION = DEFAULT_TEMPLATE_TEXT;

/**
 * The full extra instructions for one attempt: the generated length block (its numbers always match
 * the model's own submenu values), the draft-block instruction when the model has one (before the
 * format, because it points at it), the Additional instruction (default text when none is set),
 * pi's own /compact note, the tool-call note, and the no-think tag when thinking is off.
 */
export function buildInstructions(args: {
	sizes: Sizes;
	inputTokens: number;
	kind: Kind;
	options: ModelOptions;
	additionalInstruction: string | null;
	/** The segment's own size, without the framing allowance: what {SEGMENT SIZE} reports. Falls back to
	 *  inputTokens, which is the whole request figure, when a caller has nothing better to say. */
	segmentTokens?: number;
	/** The main template's USER default: used when the active is null (empty = follow it). */
	userDefault?: string | null;
	customInstructions?: string;
}): string {
	const { sizes, inputTokens, kind, options } = args;
	const segmentTokens = args.segmentTokens ?? inputTokens;
	const parts: string[] = [];
	// THE SUMMARY TEMPLATE (2026-10-06): ONE text, the one the Summary template menu shows and edits.
	// It used to be two stored texts with an order of precedence between them, and the one the menu
	// did NOT show won - the request carried a different template from the one on screen. Now the
	// menu's text is the template, the extension default rides when none is saved, and the Additional
	// instruction is what its name says: an extra paragraph after the template, never a replacement.
	parts.push(applyTemplate(args.userDefault?.trim() || DEFAULT_TEMPLATE_TEXT, options, options.inputStubs, { sizes, inputTokens, segmentTokens }));
	// The extra instruction, when it is an instruction. A stored one that is itself a whole template
	// (it carries the placeholders) is a leftover of the old two-text design and is NOT appended twice.
	const extraInstruction = args.additionalInstruction?.trim();
	if (extraInstruction && !looksLikeTemplate(extraInstruction)) parts.push(extraInstruction);
	if (args.customInstructions) parts.push(`Extra focus for this summary, from the user's own /compact note: ${args.customInstructions}`);
	if (options.thinking === "off" && kind === "local" && options.noThinkMarker) parts.push(options.noThinkMarker);
	return parts.filter(Boolean).join("\n\n");
}

/**
 * Instructions for the optional corrective pass: the same instructions plus the model's rejected
 * draft and an explicit demand to expand it to the minimum.
 */

// ---------------------------------------------------------------- config

function clamp(n: number, lo: number, hi: number): number {
	return Math.max(lo, Math.min(hi, n));
}

function numOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function round2(n: number): number {
	return Math.round(n * 100) / 100;
}

/** A fraction of reserveTokens as the menu shows it. */
function frac(n: number): string {
	return String(round2(n));
}

/** Order position of a model in the Compaction models list, -1 when not in it. */
function orderPosition(cfg: Config, ref: string): number {
	return Object.keys(cfg.models).indexOf(ref);
}

/** Kind guess from a config reference alone (no registry): llama-server refs and localhost/LAN URLs
 *  are local. Same idea as isLocalModel, which needs a live model object instead. */
function refLooksLocal(ref: string): boolean {
	return /llama|ollama|127\.0\.0\.1|localhost|192\.168\.|(^|\.)10\./i.test(ref);
}

/** Migrate one stored per-model profile (config v2 "overrides"/"defaults") to the new options. */
function modelOptionsFrom(src: any, fallbackMarker: string): ModelOptions {
	const levels = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
	const o: ModelOptions = {
		thinking: (levels.includes(src?.thinking) ? src.thinking : "off") as Level,
		timeoutMs: clamp(Math.round(numOr(src?.timeoutMs, 600_000)), 0, 7_200_000),
		summaryPercent: pctField(src?.summaryPercent, DEFAULT_OPTIONS.summaryPercent),
		summaryMinTokens: nonNegField(src?.summaryMinTokens, DEFAULT_OPTIONS.summaryMinTokens),
		summaryMaxTokens: nonNegField(src?.summaryMaxTokens, DEFAULT_OPTIONS.summaryMaxTokens),
		legacyMinPct: typeof src?.summaryMinPercent === "number" && Number.isFinite(src.summaryMinPercent) ? src.summaryMinPercent : undefined,
		legacyMaxPct: typeof src?.summaryMaxPercent === "number" && Number.isFinite(src.summaryMaxPercent) ? src.summaryMaxPercent : undefined,
		inputStubs: typeof src?.inputStubs === "boolean" ? src.inputStubs : true,
		argCap: isCharCap(src?.argCap) ? src.argCap : 500,
		preserveStubs: isPreserve(src?.preserveStubs) ? src.preserveStubs : "verbatim",
		preserveUser: isPreserve(src?.preserveUser) ? src.preserveUser : "verbatim",
		preserveReplies: isPreserve(src?.preserveReplies) ? src.preserveReplies : "detailed",
		preserveThinking: isPreserve(src?.preserveThinking) ? src.preserveThinking : "summary",
		chainMode: isChainMode(src?.chainMode) ? src.chainMode : "fold",
		noThinkMarker: typeof src?.noThinkMarker === "string" ? src.noThinkMarker : fallbackMarker,
		sampling: sanitizeSampling(src?.sampling),
	};
	// Keep 0 = unlimited; sub-minute odd values snap up to the smallest step (2 min).
	if (o.timeoutMs > 0 && o.timeoutMs < 60_000) o.timeoutMs = TIMEOUT_STEPS[0];
	return o;
}

/**
 * Accept the older config versions:
 *  - v1: { model, fallbacks, cap/extra/mandate, prompt, promptOnline }
 *  - v2: { models: [...], defaults: { local, online }, overrides: { ref: {...} } }
 * and turn both into the v3 shape: { models: { ref: ModelOptions } } with per-model options.
 */
function migrateV1andV2(raw: any): { models: Record<string, ModelOptions>; retries: number; retryDelaySeconds: number; noThinkMarker: string; additionalInstruction: string | null } {
	// --- the model list: v3 object, v2/v1 arrays, v1 single model + fallbacks
	const refs: string[] = [];
	const push = (v: unknown) => {
		if (typeof v !== "string" || !v.trim()) return;
		const ref = v.trim();
		if (!refs.includes(ref)) refs.push(ref);
	};
	if (raw?.models && typeof raw.models === "object" && !Array.isArray(raw.models)) {
		for (const ref of Object.keys(raw.models)) push(ref);
	} else {
		if (Array.isArray(raw?.models)) for (const v of raw.models) push(v);
		if (Array.isArray(raw?.chain)) for (const v of raw.chain) push(v);
		push(raw?.model);
		if (Array.isArray(raw?.fallbacks)) for (const v of raw.fallbacks) push(v);
	}
	const selected = refs.slice(0, MAX_SLOTS);

	// --- the pi-wide no-think tag: v3 name first, then the v2 name
	const noThinkMarker = typeof raw?.noThinkMarker === "string" ? raw.noThinkMarker.trim() : typeof raw?.thinkOffMarker === "string" ? raw.thinkOffMarker.trim() : DEFAULT_NO_THINK_TAG;

	// --- additional instruction: v3 name, then the v2/v1 prompt texts (the stock old text is
	//     dropped - the new default text replaces it; a hand-written old text is kept)
	const LEGACY_PROMPT_DEFAULT =
		"Make a very detailed summary of the text. Preserve all user prompts and all assistant replies in full. Summarize assistant thinking. Preserve all file links verbatim. Make a brief summary of tool calls.";
	const legacy = typeof raw?.prompt === "string" && raw.prompt.trim() ? raw.prompt : null;
	const additionalInstruction =
		typeof raw?.additionalInstruction === "string"
			? raw.additionalInstruction
			: legacy && legacy.trim() !== LEGACY_PROMPT_DEFAULT
				? legacy
				: null;

	// --- retries
	const retries = Number.isFinite(raw?.retries) ? clamp(Math.round(raw.retries), 0, 5) : 2;
	const retryDelaySeconds = Number.isFinite(raw?.retryDelaySeconds) ? clamp(Math.round(raw.retryDelaySeconds), 0, 300) : 5;

	// --- per-model options, best source first:
	//     1. a v3 models object entry
	//     2. a v2 overrides[ref] entry (thinking/timeout/fractions; old absolute caps become shares)
	//     3. the v2 kind default (local/online) when we can tell which kind the model is
	// Everything else gets the internal defaults applied at selection time.
	const models: Record<string, ModelOptions> = {};
	const overridesV2: Record<string, any> = raw?.overrides && typeof raw.overrides === "object" ? raw.overrides : {};
	const defaultsLocal = raw?.defaults?.local ?? {};
	const defaultsOnline = raw?.defaults?.online ?? raw?.defaults?.cloud ?? {};
	for (const ref of selected) {
		const looksLocal = refLooksLocal(ref);
		const v3 = raw?.models?.[ref];
		if (v3 && typeof v3 === "object") {
			models[ref] = modelOptionsFrom(v3, noThinkMarker, looksLocal ? "off" : "full");
			continue;
		}
		const kindDefault = looksLocal ? defaultsLocal : defaultsOnline;
		// The pi-wide no-think tag only ever applied to local models in v2.
		const kindMarker = looksLocal ? noThinkMarker : "";
		const o2 = overridesV2[ref];
		if (o2 && typeof o2 === "object" && (o2.thinking !== undefined || o2.timeoutMs !== undefined || o2.targetFrac !== undefined || o2.summaryBudgetTokens !== undefined)) {
			// An override rides ON TOP of the kind defaults, so an entry that only set the timeout
			// still inherits the kind's thinking and sizes.
			models[ref] = modelOptionsFrom({ ...kindDefault, ...o2 }, kindMarker);
			continue;
		}
		if (kindDefault && (kindDefault.thinking !== undefined || kindDefault.timeoutMs !== undefined || kindDefault.targetFrac !== undefined || kindDefault.summaryBudgetTokens !== undefined)) {
			models[ref] = modelOptionsFrom(kindDefault, kindMarker);
		}
	}
	return { models, retries, retryDelaySeconds, noThinkMarker, additionalInstruction };
}

/** Percent field: any number 1-99, else the default. */
function pctField(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 1 && value <= 99 ? Math.round(value) : fallback;
}

/** Every size number in this file goes through this. A NaN reaching sizesFor is what made every
 *  compaction of 2026-10-05 ride out with aim/min/max = NaN: both acceptance gates compare with <
 *  and >, both are false against NaN, so every summary was accepted whatever its length, and the
 *  request went out with maxTokens = NaN. A non-finite number never passes this point: it becomes
 *  the default, and the default is a real number. */
function tok(value: unknown, fallback = 0): number {
	return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.round(value)) : Math.max(0, Math.round(fallback));
}

/** A pre-2026-10-05 config stored the size limits as percents of the window: convert one with THIS
 *  model's window. undefined (no stored percent, or no window) means: keep the token default. */
function legacyPctToTokens(pct: number | undefined, windowTokens: number): number | undefined {
	if (typeof pct !== "number" || !Number.isFinite(pct)) return undefined;
	if (!(windowTokens > 0) || !Number.isFinite(windowTokens)) return undefined;
	return Math.max(0, Math.round((pct / 100) * windowTokens));
}

/** Non-negative token field: any finite number >= 0, else the default. */
function nonNegField(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.round(value) : fallback;
}

export function loadConfig(): Config {
	const base = DEFAULT_CONFIG;
	let raw: any;
	try {
		raw = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
	} catch {
		return structuredClone(base);
	}
	const m = migrateV1andV2(raw);
	return {
		enabled: typeof raw.enabled === "boolean" ? raw.enabled : base.enabled,
		models: m.models,
		retries: m.retries,
		retryDelaySeconds: m.retryDelaySeconds,
		log: typeof raw.log === "boolean" ? raw.log : base.log,
		transcriptPointer: typeof raw.transcriptPointer === "boolean" ? raw.transcriptPointer : base.transcriptPointer,
		// "Direct request" stopped being a user option on 2026-10-05: the template path is the only path,
		// so the stored value is ignored and the direct request always runs.
		directRequest: true,
		shapeGate: typeof raw.shapeGate === "boolean" ? raw.shapeGate : base.shapeGate,
		chatMaxTokensCap: typeof raw.chatMaxTokensCap === "number" && Number.isFinite(raw.chatMaxTokensCap) && raw.chatMaxTokensCap >= 0 ? Math.round(raw.chatMaxTokensCap) : base.chatMaxTokensCap,
		// The two sizes are token counts now; migrateScaling translates a file that still holds the old
		// percentages, so an upgrade changes nothing the operator can see or feel.
		scaling: migrateScaling(raw, base.scaling),
		autoCompact: typeof raw.autoCompact === "boolean" ? raw.autoCompact : base.autoCompact,
		autoCompactPaused: typeof raw.autoCompactPaused === "boolean" || typeof raw.autoCompactPaused === "string" ? raw.autoCompactPaused : base.autoCompactPaused,
		elision: sanitizeElision(raw?.elision),
		noThinkMarker: m.noThinkMarker,
		// 2026-10-06: the template is ONE text now - the one the Summary template menu edits. An old
		// config could hold a whole template in the "additional instruction" slot, and that copy used
		// to WIN over the menu's text, so the request carried a template nobody was looking at. Such a
		// copy moves aside into templatePreviousCopy and stops shadowing anything.
		additionalInstruction: typeof m.additionalInstruction === 'string' && looksLikeTemplate(m.additionalInstruction)
			? null
			: m.additionalInstruction,
		templatePreviousCopy:
			typeof m.additionalInstruction === 'string' && looksLikeTemplate(m.additionalInstruction)
				? m.additionalInstruction
				: typeof raw?.templatePreviousCopy === 'string' && raw.templatePreviousCopy.trim()
					? raw.templatePreviousCopy
					: null,
		categorySentences: sanitizeCategorySentences(raw?.categorySentences),
		categoryUserDefaults: sanitizeCategorySentences(raw?.categoryUserDefaults),
		templateUserDefaults: typeof raw?.templateUserDefaults === "string" && raw.templateUserDefaults.trim() ? raw.templateUserDefaults : null,
		// Two settings the loader used to forget (2026-10-21, defect #41). A field that is not copied
		// here is undefined at runtime AND vanishes from the file on the next write, so the setting
		// reads as 'I changed it and it would not stick'. activeTemplateName is which library row shows
		// ACTIVE and what Ctrl+P re-selects after a restart. The receipt block that used to ride inside
		// closes a summary. Both are checked for shape like everything else above.
		activeTemplateName:
			typeof raw.activeTemplateName === "string" && raw.activeTemplateName.trim()
				? raw.activeTemplateName
				: base.activeTemplateName,
		// Read it or the operator can never switch it on: `missingConfigKeys` fails a setting the loader
		// does not carry, which is how this field would have stayed permanently off while looking settable.
		landWithoutAbort: typeof raw.landWithoutAbort === "boolean" ? raw.landWithoutAbort : base.landWithoutAbort,
		userTemplates: Array.isArray(raw?.userTemplates)
			? (raw.userTemplates as any[])
					.filter((t) => t && typeof t.name === "string" && typeof t.text === "string")
					.map((t) => ({ name: t.name as string, text: t.text as string }))
			: [],
	};
}

/** Keep only well-formed category sentence overrides: known kinds, known levels, non-empty text. */
function sanitizeCategorySentences(raw: any): { [kind: string]: { [level: string]: string } } | null {
	if (!raw || typeof raw !== "object") return null;
	const KINDS = ["stubs", "user", "replies", "thinking"];
	const LEVELS = ["verbatim", "detailed", "summary", "brief"];
	const out: { [kind: string]: { [level: string]: string } } = {};
	for (const kind of KINDS) {
		const levels = raw?.[kind];
		if (!levels || typeof levels !== "object") continue;
		for (const level of LEVELS) {
			const v = levels?.[level];
			if (typeof v === "string" && v.trim()) (out[kind] ??= {})[level] = v.trim();
		}
	}
	return Object.keys(out).length ? out : null;
}

/** Test harnesses point COMPACTION_PLUS_CONFIG at a scratch copy and set COMPACTION_PLUS_READ_ONLY,
 *  so a harness that forgets the copy still cannot write into the operator's files. Twice now a test
 *  run polluted a real record: elision events on 2026-10-04, and 18 fake trigger lines on
 *  2026-10-21 - the second time because a static `import` runs before the harness can set the
 *  override, which is why the switch is a separate variable and not the path. */
function writesBlocked(): boolean {
	return process.env.COMPACTION_PLUS_READ_ONLY === "1";
}

function saveConfig(cfg: Config): void {
	mkdirSync(AGENT_DIR, { recursive: true });
	if (writesBlocked()) return;
	writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2) + "\n", "utf8");
}

/** Which settings in DEFAULT_CONFIG a loaded config does not carry. A field the loader forgets is
 *  undefined at runtime and disappears from the file on the next write; on 2026-10-21 that was how
 *  two settings managed to look like they would not persist. Reported once at start-up and repaired. */
export function missingConfigKeys(cfg: Config): string[] {
	return Object.keys(DEFAULT_CONFIG).filter((k) => (cfg as Record<string, unknown>)[k] === undefined);
}

/** Which top-level settings differ between two configs, for the write record. Model options are
* named per field, because 'models changed' tells a later reader nothing. */
export function changedSettings(before: Config, after: Config): string[] {
	const out: string[] = [];
	for (const key of Object.keys(after) as (keyof Config)[]) {
		if (JSON.stringify((before as Record<string, unknown>)[key] ?? null) === JSON.stringify((after as Record<string, unknown>)[key] ?? null)) continue;
		if (key !== "models") {
			out.push(String(key));
			continue;
		}
		const b = (before.models ?? {}) as Record<string, Record<string, unknown>>;
		const a = after.models ?? {};
		for (const ref of new Set([...Object.keys(b), ...Object.keys(a)])) {
			const x = b[ref] ?? {};
			const y = a[ref] ?? {};
			for (const field of new Set([...Object.keys(x), ...Object.keys(y)])) {
				if (JSON.stringify(x[field] ?? null) !== JSON.stringify(y[field] ?? null)) {
					out.push(`models.${ref.split("=").pop()}.${field}`);
				}
			}
		}
	}
	return out;
}

/**
 * One setting change, written and then checked. Added 2026-10-21 after the report that settings
 * reverted over a restart: a write that fails, or that writes a stale copy of the config, looks
 * exactly like a setting that was never changed. So every change says what it changed, and the file
 * is read back to prove the bytes on disk are the bytes that were meant to be there.
 */
function update(fn: (cfg: Config) => void): void {
	const before = loadConfig();
	const cfg = loadConfig(); // re-read, never a copy somebody held on to across a dialog
	fn(cfg);
	const changed = changedSettings(before, cfg);
	if (!changed.length) return;
	let error = "";
	try {
		saveConfig(cfg);
	} catch (e: any) {
		error = String(e?.message ?? e);
	}
	// Read it back. The file is small and this happens once per settings change, so the cost is
	// nothing next to a change the operator believes they made and did not.
	let verified = writesBlocked(); // a harness may not write, so there is nothing to confirm
	let reason = "";
	if (!error && !verified) {
		const wanted = JSON.stringify(cfg, null, 2) + "\n";
		let back = "";
		try {
			back = readFileSync(CONFIG_PATH, "utf8");
		} catch (e: any) {
			reason = String(e?.message ?? e);
		}
		if (back === wanted) verified = true;
		else if (!reason) reason = back ? "the file holds something else" : "the file came back empty";
	}
	logLine(cfg, {
		event: verified ? "config_written" : "config_write_failed",
		changed: changed.slice(0, 12).join(","),
		count: changed.length,
		...(error ? { error } : {}),
		...(reason ? { reason } : {}),
	});
	if (!verified && !writesBlocked()) {
		// Screen silence is the rule, but a setting that did not persist is the one thing the operator
		// must not discover a restart later.
		try {
			watcherCtx?.ui?.notify?.(
				`A settings change did not stick (${changed.slice(0, 3).join(", ")}${error ? ": " + error : reason ? ": " + reason : ""}).`,
				"warning",
			);
		} catch {
			/* no screen to tell - the log line above is the record */
		}
	}
}


/** One line per try, so "which model actually wrote that summary" is never a guess again. */
// ---------------------------------------------------------------- attempt log

/** Local-time stamp "YYYY-MM-DDTHH:MM:SS.mmm" - the machine's own timezone, still lexicographically
 *  sortable. (Date.toISOString() is UTC; the log must read like the wall clock it hangs next to.) */
function localStamp(): string {
	const d = new Date();
	const p = (n: number, w = 2) => String(n).padStart(w, "0");
	return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
}

function logLine(cfg: Config, record: Record<string, unknown>): void {
	if (!cfg.log) return;
	try {
		mkdirSync(AGENT_DIR, { recursive: true });
		try {
			if (statSync(LOG_PATH).size > LOG_MAX_BYTES) renameSync(LOG_PATH, LOG_PATH + ".1");
		} catch {
			/* no log file yet */
		}
		// The pid travels with every line. Two processes can write this one file - the pi that is running the
		// session and a test harness driving the same code - and on 2026-10-18 that distinction was the whole
		// question twice: once when an old build's writes were mistaken for the new one's, and once when a
		// harness's own events had to be told apart from a real compaction's.
		if (writesBlocked()) return;
		appendFileSync(LOG_PATH, JSON.stringify({ ts: localStamp(), pid: process.pid, ...record }) + "\n", "utf8");
	} catch {
		/* logging must never break compaction */
	}
}

/** The events that are bookkeeping rather than news. They stay in the file - they are how a strange
 *  compaction gets diagnosed afterwards - but they are not what the operator opens the log to read, and
 *  in a long session they pushed the result lines off the screen entirely (2026-10-21, on request:
 *  "sometimes I cannot even see the compaction info because the log is filled with some silly useless
 *  lines"). 'v' in the log view shows them again.
 *
 *  Deliberately NOT here: start, attempt, ok, fail, direct_fail, accepted_with_warnings, give_up, done,
 *  every *_failed, retry, skip, elision_error, auto_compaction_held. Skip stays because "model X was not in"
 *  the model list" is the answer to a question the operator asks; auto_compaction_held stays because when
 *  pi asks early and nothing happens, that line is the only place saying why. */
/**
 * What each log event is CALLED on screen (2026-10-21). Kept next to LOG_CHATTER, which decides
 * whether a row is shown at all: this decides what a shown row says. The two lists answer different
 * questions, so an event can be visible with a plain name (compaction_landed -> COMPACTED) or
 * hidden with one (auto_compaction_held).
 */
const LOG_ROW_NAMES: Record<string, string> = {
	start: "COMPACTION START",
	attempt: "COMPACTION TRY",
	ok: "COMPACTION SUMMARY",
	done: "COMPACTION DONE",
	compaction_landed: "COMPACTED",
	// The append path of #43 explaining itself is worth a row: it is the difference between "nothing
	// happened" and "it fell back to the interrupting route, and here is why".
	no_abort_landing_failed: "NO-ABORT LANDING FAILED",
	fail: "COMPACTION TRY FAILED",
	direct_fail: "COMPACTION TRY FAILED",
	give_up: "COMPACTION GAVE UP",
	skip: "MODEL SKIPPED",
	accepted_with_warnings: "COMPACTION NOTES",
	compaction_no_op: "NOTHING TO COMPACT",
	compaction_failed: "COMPACTION FAILED",
	auto_compaction_failed: "COMPACTION FAILED",
	auto_compaction_paused: "AUTO-COMPACTION PARKED",
	auto_compaction_skipped: "PARKED: ASK IGNORED",
	auto_compaction_parked_warning: "PARKED: ALMOST FULL",
	elision: "ELISION",
	elision_error: "ELISION FAILED",
	auto_compaction_held: "HELD",
};

const LOG_CHATTER = new Set(["user_prompts", // one per candidate that passed its checks; COMPACTED carries the figures
	// Two rows that repeat what a neighbour says (2026-10-22): `done` is written the moment a finished
	// summary is handed over and COMPACTED follows it in the same second; the notes row fires for about
	// four summaries in ten, because a model-written summary is rarely exactly the shape asked for. Both
	// stay in the file - if COMPACTED is ever missing, `done` proves the summary existed.
	"done",
	"accepted_with_warnings",
	// Nothing writes these any more (the automatic continuation was deleted 2026-10-21), but the log file
	// still holds years of them, and the results-only view must go on hiding them.
	"auto_resume_waiting",
	"auto_resume_turn",
	"auto_resume_retrying",
	"auto_resume_cancelled",
	"auto_resume_failed",
	"auto_resume_not_needed",
	"auto_resume_dropped",
	"preprocess",
	"sizes",
	"note",
	"info",
	"trim",
	"user_prompts",
	// The elision DECISIONS stay hidden (a switch thrown, a batch deferred); the elision BATCH itself
	// comes back into the view, because it is the entry that carries the numbers - what was stubbed and
	// how many tokens it saved (2026-10-21: "i want to see only compaction and elision entries there with
	// their data").
	"elision_off",
	"elision_hold",
	"elision_wait",
	"two_compactors",
	"safe_point_held",
	// The record that pi accepted the summary is part of the compaction, not bookkeeping about it:
	// it carries the size of the summary that actually landed (2026-10-21).
	// (compaction_landed left this list.)
	// WHEN the trigger fires is arithmetic, not a result: the two clocks, the hold while pi asks early,
	// the retry slot, the park being re-armed. All of it stays in the file for a later diagnosis and out
	// of the view (2026-10-21: "you can stop inserting any unnecessary log entries ... filter them out in
	// the log display").
	"auto_compaction_held",
	"auto_compaction_cancelled",
	"auto_compaction_resumed",
	"auto_trigger_retry",
	"safe_point_released",
	"trigger_clock",
	"trigger_clock_released",
	"ctx_dbg",
	"watcher",
	"heartbeat",
	// Settings writes: the screen already says when a write failed, and a per-keystroke board row in a
	// results view is noise.
	"config_written",
	"config_write_failed",
	"enable_refused",
	// auto_trigger is the watcher's own arithmetic: the estimate, the line, the window, and - while
	// pi's threshold holds the trigger, a standing-back note repeated on a throttle. What the operator opens
	// the log for is the compaction, and `start` already carries the reason it happened, so this is
	// bookkeeping for a later diagnosis, not news (2026-10-21: "i don't need to see that line").
	"auto_trigger",
]);

/**
 * The log, human-readable: date and time first, then status, model, elapsed time and the transfer
 * figures in tokens (words are not logged - they were the old sizing flaw, and tokens are what the
 * acceptance window is written in). Unknown fields are simply left out.
 */
function tailLogPretty(lines = 16, verbose = false): string {
	try {
		// Read far more lines than are shown, because filtering drops most of them: 201 of the last 400
		// entries in a real session were one bookkeeping event.
		const raw = readFileSync(LOG_PATH, "utf8").split("\n").filter(Boolean).slice(verbose ? -lines : -lines * 60);
		const all = verbose
			? raw.slice(-lines)
			: raw.filter((l: string) => {
					try {
						return !LOG_CHATTER.has(String(JSON.parse(l)?.event ?? ""));
					} catch {
						return true; /* a line we cannot read is not a line to hide */
					}
				}).slice(-lines);
		if (!all.length) return verbose ? "(the log is empty so far)" : "(no result lines yet - press v to see every line, including the bookkeeping)";
		return all
			.map((l: string) => {
				try {
					const d = JSON.parse(l);
					const when = String(d.ts ?? "");
					const stamp = `${when.slice(0, 10)} ${when.slice(11, 19)}`;
					const ref = d.ref ? String(d.ref).split("/").pop()?.slice(0, 34) : "";
					const seconds = typeof d.seconds === "number" ? `${d.seconds}s` : "";
					// Transfer figures: pi's own provider counts when present, otherwise the estimate.
					const tokIn = typeof d.input === "number" ? d.input : undefined;
					const tokOut = typeof d.output === "number" ? d.output : typeof d.measured === "number" ? d.measured : undefined;
					const transfer =
						tokIn !== undefined || tokOut !== undefined
							? `${tokIn !== undefined ? `${fmt(tokIn)} tok in` : ""}${tokIn !== undefined && tokOut !== undefined ? ", " : ""}${tokOut !== undefined ? `${fmt(tokOut)} tok out` : ""}${typeof d.reasoning === "number" && d.reasoning > 0 ? ` (thinking ${fmt(d.reasoning)})` : ""}`
							: "";
					// The acceptance bounds are ABSOLUTE TOKENS now (the percent-of-window wording printed
					// "min ?% / max ?%" after the 2026-10-05 migration killed minPct/maxPct) and the window is
					// only printed when the event actually carried one - the "of the 0 tok window" tail was noise.
					// 2026-10-07, on request: the AIM (the size the summary was asked for) is printed with the
					// acceptance bounds, so a TRY line and an OK line both show what the run was aiming at -
					// "accept 4,096-30,000 tok" alone said nothing about the target.
					const aim = typeof d.target === "number" && d.target > 0 ? `aim ${fmt(d.target)} tok` : "";
					const window =
						typeof d.min === "number" && typeof d.max === "number"
							? `${aim ? aim + " \u00b7 " : ""}accept ${fmt(d.min)}-${fmt(d.max)} tok${typeof d.window === "number" && d.window > 0 ? ` (window ${fmt(d.window)} tok)` : ""}`
							: aim;
					// The words a row starts with (2026-10-21, on request). The operator reads this view to see
					// what the compactor DID, so the internal event names get plain ones: a run begins with
					// COMPACTION START, each model tried is a COMPACTION TRY, and the moment the new context exists
					// is COMPACTED. An event that is not listed shows its own name in capitals, so a new
					// record type can be unnamed for a while but never silently nameless.
					const status = LOG_ROW_NAMES[String(d.event ?? "")] ?? String(d.event ?? "").toUpperCase();
					// An accepted-with-warnings line must say WHAT was not as asked - the warnings array is the
					// list of rule violations the summary was accepted despite (2026-10-07, on request).
					const why =
						d.reason ??
						(Array.isArray(d.warnings) && d.warnings.length
							? `accepted despite: ${d.warnings.map((w: any) => String(w)).join("; ").slice(0, 300)}`
							: d.error
								? String(d.error).split("\n")[0].slice(0, 90)
								: "");
					// A line the operator is shown should say something when it is shown. `note` used to be
					// dropped, so the gate's hold - the one line that explains why a compaction did NOT
					// happen - arrived as a bare AUTO_COMPACTION_HELD with nothing after it (2026-10-21).
					// Bookkeeping events are filtered out of the results view anyway, so putting the note
					// here only ever adds words where the operator asked to be told.
					const note =
						typeof d.note === "string" && !LOG_CHATTER.has(String(d.event ?? ""))
							? d.note.replace(/\s+/g, " ").slice(0, 400)
							: "";
					// The elision events carry their own figures.
					const elision = d.event === "elision" ? [`${fmt(d.applied ?? 0)} result${d.applied === 1 ? "" : "s"} elided`, `${fmt(d.resultsTokens ?? 0)} tok -> ${fmt(d.stubTokens ?? 0)} tok`, `saved ${fmt(d.savedTokens ?? 0)} tok`, `ctx ~${fmt(d.effTokens ?? 0)} tok -> ~${fmt(d.postBatchWire ?? 0)} tok`].join(" · ") : "";
					// A landed summary is a result with a size, and the size is the figure worth keeping
					// in view next to the aim it was asked for (2026-10-21: "only compaction and elision entries
					// there with their data').
					const summary = typeof d.summaryTokens === "number" ? `summary ${fmt(d.summaryTokens)} tok` + (typeof d.measured === "number" && typeof d.min === "number" ? ` (asked ${fmt(d.min)}-${fmt(d.max)} tok, ${fmt(d.words ?? 0)} words written)` : "") + (typeof d.quoted === "number" && typeof d.given === "number" ? `, user prompts quoted ${d.quoted} of ${d.given}` : "") + (Array.isArray(d.failures) && d.failures.length ? `, after ${d.failures.length} candidate(s) tried and lost` : "") : "";
					return [stamp, status, ref ?? "", seconds, transfer, window, elision, why, summary, note].filter(Boolean).join(" · ");
				} catch {
					return l.slice(0, 400);
				}
			})
			.join("\n");
	} catch {
		return `(no log file yet at ${LOG_PATH})`;
	}
}

// ---------------------------------------------------------------- pi's compaction settings

/** pi's own settings file (kept for reference; the extension no longer writes to it - the 2026-10-01
 *  unlink removed the last writer, writePiCompactionNumber). */
const SETTINGS_PATH = join(AGENT_DIR, "settings.json");

// ---------------------------------------------------------------- model helpers

const MAX_SLOTS = 5;

function parseRef(ref: string): { provider: string; id: string } | null {
	const at = ref.indexOf("/");
	if (at <= 0 || at === ref.length - 1) return null;
	return { provider: ref.slice(0, at), id: ref.slice(at + 1) };
}

function refOf(model: { provider: string; id: string }): string {
	return `${model.provider}/${model.id}`;
}

function isSessionRef(ref: string): boolean {
	return ref === SESSION_REF || ref === "@session";
}

/**
 * Resolve a configured reference to a model, "session" included.
 *
 * Provider ids can contain slashes and colons themselves (a llama.cpp provider registers refs like
 * "llama-server=http://127.0.0.1:<port>/<path>...gguf", so a naive split at the first "/" produces
 * nonsense). So: first try an exact match over every model pi can see, then try find() at every
 * slash position.
 */
export function lookupRef(ref: string | null, ctx: { modelRegistry: any; model?: any }): any | undefined {
	if (!ref) return undefined;
	if (isSessionRef(ref)) return ctx?.model;
	const pool: any[] = [...(ctx?.modelRegistry?.getAvailable?.() ?? [])];
	if (ctx?.model) pool.push(ctx.model);
	for (const m of pool) {
		if (m && refOf(m) === ref) return m;
	}
	for (let i = 0; i < ref.length; i++) {
		if (ref[i] !== "/") continue;
		try {
			const found = ctx?.modelRegistry?.find?.(ref.slice(0, i), ref.slice(i + 1));
			if (found) return found;
		} catch {
			/* keep trying */
		}
	}
	return undefined;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve) => {
		const done = () => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", done);
			resolve();
		};
		const timer = setTimeout(done, ms);
		signal?.addEventListener("abort", done, { once: true });
	});
}

/** True for anything served from this machine or the LAN (llama.cpp, ollama, ...). */
function isLocalModel(model: { provider: string; baseUrl?: string }): boolean {
	const url = (model.baseUrl ?? "").toLowerCase();
	if (/localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|192\.168\.|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\./.test(url)) {
		return true;
	}
	return /llama|ollama|lm-?studio|cpu-om|local/i.test(model.provider);
}

function profileFor(cfg: Config, model: { provider: string; id: string; baseUrl?: string } | undefined, ref?: string): ModelOptions {
	if (ref && cfg.models[ref]) return optionsFor(cfg, ref, model);
	if (model) {
		const found = Object.keys(cfg.models).find((r) => !isSessionRef(r) && r === refOf(model));
		if (found) return optionsFor(cfg, found, model);
	}
	return { ...DEFAULT_OPTIONS };
}
function kindOf(model: any): Kind {
	return isLocalModel(model) ? "local" : "online";
}

/**
 * How full the session model's context actually is, in tokens - the same figure pi compares against
 * contextWindow - reserveTokens when it decides to compact (getContextUsage in core/agent-session.js).
 * Asking pi rather than measuring here is deliberate: a number invented in an extension drifts away
 * from the one that actually triggers compaction. pi answers undefined when the model has no known
 * window, and tokens: null until the first answer after a compaction. Both arrive here as null,
 * meaning "unknown", which callers must not read as "empty".
 */
function realContextTokens(ctx: Ctx): number | null {
	try {
		const getUsage: any = (ctx as any)?.getContextUsage;
		if (typeof getUsage !== "function") return null;
		const u = getUsage.call(ctx);
		return u && typeof u.tokens === "number" && u.tokens > 0 ? u.tokens : null;
	} catch {
		return null;
	}
}

/** Tokens of one message, by pi's own counting (images, thinking and tool calls included). */
function msgTokens(m: any): number {
	try {
		return estimateTokens(m);
	} catch {
		return Math.ceil(messageChars(m) / 4);
	}
}

/**
 * Size of the history going into the summariser, with no request overhead in it: the raw messages, as
 * they stand in the session, before pi clips tool results. A summary longer than the content it replaces
 * buys nothing, so while the session is overflowing this is the ceiling we impose on ourselves - pi does
 * not check an extension's summary length at all, it takes whatever the extension returns.
 */
export function historyTokensOnly(preparation: any): number {
	let tokens = 0;
	const seen = new Set<any>();
	for (const group of [preparation.messagesToSummarize ?? [], preparation.turnPrefixMessages ?? []]) {
		for (const m of group) {
			if (seen.has(m)) continue;
			seen.add(m);
			tokens += msgTokens(m);
		}
	}
	// The previous summary is copied in as plain text, and the summarisation instruction rides with it.
	return tokens + Math.ceil(((preparation.previousSummary ?? "").length + 2000) / 4);
}

/**
 * Tokens of the part of every request that has nothing to do with the conversation: the system prompt
 * and the tool definitions. pi keeps both on the projected system message (its content plus
 * `toolsAdded`), and estimateTokens counts them, so this is a measurement rather than a guess. With
 * the MCP servers connected on this machine it comes to about 25,000 tokens - the amount that was
 * missing when an overflow compaction got missed on 2026-09-22.
 */
function fixedRequestTokens(ctx: Ctx): number {
	try {
		const projection: any = (ctx as any)?.sessionManager?.buildSessionProjection?.();
		const sys = (projection?.messages ?? []).find((m: any) => m?.role === "system");
		if (sys) {
			const t = msgTokens(sys);
			if (t > 0) return t;
		}
	} catch {
		/* fall through to the plain prompt text */
	}
	try {
		const text: any = (ctx as any)?.getSystemPrompt?.();
		if (typeof text === "string" && text.length) return Math.ceil(text.length / 4);
	} catch {
		/* fall through to the standing allowance */
	}
	return FIXED_REQUEST_TOKENS;
}

/** Characters of one conversation message as it goes on the wire, images included. */
function messageChars(m: any): number {
	if (typeof m?.content === "string") return m.content.length;
	let chars = 0;
	for (const b of m?.content ?? []) {
		if (b?.type === "text") chars += String(b.text ?? "").length;
		else if (b?.type === "thinking") chars += String(b.thinking ?? "").length;
		else if (b?.type === "toolCall" || b?.type === "tool_use") {
			chars += String(b.name ?? "").length + JSON.stringify(b.arguments ?? b.input ?? {}).length;
		} else if (b?.type === "image") chars += 1500 * 4; // a resized image costs roughly 1500 tokens
	}
	return chars + 24; // role, id and JSON framing
}

/** Characters that surround the conversation inside a summarisation request. */
const SUMMARISER_FRAMING_TOKENS = 1500;

/**
 * Reported token counts ran 1.24 to 1.31 times the plain characters-divided-by-four estimate on every
 * Gemini request in the log (43,630 reported against 34,560 estimated, 18,212 against 14,077), because
 * flattened tool output and JSON tokenize denser than prose. Estimates of the summariser request are
 * multiplied by this so a slot is never chosen because a number looked small.
 */
const TOKEN_HEAVINESS = 1.3;

/**
 * The conversation exactly as the summariser receives it. pi flattens the messages with convertToLlm
 * and serializeConversation, and that writer clips every tool result to 2,000 characters
 * (TOOL_RESULT_MAX_CHARS in core/compaction/utils.js). That clip alone removed 30 to 55 percent of the
 * characters of the last fourteen regions on this session (117,021 raw against 85,343 kept, 193,524
 * against 122,177): a 120,000 character file read reaches the summariser as 2,000 characters.
 */
function summariserText(preparation: any): string {
	const lists = [...(preparation?.messagesToSummarize ?? []), ...(preparation?.turnPrefixMessages ?? [])];
	try {
		let text = serializeConversation(convertToLlm(lists));
		if (preparation?.previousSummary) text += `\n${preparation.previousSummary}`;
		return text;
	} catch {
		// The flattener is unavailable: fall back to the raw messages, which overestimates.
		let chars = (preparation?.previousSummary ?? "").length;
		for (const m of lists) chars += messageChars(m);
		// Only the length matters for sizing, so the filler is a stand-in for the flattened text.
		return "x".repeat(chars);
	}
}

/**
 * Tokens of the request that carries the region. A summarisation request pays NONE of the overhead a
 * normal session request pays: compaction.js builds it as one system line plus one user message, so
 * the system prompt and the tool definitions (about 24,755 tokens with the MCP servers on this machine) are
 * not in it. Counting them here cost a slot on 2026-09-22 for no reason.
 */
export function summariserInputTokens(preparation: any, extraInstructionChars = 0): number {
	return Math.ceil(((summariserText(preparation).length + extraInstructionChars) * TOKEN_HEAVINESS) / 4) + SUMMARISER_FRAMING_TOKENS;
}

/** The size of the SEGMENT ALONE, with no framing in it (2026-10-17). {SEGMENT SIZE} used to be filled
 *  from summariserInputTokens, which adds SUMMARISER_FRAMING_TOKENS - so the model was told the material
 *  below was worth 1,500 tokens more than it is. It plans its length against the figure it is given, so
 *  the figure it is given has to be the material. Both are heuristic (chars x TOKEN_HEAVINESS / 4); the
 *  server's own count is the truth, and switching this to that count is the open follow-up. */
export function summariserSegmentTokens(preparation: any): number {
	return Math.ceil((summariserText(preparation).length * TOKEN_HEAVINESS) / 4);
}

/** Allowance for the fixed part of a request when the live projection cannot be read. */
const FIXED_REQUEST_TOKENS = 4096;

/**
 * How long the finished summary may be before the next ordinary request to the session model no longer
 * fits. After a compaction the session holds three things: the fixed overhead (system prompt plus tool
 * definitions, measured at 24,755 tokens here), the recent tail pi keeps word for word
 * (keepRecentTokens) and the new summary. Staying inside window minus reserveTokens is what stops pi
 * compacting again on the next turn.
 */
function postCompactionHeadroom(ctx: Ctx, preparation: any): number {
	const window = ctx?.model?.contextWindow ?? 0;
	if (window <= 0) return 0;
	// THE UNLINK: the extension's own scaled values, not pi's preparation settings.
	const keep = extKeepRecent(ctx);
	const reserve = extReserve(ctx);
	return Math.max(0, window - keep - reserve - fixedRequestTokens(ctx));
}

/** Output tokens per visible word, measured on this session: Gemini 3.5 Flash Lite produced 1,106
 *  visible words from 2,212 visible output tokens. Two is the honest number for a model that is not
 *  thinking, and it is the number the token targets are divided by when the prompt speaks in words. */
const TOKENS_PER_WORD = 1.33;

function wordsFor(tokens: number): number {
	return Math.round(tokens / TOKENS_PER_WORD);
}

/** The base URL of a local summariser ref (`llama-server=http://127.0.0.1:9931/<model>`), or the
 *  model's own baseUrl when the ref carries none. Undefined for an online ref. */
export function localSummariserBaseUrl(model: any, ref: string): string | undefined {
	if (!refLooksLocal(ref)) return undefined;
	const eq = ref.indexOf("=");
	const tail = eq >= 0 ? ref.slice(eq + 1) : "";
	if (tail.startsWith("http://") || tail.startsWith("https://")) {
		// The origin, scheme included. An earlier version of this helper returned "127.0.0.1:9931" with
		// the scheme stripped off, which made every fetch an invalid URL that threw immediately - so
		// server-side counting silently never ran, and every figure in the log came from the heuristic.
		try {
			return new URL(tail).origin;
		} catch {
			return undefined;
		}
	}
	const base = String(model?.baseUrl ?? "").replace(/\/+$/, "");
	return base.startsWith("http") ? base : undefined;
}

/** What this base URL answers, remembered so a server without the route is not probed every time. A
 *  missing route is remembered for an hour (it is not a transient fault, and retrying it every
 *  compaction would put a timeout in front of every request); a 5xx or a timeout for a minute. */
const countCapability = new Map<string, { how: "messages" | "tokenize" | "no"; at: number }>();
function countRoute(base: string): "messages" | "tokenize" | "no" {
	const hit = countCapability.get(base.replace(/\/+$/, ""));
	if (!hit) return "messages";
	const ttl = hit.how === "no" ? 3_600_000 : 60_000;
	return Date.now() - hit.at > ttl ? "messages" : hit.how;
}

/** The exact number of tokens this server would read for this text, or null when it cannot be trusted.
 *
 *  - First POST /v1/messages/count_tokens (Anthropic-shaped): llama.cpp has it, Strata has it, and it
 *    renders the chat template before counting, so it answers "what will the model read". Measured here:
 *    65 ms for 170,000 characters and 56,495 tokens where the heuristic said 55,250.
 *  - Then POST /tokenize, which every llama.cpp-derived server has had always and which ik_llama.cpp
 *    implements without the route above. It counts the raw text with no template, which is precisely
 *    what {SEGMENT SIZE} means: the material, nothing else.
 *  An answer is accepted only when it is finite, positive and within a factor of two of the character
 *  estimate. A server that returns 0, or counts something else entirely, must not be allowed to drive
 *  what the model is told to aim at: a wrong count is worse than an honest guess, so anything odd means
 *  the estimate is used and the record says which ruler was in force. */
export async function countTextAt(base: string, text: string, estimate: number, ms = 6_000): Promise<{ n: number; how: "messages" | "tokenize" } | null> {
	// A server that already failed both routes is not asked again for an hour: an engine with neither
	// route would otherwise pay two timeouts on every single compaction, which is slower than the guess.
	if (countRoute(base) === "no") return null;
	const sane = (n: any): boolean => {
		const v = Number(n);
		return Number.isFinite(v) && v > 0 && v >= estimate * 0.5 && v <= Math.max(estimate * 2, 4096);
	};
	const url = base.replace(/\/+$/, "");
	const route = countRoute(base);
	const post = async (path: string, body: any, extra: Record<string, string>) => {
		const r = await fetch(url + path, { method: "POST", headers: { "content-type": "application/json", ...extra }, body: JSON.stringify(body), signal: AbortSignal.timeout(ms) });
		if (r.status === 404 || r.status === 405 || r.status === 501) return "missing" as const;
		if (!r.ok) return "bad" as const;
		return ((await r.json().catch(() => null)) ?? null) as any;
	};
	if (route !== "tokenize") {
		try {
			const out = await post("/v1/messages/count_tokens", { model: "count", max_tokens: 1, messages: [{ role: "user", content: text }] }, { "anthropic-version": "2023-06-01" });
			if (out === "missing") countCapability.set(url, { how: "tokenize", at: Date.now() });
			else if (out === "bad") countCapability.set(url, { how: route === "messages" ? "messages" : route, at: Date.now() });
			else if (sane(out?.input_tokens)) {
				countCapability.set(url, { how: "messages", at: Date.now() });
				return { n: Number(out.input_tokens), how: "messages" };
			} else countCapability.set(url, { how: "tokenize", at: Date.now() });
		} catch {
			countCapability.set(url, { how: "tokenize", at: Date.now() });
		}
	}
	try {
		const out = await post("/tokenize", { text, content: text, add_special: false, parse_special: true }, {});
		if (out === "missing" || out === "bad") {
			countCapability.set(url, { how: "no", at: Date.now() });
			return null;
		}
		// llama.cpp-family servers answer with a "tokens" array; ik_llama.cpp's documents the request field
		// as "content" (we send both names, so either answers) and some forks answer only with the variant
		// array that includes special tokens, which counts one or two more and is near enough for a ruler.
		const arr = Array.isArray(out?.tokens) ? out.tokens : Array.isArray(out?.tokens_with_spc) ? out.tokens_with_spc : Array.isArray(out?.tokens_spc) ? out.tokens_spc : Array.isArray(out?.tokens_with_bos) ? out.tokens_with_bos : undefined;
		const raw = arr !== undefined ? arr.length : out?.tokens !== undefined ? out.tokens : out?.count ?? out?.size;
		if (sane(raw)) {
			countCapability.set(url, { how: "tokenize", at: Date.now() });
			return { n: Number(raw), how: "tokenize" };
		}
		countCapability.set(url, { how: "no", at: Date.now() });
	} catch {
		countCapability.set(url, { how: "no", at: Date.now() });
	}
	return null;
}


/**
 * The size window one summarisation attempt is governed by, all in tokens.
 *
 * THE LIMITS (absolute tokens since 2026-10-05; before that they were percentages of the model's
 * WINDOW, which is what broke the 2026-10-05 session: 2% of a 1,048,575-token window is an
 * acceptance floor of 20,972 tokens, which no model on earth answers, so all ~100 attempts came back
 * "too thin" and every one of them re-uploaded the whole context). min and max are now absolute
 * tokens; the AIM (what the instructions request) is still summaryPercent% of the window, clamped
 * into [min, max]:
 *   - max    = min(summaryMaxTokens, the region itself) — a summary longer than the source is
 *             pointless; the generation request is capped a little above max (GEN_MARGIN) so a
 *             summary landing exactly on max is not killed by pi's length-stop;
 *   - min    = min(summaryMinTokens, max) — a thinner summary is thrown away.
 * Then trimmed down where reality demands it: an overflow compaction (pi refuses a summary longer
 * than the history it replaces) and the room actually left in the session window after compaction.
 */
type Sizes = { min: number; target: number; max: number; genCap: number; notes: string[] };

/** The tolerance a finished summary is judged against, and its ONLY job (2026-10-20, the operator's
 *  design, replacing a rule that used this same number for two things at once). A summary of up to
 *  Summary max + GEN_MARGIN is accepted; anything longer is thrown away and the next model is asked.
 *  The ask itself is Summary max exactly, so this is not room the model is invited to write into - it
 *  is slack for the two ways an honest reply lands over the line: an online provider has been seen to
 *  come back 300-400 tokens over an instructed ceiling, and our token count is never the server's
 *  token count, because the two tokenizer files disagree. A local llama-family server treats the
 *  ceiling as a hard stop, so on those this slack almost never gets used, which is the point rather
 *  than an accident.
 *
 *  It used to also be the slack on what the ENGINE IS TOLD to generate, which is what made the number
 *  meaningless to the operator: with max at 30,000 the log reported 30,512 back at them as if it were a
 *  measurement, and before that the scratch and thinking allowances were ADDED on top of both, so a
 *  model with a 30,000 aim was told to generate 34,608 and ran off the window. Nothing may push the
 *  instruction past Summary max, because a local engine asked for more than its window holds does not
 *  fail politely - it takes the session down as a crash. If the ask and the tolerance ever need to
 *  differ again, that is a second constant with its own name, not a second meaning smuggled into this
 *  one. Raised to 1000 once, during the emergency of 2026-10-19, and put back to 512 the next day. */
const GEN_MARGIN = 512;

/** What a scratchpad is worth reserving for when the template asks for one and the model can afford
 *  it (2026-10-18). Since the draft setting was deleted the only way to want a scratchpad is to write
 *  it into the template, so that is where the wish is looked for: the text the model will be shown.
 *  Reserve nothing and the scratchpad eats into the summary itself, so a run that means to fill the
 *  ceiling is cut off in the middle of the summary and the whole attempt is thrown away. Where the
 *  model's own output permission is known the allowance is capped by it, because asking a model to
 *  emit more than it can is not a longer scratchpad, it is a truncation with extra steps. */
const SCRATCH_ALLOWANCE = 4_096;
const SCRATCH_MIN_USEFUL = 1_024;
function asksForScratch(instructions: string): boolean {
	return /<draft[\s>]/i.test(instructions) || /<analysis[\s>]/i.test(instructions) || /\bscratchpad\b/i.test(instructions);
}
/** The scratchpad share of the output budget: the allowance below, shrunk to whatever a model can
 *  actually still emit above the summary ceiling and the thinking allowance, and written down in the log
 *  when that leaves too little to matter (2026-10-18). It deliberately reads nothing but the model and
 *  the numbers already computed, because the request text does not exist where this is needed - reading
 *  it there sat in a temporal dead zone and threw "Cannot access 'instructions' before initialization"
 *  on every compaction, which re-triggered on a context that could not be compacted. */
function scratchAllowanceFor(model: any, baseCap: number): { allowance: number; note: string } {
	const limit = Number(model?.maxTokens);
	if (!Number.isFinite(limit) || limit <= 0) return { allowance: SCRATCH_ALLOWANCE, note: `scratchpad allowed ${fmt(SCRATCH_ALLOWANCE)} tok; this model states no output limit` };
	const room = Math.max(0, Math.floor(limit) - baseCap - 256);
	if (room >= SCRATCH_ALLOWANCE) return { allowance: SCRATCH_ALLOWANCE, note: "" };
	if (room >= SCRATCH_MIN_USEFUL) return { allowance: room, note: `the scratchpad gets ${fmt(room)} of the ${fmt(SCRATCH_ALLOWANCE)} tok: this model emits at most ${fmt(limit)} tok and the ceiling, the margin and the thinking allowance already want ${fmt(baseCap)}` };
	return { allowance: room, note: `THE SCRATCHPAD HAS ALMOST NO ROOM: this model emits at most ${fmt(limit)} tok, everything before it already takes ${fmt(baseCap)}, leaving ${fmt(room)} for scratch work plus the summary. Raise the model output limit (pi owns it now), lower Summary max, or the run is cut off and the attempt discarded.` };
}


/** When a model THINKS before writing the summary, the reasoning tokens ride inside the same
 * generation cap. Without extra room the thinking eats the summary's budget and the answer is cut
 * mid-sentence (stop reason "length"). Sized for llama.cpp-style local servers; a too-small
 * allowance just fails the attempt, a bigger one only wastes nothing when unused. */
/** Output-token headroom a THINKING summarizer needs on top of the summary ceiling, so the
 *  reasoning cannot push the generation into a length-stop before the summary is finished. The
 *  reasoning rides inside the same max_tokens budget as the summary itself, and on huge compaction
 *  inputs (200k+) even effort "low" reasons for many thousands of tokens - glm-flash-latest CANNOT
 *  disable reasoning at all (OpenRouter rejects reasoning.enabled=false with "Reasoning is mandatory
 *  for this endpoint"; verified 2026-10-01), so the only cure is room. This is pure permission
 *  headroom: the acceptance window still judges the summary alone (reasoning is subtracted before
 *  measuring), so a chatty model is rejected as before. (Raised 2026-10-01 after glm/qwen compaction
 *  attempts with thinking on died at the cap with the summary half written.) */
const THINKING_ALLOWANCE: Record<string, number> = { minimal: 4_096, low: 8_192, medium: 12_288, high: 16_384, xhigh: 20_480, max: 20_480 };

/** Output-token headroom a scratchpad needs on top of the summary ceiling, so that a long one cannot
 *  push the summary itself into a length-stop. The scratchpad is stripped before measuring, but it
 *  rides inside the model's output, so the generation cap governs scratchpad and summary together. */

export function sizesFor(profile: ModelOptions, opts: { inputTokens: number; historyTokens: number; headroom: number; overflow: boolean; windowTokens?: number }, prevTokens = 0): Sizes {
	const notes: string[] = [];
	// The REGION: the material being folded (the session ctx minus the preserved tail, measured
	// after the stubbing pass) - kept only for the sanity cap and the unknown-window fallback.
	const region = Math.max(0, Math.round(opts.inputTokens));
	const window = Math.max(0, Math.round(opts.windowTokens ?? 0));
	// THE AIM AND THE LIMITS. The LIMITS are ABSOLUTE TOKENS (a model cannot write a 100k-token
	// summary however big its window is, which is why the 2026-10-03 percent-of-window limits were
	// dropped: 5% of a 1M window is not an acceptance floor, it is unreachable). The AIM is still a
	// percent of the window, clamped into [min, max]. A summary longer than the source is pointless,
	// so the max is capped by the region; the min never rises above the max.
	// THE LIMITS (2026-10-05, final): ABSOLUTE TOKENS, and every number below goes through tok(), so
	// a NaN can never reach the acceptance gates again (on 2026-10-05 the migrated fields were all
	// NaN: both gates compared false against NaN, every summary was accepted whatever its length, and
	// the requests went out with maxTokens=NaN).
	const maxTok = tok(profile.summaryMaxTokens, DEFAULT_OPTIONS.summaryMaxTokens);
	const minTok = tok(profile.summaryMinTokens, DEFAULT_OPTIONS.summaryMinTokens);
	const regionCeil = Math.max(tok(region), 1_024);
	let max = Math.max(1_024, Math.min(maxTok, regionCeil));
	if (maxTok > regionCeil) notes.push("max capped to the region itself (a summary longer than the source is pointless)");
	let min = Math.min(Math.max(128, minTok), max);
	const aimPct = pctField(profile.summaryPercent, DEFAULT_OPTIONS.summaryPercent);
	let target = clamp(Math.round((aimPct * (window > 0 ? window : regionCeil)) / 100), min, max);
	if (prevTokens > 0 && profile.chainMode === "attach") {
		// The verbatim previous summary rides INSIDE the summary: raise the ceiling by its size and
		// demand it in the minimum too - a lazy model cannot pass by returning only the previous
		// text, because the minimum still includes the region-derived aim.
		const prev = tok(prevTokens);
		max = tok(max) + prev;
		min = Math.min(max, min + prev);
		target = clamp(target, min, max);
		notes.push(`chain attach: +${fmt(prev)} tok for the verbatim previous summary`);
	}
	const hist = tok(opts.historyTokens);
	if (opts.overflow && hist > 0 && max > hist) {
		max = hist;
		notes.push("overflow compaction: pi refuses a summary longer than the history it replaces");
	}
	const head = tok(opts.headroom);
	if (head > 0 && max > head) {
		max = Math.max(1024, head - 256);
		notes.push("trimmed to the room left in the session window after compaction");
	}
	if (target > max) target = max;
	if (min > target) min = target;
	// min 0 = "no minimum": any real output is accepted (the empty-summary check still applies).
	if (min > 0 && min < 128) min = 128;
	// The ask IS the allowance. genCap is kept as its own field because the fallback path and the attempt
	// log both report it, and because what the engine is capped at and what an answer is judged against
	// are two different numbers - GEN_MARGIN's whole job is now the second one.
	return { min, target, max, genCap: max, notes };
}

/**
 * pi sizes the summary as min(floor(0.8 × reserveTokens), model.maxTokens) (compaction.js), so the
 * generation cap is turned back into a reserve value here: floor(0.8 × reserve) = genCap.
 */
export function withBudget(preparation: any, genCap: number): any {
	const reserve = clamp(Math.ceil(tok(genCap, DEFAULT_RESERVE_TOKENS * 2) / 0.8), 320, 250_000);
	return { ...preparation, settings: { ...preparation.settings, reserveTokens: reserve } };
}

/**
 * How big the delivered summary really is. The provider's own output count is the truth (reasoning
 * tokens are a subset of `output` in pi's Usage, so they are subtracted); without it the text is
 * estimated slightly above pi's chars/4, because bullet summaries with numbers and paths tokenize
 * denser than prose.
 */
/** The engine is the only counter that matters (the operator's rule, 2026-10-21). Where the server
 *  tells us how many tokens it wrote, that number is the answer. Where it tells us nothing, the word
 *  count is the next best thing: counting words is free, and one number per word beats one number per
 *  character, because the ratio between words and tokens is stable for a given tokenizer. So the ratio
 *  is LEARNED per model from the pairs we do get - every time an engine reports tokens for a text whose
 *  words we counted, that model's tokens-per-word is remembered for the next time it stays silent. */
const tokPerWord = new Map<string, number>();
const TOK_PER_WORD_DEFAULT = 1.35;

function tokPerWordFor(ref: string): number {
	return tokPerWord.get(String(ref ?? "")) ?? TOK_PER_WORD_DEFAULT;
}

/** Only trust a measurement over a real stretch of text, and only inside sane bounds: a 40-word answer
 *  or a ratio of six tokens per word says more about that one reply than about the tokenizer. */
function learnTokPerWord(ref: string, tokens: number, words: number): void {
	if (!ref || words < 200 || tokens <= 0) return;
	const r = tokens / words;
	if (r < 0.6 || r > 3) return;
	tokPerWord.set(String(ref), Math.round(r * 1000) / 1000);
}

/** The size shown on the COMPACTED row: the engine's own count of what the winning model wrote when we
 *  have it, the word ruler when we do not. A character guess is never allowed to sit beside the
 *  engine's number as a second answer to the same question. */
function landedSummarySize(info: any, text: string): number {
	if (info && typeof info.measured === "number" && info.measured > 0) return info.measured;
	return measuredTokens(text, null, info?.ref ?? "").tokens;
}

export function measuredTokens(
	summary: string,
	usage: any,
	ref = "",
): { tokens: number; by: "usage" | "words" | "chars" } {
	const words = countWords(summary);
	if (usage && typeof usage.output === "number" && usage.output > 0) {
		const reasoning = typeof usage.reasoning === "number" ? usage.reasoning : 0;
		const tokens = Math.max(0, usage.output - reasoning);
		learnTokPerWord(ref, tokens, words);
		return { tokens, by: "usage" };
	}
	if (words > 0) return { tokens: Math.ceil(words * tokPerWordFor(ref)), by: "words" };
	// No words at all - a summary of nothing but symbols, or an empty reply. The character ruler is the
	// last thing left standing, and it is a last resort, not a habit.
	return { tokens: Math.ceil((summary.length * 1.15) / 4), by: "estimate" };
}

/**
 * Remove the draft scratchpad (DRAFT_FULL / DRAFT_MINI, wrapped in <draft>...</draft>) from a
 * finished summary, so only the summary itself is stored and measured. An unclosed draft (the
 * model ran out of tokens mid-scratchpad) is cut at the first CURRENT STATE heading; when no
 * heading follows, the text is left alone rather than risk deleting the summary. Returns how many
 * characters the scratchpad occupied, so a usage-based token count can be corrected: the draft
 * rode inside the provider's output figure, but it must not count against the acceptance window.
 */
export function stripRoughWork(text: string): { summary: string; strippedChars: number } {
	let s = String(text ?? "");
	let stripped = 0;
	const open = s.indexOf("<draft>");
	if (open >= 0) {
		const close = s.indexOf("</draft>", open);
		if (close >= 0) {
			const end = close + "</draft>".length;
			stripped += end - open;
			s = s.slice(0, open) + s.slice(end);
		} else {
			const rest = s.slice(open);
			const m = rest.match(/\n##\s*CURRENT STATE/);
			if (m && typeof m.index === "number" && m.index > 0) {
				stripped += m.index;
				s = s.slice(0, open) + rest.slice(m.index);
			}
		}
	}
	s = s.replace(/<\/draft>/g, "").replace(/\n{3,}/g, "\n\n").trimStart();
	return { summary: s, strippedChars: stripped };
}

/**
 * Append the raw-transcript pointer line to an extension-written summary (Claude Code's trick:
 * "read the full transcript at: ..."). The next model can recover anything the summary dropped by
 * reading the session file; the ledger rules tell the model not to copy the line forward, and the
 * next compaction appends a fresh one. Best-effort: no session path from pi, no line.
 */
export function appendTranscriptPointer(cfg: Config, ctx: Ctx, summary: string): string {
	if (!cfg.transcriptPointer) return summary;
	try {
		const file: unknown = (ctx as any)?.sessionManager?.getSessionFile?.();
		if (typeof file === "string" && file && !summary.includes("Full raw transcript of the summarized history:")) {
			return `${summary}\n\nFull raw transcript of the summarized history: ${file} - read it only if this summary is missing something you need.`;
		}
	} catch {
		/* best effort */
	}
	return summary;
}

function countWords(text: string): number {
	return String(text).trim().split(/\s+/).filter(Boolean).length;
}

// ---------------------------------------------------------------- stubs (elision + compaction input)

/** The status line pi's shell tool appends to a failing/aborted/timed-out result. The LAST match is
 *  the final verdict; a stub must always carry it (it is the single most useful line in the tail). */
const LAST_STATUS_RE = /(?:^|\n)[^\n]*(?:Command exited with code \d+|Command timed out after [^\n]*|Command aborted|Command terminated without an exit code)[^\n]*$/;

/** Plain text of a result's content blocks (images contribute only a marker - the stub replaces
 *  them wholesale anyway). */
function contentTextOf(content: any): string {
	if (typeof content === "string") return content;
	let text = "";
	for (const b of content ?? []) {
		if (b?.type === "text") text += String(b.text ?? "");
		else if (b?.type === "image") text += "[image]";
	}
	return text;
}

/** Key arguments of a tool call, as a short summary for a stub ("src/api/client.ts" / "bun test"). */
function keyArgsOf(name: string, args: any): string {
	if (!args || typeof args !== "object") return "";
	for (const key of ["path", "command", "url", "query", "pattern", "file_path"]) {
		const v = args[key];
		if (typeof v === "string" && v) {
			const s = v.replace(/\s+/g, " ").trim();
			return ` ${s.length > 120 ? s.slice(0, 117) + "..." : s}`;
		}
	}
	return "";
}

/** Find the toolCall block a result answers, walking backwards from the result's position. */
function matchingToolCall(messages: any[], index: number, toolCallId: string | undefined, toolName: string): { name: string; args: any } | undefined {
	for (let i = index; i >= 0; i--) {
		const m = messages[i];
		if (m?.role !== "assistant" || !Array.isArray(m.content)) continue;
		for (const b of m.content) {
			if (b?.type === "toolCall" && (toolCallId === undefined || b.id === toolCallId)) {
				return { name: String(b.name ?? toolName), args: b.arguments };
			}
		}
	}
	return undefined;
}

/**
 * The informative stub that replaces an elided tool result - in live requests (the elision stage)
 * and in the summarizer's input alike. Shape: tool name, key argument, the output's size, the tail
 * excerpt (the exit/status line is ALWAYS included when one exists), and the recovery hint.
 */
export function buildResultStub(
	toolName: string,
	argsSummary: string,
	text: string,
	tailChars: CharCap,
): string {
	const clean = String(text ?? "");
	const lines = clean ? clean.split("\n").length : 0;
	const status = clean.match(LAST_STATUS_RE)?.[0]?.trim();
	let tail = "";
	if (tailChars === "full") {
		tail = clean.trim();
	} else if (typeof tailChars === "number" && tailChars > 0 && clean) {
		const n = Math.min(tailChars, clean.length);
		tail = (clean.length > n ? "... " : "") + clean.slice(clean.length - n).replace(/\s+/g, " ").trim();
	}
	if (status && !tail.includes(status)) {
		tail = tail ? `${status}  (${tail})` : status;
	}
	const parts = [
		`[${toolName}${argsSummary ? ` "${argsSummary.trim()}"` : ""}`,
		`output elided: ${fmt(lines)} lines / ${fmt(clean.length)} chars`,
	];
	if (tail) parts.push(`tail: "${tail}"`);
	parts.push(`re-run or re-read to see the full output`);
	return `${parts.join(" - ")}]`;
}

/** True when a message is a tool result (live transcript or prepared region alike). */
function isToolResult(m: any): boolean {
	return m?.role === "toolResult" && m?.toolCallId !== undefined;
}

/**
 * Replace ONE tool-result message with its stub, as a NEW message object (never mutating the
 * original - the preparation object is shared across the rotation's candidates).
 */
function stubbedResult(messages: any[], index: number, m: any, tailChars: CharCap): any {
	const call = matchingToolCall(messages, index - 1, m.toolCallId, String(m.toolName ?? "tool"));
	const name = String(m.toolName ?? call?.name ?? "tool");
	const stub = buildResultStub(name, keyArgsOf(name, call?.args), contentTextOf(m.content), tailChars);
	return { ...m, content: [{ type: "text", text: stub }] };
}

/**
 * Cap tool-call argument VALUES inside assistant messages (whole files ride inside write/edit
 * arguments and arrive uncapped in the summarizer input). Copy-on-write; strings only; non-string
 * values pass through.
 */
export function capToolCallArgs(messages: any[], cap: CharCap): { messages: any[]; capped: number } {
	if (cap === "full") return { messages, capped: 0 };
	let capped = 0;
	const out = messages.map((m) => {
		if (m?.role !== "assistant" || !Array.isArray(m.content)) return m;
		const content = m.content.map((b: any) => {
			if (b?.type !== "toolCall" || !b.arguments || typeof b.arguments !== "object") return b;
			let argsChanged = false;
			const args: Record<string, unknown> = {};
			for (const [k, v] of Object.entries(b.arguments)) {
				if (typeof v === "string" && v.length > cap) {
					args[k] = `${v.slice(0, cap)}...(+${fmt(v.length - cap)} chars)`;
					argsChanged = true;
				} else {
					args[k] = v;
				}
			}
			if (!argsChanged) return b;
			capped++;
			return { ...b, arguments: args };
		});
		if (content === m.content) return m;
		return { ...m, content };
	});
	return { messages: out, capped };
}

/**
 * The summarizer's input, per the model's own options: every tool result becomes an informative
 * stub (pi's 2,000-character clip then has nothing important left to cut - the stub is short by
 * design and carries the exit/tail message the head-only clip used to destroy), and tool-call
 * argument values get capped. Returns a NEW preparation; the caller's object is untouched.
 */
export function preprocessForSummariser(prep: any, options: { inputStubs: boolean; argCap: CharCap; stubTailChars?: CharCap }): { prep: any; stubbed: number; argsCapped: number; rawChars: number; keptChars: number; savedChars: number } {
	const tail = options.stubTailChars ?? ELISION_DEFAULTS.stubTailChars;
	let stubbed = 0;
	let argsCapped = 0;
	// How much material the elision actually removed, so the log can say where the tokens went. Before
	// this, a compaction could read 49,151 tokens out of a 164,239-token context and the log had no way
	// to account for the difference beyond a count of how many results were touched.
	let rawChars = 0;
	let keptChars = 0;
	const charsOf = (list: any[]) => list.reduce((n: number, m: any) => n + contentTextOf(m?.content).length, 0);
	const out: any = { ...prep };
	for (const key of ["messagesToSummarize", "turnPrefixMessages"]) {
		const list: any[] = prep?.[key] ?? [];
		rawChars += charsOf(list);
		let working = list;
		if (options.inputStubs) {
			working = working.map((m, i) => {
				if (!isToolResult(m)) return m;
				// Results at or under pi's 2,000-char clip threshold arrive complete (exit line at the
				// end included) - their original is strictly better than a stub. Stub only what the
				// clip would actually damage.
				if (contentTextOf(m.content).length <= 2_000) return m;
				return stubbedResult(working, i, m, tail);
			});
		}
		const cappedRes = capToolCallArgs(working, options.argCap);
		working = cappedRes.messages;
		argsCapped += cappedRes.capped;
		if (options.inputStubs) {
			stubbed += working.filter((m: any, i: number) => isToolResult(m) && m !== list[i]).length;
		}
		out[key] = working;
		keptChars += charsOf(working);
	}
	return { prep: out, stubbed, argsCapped, rawChars, keptChars, savedChars: Math.max(0, rawChars - keptChars) };
}

/** What the elision removed from the material, in the same tokens the material is measured in. The
 *  elision counts characters and the ruler in force counted the material AFTER the cut, so the saving
 *  is scaled through that ratio instead of being guessed from a constant tokens-per-character. */
function elidedTokens(pre: { savedChars?: number; keptChars?: number }, material: number): number {
	const saved = Number(pre?.savedChars);
	const kept = Number(pre?.keptChars);
	if (!Number.isFinite(saved) || !Number.isFinite(kept) || kept <= 0 || material <= 0) return 0;
	return Math.max(0, Math.round((material * saved) / kept));
}

// ---------------------------------------------------------------- the elision stage

/**
 * ELISION - the rule-based first stage of context management (the harness paper's headline finding:
 * stage cheap elision before LLM summarization). In every live request, tool results outside a
 * protected recent window are replaced by informative stubs. The session file on disk keeps every
 * original; the tool CALLS stay visible; the model re-runs or re-reads when it needs an old output.
 *
 * Implementation: a `context` handler - it fires before EVERY LLM call with the full transcript,
 * and the handler's returned message list is what actually gets sent. Stored history is never
 * touched, so pi's compaction machinery (and the summarizer, which preprocesses its own input)
 * always sees the raw originals.
 *
 * Trigger and batching: the context is measured on the incoming (unstubbed) messages; when it
 * reaches `softPercent` of the model's window, every result outside the protected window becomes
 * eligible, oldest first. A NEW batch of stubs is applied only when it saves at least
 * `minSavingsTokens` - prompt caches charge a re-prefill whenever a prefix changes, so small
 * changes wait and one batch amortizes across many results. Between batches the applied set is
 * stable, so consecutive requests keep the same shape and the cache stays warm. A batch also needs
 * at least `minResultsToStub` results waiting - the per-fire prefill cost is roughly fixed (the
 * whole tail after the break point re-prefills), so batching across many results is what makes each
 * fire pay off. Batches also stop firing once the live request comes within `stopGapTokens` of pi's
 * compaction trigger - a batch there would be wasted work, since compaction resets everything
 * moments later.
 */
const elisionStateBySession = new Map<string, { applied: Map<string, { stub: string; savedTokens: number }> }>();

/** One state file per session, named with the session id - the same uuid pi uses in its own session
 *  file name (<ISO-time>_<uuid>.jsonl), so a session's stub file is findable by eye: the elision
 *  folder holds Elision-style per-session files, just keyed by the uuid part. Every file in here is
 *  read back into the one shared tool-call-id map, so a resume or a fork finds its stubs either way. */
/** The elision state persists per session (the stubbed tool-call ids + their stubs): a pi restart
 *  used to wipe it, so the same batch re-fired with identical numbers after every resume (the
 *  repeating log lines). Stored under ~/.pi/agent/pi-compact-plus-state/<sessionId>.json; files
 *  older than 3 days are ignored (a stale session's ids are stale). */
const ELISION_STATE_DIR = join(CONFIG_PATH, "..", "pi-compact-plus-state");

function elisionStatePath(sessionId: string): string {
	const safe = sessionId.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 80) || "session";
	return join(ELISION_STATE_DIR, `${safe}.json`);
}

/** ONE stub store, keyed by TOOL-CALL id (unique per call) and seeded from EVERY state file in the
 *  directory. THE 2026-10-05 BUG THIS FIXES: the store used to be keyed by SESSION id, and a file
 *  older than three days was dropped silently - so a resumed or forked session, which gets a NEW
 *  session id, started with an empty store, every already-stubbed result became a fresh candidate
 *  again, and the live wire flipped back to the raw originals and had to be re-prefilled in full.
 *  Tool-call ids survive a resume; a session id does not. */
function loadElisionState(sessionId: string): { applied: Map<string, { stub: string; savedTokens: number }> } {
	const cached = elisionStateBySession.get(sessionId);
	if (cached) return cached;
	const applied = new Map<string, { stub: string; savedTokens: number }>();
	let files: string[] = [];
	try {
		mkdirSync(ELISION_STATE_DIR, { recursive: true });
		files = readdirSync(ELISION_STATE_DIR).filter((f) => f.endsWith(".json"));
	} catch {
		/* no state dir yet - fresh session */
	}
	for (const f of files) {
		let raw: any;
		try {
			raw = JSON.parse(readFileSync(join(ELISION_STATE_DIR, f), "utf8"));
		} catch {
			continue; // an unreadable state file is not worth losing the others over
		}
		for (const [key, entry] of Object.entries(raw?.applied ?? {})) {
			const e: any = entry;
			// Stubs never age out: a session opened after a week or a month must find its stubs exactly
			// where it left them. The `at` stamp is kept for the log only.
			if (typeof e?.stub === "string" && typeof e?.savedTokens === "number") applied.set(key, { stub: e.stub, savedTokens: e.savedTokens });
		}
	}
	const state = { applied };
	elisionStateBySession.set(sessionId, state);
	return state;
}

function saveElisionState(sessionId: string, state: { applied: Map<string, { stub: string; savedTokens: number }> }): void {
	try {
		mkdirSync(ELISION_STATE_DIR, { recursive: true });
		const obj: Record<string, { stub: string; savedTokens: number; at: number }> = {};
		const at = Date.now();
		for (const [k, v] of state.applied) obj[k] = { stub: v.stub, savedTokens: v.savedTokens, at };
		writeFileSync(elisionStatePath(sessionId), JSON.stringify({ applied: obj }, null, 0), "utf8");
	} catch {
		/* state persistence is best-effort */
	}
}

/** When the last batch fired per session - the throttle that keeps elision from re-writing the
 *  prompt-cache break point every few turns (see ElisionConfig.minBatchGapSec). */
const lastBatchAtMs = new Map<string, number>();

/** THE ELISION CIRCUIT BREAKER (2026-10-05). The elision handler runs on EVERY live request, so a
 *  handler that throws would break every single request of the session, and a handler that fires a
 *  batch on every request re-uploads the whole context each time. One failure: a detailed log line,
 *  an ALL-CAPS notice on screen, and one more attempt allowed 3 minutes later. Two failures: the
 *  elision stage switches itself OFF for the session and says so on screen - it is a context
 *  helper, not something allowed to eat the session. `elision on` (or a /reload) switches it back. */
const elisionFailure = { fails: 0, off: false, retryAtMs: 0 };

/** Clear the breaker (used when the user switches elision back on). */
export function resetElisionBreaker(): void {
	elisionFailure.fails = 0;
	elisionFailure.off = false;
	elisionFailure.retryAtMs = 0;
}

/** What the SERVER holds is what matters: the last real usage report this session got back
 *  (prompt tokens including the cached prefix, plus what the model generated). pi's own number is
 *  an estimate over the RAW transcript, so on an elided session it reads ~100k higher than the
 *  request that actually goes out - which is how a 60 % trigger line came to arm at a completely
 *  different size than the one on the status bar. */
/**
 * THE context number for this extension, in one place: what the model server currently holds.
 * Order of preference, first non-zero wins:
 *   wire   - the prompt tokens the server reported for the newest request after the last compaction.
 *            Elisions are in that figure, so it is the real fill of the model's context.
 *   pi     - pi's own estimate from the same usage figure.
 *   sum    - what the NEXT request would carry, for a model that answers without usage figures, right
 *            after a compaction, or straight after a restart. Taken from pi itself: the entries pi would
 *            assemble (newest summary + the kept tail + everything added since), NOT the whole archive.
 * `all` is the whole branch next to it - what is on disk rather than what is on the wire, which on an
 * elided session is about twice as big - and `justCompacted` says the newest of those entries is the
 * summary itself: the one state where no measure exists yet. `how` says which was used, so a zero can
 * never be mistaken for an empty context.
 */
/** Tokens one session entry adds to the context. A summary rides on the entry's own text (pi keeps it in
 *  `summary`, not in a message); pass countSummary=false to leave summaries out of a total. */
function tokensOfEntry(e: any, countSummary = true): number {
	try {
		if ((e as any)?.type === "compaction") return countSummary ? Math.round((((e as any).summary ?? "").length * 1.15) / 4) : 0;
		const m: any = (e as any)?.message ?? e;
		if (!m || m.role === "system") return 0;
		return Math.round(estimateTokens(m as never));
	} catch {
		return 0;   // an entry this cannot read adds nothing rather than breaking the measure
	}
}

function measureContext(ctx: any): { wire: number; pi: number; sum: number; all: number; tokens: number; how: string; justCompacted: boolean } {
	let wire = 0;
	let piEst = 0;
	let sum = 0;
	let all = 0;
	let keptSlice = false;
	let nothingSinceCompaction = false;
	try {
		wire = wireTokensOf(ctx);
		const u = (ctx as any)?.getContextUsage?.();
		piEst = typeof u?.tokens === 'number' && u.tokens > 0 ? Math.round(u.tokens) : 0;
	} catch {
		/* measured as zero, the caller is told how it was measured */
	}
	if (!(wire > 0) && !(piEst > 0)) {
		try {
			const sm: any = (ctx as any)?.sessionManager;
			// WHAT WOULD BE SENT, TAKEN FROM PI RATHER THAN GUESSED: buildContextEntries() is how pi
			// assembles a context - the newest summary first, then the entries from its firstKeptEntryId,
			// then everything added since. The kept tail sits PHYSICALLY BEFORE the summary in the file, so
			// "from the newest compaction onward" is not it, and summing the whole branch is the archive:
			// roughly twice what the model is actually shown, which is what set this trigger off a second
			// after a compaction had landed and left the user with "Nothing to compact (session too small)".
			let slice: any[] = [];
			try {
				slice = sm?.buildContextEntries?.() ?? [];
			} catch {
				slice = [];
			}
			const branch: any[] = sm?.getBranch?.() ?? [];
			for (const e of branch) all += tokensOfEntry(e, false);
			if (slice.length) {
				// pi's own warning applies here: the kept range can swallow an OLDER summary, and only the
				// newest one is actually sent - count that one and no other, or the figure double-counts.
				let newestId: any = null;
				for (let i = branch.length - 1; i >= 0; i--) if ((branch[i] as any)?.type === "compaction") { newestId = (branch[i] as any).id; break; }
				keptSlice = newestId !== null;
				for (const e of slice) {
					if ((e as any)?.type === "compaction" && (e as any).id !== newestId) continue;
					sum += tokensOfEntry(e);
				}
				// THE STATE WITH NOTHING TO MEASURE: a summary is in what would be sent and not one answer has
				// come after it - the server has not been asked anything since it landed. Only a real number can
				// justify starting another compaction here (see the watcher's stand-down).
			nothingSinceCompaction = keptSlice && wire === 0;
			} else {
				// No pi to ask (older build, or the call refused): the archive is the only figure left.
				sum = all;
			}
		} catch {
			/* nothing to sum */
		}
	}
	const how = wire > 0 ? 'wire' : piEst > 0 ? 'pi-estimate' : sum > 0 ? (keptSlice ? 'kept-slice' : 'message-sum') : 'unmeasurable';
	const tokens = wire > 0 ? wire : piEst > 0 ? piEst : sum;
	return { wire, pi: piEst, sum, all, tokens, how, justCompacted: nothingSinceCompaction };
}

function wireTokensOf(ctx: any): number {
	try {
		const branch = ctx?.sessionManager?.getBranch?.() ?? [];
		let latestCompaction = -1;
		for (let i = 0; i < branch.length; i++) if (branch[i]?.type === "compaction") latestCompaction = i;
		let best = 0;
		for (let i = branch.length - 1; i > latestCompaction; i--) {
			const m: any = branch[i]?.role === undefined ? branch[i]?.message : branch[i];
			// PI'S OWN RULE, COPIED ON PURPOSE (2026-10-15). A response that was aborted or errored carries the
			// usage of the request that was cut short, which describes the context as it was BEFORE the summary
			// that is now in front of it. pi excludes those messages from its accounting for exactly this reason
			// ("kept pre-compaction messages have stale usage reflecting the old, larger context and would falsely
			// trigger compaction right after one just finished"), and since 2026-10-15 a compaction can be started
			// mid-run - where the abort it issues leaves precisely such a message behind. Reading it as the live
			// fill would have made every mid-run compaction immediately re-arm the trigger with the old, bigger
			// number: the re-trigger loop this file exists to have killed.
			const sr = String(m?.stopReason ?? '');
			if (sr === 'aborted' || sr === 'error') continue;
			const u: any = m?.usage;
			if (!u) continue;
			const total = Number(u.totalTokens) > 0 ? Number(u.totalTokens) : [u.input, u.output, u.cacheRead, u.cacheWrite].reduce((acc: number, v: any) => acc + (Number(v) || 0), 0);
			if (total > 0) { best = total; break; }
		}
		return best;
	} catch {
		return 0;
	}
}

export function applyElision(messages: any[], ctx: Ctx, opts?: { force?: boolean }): { messages: any[] } | undefined {
	const cfg = loadConfig();
	const el = cfg.elision;
	const sessionId = String(ctx?.sessionManager?.getSessionId?.() ?? "session");
	let state = elisionStateBySession.get(sessionId);
	// THE MASTER SWITCH (2026-10-05): "Extension active: OFF" used to switch off only the compaction
	// side - elision kept firing on its own switch, and every batch moves the prompt-cache break point,
	// i.e. re-uploads the whole context to the server (the 2026-10-05 night: batches at 717k tokens on
	// a 1M session, one after another). OFF now means OFF for both, and any stubs already applied are
	// dropped so the wire goes back to the raw originals straight away.
	if (!el.enabled) { // elision has its own switch since 2026-10-05 - independent of the compaction switch
		if (state?.applied.size) {
			state.applied.clear();
			saveElisionState(sessionId, state);
			logLine(cfg, { event: "elision_off", sid: String(sessionId).slice(-8), note: "elision cleared every applied stub (extension/elision switched off)" });
		}
		return undefined;
	}
	const window = ctx?.model?.contextWindow ?? 0;
	if (!(window > 0) || !Array.isArray(messages) || messages.length === 0) return undefined;


	// Context size of the INCOMING messages (always unstubbed - pi rebuilds them from storage).
	let effTokens = 0;
	for (const m of messages) effTokens += msgTokens(m);
	// pi's own usage-based number (what the ctx display shows) counts the system prompt, the tool
	// schemas and the overhead the message sum misses - trust the bigger of the two, or the start
	// line disagrees with the number the user watches (the 2026-10-03 no-elision bug: pi showed 98k
	// while the message sum said 70k, so a 50% line at 70k never armed).
	const usageTokens = Math.round((ctx as any)?.getContextUsage?.()?.tokens ?? 0);
	if (usageTokens > effTokens) effTokens = usageTokens;
	// What the server actually holds right now (elided), for the log line next to the raw number.
	const wireTokens = wireTokensOf(ctx);
	const softTokens = Math.floor((el.softPercent / 100) * window);

	// The protected recent window: walk from the end until protectRecentTokens is covered; nothing
	// inside it is ever stubbed (the working model keeps its freshest results in full).
	const protectedFrom = (() => {
		let t = 0;
		for (let i = messages.length - 1; i >= 0; i--) {
			t += msgTokens(messages[i]);
			if (t >= el.protectRecentTokens) return i;
		}
		return 0;
	})();

	if (!state) {
		state = loadElisionState(sessionId);
		elisionStateBySession.set(sessionId, state);
	}
	// Prune applied stubs whose result no longer exists (after a compaction or /clear).
	const present = new Set<string>();
	for (const m of messages) if (isToolResult(m)) present.add(String(m.toolCallId));
	for (const key of [...state.applied.keys()]) if (!present.has(key)) state.applied.delete(key);

	// Eligible = tool results before the protected window, oldest first, not yet applied, big
	// enough for a stub to pay for itself (the stub itself costs ~60-100 tokens).
	// THE BREAK POINT (2026-10-05: this is what made every batch re-upload the WHOLE context). The
	// prompt cache stays valid up to the first message whose content changes, so a batch that stubs the
	// OLDEST results changes the beginning of the context and forces a full re-prefill: all night on a
	// 700k session every batch cost a ~700k-token prefill instead of the tail it saves. Batches now
	// fill in NEWEST-FIRST (the block just behind the protected window) and never touch anything at or
	// before the oldest stub already on the wire, so a batch re-prefills only the tail after it.
	// PASS 1 - where is the cache break point already? It sits at the OLDEST stub on the wire. The
	// server's prompt cache is valid only up to the first message whose content differs from the last
	// request, so a batch may only change things AFTER it. (The first version of this fix looked for
	// the frontier in the same loop that collected the candidates, so it was found at index 53 while
	// indexes 2..52 had already been collected - and the next batch broke the cache at index 2 again.)
	let frontier = -1; // index of the oldest stub already on the wire, -1 = none yet
	for (let i = 0; i < messages.length && i < protectedFrom; i++) {
		if (isToolResult(messages[i]) && state.applied.has(String(messages[i].toolCallId))) { frontier = i; break; }
	}
	// PASS 2 - candidates, AFTER the break point only. Results older than the frontier are left as
	// raw originals on purpose: stubbing one of those would move the break point to the start of the
	// session and cost a full re-prefill to save a few thousand tokens. They are counted and logged.
	let behindFrontier = 0;
	const candidates: Array<{ index: number; key: string; stub: string; savedTokens: number }> = [];
	if (opts?.force || effTokens >= softTokens) {
		for (let i = 0; i < messages.length && i < protectedFrom; i++) {
			const m = messages[i];
			if (!isToolResult(m)) continue;
			const key = String(m.toolCallId);
			if (state.applied.has(key)) continue; // already stubbed - frozen
			if (frontier >= 0 && i < frontier) { behindFrontier++; continue; } // behind the break point: leave it raw
			const text = contentTextOf(m.content);
			if (el.stubTailChars !== "full" && text.length < el.stubTailChars + 600) continue; // stub would save nothing
			const st = stubbedResult(messages, i, m, el.stubTailChars);
			const saved = msgTokens(m) - msgTokens(st);
			if (saved <= 0) continue;
			candidates.push({ index: i, key, stub: st.content[0].text, savedTokens: saved });
		}
	}

	// Batch check - a batch fires only when ALL three hold:
	//   1. enough results waiting (minResultsToStub) - the per-fire prefill cost is roughly fixed
	//      (the whole tail after the break point re-prefills), so batching is what pays;
	//   2. enough savings (minSavingsTokens) - the wire must shrink meaningfully;
	//   3. the batch actually brings the live request below the stop line (a GAP below pi's
	//      compaction trigger, window - reserveTokens) - a batch that leaves the wire at the
	//      compaction doorstep is wasted churn: compaction resets everything moments later and
	//      stubs everything unconditionally anyway. When a context leap jumps past the stop line,
	//      a big enough batch can still rescue the request WITHOUT compaction (the paper's 0%
	//      managed-overflow case).
	// Of everything eligible (all of it AFTER the break point), stub the block closest to the END of the
	// context, up to maxResultsPerBatch of them; older ones stay raw on purpose, see above.
	const eligible = candidates.filter((c) => !state!.applied.has(c.key));
	const newCandidates = eligible.length > el.maxResultsPerBatch ? eligible.slice(eligible.length - el.maxResultsPerBatch) : eligible;
	const newSavings = newCandidates.reduce((sum, c) => sum + c.savedTokens, 0);
	// The batch throttle: one break-point move per gap. Between batches the applied set is stable,
	// so the wire keeps its shape and the server-side prompt cache stays warm.
	const nowMs = Date.now();
	const sinceLast = nowMs - (lastBatchAtMs.get(sessionId) ?? 0);
	const throttled = !opts?.force && lastBatchAtMs.has(sessionId) && sinceLast < Math.max(0, el.minBatchGapSec) * 1000;
	const compactionPoint = Math.max(0, window - extReserve(ctx));
	const stopTokens = Math.max(softTokens, compactionPoint - el.stopGapTokens);
	const appliedSaved = [...state!.applied.values()].reduce((s, e) => s + e.savedTokens, 0);
	const postBatchWire = effTokens - appliedSaved - newSavings;
	if (opts?.force && newCandidates.length === 0) return undefined; // nothing new to elide
	if (opts?.force || (!throttled && newCandidates.length >= el.minResultsToStub && newSavings >= el.minSavingsTokens && postBatchWire < stopTokens)) {
		for (const c of newCandidates) state!.applied.set(c.key, { stub: c.stub, savedTokens: c.savedTokens });
		lastBatchAtMs.set(sessionId, nowMs);
		saveElisionState(sessionId, state!);
		const resultsTok = newCandidates.reduce((sum, c) => sum + Math.round(msgTokens(messages[c.index])), 0);
		const stubTok = resultsTok - newSavings;
		// Everything the trigger arithmetic is made of, in one line: the trigger line (softPercent of
		// the window) and the stop line (pi's compaction point minus the stop gap) next to the actual
		// size, so "why did elision fire at 81k on a 135k window" is answerable from the log alone.
		logLine(cfg, { event: "elision", sid: String(sessionId).slice(-8), applied: newCandidates.length, eligible: eligible.length, deferred: eligible.length - newCandidates.length, oldestStubOnWire: frontier, breakPointIndex: newCandidates.length ? newCandidates[0].index : frontier, behindFrontier, resultsTokens: resultsTok, stubTokens: stubTok, savedTokens: newSavings, effTokens, postBatchWire, window, softPercent: el.softPercent, softTokens, stopTokens, rawTokens: effTokens, wireTokens, force: opts?.force === true });
		let totalResults = state!.applied.size;
		let totalSaved = 0;
		for (const [, entry] of state!.applied) totalSaved += entry.savedTokens;
		ctx?.ui?.notify?.(`Elision: ${newCandidates.length} new tool call${newCandidates.length === 1 ? "" : "s"} stubbed, ${fmt(newSavings)} tok saved. Total: ${fmt(totalResults)} tool results stubbed, ${fmt(totalSaved)} tok saved. Trigger line ${fmt(softTokens)} tok (${el.softPercent}% of a ${fmt(window)} tok window), stop line ${fmt(stopTokens)} tok.`, "info");
	} else if (throttled && newCandidates.length) {
		// Visible, once per gap: the batch is waiting, not lost.
		logLine(cfg, { event: "elision_hold", sid: String(sessionId).slice(-8), waiting: newCandidates.length, waitingTokens: newSavings, sinceLastMs: sinceLast, gapMs: Math.max(0, el.minBatchGapSec) * 1000, note: "batch held back by the elision batch gap" });
	}

	if (state.applied.size === 0) return undefined;
	let changed = false;
	const out = messages.map((m) => {
		if (!isToolResult(m)) return m;
		const entry = state!.applied.get(String(m.toolCallId));
		if (!entry) return m;
		changed = true;
		return { ...m, content: [{ type: "text", text: entry.stub }] };
	});
	return changed ? { messages: out } : undefined;
}

// ---------------------------------------------------------------- per-model sampling flags

/** The sampling flags a compaction request may carry for its model. null = ignore (send nothing). */
export interface SamplingFlags {
	temperature: number | null;
	top_p: number | null;
	top_k: number | null;
	min_p: number | null;
	presence_penalty: number | null;
	repetition_penalty: number | null;
}

export const SAMPLING_KEYS = ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"] as const;
export type SamplingKey = (typeof SAMPLING_KEYS)[number];

/** Every flag ignored - the default for every model. */
export const SAMPLING_DEFAULTS: SamplingFlags = { temperature: null, top_p: null, top_k: null, min_p: null, presence_penalty: null, repetition_penalty: null };

/** Sanity windows; out-of-range values snap to the nearest edge. */
const SAMPLING_RANGES: Record<SamplingKey, [number, number]> = {
	temperature: [0, 2],
	top_p: [0, 1],
	top_k: [0, 200],
	min_p: [0, 1],
	presence_penalty: [-2, 2],
	repetition_penalty: [0, 2],
};

/** Qwen3.8-Flash-Next's recommended INSTRUCT (non-thinking) values (unsloth.ai/docs/models/qwen3.8-next). */
export const QWEN_INSTRUCT_SAMPLING: SamplingFlags = { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 1.5, repetition_penalty: 1.0 };

export function sanitizeSampling(raw: any): SamplingFlags {
	const out = { ...SAMPLING_DEFAULTS };
	if (!raw || typeof raw !== "object") return out;
	for (const k of SAMPLING_KEYS) {
		const v = raw[k];
		if (typeof v !== "number" || !Number.isFinite(v)) continue;
		const range = SAMPLING_RANGES[k];
		out[k] = clamp(v, range[0], range[1]);
	}
	return out;
}

/** The request-body sampling params for a compaction call: every flag with a value becomes a key;
 *  ignored flags are left out. On llama-server providers repetition_penalty is spelled the way the
 *  server actually reads it (repeat_penalty); the rest pass under their own names. */
export function samplingParamsFor(ref: string, s: SamplingFlags | undefined): Record<string, unknown> | undefined {
	if (!s) return undefined;
	const local = refLooksLocal(ref);
	const out: Record<string, unknown> = {};
	for (const k of SAMPLING_KEYS) {
		const v = s[k];
		if (typeof v !== "number") continue;
		out[k === "repetition_penalty" && local ? "repeat_penalty" : k] = v;
	}
	return Object.keys(out).length ? out : undefined;
}

/** Per-request kwargs that make the THINKING EFFORT real for a local server: pi-ai sends nothing
 *  thinking-related for a plain openai-format model (its reasoning branches do not cover it), so
 *  low/medium/xhigh would otherwise be silently dropped as this reasoning_effort key. 2026-10-19, the
 *  user set the record straight about the other half: their model is served by strata with a
 *  hand-written chat template, and that engine reads no such flag - the template switches modes by
 *  looking for the no-think tag inside the message text, which buildRequestParts already writes there
 *  when thinking is off. So "off" sends no body key, exactly as it did before that detour. */
export function chatTemplateKwargsFor(thinking: Level): Record<string, unknown> | undefined {
	const level = thinking === "minimal" ? "low" : thinking === "high" ? "xhigh" : thinking;
	if (level === "low" || level === "medium" || level === "xhigh") return { reasoning_effort: level };
	// "off" is carried by the no-think tag in the text, not by a body key (2026-10-19, corrected after a
	// detour): this user's model runs under strata with a hand-written chat template, that engine reads no
	// chat-template flag at all, and the template switches modes by looking for the tag inside the message
	// - which buildRequestParts already writes there when thinking is off. A key the engine does not read
	// would only have made the request look as if thinking had been switched off when nothing was.
	// tagWritten, further down in directCompact, is the honest record of whether the tag went out.
	// "off" and "max" therefore send no thinking key of any kind; if some future server does want the
	// body key, this is the single line to put it back on.
	return undefined;
}

/**
		 * Put a ceiling on the model's own thinking, in tokens, for THIS request - the one lever that works
		 * on a server whose chat template we do not control (2026-10-20, taken from the servers themselves).
		 * Thinking gets decided twice, and the prompt only reaches the second decision: the server picks the
		 * size of the thinking channel from the REQUEST PARAMETERS before generating anything, while the
		 * no-think tag is text the model reads later, by which time the budget is already spent. So say it in
		 * the request. Every recognised spelling goes, because a server ignores the keys it does not
		 * implement and this one request has to serve three engines: reasoning_budget_tokens is Strata's cap
		 * (its run-config applies the server-wide value only when the request sends none of its own, so
		 * naming ours wins), and thinking plus chat_template_kwargs is what ik_llama.cpp's server reads.
		 * Never send 0 to Strata: its documentation says 0 means no cap rather than no thinking, which would
		 * hand the whole request back to the thinking channel - the exact shape of last night's failure. A
		 * small non-zero budget still lets a model close its own think block cleanly. The total asked for does
		 * not change with this: the summary is what gains the room thinking gives up.
	 */
	export function thinkingBudgetFor(thinking: Level): { budget: number; fields: Record<string, unknown> } {
		const budgets: Record<string, number> = { off: 256, minimal: 1024, low: 1024, medium: 4096, high: 8192, xhigh: 8192 };
		const budget = budgets[thinking] ?? 4096;
		const fields: Record<string, unknown> = {
			reasoning_budget_tokens: budget,
			thinking: thinking === "off" ? { type: "disabled" } : { type: "enabled", budget_tokens: budget },
		};
		if (thinking === "off") fields.chat_template_kwargs = { enable_thinking: false, thinking_enable: false, thinking_budget: 0 };
		else fields.chat_template_kwargs = { enable_thinking: true, thinking_budget: budget };
		return { budget, fields };
	}

/** The streamFn handed to pi's compact() so a FALLBACK request carries the same sampling flags and
 *  effort kwargs as the direct request. undefined = nothing to inject - call compact() as before. */
function samplingStreamFn(ref: string, sampling: SamplingFlags | undefined, thinking: Level): ((model: any, context: any, opts: any) => any) | undefined {
	const params = samplingParamsFor(ref, sampling);
	const kwargs = refLooksLocal(ref) ? chatTemplateKwargsFor(thinking) : undefined;
	// The same ceiling the direct request sends, so a fallback through pi's own compact() is not the
	// one path where the model may spend the whole ask on its own reasoning.
	const kBud = refLooksLocal(ref) ? thinkingBudgetFor(thinking) : undefined;
	if (!params && !kwargs && !kBud) return undefined;
	return (model, context, opts) => piStreamSimple(model, context, {
		...opts,
		samplingParams: { ...(params ?? {}), ...(kwargs ?? {}), ...((kBud?.fields as Record<string, unknown>) ?? {}) },
	});
}

/** Board label for the state of a model's flags. */
function samplingLabel(s: SamplingFlags): string {
	const n = SAMPLING_KEYS.filter((k) => typeof s[k] === "number").length;
	return n ? `custom (${n} set)` : "ignore";
}

/** A thinking level as one plain word. The stored value for "do not reason" is "off"; the operator
 *  reads and picks it as "none" (2026-10-07: the row showed the internal word). */
function thinkingWord(level: Level | string | undefined): string {
	return !level || level === "off" ? "none" : String(level);
}

/** Row help, one per flag. */
const SAMPLING_HELP: Record<SamplingKey, string> = {
	temperature: 'Sampling temperature for the compaction call only (your chat requests are never touched). Qwen3.8-Flash-Next non-thinking recommends 0.7; the server flag default is the thinking value 1.0. "ignore" = send nothing, the server default applies.',
	top_p: 'Nucleus cutoff for the compaction call only. Qwen3.8-Flash-Next non-thinking recommends 0.80 (the server thinking default is 0.95). "ignore" = server default.',
	top_k: 'Top-K for the compaction call only. Qwen3.8-Flash-Next recommends 20 in both modes. "ignore" = server default.',
	min_p: 'Min-P for the compaction call only. Qwen3.8-Flash-Next recommends 0.0 in both modes. "ignore" = server default.',
	presence_penalty: 'Presence penalty for the compaction call only. Qwen3.8-Flash-Next non-thinking recommends 1.5 (thinking 0.0). "ignore" = server default.',
	repetition_penalty: 'Repetition penalty for the compaction call only. Qwen3.8-Flash-Next recommends 1.0 in both modes. Sent as repeat_penalty (the name llama-server reads). "ignore" = server default.',
};

/** Cycle ladders for the TUI rows; the Qwen instruct value sits first where one exists. */
const SAMPLING_LADDERS: Record<SamplingKey, string[]> = {
	temperature: ["0.7", "1", "0.9", "0.8", "0.6", "0.5", "0.3", "0.2"],
	top_p: ["0.8", "0.95", "1", "0.9", "0.7"],
	top_k: ["20", "40", "60", "10", "0"],
	min_p: ["0", "0.02", "0.05", "0.1"],
	presence_penalty: ["1.5", "1", "0.5", "0.2", "0"],
	repetition_penalty: ["1", "1.05", "1.1", "1.15", "0.9"],
};

// ---------------------------------------------------------------- the direct request (fork)

/** The compaction engine's own system prompt - replaces pi's generic summarizer prompt entirely. */
const COMPACTOR_SYSTEM_PROMPT = `You are a compaction engine for a coding-agent session. Read the conversation segment and produce the turn-ledger summary described in the instructions below. Do NOT continue the conversation, do NOT answer questions asked in it, and do NOT call tools. Respond with the summary only.`;

/** Cap the string values of ONE tool-call arguments object (copy-on-write). */
function capArgsObj(args: any, cap: CharCap): any {
	if (cap === "full" || !args || typeof args !== "object") return args;
	const out: Record<string, unknown> = {};
	let changed = false;
	for (const [k, v] of Object.entries(args)) {
		if (typeof v === "string" && v.length > cap) {
			out[k] = `${v.slice(0, cap)}...(+${fmt(v.length - cap)} chars)`;
			changed = true;
		} else {
			out[k] = v;
		}
	}
	return changed ? out : args;
}

/**
 * The fork's own region serializer - the thing pi's serializeConversation does, but with
 * DETERMINISTIC TURN NUMBERS and stubs/caps applied in the same pass. Turn boundaries follow pi's
 * own rule (user message, bash execution, custom message, branch/compaction summary). Labels:
 *   [User - Turn 7]: ...        [User bash - Turn 7]: ...     [Note - Turn 7]: ...
 *   [Assistant thinking]: ...   [Assistant]: ...
 *   [Tool call bash("bun test")]:      [Tool result]: ...
 * A result longer than 2,000 characters becomes an informative stub (inputStubs on) - pi's
 * head-only clip then has nothing important left to cut. Argument VALUES are capped (argCap).
 */
export function serializeRegion(
	messages: any[],
	opts: { startTurn: number; inputStubs: boolean; argCap: CharCap; partial?: boolean },
): { text: string; lastTurn: number; hadTurns: boolean } {
	const parts: string[] = [];
	let turn = Math.max(0, Math.floor(opts.startTurn));
	let hadTurns = false;
	let partialLabeled = false;
	for (let i = 0; i < messages.length; i++) {
		const m = messages[i];
		const role = m?.role;
		if (role === "user" || role === "bashExecution" || role === "custom" || role === "branchSummary" || role === "compactionSummary") {
			const text = contentTextOf(m.content).trim();
			if (!text) continue;
			turn++;
			hadTurns = true;
			const label = role === "user" ? "User" : role === "bashExecution" ? "User bash" : role === "custom" ? "Note" : "Earlier summary";
			let tag = `Turn ${turn}`;
			if (opts.partial && !partialLabeled) {
				tag += " (the beginning of this turn - it continues in the kept messages after the summary)";
				partialLabeled = true;
			}
			parts.push(`[${label} - ${tag}]: ${text}`);
			continue;
		}
		if (role === "assistant") {
			const thinking: string[] = [];
			const texts: string[] = [];
			const calls: string[] = [];
			for (const b of m.content ?? []) {
				if (b?.type === "thinking") thinking.push(String(b.thinking ?? ""));
				else if (b?.type === "text") texts.push(String(b.text ?? ""));
				else if (b?.type === "toolCall") {
					const a = Object.entries(capArgsObj(b.arguments, opts.argCap) ?? {})
						.map(([k, v]) => `${k}=${JSON.stringify(v)}`)
						.join(", ");
					calls.push(`${b.name}(${a})`);
				}
			}
			if (thinking.length) parts.push(`[Assistant thinking]: ${thinking.join("\n")}`);
			if (texts.length) parts.push(`[Assistant]: ${texts.join("")}`);
			for (const c of calls) parts.push(`[Tool call ${c}]:`);
			continue;
		}
		if (role === "toolResult") {
			const text = contentTextOf(m.content);
			if (!text) continue;
			const out = opts.inputStubs && text.length > 2_000 ? stubbedResult(messages, i, m, ELISION_DEFAULTS.stubTailChars).content[0].text : text;
			parts.push(`[Tool result]: ${out}`);
			continue;
		}
		const text = contentTextOf(m.content).trim();
		if (text) parts.push(`[${String(role)}]: ${text}`);
	}
	return { text: parts.join("\n\n"), lastTurn: turn, hadTurns };
}

/** The latest "Turn N" number mentioned anywhere in a text (the chained-compaction counter). */
export function maxTurnInSummary(text: string): number {
	let max = 0;
	for (const m of String(text ?? "").matchAll(/\bTurn (\d+)\b/g)) max = Math.max(max, Number(m[1]));
	return max;
}

/**
 * Where the previous compaction's numbering ended. Best source: the CompactionEntry details the
 * fork itself wrote (details.lastTurnNumber); fallback: parse the previous summary's text.
 */
/** Size estimate of the previous summary (for the chain-attach ceiling raise). */
function prevSummaryTokens(prep: any): number {
	const s = prep?.previousSummary;
	return s && String(s).trim() ? Math.ceil((String(s).length * 1.3) / 4) : 0;
}

export function readPrevTurnInfo(ctx: Ctx, preparation: any): { lastTurn: number; source: "details" | "parsed" | "none" } {
	try {
		const entries: any[] = ctx?.sessionManager?.getEntries?.() ?? [];
		for (let i = entries.length - 1; i >= 0; i--) {
			const e = entries[i];
			if (e?.type === "compaction" && e?.details && typeof e.details.lastTurnNumber === "number") {
				return { lastTurn: Math.max(0, Math.floor(e.details.lastTurnNumber)), source: "details" };
			}
		}
	} catch {
		/* fall through to parsing */
	}
	const lastTurn = maxTurnInSummary(preparation?.previousSummary ?? "");
	return { lastTurn, source: lastTurn > 0 ? "parsed" : "none" };
}

/** The previous summary, introduced properly: what it covers, that its numbering is final, what to
 *  carry forward and what to skip. */
export function labeledPreviousSummary(previousSummary: string | undefined, prevLastTurn: number, nextTurn: number): string {
	if (!previousSummary?.trim()) return "";
	return [
		`<previous-summary>`,
		`This ledger was written by the previous compaction. It covers turns up to Turn ${prevLastTurn > 0 ? prevLastTurn : "?"} (the turn numbers in it are already final). Carry every turn forward - tighten old entries if space demands - and continue the numbering at Turn ${nextTurn}. Its "Full raw transcript" line is bookkeeping: do not copy it.`,
		`---`,
		previousSummary.trim(),
		`</previous-summary>`,
	].join("\n");
}

/**
 * The SHAPE GATE: the size gate rejects wrong LENGTH; this rejects wrong SHAPE. Checks: all six
 * sections present; at least one Turn label; the ledger's turn labels strictly increasing; every
 * "superseded/amended by Turn N" tag referencing a turn number that exists in the summary.
 */
export function shapeCheck(summary: string): { ok: boolean; reason?: string } {
	const s = String(summary ?? "");
	for (const h of ["CURRENT STATE", "TURN LEDGER", "FILES & DATA", "KEY DECISIONS", "NEXT STEPS", "CRITICAL CONTEXT"]) {
		if (!new RegExp(`^##\\s*${h}\\b`, "im").test(s)) return { ok: false, reason: `missing section "## ${h}"` };
	}
	const ledgerStart = s.indexOf("## TURN LEDGER");
	const ledgerEnd = s.indexOf("\n## ", ledgerStart + 5);
	const ledger = ledgerEnd > ledgerStart ? s.slice(ledgerStart, ledgerEnd) : s.slice(ledgerStart);
	const labels = [...ledger.matchAll(/^\s*Turn (\d+)\b/gm)].map((m) => Number(m[1]));
	if (!labels.length) return { ok: false, reason: "no Turn labels in the ledger" };
	for (let i = 1; i < labels.length; i++) {
		if (labels[i] <= labels[i - 1]) return { ok: false, reason: `ledger turn labels not increasing at "Turn ${labels[i]}"` };
	}
	const maxTurn = Math.max(...labels);
	// 2026-10-06: the sub-line check. A ledger whose entries are single prose lines passes every other
	// check and still loses the tool calls, the replies and the thinking, which is what users see as
	// "nothing but a heavy summary". With several turns, at least half of them must carry sub-lines.
	const led = ledger.split("\n");
	let entries = 0;
	let withSub = 0;
	for (let i = 0; i < led.length; i++) {
		if (!/^\s*Turn \d+\b/.test(led[i])) continue;
		entries += 1;
		let sub = 0;
		for (let j = i + 1; j < led.length; j++) {
			if (/^\s*Turn \d+\b/.test(led[j]) || /^#{1,3}\s/.test(led[j])) break;
			// A sub-line is an indented "label: content" line (the label is any short word or phrase,
			// because the template decides whether it reads "user:" or "user asked:") or a bulleted line.
			if (/^\s{2,}[^-\s#:][^:]{0,44}:\s*\S/.test(led[j]) || /^\s{4,}-\s+\S/.test(led[j])) sub += 1;
		}
		if (sub >= 2) withSub += 1;
	}
	if (entries >= 5 && withSub < Math.ceil(entries / 2)) {
		return { ok: false, reason: `the ledger is flat - only ${withSub} of ${entries} turn entries carry sub-lines (the user prompt, the thinking, the replies and the tool calls belong on their own lines)` };
	}
	for (const m of s.matchAll(/(?:superseded|amended) by Turn (\d+)/g)) {
		if (Number(m[1]) > maxTurn) return { ok: false, reason: `tag references Turn ${m[1]} which is not in the summary` };
	}
	return { ok: true };
}
/** How much of the TURN LEDGER really carries a user prompt (2026-10-18), and how many prompts appear
 *  anywhere in the summary (2026-10-19 - see userPromptsGiven for why that second figure exists).
 *
 *  `turns` and `withPrompt` look at the ledger: the Turn entries and, for each, whether its sub-lines hold a
 *  line labelled with the word user with real text after the colon. The label is whatever the template chose
 *  ("user:", "user asked:", "USER PROMPT:") so the tell is the word rather than the wording, and a label with
 *  nothing after it does not count, because an empty user line is how a model satisfies the shape without
 *  carrying the prompt it was told to quote.
 *
 *  `linesAnywhere` counts the same labelled lines across the whole summary, ledger or no ledger. It exists
 *  because the 00:50 summary on 2026-10-18 carried no Turn labels whatsoever - the shape gate said so out loud
 *  in its warning - and a check written only on turn labels reads "zero turns, nothing to complain about" in
 *  exactly the case where the model threw the structure away along with the prompts.
 *
 *  Counts, not a verdict, so the caller can tell a thin ledger from an empty one. Falls back to the whole
 *  summary when there is no ledger section, so a differently shaped template still gets measured instead of a
 *  silent zero. */

export function userPromptCoverage(summary: string): { turns: number; withPrompt: number; linesAnywhere: number } {
	const all = String(summary ?? "").split("\n");
	// "four letters" rather than "not empty": a user line that says nothing, shows a dash, or quotes the word
	// "ok" is the model acknowledging the shape without carrying the prompt. The same rule on the given side,
	// in userPromptsGiven, keeps that honest rather than punishing a faithful summary of a terse session.
	const real = (s: string) => s.replace(/[^A-Za-z0-9]/g, "").length >= 4;
	let inLedger = false;
	const ledger: string[] = [];
	for (const l of all) {
		if (/^##\s*TURN LEDGER\b/i.test(l)) {
			inLedger = true;
			continue;
		}
		if (inLedger && /^##\s/.test(l)) break;
		if (inLedger) ledger.push(l);
	}
	const body = ledger.length >= 4 ? ledger : all;
	let turns = 0;
	let withPrompt = 0;
	for (let i = 0; i < body.length; i++) {
		if (!/^\s*Turn \d+\b/.test(body[i])) continue;
		turns += 1;
		for (let j = i + 1; j < body.length; j++) {
			if (/^\s*Turn \d+\b/.test(body[j]) || /^#{1,3}\s/.test(body[j])) break;
			const m = body[j].match(/^\s*user[^:\n]{0,30}\s*:\s*(\S.*)$/i);
			if (m && real(m[1])) {
				withPrompt += 1;
				break;
			}
		}
	}
	let linesAnywhere = 0;
	for (const l of all) {
		const m = l.match(/^\s*[-*]?\s*user[^:\n]{0,30}\s*:\s*(\S.*)$/i);
		if (m && real(m[1])) linesAnywhere += 1;
	}
	return { turns, withPrompt, linesAnywhere };
}

/** How many prompts the summarizer was HANDED (2026-10-19). serializeRegion marks each one - "[User - Turn
 *  12]: ...", and for a /bash entry or a custom note "[User bash - Turn 12]" / "[Note - Turn 12]". Those are
 *  all things the user gave the session, so they all belong in the denominator. "[Earlier summary - Turn N]"
 *  is deliberately not counted: that is the previous summary rather than a prompt, and counting it would make
 *  a chained summary look better supplied than the model was.
 *
 *  Only prompts with at least four letters count, mirroring the summary-side rule above. A prompt whose text
 *  was trimmed by preprocessing still counts as long as any of its text survives the trim - the four-letter
 *  rule is what decides, not whether the material was shortened, and a prompt emptied to nothing was never
 *  there to quote. */
export function userPromptsGiven(material: string): number {
	let n = 0;
	const real = (s: string) => s.replace(/[^A-Za-z0-9]/g, "").length >= 4;
	for (const m of String(material ?? "").matchAll(/^\[(?:User(?: bash)?|Note) - Turn \d+[^\]]*\]:\s*(\S.*)$/gm)) {
		if (real(m[1])) n += 1;
	}
	return n;
}

/** pi's machine-readable file appendix (<read-files>/<modified-files>), from the preparation's
 *  fileOps (a Set or an array - both accepted). */
export function fileOpsAppendix(fileOps: any): string {
	try {
		const listOf = (x: any): string[] => [...(x ?? [])].map(String);
		const modified = new Set([...listOf(fileOps?.edited), ...listOf(fileOps?.written)]);
		const readFiles = listOf(fileOps?.read).filter((f) => !modified.has(f)).sort();
		const modifiedFiles = [...modified].sort();
		const sections: string[] = [];
		if (readFiles.length) sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
		if (modifiedFiles.length) sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
		return sections.length ? `\n\n${sections.join("\n\n")}` : "";
	} catch {
		return "";
	}
}

/**
 * The DIRECT summarization request: the whole thing built here - labelled previous summary, the
 * turn-numbered region (stubs + arg caps applied in the same pass), one instruction source - and
 * the model called through the registry (request-time auth), maxTokens set directly, no prompt
 * cache writes. Throws on any failure (stop reason, tool-call attempt); the caller falls back to
 * pi's compact().
 */
export async function directCompact(args: {
	preparation: any;
	model: any;
	registry: any;
	ref: string;
	kind: Kind;
	sizes: Sizes;
	options: ModelOptions;
	signal: AbortSignal | undefined;
	thinking: Level;
	prevTurn: number;
	customInstructions?: string;
	/** The main template: the ACTIVE text (or null to follow the user default / the extension
	 *  default). The direct request builds its own instructions, so these must ride in.
	 */
	additionalInstruction?: string | null;
	userDefault?: string | null;
	/** Called with the length of every piece of text that arrives (2026-10-19, item (a)). The caller
	 *  keeps the silence clock restarted for as long as the model is writing: a request that streams
	 *  is a request that is alive, whatever the wall clock says. Absent keeps the old behaviour. */
	onActivity?: (chars: number) => void;
}): Promise<{ text: string; usage: any; lastTurn: number; inputText: string; segmentBy: "count" | "estimate"; fromReasoning: boolean }> {
	const { preparation, model, registry, sizes, options } = args;
	const startTurn = args.prevTurn > 0 ? args.prevTurn : 0;
	const main = serializeRegion(preparation?.messagesToSummarize ?? [], { startTurn, inputStubs: options.inputStubs, argCap: options.argCap });
	const prefix = serializeRegion(preparation?.turnPrefixMessages ?? [], { startTurn: main.lastTurn, inputStubs: options.inputStubs, argCap: options.argCap, partial: true });
	const nextTurn = main.lastTurn + (prefix.hadTurns ? 1 : 0);
	const prevBlock = labeledPreviousSummary(preparation?.previousSummary, args.prevTurn, nextTurn);
	const regionText = [main.text, prefix.text].filter(Boolean).join("\n\n");
	const coverLine = `The conversation segment below covers Turns ${startTurn + 1}-${Math.max(main.lastTurn, prefix.lastTurn)} (already numbered - use those numbers).`;
	// The real count when the server can state one (2026-10-18): the character heuristic is within about
	// 2 % on this pack's transcript and within about 11 % on summary prose, but it is a guess, and a guess
	// the model is asked to hold itself to is a poor ruler. An online candidate cannot be counted before
	// the fact, so it keeps the estimate and the record says which ruler was used.
	const countBase = localSummariserBaseUrl(model, args.ref);
	const guessed = Math.ceil((regionText.length * TOKEN_HEAVINESS) / 4);
	const counted = countBase ? await countTextAt(countBase, [prevBlock, regionText].filter(Boolean).join("\n\n"), guessed) : null;
	const segmentBy = counted ? counted.how : countBase ? "estimate (this server gave no usable count)" : "estimate (not a local model)";
	const inputTokens = (counted?.n ?? guessed) + SUMMARISER_FRAMING_TOKENS;
	const instructions = buildInstructions({ sizes, inputTokens, segmentTokens: counted?.n ?? inputTokens - SUMMARISER_FRAMING_TOKENS, kind: args.kind, options, additionalInstruction: args.additionalInstruction ?? null, userDefault: args.userDefault, customInstructions: args.customInstructions });
	const promptText = [
		prevBlock,
		prevBlock ? coverLine : coverLine,
		`<conversation>`,
		regionText,
		`</conversation>`,
		instructions,
	].filter(Boolean).join("\n\n");
	const context = {
		systemPrompt: COMPACTOR_SYSTEM_PROMPT,
		messages: [{ role: "user", content: [{ type: "text", text: promptText }], timestamp: Date.now() }],
	};
	// The engine's instruction ceiling, and a hard one (2026-10-19, the user's rule after a crash):
	// Summary max, exactly, is what the engine is told to generate, and that figure is the model's TOTAL
	// allowance for this summary - thinking included, because a thinking level is a model setting rather
	// than extra room, and it is capped inside the request (see thinkingBudgetFor) so it spends from
	// inside the allowance instead of quietly enlarging it. The same number reaches the template as
	// {MAX TOKENS}, so what the model is told and where the server will stop it are the same figure.
	// The margin above max is not part of this ask at all; it is only the tolerance the finished answer
	// is judged against, and it is applied at the size checks, not here.
	// History, kept because these lines are where the design went wrong twice: the allowances used to be
	// ADDED on top of max (a 30,000 aim became a 34,608 request and ran off the window), and then the
	// ask was max + margin, which made the ceiling in the log look like a measurement of the reply.
	const dCeiling = sizes.max + GEN_MARGIN;
	const dScratch = scratchAllowanceFor(model, sizes.genCap);
	const dAsk = sizes.genCap;
	// Whether the no-think tag went out with this request, logged, because "thinking is off in the
	// settings" and "the engine was told" are two different claims, and last night needed the second one
	// written down somewhere a person could read it.
	const noteWritten = args.thinking === "off" && args.kind === "local" && !!options.noThinkMarker;
	const requestOptions: Record<string, unknown> = { maxTokens: dAsk, signal: args.signal, cacheRetention: "none" };
	if (model?.reasoning && args.thinking !== "off") requestOptions.reasoning = args.thinking;
	// The per-model sampling flags, plus (for a local model that takes an effort level) the effort
	// itself, ride into the request body here; the adapter merges samplingParams over the named fields.
	const dParams = samplingParamsFor(args.ref, options.sampling);
	const dKwargs = refLooksLocal(args.ref) ? chatTemplateKwargsFor(args.thinking) : undefined;
	// The ceiling on thinking is computed beside the template kwargs because both answer the same
	// question - how much of this request may go on thinking - and only one of them is a request
	// parameter rather than a line of prose.
	const dBudget = refLooksLocal(args.ref) ? thinkingBudgetFor(args.thinking) : undefined;
	// The thinking ceiling rides in the same body object as the sampling flags: the adapters merge
	// samplingParams over the named request fields, and a server that has never heard of
	// reasoning_budget_tokens simply does not see it. That silence is what makes sending all three
	// spellings safe rather than a gamble on which engine is on the other end.
	if (dParams || dKwargs || dBudget) {
		const merged: Record<string, unknown> = { ...(dParams ?? {}), ...(dKwargs ?? {}), ...((dBudget?.fields as Record<string, unknown>) ?? {}) };
		// A sampling flag of the same name is the user's own setting, so theirs wins on a collision; the
		// two chat_template_kwargs objects merge instead of one replacing the other.
		if (dKwargs && dBudget) merged.chat_template_kwargs = { ...(((dBudget.fields as any)?.chat_template_kwargs as Record<string, unknown>) ?? {}), ...(((dKwargs as any)?.chat_template_kwargs as Record<string, unknown>) ?? {}) };
		requestOptions.samplingParams = merged;
	}
	// Consuming the stream rather than waiting on its result is what makes the silence clock mean
	// anything: the events are the only place where "still writing" is visible. The result is read
	// off the same stream afterwards, which is what .result() would have waited for anyway.
	const dStream: any = registry.streamSimple(model, context, requestOptions);
	let response: any;
	if (typeof args.onActivity === "function" && typeof dStream?.[Symbol.asyncIterator] === "function") {
		for await (const ev of dStream) {
			const dText = typeof (ev as any)?.delta === "string" ? (ev as any).delta : "";
			args.onActivity(dText.length);
		}
		response = await dStream.result();
	} else {
		response = await dStream.result();
	}
	// A generation that ran out of room is not automatically worthless, and treating it that way is
	// what stopped every compaction tonight (2026-10-19, live): the local model's hidden reasoning
	// ate most of the request budget, the visible summary stopped at the cap, and the attempt was
	// thrown away the instant stopReason arrived - before a single character of it was looked at. The
	// user's reading was right that the number made no sense: 30,512 is not a measurement of what
	// the model wrote, it is our own acceptance ceiling (Summary max + GEN_MARGIN) reported back at
	// them, while the request had actually been capped higher and the model had filled it. Judge what
	// arrived instead, after it has arrived: a half-finished sentence at the end is a loss, a session
	// that cannot compact is a bigger one. So this run keeps its honest numbers, and the gate below
	// (which is the ordinary size and shape checks, run on the text as it would be after a clean stop)
	// decides.
	const cutShort = response?.stopReason === "length";
	// What the SERVER was told to generate, and everything the model wrote against it. A message about
	// being cut short must quote these, not our own acceptance ceiling, or the reader chases a number
	// that was never a measurement of anything.
	const requestCap = Number(requestOptions.maxTokens) || sizes.genCap;
	// pi reports `output` as everything the server completed, its own reasoning included, and `reasoning`
	// as the part of that which was hidden - so the two are never added together, only named separately.
	const wroteAll = Math.round(response?.usage?.output ?? 0);
	const reasonedAll = Math.round(response?.usage?.reasoning ?? 0);
	if (response?.stopReason === "error") throw new Error(response?.errorMessage || "provider error");
	if (response?.stopReason === "aborted") throw new Error("aborted");
	if ((response?.content ?? []).some((b: any) => b?.type === "toolCall")) throw new Error("the model tried to call a tool");
	let text = (response?.content ?? [])
		.filter((b: any) => b?.type === "text")
		.map((b: any) => String(b.text ?? ""))
		.join("")
		.trim();
	// AN EMPTY ANSWER IS NOT AN EMPTY REQUEST (2026-10-18). On 03:24 the local Qwen ran 594 s and the
	// record said "empty response", which threw the run away as though nothing had happened: about 440 s
	// of that was decoding at this pack's rate, i.e. several thousand tokens of reasoning the answer
	// channel never carried. So the reasoning channel is now offered to the gates instead of the bin - a
	// summary the model wrote in its head is worth the rough-work stripper it has to go through - and a
	// run with NOTHING in either channel fails carrying the usage counts and the stop reason, so the next
	// one of those says what actually happened instead of repeating itself.
	let fromReasoning = false;
	if (!text) {
		const scratch = (response?.content ?? [])
			.filter((b: any) => b?.type === "thinking" || b?.type === "reasoning")
			.map((b: any) => String(b?.text ?? b?.thinking ?? b?.reasoning ?? ""))
			.join("\n\n")
			.trim();
		if (scratch.length > 200) {
			text = scratch;
			fromReasoning = true;
		} else {
			const dead: any = new Error(scratch ? "empty response (the reasoning channel held only " + scratch.length + " characters, no summary)" : "empty response");
			dead.usage = response?.usage;
			dead.stopReason = response?.stopReason;
			throw dead;
		}
	}
	// Running out of room is a FAILURE, and it has to stay one (2026-10-20, on the user's evidence
	// rather than my inference). On 2026-10-19 I turned this exact case into carry on with whatever
	// arrived, and the result was the summary of last night: the room went on a draft block and on
	// hidden reasoning, so what arrived was a short write-up OF THE MODEL'S OWN DRAFT instead of the
	// conversation. Accepting that writes a lobotomized ledger into the session and the user loses
	// the whole transcript from the model's view; throwing this attempt moves the ask on to the next
	// model, which costs minutes. Minutes are the cheaper loss, and a truncated run is exactly what
	// produces a stub, so there is no judgement of the text that makes this safe to keep.
	if (cutShort) {
		throw new Error(
				`generation was cut short at the request cap of ${fmt(requestCap)} tok: the model wrote ${fmt(wroteAll)} tok against it, of which ${fmt(reasonedAll)} tok was hidden reasoning, so the summary that arrived is whatever room was left over. Two things eat that room and neither is excluded by the size numbers written into the template: hidden reasoning - a thinking-by-default chat template keeps reasoning while our request sends it nothing about thinking, and the server then takes a share of the request budget for it - and a draft or scratchpad block, which the template used to ask for and which costs a second full pass of output tokens. Switch Thinking off for this model, keep draft wording out of the summary template, and move the trigger earlier so the segments get smaller`,
		)
	}
	// The exact conversation text the request carried - the stub-coverage gate counts the stub
	// markers in it against the copies in the summary.
	return { text, usage: response.usage, lastTurn: Math.max(main.lastTurn, prefix.lastTurn), inputText: `${main.text}\n${prefix.text}`, segmentBy, fromReasoning };
}

/**
 * Shrink the summariser input so that it fits a slot's context window. The NEWEST messages are kept -
 * they are what the model has most recently reasoned about, and the point of a compaction summary is to
 * let the work continue, not to archive week-old tool output.
 *
 * Cut points are only taken at a message that starts a turn (a user message, a bash run, a summary), so
 * a tool call is never separated from the result that answers it. Returns undefined when even the newest
 * slice that keeps a sensible amount of text is too big.
 */
export function trimToTokenBudget(preparation: any, budgetTokens: number): any | undefined {
	const firstList: any[] = preparation.messagesToSummarize ?? [];
	const secondList: any[] = preparation.turnPrefixMessages ?? [];
	// Work on one flat list so the ordering is preserved while trimming from the front.
	const all = [...firstList, ...secondList];
	if (all.length === 0) return preparation;

	// Drop from the front until the remaining tail fits, stopping at a turn start.
	let start = 0;
	let total = 0;
	for (let i = all.length - 1; i >= 0; i--) {
		total += msgTokens(all[i]);
		if (total > budgetTokens) break;
		start = i;
	}
	if (total > budgetTokens) {
		// Walk forward from the cut to the next turn-start message so the slice begins cleanly.
		while (start < all.length && !isTurnStart(all[start])) start++;
	}
	const kept = all.slice(start);
	if (kept.length === 0) return undefined;
	let keptTokens = 0;
	for (const m of kept) keptTokens += msgTokens(m);
	if (keptTokens > budgetTokens) return undefined; // even the cleanest tail is too big
	return {
		...preparation,
		messagesToSummarize: kept,
		turnPrefixMessages: [],
		_droppedFromTrim: all.length - kept.length,
	};
}

/** Whether a message begins a turn (same rule pi uses in isTurnStartMessage). */
function isTurnStart(m: any): boolean {
	switch (m?.role) {
		case "user":
		case "bashExecution":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return true;
		default:
			return false;
	}
}

function combineSignals(signal: AbortSignal | undefined, ms: number, onTimeout?: () => void): { signal: AbortSignal | undefined; dispose: () => void } {
	if (!ms || ms <= 0) return { signal, dispose: () => {} };
	const ctrl = new AbortController();
	const timer = setTimeout(() => {
		onTimeout?.();
		ctrl.abort(new Error(`timed out after ${Math.round(ms / 1000)}s`));
	}, ms);
	const onAbort = () => ctrl.abort((signal as any)?.reason);
	signal?.addEventListener("abort", onAbort, { once: true });
	return {
		signal: ctrl.signal,
		dispose: () => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		},
	};
}

/** What "timed out" ought to mean (2026-10-19, item (a) of the 19:15 report). The plain wall-clock
 *  ceiling above cannot tell a slow model from a dead request: it cut a generation that was still
 *  producing at exactly ten minutes, and the online model that took over then finished the same
 *  summary and billed 29.2k output tokens. Two clocks say what anybody meant. SILENCE is the one the
 *  operator sets, and every character that arrives restarts it, so a model that keeps writing keeps
 *  its request. The BACKSTOP is the one nobody sets, because a stream that produces forever is still
 *  a stream that will never finish. Whichever expires first aborts, and the record says which it was:
 *  "it went silent" and "it was still writing when the ceiling arrived" are different faults with
 *  different cures, and "timed out" hid the difference.
 */
function silenceWatch(
	signal: AbortSignal | undefined,
	opts: { silenceMs: number; backstopMs: number; onTimeout: (why: "silence" | "backstop") => void },
): { signal: AbortSignal | undefined; dispose: () => void; beat: (chars: number) => void } {
	const ctrl = new AbortController();
	let idle: ReturnType<typeof setTimeout> | undefined;
	let lastArm = 0;
	const arm = () => {
		if (!opts.silenceMs || opts.silenceMs <= 0) return;
		clearTimeout(idle);
		lastArm = Date.now();
		idle = setTimeout(() => {
				opts.onTimeout("silence");
				ctrl.abort(new Error(`nothing arrived for ${secs(opts.silenceMs)}`));
			}, opts.silenceMs);
	};
	// A streaming reply says delta twenty times a second; re-arming a timer twenty times a second is
	// waste. The throttle is capped at a third of the window, though, because a one second slop on a
	// clock measured in minutes is nothing while the same slop on a short test clock would abort a
	// model that was writing all along.
	const beat = (chars: number) => {
		if (chars <= 0) return;
		if (Date.now() - lastArm < Math.min(1000, Math.max(1, Math.round(opts.silenceMs / 3)))) return;
		arm();
	};
	arm();
	const backstop =
		opts.backstopMs > 0
			? setTimeout(() => {
					opts.onTimeout("backstop");
					ctrl.abort(new Error(`still writing after ${secs(opts.backstopMs)}`));
				}, opts.backstopMs)
			: undefined;
	// A cancel from outside is not a timeout: stop both clocks first, or the person who pressed Escape is
	// recorded as a model that went silent, and the next model is blamed for nothing.
	const onAbort = () => {
		clearTimeout(idle);
		if (backstop) clearTimeout(backstop);
		ctrl.abort((signal as any)?.reason);
	};
	signal?.addEventListener("abort", onAbort, { once: true });
	return {
		signal: ctrl.signal,
		beat,
		dispose: () => {
			clearTimeout(idle);
			if (backstop) clearTimeout(backstop);
			signal?.removeEventListener("abort", onAbort);
		},
	};
}

/** How fast this model wrote the last time we measured it, in summary tokens per second. Kept for
 *  the life of the process, which is exactly the horizon over which "this box does 15 tok/s" holds. */
const observedRate = new Map<string, number>();

/** The ceiling on a stream that never stops: twice what the last measured rate says this summary
 *  should take, never under twenty minutes, never over two hours. A model that has never been
 *  measured is assumed slow (4 tokens a second locally, 25 online), because guessing high costs a
 *  wait while guessing low throws away a summary that was nearly finished - the mistake this
 *  change exists to stop. "Unlimited" on the board means unlimited: no ceiling behind it either. */
function backstopMsFor(kind: Kind, targetTokens: number, silenceMs: number, ref: string): number {
	if (silenceMs <= 0) return 0;
	const seen = observedRate.get(ref);
	const tps = seen && seen > 0.2 ? seen : kind === "local" ? 4 : 25;
	const estimate = targetTokens > 0 ? Math.round((targetTokens / tps) * 1000) * 2 : 0;
	return Math.min(7_200_000, Math.max(1_200_000, estimate, silenceMs));
}
function fmt(n: any): string {
	return typeof n === "number" ? n.toLocaleString("en-US") : "?";
}

function secs(ms: number): string {
	if (!ms) return "no limit";
	if (ms % 60000 === 0) return `${ms / 60000} min`;
	return `${Math.round(ms / 1000)} s`;
}

function shortName(id: string): string {
	return String(id ?? "").replace(/^.*[\\/]/, "").replace(/\.gguf$/i, "");
}

// ---------------------------------------------------------------- thinking levels

/** Rough thinking size per level, only used when a token budget is given by hand. */
const ALL_LEVELS: Level[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

const THINKING_LADDER: Array<{ level: Level; capacity: number }> = [
	{ level: "off", capacity: 0 },
	{ level: "minimal", capacity: 1024 },
	{ level: "low", capacity: 4096 },
	{ level: "medium", capacity: 8192 },
	{ level: "high", capacity: 16384 },
	{ level: "xhigh", capacity: 24576 },
	{ level: "max", capacity: Number.MAX_SAFE_INTEGER },
];

/** Levels this model really accepts. Mirrors pi's own rule for `thinkingLevelMap`. */
function supportedThinkingLevels(model: any): Level[] {
	if (!model?.reasoning) return ["off"];
	const map: Record<string, unknown> = model.thinkingLevelMap ?? {};
	return THINKING_LADDER.map((s) => s.level).filter((level) =>
		map[level] === null ? false : level === "xhigh" || level === "max" ? map[level] !== undefined : true,
	);
}

/** Translate a hand-written token budget into the smallest level that can hold it. */
function thinkingForBudget(model: any, budgetTokens: number, available: Level[]): Level {
	const want = Math.max(0, Math.floor(budgetTokens));
	const ladder = THINKING_LADDER.filter((s) => available.includes(s.level));
	return (ladder.find((s) => s.capacity >= want) ?? ladder[ladder.length - 1])?.level ?? "off";
}

/** What the request will actually ask for, and whether it differs from what the config says. */
function effectiveThinking(wanted: Level, model: any): { level: Level; note: string } {
	const available = supportedThinkingLevels(model);
	let level: Level;
	let note = "";
	level = available.includes(wanted) ? wanted : available[0];
	if (level !== wanted) note = ` (this model has no "${wanted}", so ${level})`;
	return { level, note };
}

// ---------------------------------------------------------------- rotation

type Candidate = { ref: string; model: any; profile: ModelOptions; kind: Kind };

/** The configured references turned into real models, in order; anything unknown is reported back. */
/** Why a configured summariser is not available right now, in words a person can act on. The whole
 *  story used to be "not in the model list", which names the place the model was missing from and never
 *  says what was there instead (2026-10-19: the local entry was correct, pi simply had the other GGUF
 *  loaded, and the rotation walked on to an online endpoint that then timed out). A local provider is a
 *  server that answers with the one model it has loaded, so saying which one is the whole answer. */
export function missingWhy(ref: string, ctx: { modelRegistry: any; model?: any }): string {
	const pool: any[] = [...(ctx?.modelRegistry?.getAvailable?.() ?? []), ...(ctx?.model ? [ctx.model] : [])].filter(Boolean);
	const refs = pool.map((m: any) => refOf(m)).filter(Boolean) as string[];
	// Where a ref stops naming the provider. pi writes "provider/id" and the id itself can hold slashes
	// (openrouter/deepseek/v4), while a local provider is "llama-server=http://host:port" and holds them
	// too - so neither "the last slash" (blamed a provider called "openrouter/deepseek") nor "the first
	// slash" (blamed "llama-server=http:") is right. A URL keeps its scheme and host; anything else cuts
	// at the first slash.
	const providerOf = (r: string): string => {
		const scheme = r.indexOf("://");
		if (scheme > 0) {
			const after = r.indexOf("/", scheme + 3);
			if (after > 0) return r.slice(0, after);
		}
		const slash = r.indexOf("/");
		return slash > 0 ? r.slice(0, slash) : r;
	};
	// The provider is taken to be the longest prefix that the models pi DOES have also start with; only
	// when nothing matches is the plain guess used.
	const prefixes = Array.from(new Set(refs.map(providerOf))).filter(Boolean);
	const provider =
		prefixes.filter((p) => ref === p || ref.startsWith(p + "/")).sort((a, b) => b.length - a.length)[0] ?? providerOf(ref);
	const offered = refs.filter((r) => r === provider || r.startsWith(provider + "/"));
	const short = (r: string) => (r.startsWith(provider + "/") ? r.slice(provider.length + 1) : r);
	const wanted = short(ref);
	// A local provider is "name=http://host:port" and shortName() would cut that to the port, so drop only
	// the scheme and let the line read the way the row on the settings board reads.
	const plabel = provider.replace(/^([\w.-]+=)https?:\/\//, "$1").replace(/\/+$/, "");
	if (!offered.length) {
		const others = Array.from(new Set(prefixes)).slice(0, 3);
		return others.length
			? `pi has nothing loaded from "${plabel}"; it offers ${others.join(", ")}`
			: "pi has no models loaded at all right now";
	}
	// The interesting case is the near miss: the same model under another id, or the same file under
	// another quant. Naming it says what the entry ought to read, which is what was missing on 2026-10-19.
	const tail = (r: string) => r.slice(r.lastIndexOf("/") + 1);
	const near = offered.find(
		(r) => r !== ref && (tail(r).toLowerCase() === wanted.toLowerCase() || (wanted.length > 12 && tail(r).toLowerCase().startsWith(wanted.toLowerCase().slice(0, 12)))),
	);
	if (near) return `${plabel} is loaded, this exact entry is not; pi calls it "${short(near)}"`;
	return `${plabel} is loaded, not "${wanted}"; it offers ${offered.slice(0, 2).map(short).join(", ")}${offered.length > 2 ? ` and ${offered.length - 2} more` : ""}`;
}

function buildRotation(cfg: Config, ctx: { modelRegistry: any; model?: any }): { candidates: Candidate[]; missing: string[] } {
	const candidates: Candidate[] = [];
	const missing: string[] = [];
	for (const ref of Object.keys(cfg.models).slice(0, MAX_SLOTS)) {
		const model = lookupRef(ref, ctx);
		const profile = profileFor(cfg, model, ref);
		if (!model) {
			missing.push(ref);
			continue;
		}
		candidates.push({ ref, model, profile, kind: isLocalModel(model) ? "local" : "online" });
	}
	return { candidates, missing };
}

function displayRef(ref: string, model: any): string {
	if (isSessionRef(ref)) return `session (pi's current model: ${shortName(model?.id ?? "unknown")})`;
	return ref;
}

type AttemptOutcome =
	| { ok: true; compaction: any; ref: string; model: any; kind: Kind; measured: number; target: number; min: number; max: number; words: number; seconds: number }
	| { ok: false; error: string };

/**
 * Ask each model in the rotation, in order, until one writes a summary that is neither too thin nor
 * too long. pi's own retry loop is handed to compact(), so a busy endpoint is retried inside the call
 * with the configured number of extra tries and spacing; every try is written to the log.
 */
export async function summarizeWithRotation(
	preparation: any,
	customInstructions: string | undefined,
	cfg: Config,
	ctx: any,
	signal: AbortSignal | undefined,
): Promise<{ ok: true; result: Extract<AttemptOutcome, { ok: true }> } | { ok: false; error: string }> {
	const { candidates, missing } = buildRotation(cfg, ctx);
	// Each skip now says what the provider offered instead of only what it lacked (2026-10-19, item (b)).
	for (const ref of missing) logLine(cfg, { event: "skip", ref, reason: missingWhy(ref, ctx).slice(0, 140) });
	if (!candidates.length) {
		return { ok: false, error: missing.length ? `no model found for ${missing.join(", ")}` : "no models configured" };
	}

	const prevTurn = readPrevTurnInfo(ctx, preparation);
	const failures: string[] = [];
	// The report on screen wants to name the models that were tried and lost, and `failures` is a local of
	// this function. Pointing a module-level name at the same array costs one line and no plumbing. It MUST
	// sit below the declaration: assigned above it, the read of `failures` is a temporal-dead-zone error,
	// which threw out of the compaction handler on 2026-10-22 and let pi write its own short summary.
	lastRotationFailures = failures;

	// The session model's window decides how big the summary may be; see postCompactionHeadroom.
	// A 29,000 token summary on 2026-09-16 pushed the next request past the window and killed the
	// session with "Request timed out", which is what that check exists to prevent.
	const activeWindow = ctx.model?.contextWindow ?? 0;
	// "Already overflowing" is decided from the provider's own count of the live context, not from the
	// size of the region being summarised. The two differ by the fixed request overhead, and using the
	// region size is what made an overflow compaction look like an ordinary one.
	const actual = realContextTokens(ctx);
	const reserveTokens = extReserve(ctx);
	const overflow = activeWindow > 0 && actual !== null && actual + reserveTokens >= activeWindow;
	const historyTokens = historyTokensOnly(preparation);
	const headroom = overflow ? 0 : postCompactionHeadroom(ctx, preparation);
	if (overflow) {
		logLine(cfg, { event: "info", note: "overflow compaction", actual, window: activeWindow, historyTokens: historyTokensOnly(preparation) });
	}
	logLine(cfg, {
		event: "sizes",
		reserveTokens,
		headroom,
	});

	for (const c of candidates) {
		// What this slot actually receives. Only an oversized slot gets a trimmed copy, so a normal
		// compaction is passed to pi completely untouched.
		let prep: any = preparation;
		const auth = await ctx.modelRegistry.getApiKeyAndHeaders(c.model);
		if (!auth?.ok) {
			logLine(cfg, { event: "skip", ref: c.ref, reason: `no credentials: ${auth?.error ?? "unknown"}` });
			failures.push(`${c.ref}: no credentials`);
			continue;
		}
		const profile = c.profile;
		// The summarizer's input, per this model's own options: informative stubs (pi's 2,000-char
		// head-only clip then has nothing important left to cut - the exit/tail message survives)
		// and capped tool-call argument values. The raw preparation is never mutated.
		const pre = preprocessForSummariser(prep, { ...profile, stubTailChars: cfg.elision.stubTailChars });
		prep = pre.prep;
		if (pre.stubbed || pre.argsCapped) logLine(cfg, { event: "preprocess", ref: c.ref, stubbed: pre.stubbed, argsCapped: pre.argsCapped, rawChars: pre.rawChars, keptChars: pre.keptChars, savedChars: pre.savedChars });
		// What the summariser is really sent: the flattened region (tool results clipped to 2,000
		// characters by pi) plus the previous summary. No system prompt, no tool definitions.
		const inputTokens = summariserInputTokens(prep);
		const sizes = sizesFor(profile, {
			inputTokens,
			historyTokens: historyTokensOnly(preparation),
			headroom: overflow ? 0 : postCompactionHeadroom(ctx, preparation),
			overflow,
			windowTokens: ctx?.model?.contextWindow ?? 0,
		}, prevSummaryTokens(preparation));
		for (const note of sizes.notes) logLine(cfg, { event: "note", ref: c.ref, note });

		const window = c.model.contextWindow ?? 0;
		// The region has to fit together with the whole reply this model is allowed to write.
		if (window > 0 && inputTokens + sizes.genCap + SUMMARISER_FRAMING_TOKENS > window) {
			// An overflow rescue is different: the history is on fire and we would rather send too much
			// and be refused than clip the summary into a stub. So trim the history until it fits, keeping
			// the newest messages (which the model has most recently seen and can least afford to lose),
			// and let the rest go unsummarised rather than skipping the slot. This is what saved the
			// 2026-09-22 case where the region was 107,424 tokens and the slot's window was 131,072.
			const room = window - sizes.genCap - SUMMARISER_FRAMING_TOKENS;
			if (!overflow || room < 8192) {
				logLine(cfg, { event: "skip", ref: c.ref, reason: `region ~${fmt(inputTokens)} tokens plus room for the reply does not fit a window of ${fmt(window)}` });
				failures.push(`${c.ref}: window too small`);
				continue;
			}
			const trimmed = trimToTokenBudget(preparation, room);
			if (!trimmed) {
				logLine(cfg, { event: "skip", ref: c.ref, reason: `region ~${fmt(inputTokens)} tokens cannot be trimmed under ${fmt(room)}` });
				failures.push(`${c.ref}: region too large to trim`);
				continue;
			}
			logLine(cfg, { event: "trim", ref: c.ref, from: inputTokens, to: room, window, note: "oldest messages dropped from the summariser input" });
			prep = trimmed;
		}

		const th = effectiveThinking(profile.thinking, c.model);
		// A local chat template can hold thinking on whatever the request asks for. When we asked for
		// thinking off, also say it inside the message, because that is where such a template looks.
		// Counted on the material that is actually going out, after any overflow trim, so the figure the
		// model is shown describes its real reading matter rather than a guess about a tokenizer's habits.
		// A server that answers nothing plausible falls back to the heuristic and the record says so.
		const regionForCount = summariserText(prep);
		const guessedRegion = Math.ceil((regionForCount.length * TOKEN_HEAVINESS) / 4);
		const countBase2 = localSummariserBaseUrl(c.model, c.ref);
		const countedRegion = countBase2 ? await countTextAt(countBase2, regionForCount, guessedRegion) : null;
		const realInput = (countedRegion?.n ?? guessedRegion) + SUMMARISER_FRAMING_TOKENS;
		const segmentBy2 = countedRegion ? countedRegion.how : countBase2 ? "estimate (this server gave no usable count)" : "estimate (not a local model)";
		const realSizes = prep === preparation ? sizes : sizesFor(profile, { inputTokens: realInput, historyTokens: historyTokensOnly(prep), headroom: overflow ? 0 : postCompactionHeadroom(ctx, prep), overflow, windowTokens: ctx?.model?.contextWindow ?? 0 }, prevSummaryTokens(prep));
		// The scratchpad share of the output budget rides on the cap here. It must NOT look at the built
		// instructions: those are assembled below this line, and reading them here sat in their temporal
		// dead zone - every compaction in this session threw "Cannot access 'instructions' before
		// initialization" and pi kept re-triggering on a context it could not compact. The allowance is
		// therefore unconditional now (the template asks for scratch work anyway) and is clamped by what
		// this model is permitted to emit, which is the only hard ceiling there is.
		// The fallback through pi's own compact() carries the same total allowance as a direct request:
		// Summary max and nothing beyond it. The thinking and scratch allowances are no longer added on
		// top (2026-10-20, the operator's design) - thinking is capped in the request body, so it spends
		// from inside the allowance, and a candidate that needs more than the allowance for both is a
		// candidate to move off rather than one to run past the window. dCeilingFallback is only the
		// tolerance the answer is judged against, never a licence to ask for more.
		const dCeilingFallback = realSizes.max + GEN_MARGIN;
		const scratch = scratchAllowanceFor(c.model, realSizes.genCap);
		const genCapFinal = realSizes.genCap;
		if (scratch.note) logLine(cfg, { event: "note", ref: c.ref, note: scratch.note });
		let instructions = buildInstructions({
			sizes: realSizes,
			inputTokens: realInput,
			segmentTokens: countedRegion?.n ?? summariserSegmentTokens(prep),
			segmentBy: segmentBy2,
			kind: c.kind,
			options: profile,
			additionalInstruction: cfg.additionalInstruction,
			userDefault: cfg.templateUserDefaults,
			customInstructions,
		});
		const started = Date.now();
		// The ceiling on a model that never stops writing, worked out against the aim rather than the
		// max: a summary that overruns the aim is judged later by the size checks, it is no reason to
		// cut a request that is still producing.
		const backstop = backstopMsFor(c.kind, realSizes.target, profile.timeoutMs, c.ref);
		logLine(cfg, {
			// ONE OPENING ROW, NOT TWO (2026-10-23): this is the row that says a compaction is happening,
			// which model is about to write it, and how full the session was when it started. The separate
			// COMPACTION START row is gone - on the crash loop of 2026-10-22 it was the only row written at
			// all, and it named no model, which is why the operator could not see what was happening.
			event: "attempt",
			trigger: lastCompactionTrigger,
			contextFill: realContextTokens(ctx),
			sessionWindow: ctx?.model?.contextWindow ?? 0,
			reserveTokens: extReserve(ctx),
			keepRecentTokens: extKeepRecent(ctx),
			setReserveTokens: cfg.scaling.reserveTokens,
			sessionModel: ctx.model ? refOf(ctx.model) : "?",
			// Which ceiling on the model's own thinking actually went out with this request (2026-10-20). The
			// settings value and the number the engine honoured are different claims, and only one of them is
			// in this line; that difference is what last night's forensics had to reconstruct by arithmetic.
			thinkingBudgetTok: refLooksLocal(c.ref) ? thinkingBudgetFor(th.level).budget : undefined,
			ref: c.ref,
			kind: c.kind,
			inputTokens: realInput,
			min: realSizes.min,
			target: realSizes.target,
			max: realSizes.max,
			minTok: realSizes.min,
			maxTok: realSizes.max,
			// Non-empty only for a template that declared its rules written out by hand: which of them the
			// rendered text then left no trace of. An ordinary run logs nothing here.
			absentRules: templateAbsentRules.length ? templateAbsentRules.join(" ") : undefined,
			window: ctx?.model?.contextWindow ?? 0,
			genCap: genCapFinal,
			thinking: th.level,
			// Two numbers with two meanings (2026-10-19): timeoutMs is how long to wait with NOTHING
			// arriving, backstopMs is how long to wait however much is arriving. Both are recorded, so a
			// fail line that names its clock can be checked against what that clock was set to.
			timeoutMs: profile.timeoutMs,
			backstopMs: backstop,
			extraTries: cfg.retries,
		});
		// Which clock stopped it, not merely that one did: silence means the request was dead, backstop
		// means the model was still writing when the ceiling arrived. dChars is the proof of production.
		let timedOut: "" | "silence" | "backstop" = "";
		let dChars = 0;
		const combined = silenceWatch(signal, {
			silenceMs: profile.timeoutMs,
			backstopMs: backstop,
			onTimeout: (why) => void (timedOut = why),
		});
		// --- the DIRECT request (pi-compact-plus): the whole request built here. On failure we move
		// to the NEXT candidate; pi's compact() runs only at the very end, after every candidate has
		// failed (or in legacy mode, directRequest off).
		if (cfg.directRequest) {
			try {
				const raw = await directCompact({
					preparation: prep,
					model: c.model,
					registry: ctx.modelRegistry,
					sizes: realSizes,
					options: profile,
					signal: combined.signal,
					thinking: th.level,
					prevTurn: prevTurn.lastTurn,
					additionalInstruction: cfg.additionalInstruction,
					userDefault: cfg.templateUserDefaults,
					customInstructions,
					ref: c.ref,
					kind: c.kind,
					// Every character restarts the silence clock and is counted, so a record about being cut
					// short can say how much had already arrived.
					onActivity: (n: number) => {
						dChars += n;
						combined.beat(n);
					},
				});
				const dSeconds = Math.round(((Date.now() - started) / 1000) * 10) / 10;
				const cleaned = stripRoughWork(raw.text);
				const summary: string = cleaned.summary;
				const words = countWords(summary);
				const u: any = raw.usage ?? {};
				let measured = measuredTokens(summary, u, c.ref);
				// Thinking on means the engine's output figure includes the hidden part, so the count comes
				// from the words instead - the same ruler as everywhere else, not a character guess.
				if (th.level !== "off") measured = measuredTokens(summary, null, c.ref);
				if (cleaned.strippedChars > 0 && measured.by === "usage") {
					measured = { tokens: Math.max(0, measured.tokens - Math.ceil((cleaned.strippedChars * 1.15) / 4)), by: "usage" };
				}
				const split = { output: u.output, reasoning: u.reasoning };
				const dFail = (reason: string): Error => {
					logLine(cfg, { event: "fail", note: "moving to the next candidate; pi's compact() runs only after every candidate failed", ref: c.ref, via: "direct", seconds: dSeconds, measured: measured.tokens, min: realSizes.min, max: realSizes.max, words, reason, ...split });
					// ONE ROW PER FAILED CANDIDATE (2026-10-23). The rich row written just above IS the record
					// of this failure; the catch below used to write a second, thinner one for the same thing,
					// and the log filled up with pairs of nearly identical lines.
					const dErrLogged: any = new Error(reason);
					dErrLogged.alreadyLogged = true;
					return dErrLogged;
				};
				if (timedOut)
				throw dFail(
						timedOut === "backstop"
							? `still writing when the ${secs(backstop)} ceiling arrived, after ${fmt(dChars)} characters had come in - half a summary either way`
							: `nothing arrived for ${secs(profile.timeoutMs)}, after ${fmt(dChars)} characters had come in - the request was cut as dead`,
				);
				if (!summary || summary.trim().length < 80) throw dFail("empty summary");
				if (measured.tokens < realSizes.min) throw dFail(`too thin - the summary is ~${fmt(measured.tokens)} tok vs the acceptance floor is ${fmt(realSizes.min)} tok absolute (aim ${fmt(realSizes.target)} tok, ceiling ${fmt(realSizes.max)} tok). Raise this model's Summary min, or lower the aim %.`);
				// The ceiling for REJECTION is the ceiling the request was actually capped at - max plus GEN_MARGIN
				// (2026-10-17). Throwing away twenty minutes of writing because a summary came in a hair over the
				// number we told the model to stay under was its own kind of waste: anything between max and the cap
				// it was given is a summary that fits the allowance, and it is still reported as such in the log.
				if (measured.tokens > realSizes.max + GEN_MARGIN) throw dFail(`too long - the summary is ~${fmt(measured.tokens)} tok vs the acceptance ceiling is ${fmt(realSizes.max)} tok absolute (aim ${fmt(realSizes.target)} tok). Raise this model's Summary max, or lower the aim %.`);
				// 2026-10-06: the accuracy checks no longer REJECT. A summary that breaks the shape rules,
				// paraphrases the tool calls or loses the previous-summary section is kept anyway - what is
				// written down here is what it did not do, and the next model is not asked. Only the length
				// limits above still reject: too thin or too long is unusable, everything else is usable but
				// imperfect, and re-uploading the whole context to try again costs more than it saves.
				const warnings: string[] = [];
				if (cfg.shapeGate) {
					const sh = shapeCheck(summary);
					if (!sh.ok) warnings.push(`shape: ${sh.reason}`);
				}
				// The STUB-COVERAGE check: with preserveStubs=verbatim the summary should carry the tool
				// calls, not narrate them. Count the stub markers the request carried vs the copies in the
				// summary; a model that paraphrases instead of copying is recorded as a warning (2026-10-06:
				// warned, not rejected).
				const inStubs = (String(raw.inputText ?? "").match(/output elided:/g) ?? []).length;
				const gotStubs = (summary.match(/output elided:/g) ?? []).length;
				if (profile.preserveStubs === "verbatim" && inStubs >= 4 && gotStubs < Math.ceil(inStubs * 0.8)) {
					warnings.push(`stub coverage ${gotStubs}/${inStubs} below 80% with preserveStubs=verbatim (the model paraphrased the tool calls)`);
				}
				// THE USER-PROMPT CHECK, measured against the material rather than the summary (2026-10-19).
				// Every model on this user's list is set to preserveUser = verbatim, so a ledger without the
				// user's own words is not a style - it is the loss of everything they asked for, and it was
				// passing: the summary that landed at 23:02 on 2026-10-18 carried one user line where the run
				// six hours earlier carried twelve, and the one at 00:50 quoted three out of some forty, said
				// "no Turn labels in the ledger" in its own shape warning, and was accepted anyway. Counting
				// turns inside the summary cannot see that case - with no labels there are no turns to be sad
				// about - so the denominator is the segment we sent: every prompt in it is marked
				// "[User - Turn N]" by serializeRegion, and every prompt quoted in the summary is a line whose
				// label says user. Both sides count only text of at least four letters, so a session run on
				// "ok" and "go on" cannot convict an honest summary.
				//
				// Nothing quoted is a rejection. Quoting under a quarter of what was handed over, in a segment
				// large enough for a quarter to be more than a rounding artefact, is a rejection too: the next
				// model in the rotation costs one more upload, and a session that has lost its instructions
				// costs the session. Anything short of that but still short is a warning, because a thin ledger
				// is usually a model that ran out of room rather than one that ignored the instruction.
				const upc = userPromptCoverage(summary);
				const given = userPromptsGiven(raw.inputText ?? "");
				const wantsPrompts = profile.preserveUser === "verbatim" || profile.preserveUser === "detailed";
				if (wantsPrompts && given >= 2 && upc.linesAnywhere === 0) {
					throw dFail(`no user prompts at all - the segment offered ${given} and the summary quotes none, while this model is asked to copy them verbatim`);
				}
				if (wantsPrompts && given >= 6 && upc.linesAnywhere * 4 < given) {
					throw dFail(`barely any user prompts - the segment offered ${given} and the summary quotes ${upc.linesAnywhere}, under a quarter of what it was handed`);
				}
				if (given >= 2 && upc.linesAnywhere < given) {
					warnings.push(`user prompts quoted in ${upc.linesAnywhere} of ${given} (${given - upc.linesAnywhere} not carried through)`);
				}
				if (upc.turns >= 2 && upc.withPrompt === 0 && upc.linesAnywhere > 0) {
					warnings.push(`the ledger's Turn entries carry no user line, though prompts are quoted elsewhere in the summary (${upc.linesAnywhere} of ${given})`);
				}
				// The UNDERSHOOT, visible now instead of buried in the log: a summary at a third of its aim left
				// most of the segment out, and the reader deserves to see that written down rather than to find
				// it a dozen requests later.
				if (realSizes.target > 0 && measured.tokens < realSizes.target * 0.6) {
					warnings.push(`undershot the aim: ${fmt(measured.tokens)} of ${fmt(realSizes.target)} tokens (${Math.round((100 * measured.tokens) / realSizes.target)}%)`);
				}
				// The CHAIN-ATTACH check: with chainMode=attach the previous summary should appear,
				// verbatim, in its own section. Warned about since 2026-10-06, never a rejection.
				if (profile.chainMode === "attach" && preparation.previousSummary?.trim() && !/^##\s*PREVIOUS SUMMARY\b/im.test(summary)) {
					warnings.push("chain attach: the PREVIOUS SUMMARY section is missing");
				}
				if (warnings.length) {
					logLine(cfg, { event: "accepted_with_warnings", ref: c.ref, via: "direct", measured: measured.tokens, min: realSizes.min, max: realSizes.max, warnings });
				}
				// What this model just proved about its own speed, in the units the backstop thinks in.
				if (dSeconds > 0 && measured.tokens > 0) observedRate.set(c.ref, measured.tokens / dSeconds);
				logLine(cfg, { event: "ok", ref: c.ref, via: "direct", kind: c.kind, tokPerSec: dSeconds > 0 ? Math.round((measured.tokens / dSeconds) * 10) / 10 : undefined, seconds: dSeconds, measured: measured.tokens, measuredBy: measured.by, segmentBy: raw.segmentBy, fromReasoning: raw.fromReasoning || undefined, roughWorkStripped: cleaned.strippedChars || undefined, min: realSizes.min, target: realSizes.target, max: realSizes.max, words, stubsIn: inStubs || undefined, stubsCopied: gotStubs || undefined, input: u.input, output: u.output, reasoning: u.reasoning });
				logLine(cfg, { event: "user_prompts", ref: c.ref, quoted: upc.linesAnywhere, given, ledgerTurns: upc.turns, ledgerQuoted: upc.withPrompt });
				const compaction = {
					summary: summary + fileOpsAppendix(preparation.fileOps),
					firstKeptEntryId: prep.firstKeptEntryId,
					tokensBefore: prep.tokensBefore,
					usage: u,
					details: { lastTurnNumber: Math.max(raw.lastTurn, maxTurnInSummary(summary)) },
				};
				combined.dispose();
				return { ok: true, result: { ok: true, compaction, ref: c.ref, model: c.model, kind: c.kind, measured: measured.tokens, target: realSizes.target, min: realSizes.min, max: realSizes.max, words, seconds: dSeconds, material: realInput - SUMMARISER_FRAMING_TOKENS, elided: elidedTokens(pre, realInput - SUMMARISER_FRAMING_TOKENS) } };
			} catch (dErr: any) {
				if (signal?.aborted) {
					logLine(cfg, { event: "note", note: "cancelled", ref: c.ref });
					return { ok: false, error: "cancelled" };
				}
				// `clock` and `chars` are what this line could not say on 2026-10-19, where the whole record was
					// error "aborted": the library swallows the reason our own clock gave, so the two numbers that
					// distinguish a dead request (a clock fired, nothing arriving) from a slow one (the ceiling
					// fired while characters were still coming) are recorded here instead of in its message.
				if (!dErr?.alreadyLogged) logLine(cfg, { event: "direct_fail", ref: c.ref, clock: timedOut || undefined, chars: dChars || undefined, error: String(dErr?.message ?? dErr).slice(0, 220), stop: dErr?.stopReason, inTok: dErr?.usage?.input, outTok: dErr?.usage?.output, reasonTok: dErr?.usage?.reasoning, note: "moving to the next candidate; pi's compact() runs only after every candidate failed" });
				failures.push(`${c.ref}: direct: ${String(dErr?.message ?? dErr).split("\n")[0].slice(0, 100)}`);
				combined.dispose();
				continue;
			}
		}
		try {
			const compaction = await compact(
				withBudget(prep, genCapFinal),
				c.model,
				auth.apiKey,
				auth.headers,
				instructions,
				combined.signal,
				th.level,
				samplingStreamFn(c.ref, profile.sampling, th.level),
				auth.env,
				{ enabled: cfg.retries > 0, maxRetries: cfg.retries, baseDelayMs: cfg.retryDelaySeconds * 1000 },
				{
					onRetryScheduled: (attempt: number, maxAttempts: number, delayMs: number, errorMessage: string) =>
						logLine(cfg, { event: "retry", ref: c.ref, attempt, maxAttempts, delayMs, error: String(errorMessage).slice(0, 300) }),
				},
				undefined,
			);
			const seconds = Math.round(((Date.now() - started) / 1000) * 10) / 10;
			// Rough work a model writes for itself (a scratchpad in <draft> tags) is discarded here, before measuring or
			// storing: the acceptance window judges the summary the next request will actually see.
			const cleaned = stripRoughWork(compaction?.summary ?? "");
			const summary: string = cleaned.summary;
			const words = countWords(summary);
			const u: any = compaction?.usage ?? {};
			let measured = measuredTokens(summary, u, c.ref);
			// Thinking on means the engine's output figure includes the hidden part, so the count comes
			// from the words instead - the same ruler as everywhere else, not a character guess.
			if (th.level !== "off") measured = measuredTokens(summary, null, c.ref);
			// That rough work rode inside the provider's output count; take its estimate back out so the
			// acceptance window measures only the summary that will be stored.
			if (cleaned.strippedChars > 0 && measured.by === "usage") {
				measured = { tokens: Math.max(0, measured.tokens - Math.ceil((cleaned.strippedChars * 1.15) / 4)), by: "usage" };
			}
			// Every rejection records the split between visible text and private reasoning, because a
			// model that thinks when it should not is otherwise invisible: it burns the whole budget
			// and returns nothing.
			const split = { output: u.output, reasoning: u.reasoning };
			// A summary cut off by our own timeout is half a summary and must not be kept.
			// (pi already throws when the model stops at its token ceiling.)
			if (timedOut) {
				// pi's own compact() path keeps the stream inside the library, so nothing can report
				// production there and only the silence clock can fire - the old behaviour, with a truer sentence.
				logLine(cfg, {
					event: "fail",
					ref: c.ref,
					seconds,
					measured: measured.tokens,
					reason:
						timedOut === "backstop"
							? `still writing when the ${secs(backstop)} ceiling arrived`
							: `nothing arrived for ${secs(profile.timeoutMs)}`,
					timeoutMs: profile.timeoutMs,
					backstopMs: backstop,
					...split,
				});
				failures.push(`${c.ref}: nothing finished within ${secs(timedOut === "backstop" ? backstop : profile.timeoutMs)}`);
				continue;
			}
			if (!summary || summary.trim().length < 80) {
				logLine(cfg, { event: "fail", ref: c.ref, seconds, reason: "empty summary", ...split });
				failures.push(`${c.ref}: empty summary`);
				continue;
			}
						if (measured.tokens < realSizes.min) {
				logLine(cfg, { event: "fail", ref: c.ref, seconds, measured: measured.tokens, min: realSizes.min, max: realSizes.max, words, reason: "too thin - well under the minimum and would leave the next turn blind", ...split });
				failures.push(`${c.ref}: ~${fmt(measured.tokens)} tok, needed at least ${fmt(realSizes.min)}`);
				continue;
			}
			if (measured.tokens > realSizes.max + GEN_MARGIN) {
				logLine(cfg, { event: "fail", ref: c.ref, seconds, measured: measured.tokens, min: realSizes.min, max: realSizes.max, words, reason: "too long - over the accepted ceiling", ...split });
				failures.push(`${c.ref}: ~${fmt(measured.tokens)} tok, ceiling is ${fmt(realSizes.max)}`);
				continue;
			}
			// The same lesson from pi's own compact() path, where the timing is all we get.
			if (seconds > 0 && measured.tokens > 0) observedRate.set(c.ref, measured.tokens / seconds);
			logLine(cfg, {
				event: "ok",
				ref: c.ref,
				kind: c.kind,
				tokPerSec: seconds > 0 && measured.tokens > 0 ? Math.round((measured.tokens / seconds) * 10) / 10 : undefined,
				seconds,
				measured: measured.tokens,
				measuredBy: measured.by,
				roughWorkStripped: cleaned.strippedChars || undefined,
				min: realSizes.min,
				target: realSizes.target,
				max: realSizes.max,
				words,
				input: u.input,
				output: u.output,
				reasoning: u.reasoning,
			});
			return {
				ok: true,
				result: { ok: true, compaction: { ...compaction, summary }, ref: c.ref, model: c.model, kind: c.kind, measured: measured.tokens, input: u.input, output: u.output, target: realSizes.target, min: realSizes.min, max: realSizes.max, words, seconds, material: realInput - SUMMARISER_FRAMING_TOKENS, elided: elidedTokens(pre, realInput - SUMMARISER_FRAMING_TOKENS) },
			};
		} catch (err: any) {
			const seconds = Math.round(((Date.now() - started) / 1000) * 10) / 10;
			const message = String(err?.message ?? err).slice(0, 400);
			logLine(cfg, { event: "fail", ref: c.ref, seconds, error: message });
			failures.push(`${c.ref}: ${message.split("\n")[0].slice(0, 120)}`);
			if (signal?.aborted) return { ok: false, error: "cancelled" };
		} finally {
			combined.dispose();
		}
	}
	return { ok: false, error: failures.join(" | ") || "every model in the rotation failed" };
}


// ---------------------------------------------------------------- small display + key helpers

/** A menu option is a title plus one explaining line (used by the fallback menus). */
type Row = { label: string; hint?: string; run: () => void | Promise<void> };

function opt(title: string, hint: string): string {
	return hint ? `${title}\n${hint}` : title;
}

/** The names this file uses for its keys, and pi's name for the same key. pi's parser answers in its
 *  own vocabulary; the rest of the file was written around these shorter names, so they are translated
 *  here rather than renaming every comparison in every screen. */
const KEY_ALIASES: [string, string][] = [
	["escape", "escape"],
	["enter", "enter"],
	["up", "up"],
	["down", "down"],
	["left", "left"],
	["right", "right"],
	["home", "home"],
	["end", "end"],
	["pageUp", "pgup"],
	["pageDown", "pgdn"],
	["delete", "delete"],
	["backspace", "backspace"],
];

/**
 * Which key is this keystroke? 2026-10-21, and this is the fix for "no key works in the settings".
 *
 * This function used to compare bytes: Escape was the single character 0x1b, Up was the four
 * characters ESC [ A, Ctrl+A was the byte 0x01. That stopped being true when pi turned on the Kitty
 * keyboard protocol (pi 1.3.0, and Windows Terminal accepts it): a keystroke now arrives as an escape
 * sequence, a held key sends release and repeat events between the presses, and a lone Escape is
 * encoded as well. Every hand-decoded comparison in the file went quiet in the same moment, which is
 * exactly what the operator reported - Ctrl+ shortcuts dead, Escape not leaving a screen, holding an
 * arrow registering only the first press.
 *
 * The encoding is pi's business now, so pi decodes it: matchesKey() knows both the old sequences and
 * the new ones. A key release is answered with "" so the screens ignore it, otherwise a held arrow
 * would step twice per repeat, once for the press and once for the release.
 */
function keyOf(data: string): string {
	if (!data) return "";
	try {
		if (isKeyRelease(data)) return "";
	} catch {
		/* an older pi without the release query: nothing to skip */
	}
	for (const [piName, ours] of KEY_ALIASES) {
		try {
			if (matchesKey(data, piName as never)) return ours;
		} catch {
			/* a name this pi does not know: fall through to the byte comparison below */
		}
	}
	// Still worth answering the plain characters directly: a terminal that never negotiated the new
	// protocol, and a unit test that hands us bytes, both arrive here.
	if (data === "\x1b") return "escape";
	if (data === "\r" || data === "\n") return "enter";
	if (data === " ") return "space";
	if (data === "\x7f" || data === "\b") return "backspace";
	if (data.startsWith("\x1b[")) {
		switch (data.slice(0, 4)) {
			case "\x1b[A":
				return "up";
			case "\x1b[B":
				return "down";
			case "\x1b[C":
				return "right";
			case "\x1b[D":
				return "left";
			case "\x1b[H":
				return "home";
			case "\x1b[F":
				return "end";
		}
		if (data === "\x1b[5~") return "pgup";
		if (data === "\x1b[6~") return "pgdn";
		if (data === "\x1b[3~") return "delete";
		return "";
	}
	if (data.length === 1) return data.toLowerCase();
	return "";
}

/** Is this keystroke this control combination, in whichever encoding the terminal is using today?
 *  Wraps pi's matcher so the screens read as sentences and an older pi cannot throw. */
function chord(data: string, ...piNames: string[]): boolean {
	for (const n of piNames) {
		try {
			if (matchesKey(data, n as never)) return true;
		} catch {
			/* unknown name for this pi: try the next one */
		}
	}
	return false;
}

function cut(text: string, width: number): string {
	if (width <= 0) return "";
	return text.length <= width ? text : text.slice(0, Math.max(0, width - 1)) + "…";
}

/** A model's short name in a menu row (no model object needed, so it works for models that are not
 *  the one pi is running). */
function menuModelLabel(ref: string): string {
	if (!ref) return "(empty)";
	const tail = ref.includes("/") ? ref.slice(ref.lastIndexOf("/") + 1) : ref;
	return tail.replace(/\.gguf$/i, "").slice(0, 34);
}

function slotLabel(ref: string | null, model: any): string {
	if (!ref) return "(empty)";
	if (!isSessionRef(ref)) return ref;
	const raw = String(model?.id ?? "the model pi is running");
	const short = raw.replace(/^.*[\\/]/, "").replace(/\.gguf$/i, "");
	return `${SESSION_REF} — ${short}`;
}
// ---------------------------------------------------------------- option cycles

/** Timeout periods in the exact order they toggle. 0 = unlimited. */
function nextValue(values: string[], current: string): string {
	const i = values.indexOf(current);
	return values[(i + 1) % values.length];
}

/** Seconds as the menu shows them ("2 min" ... "unlimited"). */
function timeoutLabel(ms: number): string {
	if (!ms) return "unlimited";
	const s = Math.round(ms / 1000);
	if (s % 60 === 0) return `${s / 60} min`;
	return `${s} s`;
}

// ---------------------------------------------------------------- per-model settings

/**
 * The internal defaults for a model that was never selected: it is only ever asked with these if
 * it is put in the Compaction models list and the user never opens its submenu.
 */
/**
 * The options a model gets when it JOINS the try order (2026-10-21, defect #53).
 *
 * Every new model started with every sampling flag on "ignore". For an online model that is right - the
 * provider's server already knows how that model should be driven. For a local Qwen it is wrong in a way
 * the operator pays for on the next compaction: the server default is a chat-tuned temperature with no
 * top_k and no min_p, which is how a local model produces a wandering summary. Qwen publishes instruct
 * values, so a model whose name says Qwen joins with those, and every other model joins exactly as before.
 *
 * A model already in the config is never touched - including one whose flags the operator set back to
 * ignore on purpose. This is a starting point, shown on the Sampling flags page and changed there.
 */
export function defaultOptionsFor(ref: string): ModelOptions {
	const o: ModelOptions = { ...DEFAULT_OPTIONS, sampling: { ...SAMPLING_DEFAULTS } };
	if (/qwen/i.test(ref)) o.sampling = { ...QWEN_INSTRUCT_SAMPLING };
	return o;
}

export const DEFAULT_OPTIONS: ModelOptions = {
	sampling: { ...SAMPLING_DEFAULTS },
	thinking: "off",
	timeoutMs: 600_000, // 10 min
	summaryPercent: 10,
	summaryMinTokens: 4096,
	summaryMaxTokens: 30_000,
	inputStubs: true,
	argCap: 500,
	preserveStubs: "verbatim",
	preserveUser: "verbatim",
	preserveReplies: "detailed",
	preserveThinking: "summary",
	chainMode: "fold",
	noThinkMarker: "",
};

export function optionsFor(cfg: Config, ref: string, model: any): ModelOptions {
	// ONE source of truth: the same normaliser the config loader uses. (The 2026-10-05 breakage was
	// exactly this function keeping its own copy of the size fields: it went on writing the old
	// percent keys, so every menu edit was saved, read back as a field nobody read and silently lost.)
	const stored = cfg.models[ref];
	const local = isLocalModel(model ?? {});
	const o = modelOptionsFrom(stored, local ? cfg.noThinkMarker : "", local ? "off" : "full");
	// A pre-2026-10-05 config stored the limits as percents of the window: convert them with THIS
	// model's window, so a stored "min 5%" on a 192k model becomes 9,600 tokens, not NaN.
	const win = Number(model?.contextWindow);
	const lmin = legacyPctToTokens(o.legacyMinPct, win);
	if (lmin !== undefined) o.summaryMinTokens = lmin;
	const lmax = legacyPctToTokens(o.legacyMaxPct, win);
	if (lmax !== undefined) o.summaryMaxTokens = lmax;
	o.summaryMinTokens = tok(o.summaryMinTokens, DEFAULT_OPTIONS.summaryMinTokens);
	o.summaryMaxTokens = Math.max(o.summaryMinTokens, tok(o.summaryMaxTokens, DEFAULT_OPTIONS.summaryMaxTokens));
	o.summaryPercent = pctField(o.summaryPercent, DEFAULT_OPTIONS.summaryPercent);
	delete o.legacyMinPct;
	delete o.legacyMaxPct;
	return o;
}

/** Selected models get their stored options; the rest display (and would use) the defaults. */
function optionsOfAnyModel(cfg: Config, ref: string, model: any): ModelOptions {
	if (cfg.models[ref]) return optionsFor(cfg, ref, model);
	return { ...DEFAULT_OPTIONS };
}

/**
 * The settings that stopped being per-model on 2026-10-06: the summary aim and its floor and ceiling,
 * what goes into the summarizer input, what the summary keeps, the chain mode and the draft block.
 * Editing one of them writes EVERY model in the try order at once - they are one global setting now.
 * What stays per-model: the thinking level, the no-think tag, the timeout and the sampling flags.
 */
export const GLOBAL_MODEL_KEYS = [
	"summaryPercent", "summaryMinTokens", "summaryMaxTokens", "inputStubs", "argCap",
	"preserveStubs", "preserveUser", "preserveReplies", "preserveThinking", "chainMode",
] as const;

function setModelOptions(ref: string, patch: Partial<ModelOptions>): void {
	const shared: Partial<ModelOptions> = {};
	const mine: Partial<ModelOptions> = {};
	for (const [k, v] of Object.entries(patch) as [string, unknown][]) {
		if ((GLOBAL_MODEL_KEYS as readonly string[]).includes(k)) (shared as Record<string, unknown>)[k] = v;
		else (mine as Record<string, unknown>)[k] = v;
	}
	update((c) => {
		// One edit, every model: the global settings are stored the same way on every entry, so the
		// per-model map keeps only what is genuinely per-model.
		if (Object.keys(shared).length > 0) {
			for (const r of Object.keys(c.models)) c.models[r] = { ...c.models[r] ?? {}, ...shared };
		}
		if (Object.keys(mine).length > 0) {
			const prev = c.models[ref] ?? {};
			c.models[ref] = { ...prev, ...mine };
		}
	});
}

function rotateAdd(ref: string, options?: ModelOptions): void {
	update((c) => {
		if (c.models[ref]) return;
		c.models[ref] = options ?? defaultOptionsFor(ref);
	});
}

function rotateRemove(ref: string): void {
	update((c) => {
		delete c.models[ref];
	});
}

/** THE ONE-COMPACTOR RULE (2026-10-07, on request). A second context manager in the same session is not
 *  a cosmetic problem: pi has no plugin priority mechanism, the `context` event is a pipeline, and two
 *  compressors rewrite the same message list and clobber each other's work - the other manager's own
 *  README says to keep exactly one installed. So this extension refuses to switch itself ON while another
 *  compaction extension is present, and says what to do about it. Detection, strongest first: the
 *  sidecar file the other manager writes next to the LIVE session log (proof it is running, not just
 *  installed), pi's `packages` list in settings.json, and the extensions / package folders. */
const OTHER_COMPACTOR_NAMES = ["billion-context", "acp-context", "pi-compactor", "pi-compact-pro", "context-mode", "pi-context-manager", "pi-compressor"];

function otherCompactorDetected(): string | null {
	try {
		const sf = (process.env.PI_SESSION_FILE ?? "").trim();
		if (sf) {
			statSync(`${sf}.acp.json`);
			return "billion-context-pi - its sidecar is next to this session's log, so it is running here";
		}
	}
	catch { /* no sidecar: not running on this session (statSync throws when the file is missing) */ }
	try {
		const raw = JSON.parse(readFileSync(join(AGENT_DIR, "settings.json"), "utf8"));
		const pkgs: unknown[] = Array.isArray(raw?.packages) ? raw.packages : [];
		for (const p of pkgs) {
			const s = String(p).toLowerCase();
			for (const n of OTHER_COMPACTOR_NAMES) if (s.includes(n)) return String(p);
		}
	}
	catch { /* settings.json unreadable: nothing detected from it */ }
	for (const dir of [join(AGENT_DIR, "extensions"), join(AGENT_DIR, "npm", "node_modules")]) {
		try {
			for (const e of readdirSync(dir)) {
				const s = String(e).toLowerCase();
				for (const n of OTHER_COMPACTOR_NAMES) if (s.includes(n)) return `${e} (in ${dir})`;
			}
		}
	catch { /* folder missing */ }
	}
	return null;
}

/** THE GATE used by every path that would switch this extension's compaction ON: the menu row, the
 *  settings screen, the fallback board and the /compact-plus on command. Returns the name of the other
 *  manager when the switch must be refused (and says so on screen + in the log), null when it may go on. */
// Exported for the safety-net rig (TMP/pcp_nets.mts): these paths have saved the session twice and
// have never once been seen running, which is worth more than a comment saying they look right.
export function enableBlockedByOtherCompactor(): string | null {
	if (loadConfig().enabled) return null; // already on: nothing is being enabled here
	const other = otherCompactorDetected();
	if (!other) return null;
	try {
		logLine(loadConfig(), { event: "enable_refused", note: `another context manager is present: ${other}` });
	}
	catch { /* the log must never block a menu */ }
	watcherCtx?.ui?.notify?.(
		`NOT ENABLED - another context manager is present (${other}). Only ONE compaction extension can run in a session: they rewrite the same context and destroy each other's work. Disable that one first - take it out of "packages" in pi's settings.json (or out of the extensions folder) and /reload - then switch this extension on.`,
		"warning",
	);
	return other;
}

/** The startup half of the rule: the config can say `enabled: true` while another manager is installed
 *  (a hand edit, a config copied from another machine, the other manager installed afterwards). Warn
 *  once per session and say what to switch off; do NOT write anybody's settings behind the user's back. */
let otherCompactorWarned = false;
export function warnOnceAboutTwoCompactors(ctx: any): void {   // exported for the safety-net rig
	if (otherCompactorWarned) return;
	otherCompactorWarned = true;
	try {
		if (!loadConfig().enabled) return;
		const other = otherCompactorDetected();
		if (!other) return;
		logLine(loadConfig(), { event: "two_compactors", note: `this extension is enabled while another context manager is present: ${other}` });
		ctx?.ui?.notify?.(
			`WARNING - TWO COMPACTORS: this extension's compaction is ON while ${other} is present. They rewrite the same context and destroy each other's work. Switch this extension OFF (with both switches off it is completely inert and touches nothing) - or remove the other manager and /reload.`,
			"warning",
		);
	}
	catch { /* a warning never breaks a request */ }
}

function toggleEnabled(): boolean {
	if (enableBlockedByOtherCompactor()) return false;
	update((c) => void (c.enabled = !c.enabled));
	return true;
}

function cycleRetries(dir: number): void {
	update((c) => {
		c.retries = clamp(c.retries + dir, 0, 5);
	});
}

function cycleRetryWait(dir: number): void {
	const steps = [3, 5, 10, 15, 20, 30, 60];
	const current = steps.reduce((best, s) => (Math.abs(s - c_retryWait()) < Math.abs(s - best) ? s : best), steps[0]);
	function c_retryWait(): number {
		return loadConfig().retryDelaySeconds;
	}
	const i = steps.indexOf(current);
	update((c) => void (c.retryDelaySeconds = steps[clamp(i + dir, 0, steps.length - 1)]));
}

function cycleNoThinkMarker(): void {
	update((c) => {
		c.noThinkMarker = c.noThinkMarker ? "" : DEFAULT_NO_THINK_TAG;
	});
}

/** Live reserveTokens from pi's own compaction settings (falls back to pi's documented default). */
/**
 * WHERE AUTOMATIC COMPACTION FIRES, in tokens, for one model window (2026-10-21).
 *
 * The operator sets a trigger point in tokens. That number alone is unsafe, because a model with a
 * smaller window runs out of room long before it: a 139,400 target on a 64,000-token model would
 * simply never fire. So the point is the EARLIER of two numbers - the target, and the point this
 * model has to compact at anyway.
 *
 *
	* pi's ask point is NOT part of this arithmetic any more (2026-10-22, the operator's decision). It
	* used to cap the line, because a pi ask held below this extension's point costs a full pass over
	* the context on every turn in between. The timer is the trigger now, so there is no ask to hold;
	* the alarm is set to ring above this line, and the gate in the hook is left for the drift case
	* (pi's reserveTokens edited while pi runs, a different model loaded), where it says so.
 */
/** The reserve used when the operator set no number: a tenth of the window, never less than a
	 * normal summary plus a tail. Derived on every call, so it follows the model in use.
 */
export function autoReserveFor(window: number): number {
	const win = Math.floor(window || 0);
	if (!(win > 0)) return DEFAULT_RESERVE_TOKENS;
	return Math.max(16_384, Math.round(win * 0.1));
}

/**
	* WHERE automatic compaction fires, for any window: the window minus the reserve the operator set.
	*
	* This is the extension's own decision, and since 2026-10-22 the operator's own timer is what acts
	* on it (option A, their choice): the timer measures the context from the usage figures the server
	* already reported - which costs nothing to look at - and starts a compaction at this line, at a
	* safe point, while the session is running. pi's threshold is the ALARM: it lives above this line,
	* rings only if the timer failed, and this extension still answers it with its own summary.
	*
	* pi's own reserveTokens therefore no longer moves this line. It used to: the point was capped at
	* pi's ask point so that an ask could never be held. The cap was honest arithmetic - a held ask costs
	* one full pass over the context per turn - but it meant a smaller-context model compacted at pi's
	* number instead of the one on the board, and the operator rejected that trade (2026-10-22). With
	* the timer as the trigger there is nothing to hold: the line is the number on the board.
 */
export function triggerPointFor(o: { window: number; reserveTokens: number }): number {
	const win = Math.floor(o.window || 0);
	if (!(win > 0)) return 0;
	const reserve = o.reserveTokens > 0 ? Math.floor(o.reserveTokens) : autoReserveFor(win);
	// The only cap is arithmetic: a line with no room above it would fire on every request, so a
	// reserve smaller than 2% of the window is widened to 2%. Nothing to do with pi's settings.
	const minReserve = Math.max(2_048, Math.round(win * 0.02));
	return Math.max(1024, win - Math.max(reserve, minReserve));
}

/** The same for the model this session is using right now. */
export function triggerPoint(ctx: Ctx): number {
	const window = (ctx as any)?.model?.contextWindow ?? 0;
	if (!(window > 0)) return 0;
	return triggerPointFor({ window, reserveTokens: loadConfig().scaling.reserveTokens });
}

/**
 * The room the trigger point leaves free at the top of the window - the summary plus the messages
 * that follow it. Derived, never set: it is window minus the trigger point, and both settings and
 * every size share read it. Recomputed on every call, so a model switch moves the line at once.
 */
export function extReserve(ctx: Ctx): number {
	const window = (ctx as any)?.model?.contextWindow ?? 0;
	if (!(window > 0)) return DEFAULT_RESERVE_TOKENS;
	return Math.max(1024, window - triggerPoint(ctx));
}

/** The verbatim tail: what compaction keeps word for word and what elision protects. */
export function extKeepRecent(ctx: Ctx): number {
	const window = (ctx as any)?.model?.contextWindow ?? 0;
	if (!(window > 0)) return DEFAULT_KEEP_RECENT_TOKENS;
	// A tail bigger than the room available is nonsense; the window is the only limit worth having.
	return clamp(loadConfig().scaling.preserveTokens, 1024, Math.max(1024, window - extReserve(ctx)));
}

/** Short effective-settings line for a model row. */
function modelRowValue(cfg: Config, ref: string, model: any, reserveTokens: number): string {
	const o = optionsOfAnyModel(cfg, ref, model);
	const place = orderPosition(cfg, ref);
	const pos = place >= 0 ? `#${place + 1}` : "—";
	// 2026-10-07: the right-hand text is the four options this model's screen really holds - thinking,
	// no-think tag, timeout, sampling flags. It used to repeat aim/min/max, which are no longer
	// per-model settings and made the column read as garbage.
	const tag = o.noThinkMarker && o.noThinkMarker.trim() ? "on" : "off";
	return `${pos} · thinking ${thinkingWord(o.thinking)} · no-think tag ${tag} · timeout ${timeoutLabel(o.timeoutMs)} · sampling ${samplingLabel(o.sampling)}`;
}

function modelRowLabel(placeInOrder: number, ref: string, model: any): string {
	const name = model ? (isSessionRef(ref) ? `session (${shortName(String(model.id))})` : isLocalModel(model) ? shortName(String(model.id)) : String(model.id)) : ref;
	const badge = placeInOrder >= 0 ? `[${placeInOrder + 1}]` : "   ";
	return `${badge} ${name}`;
}


/** Put the current value into a cycle list so the toggle starts from it. */
function withCurrent(values: string[], current: string): string[] {
	const c = String(current);
	return values.includes(c) ? values : [c, ...values];
}

/** "2 min" / "45 s" / "unlimited" -> milliseconds (0 for unlimited). */
function parseDuration(text: string): number {
	const t = String(text).trim().toLowerCase();
	if (!t || t === "unlimited" || t === "0") return 0;
	if (t.includes("min")) return Math.round(Number(t.replace(/[^0-9.]/g, "")) * 60_000) || 0;
	if (t.endsWith("s")) return Math.round(Number(t.replace(/[^0-9.]/g, "")) * 1000) || 0;
	return Math.round(Number(t) * 1000) || 0;
}

/** Move a model of the list to a position (0 = first), pushing the others back. A place beyond the
 *  end lands on the end. */
function rotateTo(ref: string, place: number): void {
	update((c) => {
		const keys = Object.keys(c.models);
		const i = keys.indexOf(ref);
		if (i < 0) return;
		const target = Math.min(place, keys.length - 1);
		if (target === i) return;
		const ordered = keys.slice();
		ordered.splice(i, 1);
		ordered.splice(target, 0, ref);
		const next: Record<string, ModelOptions> = {};
		for (const k of ordered) next[k] = c.models[k];
		c.models = next;
	});
}

/** Move a model of the list up or down one place in the try order. */
function rotateMove(ref: string, dir: number): void {
	update((c) => {
		const keys = Object.keys(c.models);
		const i = keys.indexOf(ref);
		if (i < 0) return;
		const j = clamp(i + dir, 0, keys.length - 1);
		if (j === i) return;
		const ordered = keys.slice();
		ordered.splice(i, 1);
		ordered.splice(j, 0, ref);
		const next: Record<string, ModelOptions> = {};
		for (const k of ordered) next[k] = c.models[k];
		c.models = next;
	});
}

/** An explicit "provider/model" command argument - always contains a slash. */
function refArg(parts: string[]): string | null {
	const last = parts[parts.length - 1];
	if (!last || !last.includes("/")) return null;
	return last;
}
/**
 * The models to offer: the ones pi itself offers in /model (models whose provider has a working
 * login), plus the model this session is running, plus anything already in the list so it can always
 * be seen and taken out.
 */
function poolOf(ctx: Ctx): { ref: string; model: any }[] {
	const cfg = loadConfig();
	const seen = new Map<string, any>();
	const available: any[] = [...(ctx.modelRegistry?.getAvailable?.() ?? [])];
	if (ctx.model && !available.some((m: any) => m && refOf(m) === refOf(ctx.model))) available.push(ctx.model);
	for (const m of available) {
		if (m) seen.set(refOf(m), m);
	}
	for (const ref of Object.keys(cfg.models)) {
		if (!seen.has(ref)) seen.set(ref, lookupRef(ref, ctx));
	}
	// One row per model. If a ref in the list (for example "session") points at a model that is also
	// in the available list, keep the ref you already use instead of showing it twice.
	const entries = [...seen.entries()].map(([ref, model]) => ({ ref, model }));
	const keep = new Map<string, { ref: string; model: any }>();
	for (const e of entries) {
		const key = e.model ? refOf(e.model) : `ref:${e.ref}`;
		const prev = keep.get(key);
		if (!prev || (!cfg.models[prev.ref] && cfg.models[e.ref])) keep.set(key, e);
	}
	const list = [...keep.values()];
	const selected = Object.keys(cfg.models);
	list.sort((a, b) => {
		const ia = selected.indexOf(a.ref);
		const ib = selected.indexOf(b.ref);
		if (ia >= 0 && ib >= 0) return ia - ib;
		if (ia >= 0) return -1;
		if (ib >= 0) return 1;
		return a.ref.localeCompare(b.ref);
	});
	return list;
}

/** The try order, human-readable: "[1] gemini-3.5-flash-lite → [2] Qwen3.8-Flash-Next". */
function rotationText(cfg: Config, ctx?: any): string {
	const parts = Object.keys(cfg.models)
		.slice(0, MAX_SLOTS)
		.map((r, i) => {
			const model = ctx ? lookupRef(r, ctx) : undefined;
			return `[${i + 1}] ${model ? (isLocalModel(model) ? shortName(String(model.id)) : String(model.id)) : shortName(r.split("/").slice(1).join("/")) || r}`;
		});
	return parts.length ? parts.join(" → ") : "(empty - pi uses the session model)";
}

// ---------------------------------------------------------------- TUI: compaction models (order list)

/**
 * The Compaction models screen: one list holding the selected models first (with their position
 * badge) and every other model pi can reach below them, alphabetically.
 *
 * Keys:
 *   digits 1-9   on a SELECTED model: move it to that position (0 = the end); on an UNSELECTED
 *                model: filter the list (names contain digits)
 *   space        on a SELECTED model: remove it from the order (it drops to its alphabetical
 *                place among the unselected; its other settings are kept); on an UNSELECTED
 *                model: a space in the filter
 *   enter        SELECTED model: open its options submenu; UNSELECTED model: add it to the END
 *                of the order with default options
 *   type         filter the list
 *   Esc          back to the board
 */
export class OrderList implements Component, Focusable {
	private list: SettingsList | null = null;
	private items: SettingItem[] = [];
	private isFocused = false;
	/** True while a model's options screen has replaced the list: all input goes to it. */
	private inSubmenu = false;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		private done: (selectedValue?: string) => void,
		private restore?: { optionsRef?: string; optionKey?: string; subKey?: string },
		/** When set, the model options screen shows only these rows (the 2026-10-06 split). */
		private filter?: readonly string[],
	) {
		this.rebuild();
		if (this.restore?.optionsRef) {
			this.highlightRef(this.restore.optionsRef);
			this.openHighlighted(this.restore.optionKey, this.restore.subKey);
		}
	}

	// -- Component -------------------------------------------------------------

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
		// 2026-10-20, learned the hard way with a keystroke harness (TMP/pcp_keys.mts): the model options
		// screen is mounted by SWAPPING this.list for it, not by the list widget's own submenu, so the
		// widget reports no submenu for the whole time the options screen is legitimately open. A guard here
		// that trusted widget.submenuComponent released the capture on the first keystroke and threw the
		// operator straight back out of the options screen - the harness caught that before it was ever
		// loaded. If this capture is ever hardened, it needs its own bookkeeping (keep the child screen and
		// clear the flag when it goes away), not the widget's state.
		// While a model's options screen is open it owns the keyboard wholesale - digits and space
		// belong to its own rows there, not to the list underneath.
		if (this.inSubmenu) {
			this.list?.handleInput(data);
			this.tui.requestRender();
			return;
		}
		// Digits and Space act on SELECTED models only (reposition / remove from the order). On an
		// UNSELECTED model they pass through to the list's filter: model names contain digits
		// ("4.1", "3.5"), and typing them must filter, not silently add the highlighted model.
		// (Bug fix 2026-10-01: digits used to reposition-or-add whatever was highlighted, so a
		// digit typed to filter a name ended up adding some unrelated model to the order.)
		const k = keyOf(data);
		if ((k >= "1" && k <= "9") || k === "0" || k === "space") {
			const ref = this.highlightRef();
			if (ref && loadConfig().models[ref]) {
				if (k === "space") this.unselect();
				else this.reposition(k === "0" ? Number.MAX_SAFE_INTEGER : Number(k) - 1);
				return;
			}
			this.list?.handleInput(data);
			this.tui.requestRender();
			return;
		}
		if (k === "enter") {
			this.openHighlighted();
			return;
		}
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	/** Enter: a SELECTED model opens its options submenu; an UNSELECTED model joins the END of the
	 *  order with default options (rotateAdd appends). */
	private openHighlighted(initialKey?: string, openSub?: string): void {
		const ref = this.highlightRef();
		if (!ref) return;
		const cfg = loadConfig();
		if (!cfg.models[ref]) {
			rotateAdd(ref);
			this.rebuild();
			this.highlightRef(ref);
			this.ctx.ui?.notify?.(`${ref.split("/").pop()} added to the Compaction models list (last place).`, "info");
			return;
		}
		const model = lookupRef(ref, this.ctx);
		// Swap this list for the model's options screen; Esc in it rebuilds this list.
		const parent = this;
		this.inSubmenu = true;
		const screen = modelOptionsScreen(
			this.ctx,
			ref,
			model,
			() => {
				parent.inSubmenu = false;
				parent.rebuild();
			},
			(action: string) => {
				// Exit the WHOLE submenu chain carrying the action: this submenu returns it to the
				// board, the board finishes with it, and the command loop performs it over the plain
				// chat - where pi's overlays are safe - then reopens the board (restoring the exact
				// screen and row). Free-number edits vs the read-only preview differ in the action.
				if (action === "preview-description") this.done(`__exit:preview-description:${ref}`);
				else this.done(`__exit:edit-model-number:${ref}:${action}`);
			},
			initialKey,
			this.filter,
			openSub,
		);
		this.list = screen as unknown as SettingsList;
		this.tui.requestRender();
	}

	render(width: number): string[] {
		const inner = Math.max(40, width);
		const lines: string[] = [];
		lines.push("");
		lines.push(this.themeHeader());
		lines.push("");
		if (this.list) lines.push(...this.list.render(inner));
		return lines.map((l) => truncateToWidth(l, inner));
	}

	private themeHeader(): string {
		return "Type to filter (letters and digits) · digits/Space act on SELECTED models only (1-9 place, 0 end, Space remove) · Enter: selected = options, unselected = add to end · Esc leaves";
	}

	// -- Focusable -------------------------------------------------------------

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	// -- internals --------------------------------------------------------------

	private highlightIndex(): number {
		return ((this.list as unknown as { selectedIndex?: number })?.selectedIndex) ?? 0;
	}

	private reposition(place: number): void {
		const ref = this.highlightRef();
		if (!ref) return;
		const cfg = loadConfig();
		const selected = Object.keys(cfg.models);
		if (!cfg.models[ref]) {
			// Selecting an unlisted model: it joins the order with its default options, at the named
			// place (or the end when the number is bigger than the list).
			rotateAdd(ref);
			const size = Object.keys(loadConfig().models).length;
			rotateTo(ref, Math.min(place, size - 1));
			this.rebuild();
			this.highlightRef(ref);
			return;
		}
		const currentIndex = selected.indexOf(ref);
		// "7" with 2 selected = the end. "0" is always the end.
		const target = place >= selected.length ? selected.length - 1 : Math.max(0, place);
		if (target !== currentIndex) rotateTo(ref, target);
		this.rebuild();
		this.highlightRef(ref);
	}

	/** Put the selection highlight on a given model row after a rebuild. */
	private highlightRef(want: string): void;
	/** Read which model row the selection highlight is on. */
	private highlightRef(): string | null;
	private highlightRef(want?: string): string | null | void {
		if (want !== undefined) {
			// Mutating the order: drop the filter first, so the index below lands in items
			// coordinates (while a filter is active, selectedIndex addresses the filtered list).
			this.clearSearch();
			const i = this.items.findIndex((it) => it.id === `model:${want}`);
			if (i >= 0) (this.list as unknown as { selectedIndex: number }).selectedIndex = i;
			return;
		}
		// BUG FIX 2026-10-01: with the search filter active, list.selectedIndex indexes the FILTERED
		// display list, not this.items - mapping it onto this.items returned the WRONG model (the
		// first unfiltered row), so Enter/digits acted on some unrelated gemini. Read the same
		// display list the highlight is drawn from.
		const display = (this.list as unknown as { getDisplayItems?: () => SettingItem[] })?.getDisplayItems?.() ?? this.items;
		const shown = display.filter((it) => String(it.id ?? "").startsWith("model:")).map((it) => String(it.id).slice("model:".length));
		return shown[this.highlightIndex()] ?? null;
	}

	/** Reset the list's search box (used before index-based highlight moves). */
	private clearSearch(): void {
		const list = this.list as unknown as { searchEnabled?: boolean; searchInput?: { setValue: (s: string) => void }; applyFilter?: (s: string) => void };
		if (list?.searchEnabled && list.searchInput && list.applyFilter) {
			list.searchInput.setValue("");
			list.applyFilter("");
		}
	}

	private unselect(): void {
		const ref = this.highlightRef();
		if (!ref) return;
		const cfg = loadConfig();
		if (cfg.models[ref]) rotateRemove(ref);
		this.rebuild();
	}

	private rebuild(): void {
		this.inSubmenu = false;
		const cfg = loadConfig();
		const reserveTokens = extReserve(this.ctx);
		const selectedRefs = Object.keys(cfg.models);
		// Position badges follow the JSON order (the try order).
		const items: SettingItem[] = [];
		for (const ref of selectedRefs) {
			const model = lookupRef(ref, this.ctx);
			items.push(this.row(ref, model, orderPosition(cfg, ref), reserveTokens));
		}
		// Everyone else pi can reach, alphabetically, with their (default) options visible.
		const rest = poolOf(this.ctx)
			.filter((p) => !cfg.models[p.ref])
			.sort((a, b) => a.ref.localeCompare(b.ref));
		for (const p of rest) items.push(this.row(p.ref, p.model, -1, reserveTokens));
		if (!items.length) {
			items.push({
				id: "none",
				label: "No models available:",
				currentValue: "",
				description: "pi has no reachable models right now. Start a provider login or the local server, then come back.",
			});
		}
		this.items = items;
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 18),
			getSettingsListTheme(),
			(_id: string, _value: string) => {
				/* values cycle through handleInput below; Enter is handled as open */
			},
			() => this.done(),
			{ enableSearch: true },
		);
		this.tui.requestRender();
	}

	private row(ref: string, model: any, place: number, reserveTokens: number): SettingItem {
		const cfg = loadConfig();
		const selected = Boolean(cfg.models[ref]);
		return {
			id: `model:${ref}`,
			label: modelRowLabel(place, ref, model),
			currentValue: selected ? modelRowValue(cfg, ref, model, reserveTokens) : modelRowValue(cfg, ref, model, reserveTokens),
			description: selected
				? "Enter opens its options (thinking, timeout, summary sizes, no-think tag). Space removes it from the order. Digits move it to that place."
				: "Not in the Compaction models order - it shows the default options. Press a digit (or 0) to add it at that place; Space on a selected model removes it. Enter opens options only for models in the order.",
		};
	}
}

// ---------------------------------------------------------------- TUI: per-model options submenu
// Which per-model rows stay on the model screen (2026-10-06) and which moved into the boards
// "Compaction settings" submenu. One builder makes every row; each screen shows its own subset, so a
// model keeps its reasoning and sampling controls here and its compaction settings sit with the rest
// of the compaction settings.
export const MODEL_KEEP_ROWS = ["thinking", "nothink", "timeout", "sampling"] as const;
export const MODEL_MOVED_ROWS = ["summaryPct", "summaryMin", "summaryMax", "inputstubs", "argcap", "preservestubs", "preserveuser", "preservereply", "preservethinking", "chain", "previewDesc"] as const;

/**
 * One selected model's options. Every row toggles through its values on Enter/Space:
 *   1. thinking (off..max, adjusted to what the model accepts)
 *   2. draft block (off / mini / full - a hidden analysis before the summary, stripped afterwards)
 *   3. timeout (2 min .. 60 min, unlimited)
 *   4. summary aim: percent of the ctx window (the same base as the min/max limits)
 *   5. summary floor (tokens)
 *   6. summary ceiling (tokens)
 *   7. no-think tag (the marker or empty)
 * Esc goes back; changes save the moment they are made.
 */
export function modelOptionsScreen(
	ctx: Ctx,
	ref: string,
	model: any,
	onDone: () => void,
	onExit?: (action: string) => void,
	initialKey?: string,
	only?: readonly string[],
	/** Open this row's submenu as soon as the screen is built (2026-10-21, defect #52). */
	openSub?: string,
): Component {
	const buildItems = (): SettingItem[] => {
		const cfg = loadConfig();
		const o = optionsFor(cfg, ref, model);
		const reserveTokens = extReserve(ctx);
		// 2026-10-07 BUG FIX: the list used to contain only the levels the current model reports, so the
		// row jumped between a few values and e.g. "max" was unreachable. A level the model does not have
		// is already lowered at request time by effectiveThinking, so the row offers the whole ladder and
		// shows "off" as the word "none".
		const levels = [...THINKING_CYCLE].map(thinkingWord);
		const th = model ? effectiveThinking(o.thinking, model) : { level: o.thinking, note: "" };
		const rows: SettingItem[] = [];
		rows.push({
			id: "thinking",
			label: "Thinking method:",
			currentValue: thinkingWord(o.thinking),
			// 2026-10-07 BUG FIX: the row DISPLAYS "none" for off (thinkingWord) but its values array held the
			// stored word "off", so the cycle could never find the current value: Enter re-applied the first
			// element and the row looked stuck. The ladder now uses the same words the row shows (none, minimal,
			// low, medium, high, xhigh, max); the handler maps "none" back to the stored "off".
			values: withCurrent(levels.map((l) => thinkingWord(l)), thinkingWord(o.thinking)),
			description: `Enter/Space cycles: ${levels.join(" → ")}. Private reasoning before it writes - it costs time, it does not make the summary longer. "off" also writes the no-think tag (row below) into the request for local models. A level the model does not have falls back to the nearest it has.`,
		});
		rows.push({
			id: "timeout",
			label: "Summary timeout period:",
			currentValue: timeoutLabel(o.timeoutMs),
			values: [EDIT_NUMBER, "unlimited"],
			description:
				"Enter opens a free numeric entry in minutes (0 = unlimited); cycling reaches 'unlimited'. This is how long to wait with NOTHING arriving: every character that comes in restarts the clock, so a model that keeps writing keeps its request however slow it is. A second ceiling nobody sets - twice what this model's last measured speed says the summary should take, never under 20 minutes, never over 2 hours - stops a stream that produces forever. When either one fires the next model in the Compaction models list is asked, and the log says which clock it was.",
		});
		rows.push({
			id: "summaryPct",
			label: "Summary aim (% of ctx window):",
			currentValue: `${o.summaryPercent}% (now: ~${fmt(Math.round((o.summaryPercent * (ctx?.model?.contextWindow ?? 0)) / 100))} tok)`,
			values: [EDIT_NUMBER],
			description:
				"THE SUMMARY AIM: this percent of the context GETTING COMPACTED - the session ctx minus the preserved tail, after the stubbing pass. The value on the right is what this percent aims for right now (example: ctx 441k, preserve 100k, region 341k; 15% aims 51,150 tok). The acceptance bounds are the min/max rows below. Enter opens a free numeric entry (1-99). Default 10.",
		});
		rows.push({
			id: "summaryMin",
			label: "Summary min (absolute tokens):",
			currentValue: `${fmt(o.summaryMinTokens)} tok (absolute floor)`,
			values: [EDIT_NUMBER],
			description:
				"THE ACCEPTANCE MINIMUM: a summary shorter than this is thrown away and the next model is asked. Percent of the model's ctx window, so it scales when you switch models - the value on the right is what it is right now. Never rises above the max, and yields to it on tiny regions. Enter opens a free numeric entry (1-99). Default 5.",
		});
		rows.push({
			id: "summaryMax",
			label: "Summary max (absolute tokens):",
			currentValue: `${fmt(o.summaryMaxTokens)} tok (absolute ceiling)`,
			values: [EDIT_NUMBER],
			description:
				"Upper clamp on the summary aim - generation-time guard (a 968k region at 10% would ask for a 97k summary; the ceiling caps it) and the hard cap on the accepted summary. Free entry. Default 32,768.",
		});
		rows.push({
			id: "inputstubs",
			label: "Summarizer input:",
			currentValue: o.inputStubs ? "stubs" : "raw",
			values: withCurrent(["stubs", "raw"], o.inputStubs ? "stubs" : "raw"),
			description:
				'"stubs" = every tool result in the summarizer input becomes an informative stub (tool, key argument, output size, tail message with the exit code, re-run hint) - pi\'s 2,000-character head-only clip then has nothing important left to cut, and the request gets much smaller. "raw" = pi\'s own input (long results clipped to their first 2,000 characters, exit code lost). Default: stubs.',
		});
		rows.push({
			id: "argcap",
			label: "Tool-call arg cap:",
			currentValue: `${charCapLabel(o.argCap)}${o.argCap === "full" ? "" : " chars"}`,
			values: [EDIT_NUMBER, "full"],
			description:
				'Caps tool-call argument VALUES in the summarizer input (whole files ride inside write/edit arguments). Truncated values keep their beginning plus a marker. Enter opens a free numeric entry in chars (0 = full); cycling reaches "full". Default 500 chars (OMP\'s value).',
		});
		rows.push({
			id: "preservestubs",
			label: "Tool stubs in summary:",
			currentValue: o.preserveStubs,
			values: withCurrent(PRESERVE_CYCLE, o.preserveStubs),
			description:
				`How the summary keeps tool calls. The EXACT sentence the model reads at each level: ${ladderText("stubs", o.preserveStubs)}. Switching this option switches that sentence in the template automatically. Default: verbatim (stubs are already compact).`,
		});
		rows.push({
			id: "preserveuser",
			label: "User prompts:",
			currentValue: o.preserveUser,
			values: withCurrent(PRESERVE_CYCLE, o.preserveUser),
			description:
				`How the summary keeps user prompts. The EXACT sentence the model reads at each level: ${ladderText("user", o.preserveUser)}. Default: verbatim - user feedback and reversals are the first thing a later turn must not lose.`,
		});
		rows.push({
			id: "preservereply",
			label: "Assistant replies:",
			currentValue: o.preserveReplies,
			values: withCurrent(PRESERVE_CYCLE, o.preserveReplies),
			description:
				`How the summary keeps assistant replies (marked [Assistant] in the input - separate from thinking). The EXACT sentence the model reads at each level: ${ladderText("replies", o.preserveReplies)}. Default: detailed.`,
		});
		rows.push({
			id: "preservethinking",
			label: "Assistant thinking:",
			currentValue: o.preserveThinking,
			values: withCurrent(PRESERVE_CYCLE, o.preserveThinking),
			description:
				`How the summary keeps assistant thinking (marked [Assistant thinking] in the input - separate from replies). The EXACT sentence the model reads at each level: ${ladderText("thinking", o.preserveThinking)}. Default: summary.`,
		});
		rows.push({
			id: "chain",
			label: "Chain previous summary:",
			currentValue: o.chainMode,
			values: withCurrent(["fold", "attach", "skip"] as string[], o.chainMode),
			description:
				"What happens to the PREVIOUS compaction's summary. fold (default): the model integrates it and continues its turn numbering - compact but lossy. attach: the new summary must copy the previous summary VERBATIM into a '## PREVIOUS SUMMARY' section - nothing is ever lost between compactions; the size limits are raised by the previous summary's size to make room. skip: drop it - maximum compression, history before the previous compaction ends there.",
		});
		rows.push({
			id: "previewDesc",
			label: "Preview final description",
			currentValue: "Press enter to preview",
			values: ["show"],
			description:
				"The COMPLETE description THIS model would receive on the next compaction: the size block and the summary template with this model's {TOKENS} substituted (its thinking level, its preserve levels, its chain mode, the category sentences as customized on the board). Read-only preview - nothing is sent; edits in the editor are discarded.",
		});
		rows.push({
			id: "nothink",
			label: "No-think tag:",
			currentValue: o.noThinkMarker || "(empty)",
			values: withCurrent([DEFAULT_NO_THINK_TAG, "(empty)"], o.noThinkMarker || "(empty)"),
			description:
				"Written into the summary request when thinking is 'off'. Some local chat templates keep thinking switched on no matter what the request says; this tag in the message is what such a template obeys. Empty writes nothing. The pi-wide default tag is set on the board (No-think tag row); this row overrides it for this model only.",
		});
		rows.push({
			id: "sampling",
			label: "Sampling flags:",
			currentValue: samplingLabel(o.sampling),
			// 2026-10-07 BUG FIX: a non-empty `values` made pi's list treat this as a cycling row, so Enter
			// switched the row between "ignore all" and "open flags" instead of opening the flags page.
			// Empty values = Enter opens the submenu.
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				samplingScreen(ref, () => done(), (flag: string) => done(flag)) as unknown as Component,
			description:
				'Extra sampling parameters sent with THIS model\'s compaction request only (your chat requests are never touched): temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty - each a number or "ignore". Enter opens the flags page (Esc comes back); Space cycles here, "ignore all" clears every flag. All flags default to ignore; the Qwen3.8-Flash-Next entry ships with its recommended non-thinking values (temp 0.7, top_p 0.8, top_k 20, min_p 0, presence 1.5, repeat 1.0). Free numeric entry: /compact-plus sampling',
		});
		// 2026-10-06: this screen shows only its own group (thinking, no-think, timeout, sampling). The
		// compaction rows moved to the boards Compaction settings submenu, which asks for the same rows
		// through MODEL_MOVED_ROWS. The row that removed a model from the order is gone: Space on the list
		// already does that.
		return only ? rows.filter((r) => only.includes(r.id)) : rows;
	};

	const items = buildItems();
	const list = new SettingsList(
		items,
		Math.min(items.length + 2, 12),
		getSettingsListTheme(),
		(id: string, value: string) => {
			const o = optionsFor(loadConfig(), ref, model);
			const reserveTokens = extReserve(ctx);
			if (id === "thinking") {
				setModelOptions(ref, { thinking: (value === "none" ? "off" : value) as Level });
			} else if (id === "timeout") {
				if (value === EDIT_NUMBER) { if (onExit) { onExit(id); return; } rebuildList(); }
				else setModelOptions(ref, { timeoutMs: parseDuration(value) });
			} else if (id === "summaryPct" || id === "summaryMin" || id === "summaryMax") {
				// Free numeric entry. pi's input overlay must NOT open while this screen is mounted:
				// the screen's list keeps consuming the keys, the input's promise hangs, and the empty
				// overlay sticks on top of everything (bug 2026-10-01). The request therefore travels
				// UP the submenu chain - this screen exits, the OrderList exits with the action, the
				// board finishes with it - and the command loop runs the edit over the plain chat (the
				// same proven path the board's own number rows use) and reopens the board.
				if (onExit) {
					onExit(id);
					return;
				}
				rebuildList();
			} else if (id === "inputstubs") {
				setModelOptions(ref, { inputStubs: value === "stubs" });
			} else if (id === "argcap") {
				if (value === EDIT_NUMBER) { if (onExit) { onExit(id); return; } rebuildList(); }
				else setModelOptions(ref, { argCap: value === "full" ? "full" : Number(String(value).replace(/[^0-9]/g, "")) });
			} else if (id === "preservestubs") {
				setModelOptions(ref, { preserveStubs: (PRESERVE_CYCLE as string[]).includes(value) ? (value as Preserve) : "verbatim" });
			} else if (id === "preserveuser") {
				setModelOptions(ref, { preserveUser: (PRESERVE_CYCLE as string[]).includes(value) ? (value as Preserve) : "verbatim" });
			} else if (id === "preservereply") {
				setModelOptions(ref, { preserveReplies: (PRESERVE_CYCLE as string[]).includes(value) ? (value as Preserve) : "detailed" });
			} else if (id === "preservethinking") {
				setModelOptions(ref, { preserveThinking: (PRESERVE_CYCLE as string[]).includes(value) ? (value as Preserve) : "summary" });
			} else if (id === "chain") {
				setModelOptions(ref, { chainMode: isChainMode(value) ? value : "fold" });
			} else if (id === "previewDesc") {
				// The editor overlay must open over the plain chat: exit the whole submenu chain with
				// the request; the command loop assembles and shows it, then reopens this screen here.
				if (onExit) {
					onExit("preview-description");
					return;
				}
				rebuildList();
			} else if (id === "nothink") {
				setModelOptions(ref, { noThinkMarker: value === "(empty)" ? "" : value });
			} else if (id === "sampling") {
				if (value === "ignore all") {
					setModelOptions(ref, { sampling: { ...SAMPLING_DEFAULTS } });
					const row = items.find((i) => i.id === "sampling");
					if (row) row.currentValue = samplingLabel(SAMPLING_DEFAULTS);
				} else if (onExit && (SAMPLING_KEYS as readonly string[]).includes(String(value))) {
					// A flag name came back from the flags page: the number box must open over the plain
					// chat (pi's input overlay hangs while this screen is mounted - bug 2026-10-01), so the
					// request travels up the chain; the command loop opens the box and restores this screen.
					onExit(String(value));
					return;
				}
			} else if (id === "remove") {
				rotateRemove(ref);
				onDone();
				return;
			}
			rebuildList();
		},
		() => onDone(),
	);

	function rebuildList(): void {
		const fresh = buildItems();
		items.length = 0;
		items.push(...fresh);
	}
	if (initialKey) {
		const i = items.findIndex((it) => it.id === initialKey);
		if (i >= 0) (list as unknown as { selectedIndex: number }).selectedIndex = i;
	}
	// COME BACK WHERE YOU WERE (2026-10-21, defect #52, the operator's report: changing a sampling flag and
	// pressing Enter "drops them back into the previous menu"). The flag itself has to be edited over the
	// plain chat - pi's input box hangs while a screen is mounted - so this screen is rebuilt afterwards.
	// Landing on the row that opens the flags page is still one menu up from the page the operator was
	// using, so the page is opened on arrival. Highlighting the row and feeding it Enter is pi's own
	// public way into a submenu; no private internals, and if the list ever stops offering it the screen
	// simply comes back as it used to.
	if (openSub) {
		const nav = list as unknown as { selectItem?: (id: string) => void; handleInput?: (d: string) => void };
		nav.selectItem?.(openSub);
		nav.handleInput?.("\r");
	}
	return list;
}

/**
 * The sampling flags sub-page of a model's options: one row per flag, Enter/Space cycles through a
 * value ladder (the Qwen instruct value first) plus "ignore"; the current value is always in the
 * cycle even when it was typed freely via the CLI. Esc comes back to the model's options.
 */
function samplingScreen(ref: string, onDone: () => void, onFlag?: (key: string) => void): Component {
	const buildItems = (): SettingItem[] => {
		const o = optionsFor(loadConfig(), ref, undefined);
		const rows: SettingItem[] = [];
		for (const key of SAMPLING_KEYS) {
			const cur = o.sampling[key];
			const curLabel = typeof cur === "number" ? String(cur) : "ignore";
			rows.push({
				id: key,
				label: key,
				currentValue: curLabel,
				// 2026-10-07 BUG FIX: these rows cycled a built-in ladder, which read as random values. Enter
				// now opens a box with the current value in it, like every other number in this extension.
				values: [EDIT_NUMBER],
				description: `${SAMPLING_HELP[key]} Allowed range: ${SAMPLING_RANGES[key][0]} to ${SAMPLING_RANGES[key][1]}, or 0 / empty for "ignore".`,
			});
		}
		return rows;
	};
	const items = buildItems();
	return new SettingsList(
		items,
		Math.min(items.length + 2, 10),
		getSettingsListTheme(),
		(id: string, value: string) => {
			if (value === EDIT_NUMBER) {
				// Hand the flag up: the box opens over the plain chat, then this screen is restored.
				onFlag?.(id);
				return;
			}
			const o = optionsFor(loadConfig(), ref, undefined);
			const next: number | null = value === "ignore" ? null : clamp(Number(value), SAMPLING_RANGES[id as SamplingKey][0], SAMPLING_RANGES[id as SamplingKey][1]);
			setModelOptions(ref, { sampling: { ...o.sampling, [id]: next } });
			const row = items.find((i) => i.id === id);
			if (row) row.currentValue = value;
		},
		() => onDone(),
	);
}

// ------------------------------------------------- TUI: the category sentence screens

/** The four content categories, their row labels and their template tokens. */
const CATEGORY_KINDS: { kind: "stubs" | "user" | "replies" | "thinking"; label: string; token: string }[] = [
	{ kind: "stubs", label: "Tool calls", token: "TOOL CALLS SUMMARY" },
	{ kind: "user", label: "User prompts", token: "USER PROMPT SUMMARY" },
	{ kind: "replies", label: "Assistant replies", token: "ASSISTANT REPLIES SUMMARY" },
	{ kind: "thinking", label: "Assistant thinking", token: "ASSISTANT THINKING SUMMARY" },
];
const CATEGORY_LEVELS: Preserve[] = ["verbatim", "detailed", "summary", "brief"];
/** Ctrl+R: reset the highlighted sentence to the EXTENSION default (in-screen confirm). */
const KEY_RESET_SENTENCE = "\x12";
/** Ctrl+U: restore the ACTIVE sentence from the USER default (in-screen confirm). */
const KEY_USER_DEFAULT = "\x15";
/** Ctrl+S: save the CURRENT sentence as the USER default (in-screen confirm). */
const KEY_SAVE_DEFAULT = "\x13";

/** The four levels' editable sentences for ONE category. Enter opens the editor (routed through
 *  the board loop - pi's editor overlay must never open over a mounted screen); Ctrl+R switches
 *  the list into an in-screen confirmation: Enter confirms the reset, Escape cancels. */
class CategoryDescList implements Component, Focusable {
	private list: SettingsList | null = null;
	private isFocused = false;
	/** The slot awaiting confirmation: which level, and which action (ext | user | save). */
	private pending: { level: string; action: "ext" | "user" | "save" } | null = null;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		private kind: "stubs" | "user" | "replies" | "thinking",
		private kindLabel: string,
		private token: string,
		private onDone: () => void,
		private onEdit: (level: string) => void,
		private onPreview: (target: "ext" | "user") => void,
		private initialLevel?: string,
	) {
		this.rebuild();
	}

	private rows(): SettingItem[] {
		const cfg = loadConfig();
		const base: SettingItem[] = CATEGORY_LEVELS.map((level) => {
			const sentence = sentenceFor(this.kind, level);
			const ud = userDefaultLayer(this.kind, level);
			return {
				id: level,
				label: `${level} · ${layerTag(this.kind, level)}`,
				currentValue: sentence.length > 56 ? `${sentence.slice(0, 56)}…` : sentence,
				values: ["edit sentence"],
				description: `${sentence}\n\nUser default: ${ud === "ext" ? "same as the extension default" : ud === "custom" ? "custom (differs from the extension default)" : "none saved - Ctrl+U falls back to the extension default"}.\n\nEnter = edit (saves as the active sentence; inside the editor submit :extension or :user alone to restart from that default).\nCtrl+S = save as user default.\nCtrl+U = reset to user default.\nCtrl+R = reset to extension default.\nAll three ask first: Enter confirms, Escape cancels.\nThe active sentence substitutes into the template's {${this.token}} token for every model set to ${level}.`,
			};
		});
		if (this.pending) {
			const { level, action } = this.pending;
			const ext = PRESERVE_SENTENCES[level as Preserve][this.kind];
			const usr = cfg.categoryUserDefaults?.[this.kind]?.[level];
			const what =
				action === "ext"
					? `Reset ${this.kindLabel} - ${level} to the extension default`
					: action === "user"
						? `Reset ${this.kindLabel} - ${level} to the user default`
						: `Save the current sentence as the user default`;
			const detail =
				action === "ext"
					? `The active sentence is replaced by the built-in extension default:\n${ext}\n(your saved user default is kept and can be restored with Ctrl+U)`
					: action === "user"
						? usr
							? `The active sentence is replaced by your saved user default:\n${usr}`
							: `No user default is saved for this slot - this reset is a no-op. Save one with Ctrl+S first.`
						: `The user default becomes:\n${sentenceFor(this.kind, level as Preserve)}\n(the active sentence does not change)`;
			return [
				{
					id: "__confirm",
					label: `${what} - Enter = confirm`,
					currentValue: "Enter = confirm, Escape = cancel",
					values: ["go"],
					description: `${detail}\n\nEnter = do it. Escape = cancel and keep everything as it is.`,
				},
				{
					id: "__cancel",
					label: "Cancel:",
					currentValue: "Enter or Escape = keep everything as it is",
					values: ["go"],
					description: "Escape (or Enter here) returns to the sentences without changing anything.",
				},
			];
		}
		base.push(
			{
				id: "__previewUser",
				label: "Preview all user defaults",
				currentValue: "read-only",
				values: ["preview"],
				description: "Your saved user defaults for this category, one per level. '(no user default saved)' means that slot falls back to the extension default.",
			},
			{
				id: "__previewExt",
				label: "Preview all extension defaults",
				currentValue: "read-only",
				values: ["preview"],
				description: "The built-in extension default sentences for this category - what Ctrl+R resets to.",
			},
		);
		return base;
	}

	private highlightedLevel(): string | undefined {
		const l = this.list as unknown as { getDisplayItems?: () => SettingItem[]; selectedIndex: number } | null;
		const id = l?.getDisplayItems?.()[l.selectedIndex]?.id;
		return id && CATEGORY_LEVELS.includes(id as Preserve) ? id : undefined;
	}

	private applyPending(): void {
		const { level, action } = this.pending!;
		update((c) => {
			if (action === "ext") {
				const m = c.categorySentences?.[this.kind];
				if (m) delete m[level];
			} else if (action === "user") {
				const usr = c.categoryUserDefaults?.[this.kind]?.[level];
				c.categorySentences = c.categorySentences ?? {};
				c.categorySentences[this.kind] = c.categorySentences[this.kind] ?? {};
				if (usr) c.categorySentences[this.kind][level] = usr;
				else delete c.categorySentences[this.kind][level];
			} else {
				c.categoryUserDefaults = c.categoryUserDefaults ?? {};
				c.categoryUserDefaults[this.kind] = c.categoryUserDefaults[this.kind] ?? {};
				c.categoryUserDefaults[this.kind][level] = sentenceFor(this.kind, level as Preserve);
			}
			for (const bag of [c.categorySentences, c.categoryUserDefaults] as const) {
				if (!bag) continue;
				if (Object.keys(bag[this.kind] ?? {}).length === 0) delete bag[this.kind];
			}
			if (c.categorySentences && !Object.keys(c.categorySentences).length) c.categorySentences = null;
			if (c.categoryUserDefaults && !Object.keys(c.categoryUserDefaults).length) c.categoryUserDefaults = null;
		});
	}

	private rebuild(): void {
		const items = this.rows();
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 8),
			getSettingsListTheme(),
			(id: string, value: string) => {
				if (this.pending) {
					const action = this.pending.action;
					const level = this.pending.level;
					this.pending = null;
					if (id === "__confirm") {
						this.applyPending();
						this.rebuild();
						this.ctx.ui?.notify?.(
							action === "ext"
								? `${this.kindLabel} - ${level}: active sentence reset to the extension default.`
								: action === "user"
									? `${this.kindLabel} - ${level}: active sentence reset to the user default.`
									: `${this.kindLabel} - ${level}: saved as the user default.`,
							"info",
						);
					} else {
						this.rebuild();
					}
					return;
				}
				if (value === "edit sentence") this.onEdit(id);
				else if (value === "preview") this.onPreview(id === "__previewExt" ? "ext" : "user");
			},
			() => this.onDone(),
		);
		if (!this.pending && this.initialLevel) {
			const i = items.findIndex((it) => it.id === this.initialLevel);
			if (i >= 0) (this.list as unknown as { selectedIndex: number }).selectedIndex = i;
		}
	}

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
		if (!this.pending && (data === KEY_RESET_SENTENCE || data === KEY_USER_DEFAULT || data === KEY_SAVE_DEFAULT)) {
			const level = this.highlightedLevel();
			if (level) {
				this.pending = { level, action: data === KEY_RESET_SENTENCE ? "ext" : data === KEY_USER_DEFAULT ? "user" : "save" };
				this.rebuild();
				this.tui.requestRender();
				return;
			}
		}
		if (chord(data, "escape") && this.pending) {
			this.pending = null;
			this.rebuild();
			this.tui.requestRender();
			return;
		}
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	render(width: number): string[] {
		const inner = Math.max(40, width);
		const lines: string[] = ["", `${this.kindLabel} - the four level sentences (this category's words substitute into {${this.token}}):`, ""];
		lines.push(...(this.list?.render(inner) ?? []));
		return lines.map((l) => truncateToWidth(l, inner));
	}
}

/** The placeholders a template may contain, in the order the request fills them. Listed at the top of
 *  the template editor so the insertable tags are visible while typing (2026-10-07 request). */
const TEMPLATE_TAGS = [
	"{SEGMENT SIZE}",
	// A template carrying every rule as its own prose puts this marker in its text: nothing is then appended
	// afterwards, neither the rules it omits nor the ledger-shape paragraph. See applyTemplate.
	RULES_WRITTEN_OUT,
	"{LENGTH INSTRUCTIONS}",
	// The numbers the length block is made of, for a template that writes the block out in its own words.
	// Substituted wherever they appear; see INLINE_LENGTH_TOKENS and applyTemplate.
	...INLINE_LENGTH_TOKENS,
	"{PREVIOUS SUMMARY SECTION}",
	"{USER PROMPT SUMMARY}",
	"{ASSISTANT REPLIES SUMMARY}",
	"{ASSISTANT THINKING SUMMARY}",
	"{TOOL CALLS SUMMARY}",
];

/** The next free template name: one above the highest "#N" in the list. Names are never typed. */
function nextUserTemplateName(templates: { name: string }[]): string {
	let max = 0;
	for (const t of templates) {
		const m = /#(\d+)\s*$/.exec(t.name);
		if (m) max = Math.max(max, Number(m[1]));
	}
	return `User template #${max + 1}`;
}

/** Puts text on the system clipboard through the platform's tool and reports what happened. */
function copyTextToClipboard(ctx: Ctx, text: string, label: string): void {
	const tool =
		process.platform === "win32"
			? { cmd: "clip", args: [] as string[] }
			: process.platform === "darwin"
				? { cmd: "pbcopy", args: [] as string[] }
				: { cmd: "xclip", args: ["-selection", "clipboard"] };
	try {
		const res = spawnSync(tool.cmd, tool.args, { input: text, encoding: "utf8" });
		const ok = res.status === 0;
		ctx.ui.notify(ok ? `${label}: the template text is on the clipboard.` : `Could not copy it (${tool.cmd} exited ${res.status ?? "with no status"}).`, ok ? "info" : "error");
	} catch (e) {
		ctx.ui.notify(`Could not copy it: ${(e as Error)?.message ?? e}`, "error");
	}
}

/**
 * The user's template library (2026-10-07 request): any number of saved templates, named by the
 * extension. Enter opens one in pi's editor - the normal editing box, where Enter saves, Shift+Enter is
 * a newline and Esc leaves the text unchanged - with the template's name and the insertable tags shown
 * above it. The keys on the header line work on the highlighted row. The ACTIVE template is a copy
 * assigned from here (Ctrl+A); it is not edited in place, which is why the Templates menu shows it as a
 * preview.
 */
/**
 * The template library's dialogs, run by the board loop. They CANNOT run inside the library screen: a
 * modal dialog (editor, confirm) opened while another custom screen is mounted leaves the terminal with
 * no focused input - the 2026-10-07 report was an editor that took the keystrokes, gave nothing back, and
 * needed a force close. So the library only asks, and the work happens here, over the plain chat, with the
 * board closed; the return value reopens the board on the library list. "" = the built-in default.
 */
async function templateLibraryAction(ctx: Ctx, action: string): Promise<{ templateMenu: boolean; templateKey: string }> {
	const at = action.indexOf(":");
	const kind = at < 0 ? action : action.slice(0, at);
	const name = at < 0 ? "" : decodeURIComponent(action.slice(at + 1));
	const back = { templateMenu: true, templateKey: "userTemplates" };
	const header = (n: string) => `User template - ${n}\nTags you can insert: ${TEMPLATE_TAGS.join("   ")}\nEnter saves - Shift+Enter is a newline - Esc leaves the template unchanged`;
	const cfg = loadConfig();
	if (kind === "tpl-new") {
		const n = nextUserTemplateName(cfg.userTemplates);
		const text = await editText(ctx, header(n), "");
		const stored = text ?? "";
		// Written onto a FRESH read (defect #41): the editor above can be open for minutes, and an
		// object loaded before it carries the whole settings screen as it looked then.
		update((c) => {
			c.userTemplates = [...c.userTemplates, { name: n, text: stored }];
		});
		ctx.ui.notify(text === undefined ? `${n} added to the library EMPTY (Esc in the editor does not delete it).` : `${n} saved to the library. Ctrl+A on its row makes it the active template.`, "info");
		return back;
	}
	if (kind === "tpl-previewext") {
		// 2026-10-07: the built-in default is READ-ONLY by request - it used to open in the editor with
		// a note that the text is discarded, which read like an editable template. It opens in the same
		// read-only preview screen the other previews use; Ctrl+K is the way to get an editable copy.
		await previewTemplate(ctx, "ext");
		ctx.ui.notify("The built-in default is fixed - preview only. Ctrl+K on its row saves it as a new user template you can edit, rename and delete.", "info");
		return back;
	}

	if (kind === "tpl-dup") {
		const src = name ? cfg.userTemplates.find((t) => t.name === name)?.text ?? "" : DEFAULT_TEMPLATE_TEXT;
		const n = nextUserTemplateName(cfg.userTemplates);
		const text = await editText(ctx, `${header(n)}\nCopied from ${name || "the extension default"} - editing the copy does not change the original`, src);
		if (text === undefined) {
			ctx.ui.notify("Cancelled - no copy was made.", "info");
			return back;
		}
		update((c) => {
			c.userTemplates = [...c.userTemplates, { name: n, text }];
		});
		ctx.ui.notify(`${n} saved as a copy of ${name || "the extension default"}.`, "info");
		return back;
	}
	if (kind === "tpl-del") {
		if (!name) {
			ctx.ui.notify("The built-in default is part of the extension - it cannot be deleted.", "info");
			return back;
		}
		if (!cfg.userTemplates.some((t) => t.name === name)) {
			ctx.ui.notify(`"${name}" is no longer in the list.`, "warning");
			return back;
		}
		const ok = await ctx.ui.confirm(`Delete ${name}?`, "The template leaves the library. What is sent does not change: the active template keeps the text it was given. Press y (or Enter) to delete, Esc to keep it.");
		if (!ok) {
			ctx.ui.notify(`${name} kept.`, "info");
			return back;
		}
		// 2026-10-20: if the deleted entry was the active one, the choice goes back to the extension default
		// openly, rather than leaving a name that resolves to nothing and a list that shows no active row.
		update((c) => {
			c.userTemplates = c.userTemplates.filter((t) => t.name !== name);
			if (c.activeTemplateName === name) {
				c.activeTemplateName = null;
				c.templateUserDefaults = null;
			}
		});
		ctx.ui.notify(`${name} deleted.`, "info");
		return back;
	}
	if (!name) {
		// The built-in default is fixed: it opens in the same editor so it can be read and copied, and the
		// result is dropped on purpose, with the notice saying so (the 2026-10-07 request).
		await editText(ctx, `Extension default template - built in, fixed\nTags you can insert: ${TEMPLATE_TAGS.join("   ")}\nAnything typed here is DISCARDED on submit - Ctrl+K saves it as a new user template`, DEFAULT_TEMPLATE_TEXT);
		ctx.ui.notify("The built-in default cannot be changed, so what you typed was NOT saved. Ctrl+K on that row saves its text as a new user template you can edit.", "info");
		return back;
	}
	const found = cfg.userTemplates.find((t) => t.name === name);
	if (!found) {
		ctx.ui.notify(`"${name}" is no longer in the list.`, "warning");
		return back;
	}
	const text = await editText(ctx, header(name), found.text);
	if (text === undefined) {
		ctx.ui.notify(`${name} left unchanged.`, "info");
		return back;
	}
	// Was this the active template before this edit - by name, or by the text it still matched (entries
	// activated before names existed)? Same rule as the library's own editor, 2026-10-20.
	let wasActive = false;
	update((c) => {
		// Asked of the config as it is NOW, not of the copy loaded before the editor opened: the answer
		// decides whether the row keeps its ACTIVE marker (2026-10-20 report, defect #41 for the stale write).
		wasActive = c.activeTemplateName === name || (!!c.templateUserDefaults?.trim() && found.text.trim() === c.templateUserDefaults.trim());
		c.userTemplates = c.userTemplates.map((t) => (t.name === name ? { ...t, text } : t));
		if (wasActive) {
			c.activeTemplateName = name;
			c.templateUserDefaults = text;
		}
		if (c.activeTemplateName === name) c.templateUserDefaults = text;
	});
	ctx.ui.notify(`${name} saved${wasActive ? " - still the active template" : ""}.`, "info");
	return back;
}

class UserTemplateMenu implements Component, Focusable {
	private list: SettingsList | null = null;
	private items: SettingItem[] = [];
	private isFocused = false;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		// No argument = the library was left (Esc). With an action = "do this over the plain chat and bring
		// the library back" - see templateLibraryAction for why the work cannot happen here.
		private onDone: (action?: string) => void,
	) {
		this.rebuild();
	}

	/** The text the summarizer gets: the saved user default when there is one, else the built-in text. */
	private activeText(): string {
		const t = loadConfig().templateUserDefaults?.trim();
		return t && t.length ? t : DEFAULT_TEMPLATE_TEXT.trim();
	}

	private rebuild(): void {
		const cfg = loadConfig();
		const active = this.activeText();
		const extActive = active === DEFAULT_TEMPLATE_TEXT.trim();
		// The ACTIVE marker follows the name, not the text (2026-10-20, the operator's design): editing the
		// active template used to detach it from its own row, so the list showed nothing active at all. A
		// config that never stored a name still resolves, by matching the text once here.
		const activeName = cfg.activeTemplateName && cfg.userTemplates.some((t) => t.name === cfg.activeTemplateName)
			? cfg.activeTemplateName
			: cfg.userTemplates.find((t) => t.text.trim() === active)?.name ?? null;
		// The model that would be asked: number 1 in the compaction try order. Its own settings fill the tags
		// of the request preview below (2026-10-07, on request).
		const firstRef = Object.keys(cfg.models)[0] ?? "";
		const items: SettingItem[] = [
			{
				id: "__req",
				label: "View the summary request (template, prompt, session digest)",
				currentValue: firstRef ? firstRef.split("/").pop() ?? "?" : "no model",
				values: ["open"],
				description: "ENTER OPENS IT: the WHOLE text the summarizer gets, exactly as it is sent - the active template with every tag filled in for this session (the size block, the preserve rules), the previous-summary line, the cover line, the conversation region between the tags, and the system prompt that rides above it all. Read-only, nothing is sent. The name on the right is the first model in the compaction try order: it is the one that would be asked, and its settings are what fill the tags.",
			},
			{
				id: "__reqfull",
				label: "View the summary request with the full conversation",
				currentValue: firstRef ? firstRef.split("/").pop() ?? "?" : "no model",
				values: ["open"],
				description: "The same request as the row above, with the conversation region UNABRIDGED - every message of the region, tool calls and results included, exactly as the summarizer receives them. It is long: in a session full of code and logs this is a wall of text, which is why the digest version is the one on top.",
			},
			{
				id: "__ext",
				label: "View the default template",
				currentValue: `${extActive ? "ACTIVE • " : ""}${DEFAULT_TEMPLATE_TEXT.split("\n").length} lines`,
				values: ["open"],
				description: "ENTER OPENS IT: the template built into the extension - fixed, the reference you can always go back to. Read-only. Ctrl+K saves its text as a new user template you can edit, Ctrl+A makes it the active template, Ctrl+Y copies it.",
			},
		];
		for (const t of cfg.userTemplates) {
			items.push({
				id: `t:${t.name}`,
				label: t.name,
				// The active template showed only "ACTIVE" - the line count was dropped for it, so the column
				// read differently on exactly the row you use (2026-10-07 report). One format for every row.
				currentValue: `${!extActive && (t.name === activeName || t.text.trim() === active) ? "ACTIVE • " : ""}${t.text.split("\n").length} lines`,
				values: ["open"],
				description: "Enter opens it in the editor (Enter saves, Esc discards). Ctrl+P makes this the active template, Ctrl+K copies it under a new name, Ctrl+D deletes it after asking, Ctrl+Y puts its text on the clipboard.",
			});
		}
		this.items = items;
		this.list = new SettingsList(items, Math.min(items.length + 2, 14), getSettingsListTheme(), (id: string, value: string) => {
			if (value !== "open") return;
			// The two request rows are not templates: they open the real request text (the board loop's
			// preview-request: / preview-request-full: handlers). 2026-10-07, on request.
			if (id === "__req" || id === "__reqfull") {
				this.onDone(`${id === "__req" ? "preview-request" : "preview-request-full"}:${encodeURIComponent(firstRef)}`);
				return;
			}
			// 2026-10-07 CRASH FIX: the editor is a modal dialog. Opening one while this list is mounted
			// leaves the terminal with no focused input - the "empty prompt, I cannot type anything, force
			// close" report. Every dialog goes to the board loop now, which runs it over the plain chat.
			this.onDone(id === "__ext" ? "tpl-previewext" : `tpl-edit:${encodeURIComponent(id.slice(2))}`);
		}, () => this.onDone());
	}

	/** The highlighted row: a template's name, null for the built-in default, undefined for the header. */
	private selectedTarget(): string | null | undefined {
		const i = (this.list as unknown as { selectedIndex?: number })?.selectedIndex ?? 0;
		const it = this.items[i];
		if (!it || it.id === "keys" || it.id === "__req" || it.id === "__reqfull") return undefined;
		return it.id === "__ext" ? null : it.id.slice(2);
	}

	// 2026-10-07: DEAD PATH, same reason as duplicate() above - the built-in default's editor runs in the
	// board loop now (templateLibraryAction, "tpl-edit" with an empty name).
	private async openExtensionDefault(): Promise<void> {
		const saved = await editText(this.ctx, `Extension default template - built in, fixed\nTags you can insert: ${TEMPLATE_TAGS.join("   ")}`, DEFAULT_TEMPLATE_TEXT);
		if (saved === undefined) return;
		this.ctx.ui.notify("The built-in default cannot be changed, so what you typed was NOT saved. Ctrl+K on that row saves its text as a new user template you can edit.", "info");
	}

	private async editTemplate(name: string): Promise<void> {
		const found = loadConfig().userTemplates.find((t) => t.name === name);
		if (!found) {
			this.ctx.ui.notify(`"${name}" is no longer in the list.`, "warning");
			this.rebuild();
			return;
		}
		const saved = await editText(this.ctx, `${name}\nTags you can insert: ${TEMPLATE_TAGS.join("   ")}`, found.text);
		if (saved === undefined) {
			this.ctx.ui.notify(`${name}: left exactly as it was (Esc discards, Enter saves).`, "info");
			return;
		}
		const cfg = loadConfig();
		// Was THIS the active template before the edit? Answer from the state as it stands now: the stored
		// name, or - for any library entry activated before names existed - the text it still matches.
		// Fixing the 2026-10-20 report: editing the active template used to leave the list with nothing
		// active, because the identity was the text and the edit had just changed it.
		let wasActive = false;
		update((c) => {
			wasActive = c.activeTemplateName === name || (!!c.templateUserDefaults?.trim() && found.text.trim() === c.templateUserDefaults.trim());
			c.userTemplates = c.userTemplates.map((t) => (t.name === name ? { name, text: saved } : t));
			if (wasActive) {
				c.activeTemplateName = name;
				c.templateUserDefaults = saved;
			}
		});
		this.ctx.ui.notify(`${name} saved (${saved.split("\n").length} lines)${wasActive ? " - still the active template" : ". Ctrl+P makes it the active template."}.`, "info");
		this.rebuild();
	}

	private async setActive(name: string | null): Promise<void> {
		const cfg = loadConfig();
		if (name !== null && !cfg.userTemplates.some((t) => t.name === name)) {
			this.ctx.ui.notify(`"${name}" is no longer in the list.`, "warning");
			this.rebuild();
			return;
		}
		// "" is the pinned extension default; any other text becomes the active template as-is. An empty
		// template therefore pins the built-in text - say so, it must not look like nothing happened.
		const text = name === null ? "" : cfg.userTemplates.find((t) => t.name === name)!.text;
		if (!text.trim() && name !== null) {
			update((c) => {
				c.templateUserDefaults = "";
				c.activeTemplateName = name;
			});
			this.ctx.ui.notify(`${name} is empty, so making it active pins the EXTENSION default (that is what an empty template means). Put text in it, then Ctrl+A again.`, "warning");
			this.rebuild();
			return;
		}
		// The name is what the list marks ACTIVE and what keeps the choice attached through an edit
		// (2026-10-20, the operator's design). null means the extension default, which the list shows on its
		// own row. Before 2026-10-21 this choice was also silently lost at the next restart, because the
		// loader did not copy the key back out of the file - see loadConfig.
		update((c) => {
			c.templateUserDefaults = text;
			c.activeTemplateName = name;
		});
		this.ctx.ui.notify(name === null ? "The active template is the extension default again." : `${name} is now the ACTIVE template - this is the text the summarizer gets.`, "info");
		this.rebuild();
	}

	// 2026-10-07: DEAD PATH, kept only so the modal editor below is never reached from this screen again.
	// Ctrl+K emits tpl-dup: to the board loop now - an editor opened over a mounted list kills the focus.
	private duplicate(name: string | null): void {
		const cfg = loadConfig();
		const text = name === null ? DEFAULT_TEMPLATE_TEXT : cfg.userTemplates.find((t) => t.name === name)?.text;
		if (text === undefined) return;
		const newName = nextUserTemplateName(cfg.userTemplates);
		update((c) => {
			c.userTemplates = [...c.userTemplates, { name: newName, text }];
		});
		this.ctx.ui.notify(`Saved as ${newName}. Enter opens it for editing.`, "info");
		this.rebuild();
	}

	private createNew(): void {
		this.onDone("tpl-new"); // created and edited by the board loop - see the note on the row handler
	}

	private removeTemplate(name: string | null): void {
		// The asking and the deleting happen in the board loop: a confirm is a modal dialog, and a dialog
		// opened over this mounted list leaves the TUI with no focused input (2026-10-07 crash).
		this.onDone(`tpl-del:${encodeURIComponent(name ?? "")}`);
	}

	handleInput(data: string): void {
		// The library keys (the terminal bytes for Ctrl+N / Ctrl+K / Ctrl+D / Ctrl+A / Ctrl+Y), all on
		// the highlighted row. Everything else, Escape included, goes to the list (Esc = back).
		if (chord(data, "ctrl+n")) {
			this.createNew();
			return;
		}
		if (chord(data, "ctrl+k", "ctrl+d", "ctrl+a", "ctrl+p")) {
			// Ctrl+P (\x10) was added on 2026-10-20 because Ctrl+A stopped reaching this screen: pi binds
			// ctrl+a to the editor action cursor-home, and whatever pi does with it, the byte never arrives
			// here to be argued with. Ctrl+A stays accepted in case a terminal still delivers it, and Ctrl+P
			// is the one that works. The key is named in the header line and in every row description, so the
			// operator is never hunting for an undocumented binding.
			const t = this.selectedTarget();
			if (t === undefined) return;
			if (chord(data, "ctrl+k")) this.onDone(`tpl-dup:${encodeURIComponent(t ?? "")}`); // modal, board loop
			else if (chord(data, "ctrl+d")) this.removeTemplate(t);
			else void this.setActive(t);
			return;
		}
		if (chord(data, "ctrl+y")) {
			const t = this.selectedTarget();
			if (t === undefined) return;
			copyTextToClipboard(this.ctx, t === null ? DEFAULT_TEMPLATE_TEXT : loadConfig().userTemplates.find((x) => x.name === t)?.text ?? "", t === null ? "Extension default" : t);
			return;
		}
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	invalidate(): void {
		this.list?.invalidate();
	}

	render(width: number): string[] {
		// 2026-10-07: the key list used to be the FIRST ROW of the library, so it could be highlighted
		// and 'opened' - it is an instruction, not an option. Printed above the list now, with the count.
		const head = truncateToWidth(
			`  Enter opens - Ctrl+N new - Ctrl+K duplicate - Ctrl+D delete - Ctrl+P set active - Ctrl+Y clipboard - ${loadConfig().userTemplates.length} saved`,
			Math.max(20, width),
		);
		return this.list ? [head, "", ...this.list.render(width)] : [];
	}

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
		(this.list as unknown as { focused?: boolean }).focused = value;
	}
}

/** The summary template's submenu: edit, read-only previews (active / user / extension), and the
 *  user-default actions. Every row exits through the board loop - editors and pi's select live
 *  over the plain chat, never over a mounted screen. */
class TemplateMenu implements Component, Focusable {
	private list: SettingsList | null = null;
	private isFocused = false;
	private inUserTemplates = false;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		private onAction: (action: string) => void,
		private restoreKey?: string,
	) {
		this.rebuild();
		// 2026-10-07 FLATTENED (on request): this menu used to hold two rows - "User templates" and
		// "Preview the active template" - so a whole submenu level existed to reach one option. The list of
		// templates IS this menu now: the summary request on top, the extension default below it, then the
		// user templates. There is no level below it, so leaving the list closes the menu.
		this.openUserTemplates();
	}

	/** The library, shown in place of this menu's list (the pattern the categories screen uses). */
	private openUserTemplates(): void {
		this.inUserTemplates = true;
		this.list = new UserTemplateMenu(this.tui, this.ctx, (action) => {
			// No action (Esc) = swap back to this menu's own list, inside the same mounted screen - safe.
			// With an action = hand it to the board: the board closes, the loop runs the dialog over the
			// plain chat, and the board reopens on the library. Dialogs never open over a mounted screen.
			// No level below the list any more: Esc closes the menu, an action goes to the board loop.
			this.onAction(action ?? "__close");
		}) as unknown as SettingsList;
	}

	private stateLabel(): string {
		const cfg = loadConfig();
		// One template (2026-10-06): the text this menu edits is the text the summarizer reads. There
		// used to be two stored texts with an order between them, and the one this menu did NOT show
		// won, so the request carried a template nobody was looking at.
		const t = cfg.templateUserDefaults?.trim();
		if (!t) return 'the extension default (nothing of yours saved)';
		if (t === DEFAULT_TEMPLATE_TEXT.trim()) return 'the extension default (saved unchanged)';
		return t.split('\n').length + ' lines (yours - this is what is sent)';
	}

	private rows(): SettingItem[] {
		const cfg = loadConfig();
		return [
			{
				id: "userTemplates",
				label: "User templates",
				currentValue: `${cfg.userTemplates.length} saved`,
				values: ["open"],
				description: "Your own templates, as many as you want, named by the extension ('User template #N'). Enter opens the list: Enter on a template opens it in the editor (Enter saves, Shift+Enter is a newline, Esc discards), Ctrl+N makes a new one and opens it, Ctrl+K duplicates the highlighted one under a new name, Ctrl+D deletes it after asking, Ctrl+A makes it the ACTIVE template, Ctrl+Y copies its text to the clipboard. The active template is a copy assigned from there - the previews below show what is being sent right now.",
			},
			// Two previews only (2026-10-07, on request): the template in use, and the built-in one. The
			// old "preview the request a model gets" row showed the same text as the active preview, and the
			// "kept copy" preview showed a saved file, not a template being sent - both are gone.
			// `values: ["preview"]` is what makes Enter fire the list's value handler for this row; the
			// handler used to have no branch for these ids, which is why a preview never opened.
			{
				id: "previewActive",
				label: "Preview the active template:",
				currentValue: "read-only",
				values: ["preview"],
				description: "ENTER OPENS IT: the template in use right now, word for word, as the summarizer reads it (the {TOKENS} are still tokens here - they are filled per model in the request). Esc closes it and brings this menu back on the same row.",
			},
			// 2026-10-07: "Keep a copy of this template" and "Restore the kept copy" removed on request -
			// leftovers from before the library existed. Ctrl+A assigns a template as the active one, and
			// "Reset to extension default" is the one revert. The user-default field stays in the config as
			// the fallback an unassigned template follows; only the two menu rows are gone.
		];
	}

	private rebuild(): void {
		const items = this.rows();
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 10),
			getSettingsListTheme(),
			(id: string, value: string) => {
				if (id === "userTemplates") this.openUserTemplates();
				else if (value === "open" && id === "edit") this.onAction("edit-instruction");
				// The two previews. They travel up as __exit actions: the board has to close for the text
				// screen to take the terminal, and the command loop reopens the board on this row after.
				// No __exit: prefix here: the row's submenu factory in the board adds it, and the board strips
				// exactly one. These rows used to add it themselves, so the action arrived as
				// "__exit:preview-template:active", matched no handler, and the board closed with a warning
				// (2026-10-07 report: "both previews don't work, it throws me back to the main menu").
				if (id === "previewActive") this.onAction("preview-template:active");
				else if (id === "previewExt") this.onAction("preview-template:ext");
				else if (id === "previewUser") this.onAction("preview-template:user");
				else if (value === "go") this.onAction(id === "saveUser" ? "save-template-user-default" : id === "resetUser" ? "reset-template-user-default" : "reset-template-extension-default");
			},
			() => this.onAction("__close"),
		);
	}

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	render(width: number): string[] {
		const inner = Math.max(40, width);
		const lines: string[] = ["", "Summary template - previews of what is sent, your template library, the resets:", ""];
		lines.push(...(this.list?.render(inner) ?? []));
		return lines.map((l) => truncateToWidth(l, inner));
	}
}

/** The categories screen: the four content categories; Enter opens a category's sentences. */
class CategoryPick implements Component, Focusable {
	private list: SettingsList | null = null;
	private isFocused = false;
	private inSubmenu = false;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		private onDone: () => void,
		private onEdit: (kind: string, level: string) => void,
		private onPreview: (kind: string, target: "ext" | "user") => void,
		private restoreKind?: string,
		private restoreLevel?: string,
	) {
		this.rebuild();
		if (this.restoreKind) this.openKind(this.restoreKind as any, this.restoreLevel);
	}

	private rebuild(): void {
		const items: SettingItem[] = CATEGORY_KINDS.map((k) => {
			const changedCount = CATEGORY_LEVELS.filter((l) => sentenceLayer(k.kind, l, sentenceFor(k.kind, l)) !== "ext").length;
			return {
				id: k.kind,
				label: k.label,
				currentValue: changedCount ? `${changedCount}/4 differ from the extension defaults` : "all match the extension defaults",
				values: ["open"],
				description: `The editable sentence for each level (verbatim / detailed / summary / brief) of how the summary keeps ${k.label.toLowerCase()}. Substituted into the template's {${k.token}} token. Enter opens the four sentences; the bottom two rows preview your user defaults and the extension defaults read-only.`,
			};
		});
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 8),
			getSettingsListTheme(),
			(id: string, value: string) => {
				if (value === "open") this.openKind(id as any);
			},
			() => this.onDone(),
		);
	}

	private openKind(kind: "stubs" | "user" | "replies" | "thinking", level?: string): void {
		const meta = CATEGORY_KINDS.find((k) => k.kind === kind)!;
		this.inSubmenu = true;
		this.list = new CategoryDescList(this.tui, this.ctx, kind, meta.label, meta.token, () => {
			this.inSubmenu = false;
			this.rebuild();
		}, (level2: string) => this.onEdit(kind, level2), (target) => this.onPreview(kind, target), level) as unknown as SettingsList;
	}

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	render(width: number): string[] {
		const inner = Math.max(40, width);
		const lines: string[] = ["", "Category templates - Enter opens a category's four editable level sentences; Esc goes back:", ""];
		lines.push(...(this.list?.render(inner) ?? []));
		return lines.map((l) => truncateToWidth(l, inner));
	}
}

/** The editor for ONE category sentence, over the plain chat (never over a mounted screen).
 *  pi owns the editor's keys, so the resets ride as TEXT COMMANDS: submitting the lone word
 *  ":extension" or ":user" reopens the editor prefilled from that default; a normal submit saves
 *  the text as the ACTIVE sentence; empty text falls back to the extension default. */
async function editCategorySentence(ctx: Ctx, kind: string, level: string): Promise<void> {
	const meta = CATEGORY_KINDS.find((k) => k.kind === kind);
	const label = meta?.label ?? kind;
	const ext = PRESERVE_SENTENCES[level as Preserve]?.[kind as "stubs" | "user" | "replies" | "thinking"] ?? "";
	let prefill = sentenceFor(kind as any, level as Preserve);
	const title = `Category sentence — ${label} — ${level} (currently: ${layerTag(kind as any, level as Preserve)}). Substituted into {${meta?.token}} for every model at this level. Submit = save as the ACTIVE sentence; empty = extension default; submit the lone word :extension or :user to restart from that default.`;
	for (;;) {
		const text = await editText(ctx, title, prefill);
		if (text === undefined) return;
		const t = String(text).trim();
		if (t === ":extension") {
			prefill = ext;
			ctx.ui?.notify?.(`Restarted from the EXTENSION default (${label} - ${level}). Edit and submit to save it as the ACTIVE sentence.`, "info");
			continue;
		}
		if (t === ":user") {
			const usr = loadConfig().categoryUserDefaults?.[kind]?.[level];
			prefill = usr ?? ext;
			ctx.ui?.notify?.(usr ? `Restarted from your USER default (${label} - ${level}).` : `No USER default saved for this slot - restarted from the extension default instead.`, "info");
			continue;
		}
		update((c) => {
		if (!t || t === PRESERVE_SENTENCES[level as Preserve]?.[kind as "stubs" | "user" | "replies" | "thinking"]) {
			const m = c.categorySentences?.[kind];
			if (m) delete m[level];
			if (c.categorySentences && Object.keys(c.categorySentences[kind] ?? {}).length === 0) delete c.categorySentences[kind];
			if (c.categorySentences && !Object.keys(c.categorySentences).length) c.categorySentences = null;
		} else {
			c.categorySentences = c.categorySentences ?? {};
			c.categorySentences[kind] = c.categorySentences[kind] ?? {};
			c.categorySentences[kind][level] = t;
		}
	});
	ctx.ui?.notify?.(t ? `Saved: ${label} — ${level}.` : `${label} — ${level}: the built-in default is used again.`, "info");
		return;
	}
}

/** The per-model preview: the COMPLETE description this model would receive, opened read-only. */
async function previewDescription(ctx: Ctx, ref: string): Promise<void> {
	const cfg = loadConfig();
	const model = lookupRef(ref, ctx);
	const profile = optionsFor(cfg, ref, model);
	const projection: any = ctx?.sessionManager?.buildSessionProjection?.();
	const msgs: any[] = (projection?.messages ?? []).filter((m: any) => m?.role !== "system");
	const chars = msgs.reduce((a: number, m: any) => a + contentTextOf(m.content).length, 0);
	const inputTokens = Math.ceil((chars * 1.3) / 4) + SUMMARISER_FRAMING_TOKENS;
	const sizes = sizesFor(profile, { inputTokens, historyTokens: inputTokens, headroom: 0, overflow: false, windowTokens: ctx?.model?.contextWindow ?? 0 }, 0);
	const text = buildInstructions({
		sizes,
		inputTokens,
		segmentTokens: inputTokens - SUMMARISER_FRAMING_TOKENS,
		kind: isLocalModel(model) ? "local" : "online",
		options: profile,
		additionalInstruction: cfg.additionalInstruction,
		userDefault: cfg.templateUserDefaults,
		customInstructions: undefined,
	});
	await uiPreview(ctx,
		`READ-ONLY PREVIEW — this is what ${ref} would receive. Nothing is sent; ANY changes made in this editor are DISCARDED when you close it (the submit/newline/external-editor keys below belong to the template editors, not to this preview). The real request adds the conversation region (numbered turns + stubs + the previous summary) below this description.`,
		text,
	);
}


/** The auto-compaction submenu: the live trigger point (read-only) and the three settings that
 *  define it. The editor rows exit through the board loop (pi's overlays are safe only there). */
/** The elision submenu: the on/off toggle (in-screen) and the six settings, each edited through
 *  the board loop. */
class ElisionMenu implements Component, Focusable {
	private list: SettingsList | null = null;
	private isFocused = false;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		private onAction: (action: string) => void,
	) {
		this.rebuild();
	}

	private rows(): SettingItem[] {
		const cfg = loadConfig();
		const e = cfg.elision;
		const window = this.ctx?.model?.contextWindow ?? 0;
		const point = window - extReserve(this.ctx);
		return [
			{
				id: "elisionOn",
				label: "Elision on/off:",
				currentValue: e.enabled ? "on" : "off",
				values: ["on", "off"],
				description: "Old tool results are replaced by short stubs in the requests the model receives, so the context stays much lower for much longer. Your screen and the session file always keep the originals - stubs exist only on the wire. Enter/Space toggles it.",
			},
			{
				id: "elisionStart",
				label: "Start auto-elision at % of ctx:",
				currentValue: `${e.softPercent}% (now: ${fmt(Math.round((e.softPercent * window) / 100))} tok)`,
				values: ["edit"],
				description: "Elision begins when the context reaches this percent of the model's window (the value on the right is what that is right now). Enter opens a free numeric entry (1-99). Default 20.",
			},
			{
				id: "elisionProtect",
				label: "Don't elide last x tok of ctx:",
				currentValue: `${fmt(e.protectRecentTokens)} tok`,
				values: ["edit"],
				description: "A protected window counted back from the newest message: tool results inside it are never stubbed, so the recent work stays fully intact. Enter opens a free numeric entry. Default 20,000.",
			},
			{
				id: "elisionTail",
				label: "Preserve last x chars of results:",
				currentValue: `${typeof e.stubTailChars === "number" ? fmt(e.stubTailChars) : "full"} chars`,
				values: ["edit"],
				description: "How much of each tool result's ending survives inside its stub (the outcome usually lives there). Enter opens a free numeric entry (0 = none). Default 300.",
			},
			{
				id: "elisionSavings",
				label: "Auto-elide if it saves x tok:",
				currentValue: `${fmt(e.minSavingsTokens)} tok`,
				values: ["edit"],
				description: "A sweep is only taken if it actually saves at least this many tok - no busywork. Enter opens a free numeric entry. Default 2,000.",
			},
			{
				id: "elisionResults",
				label: "Auto-elide only at min results:",
				currentValue: `${e.minResultsToStub} results`,
				values: ["edit"],
				description: "A sweep waits until at least this many tool results qualify - no churn for one lonely old test run. Enter opens a free numeric entry. Default 4.",
			},
		{
			id: "elisionBatchGap",
			label: "Elision batch gap:",
			currentValue: `${e.minBatchGapSec} s`,
			values: ["edit"],
			description: "Minimum seconds between two sweeps. Every sweep moves the prompt-cache break point, so the context after it is read by the server again - without a gap a busy session pays for that over and over (2026-10-05: batches a minute apart on a 700k session). Seconds.",
		},
		{
			id: "elisionCap",
			label: "Max tool results per batch:",
			currentValue: `${e.maxResultsPerBatch} results`,
			values: ["edit"],
			description: "How many results one sweep may stub; the rest wait for the next batch. One big sweep moves the break point once and costs one re-read; a hundred small sweeps cost a hundred.",
		},
			{
				id: "elisionStop",
				label: "Stop eliding x tok before trigger:",
				currentValue: `${fmt(e.stopGapTokens)} tok (now: ${fmt(Math.max(0, point - e.stopGapTokens))} tok)`,
				values: ["edit"],
				description: "A safety distance from the auto-compaction point: elision will not push the context closer than this (the value on the right is the effective stop line right now) - near compaction, sweeping would be wasted anyway. Enter opens a free numeric entry. Default 4,000.",
			},
		];
	}

	private rebuild(): void {
		const items = this.rows();
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 10),
			getSettingsListTheme(),
			(id: string, value: string) => {
				if (id === "elisionOn") {
					update((c) => void (c.elision.enabled = value === "on"));
					this.rebuild();
					return;
				}
				if (value === "edit") this.onAction(`__exit:edit-number:${id}`);
			},
			() => this.onAction("__close"),
		);
	}

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	render(width: number): string[] {
		const inner = Math.max(40, width);
		const lines: string[] = ["", "Elision - old tool results become stubs on the wire; originals stay on disk:", ""];
		lines.push(...(this.list?.render(inner) ?? []));
		return lines.map((l) => truncateToWidth(l, inner));
	}
}

/**
 * The "Compaction settings" submenu (2026-10-06): the two compaction switches, the retry knobs, the
 * local no-think tag and - behind one more row - the per-model compaction options.
 *
 * THE RULE THE TWO SWITCHES OBEY: exactly ONE compactor is ever on. Turning this extension's
 * compaction on writes pi's own compaction OFF; turning pi's on writes this extension's OFF. Both off
 * is allowed - then nothing compacts and the session just grows. Elision is independent of both and
 * lives in its own submenu. pi's switch is a value in pi's settings.json that pi reads at startup, so
 * changing it does nothing until the user reloads; every change here says so.
 */
class CompactionMenu implements Component, Focusable {
	private list: SettingsList | null = null;
	private isFocused = false;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		private onAction: (action: string) => void,
	) {
		this.rebuild();
	}

	private rows(): SettingItem[] {
		const cfg = loadConfig();
		// The settings that are one setting for every model in the try order: the first model stands for
		// all of them, because setModelOptions writes those keys to every model in the order.
		const sharedRef = Object.keys(cfg.models)[0] ?? "";
		const o = optionsFor(cfg, sharedRef, lookupRef(sharedRef, this.ctx));
		return [
			{
				id: "compactionOn",
				label: "Compaction on/off:",
				currentValue: cfg.enabled ? "on" : "off",
				values: ["on", "off"],
				description:
					"ON = the plugin writes the summary with the first model in the Compaction models list that answers, and the size check runs on its own (see 'When auto-compaction fires' below). OFF = this extension never compacts, and nothing of it touches pi's settings.",
			},
			{
				id: "reserveTok",
				label: "Autocompaction trigger point:",
				currentValue: (() => {
					const win = (this.ctx as any)?.model?.contextWindow ?? 0;
					const set = cfg.scaling.reserveTokens;
					const eff = extReserve(this.ctx);
					// The number that was typed is the number that is shown. It used to show the reserve IN FORCE
					// only, which was honest but unreadable as an editor: setting 30,000 while pi's own reserve was
					// 30,600 printed 30,600 back, and the operator reasonably read that as the edit not taking
					// (2026-10-22). The line is the setting now, so the two can only differ when no number is set.
					if (!(win > 0)) return set > 0 ? `${fmt(set)} tok before a full window` : `automatic: ${fmt(eff)} tok before a full window`;
					const head = `${fmt(eff)} tok before the ${fmt(win)} tok window (fires at ${fmt(triggerPoint(this.ctx))} tok)`;
					return set > 0
						? head + (eff !== set ? ` (your ${fmt(set)} cannot fit this window; ${fmt(eff)} is used)` : "")
						: head + ` (no number set: a tenth of the window, floor 16,384)`;
				})(),
				values: [EDIT_NUMBER],
				description:
					"Automatic compaction fires once the context is this close to the top of the window: the line " +
						"is the model's context window minus this number, so one value means the same on every model. " +
						"0 leaves the choice to pi's own reserveTokens. This is the mirror of that pi setting and the " +
						"two do interact: if pi's reserveTokens is LARGER than this, pi asks to compact first and this " +
						"extension holds the ask until its own line, which costs a full pass over the context each turn " +
						"until then - the pi's own reserveTokens row below shows both points. The line moves the moment " +
						"this number is saved, so set it while the context is small rather than in the middle of a run.",
			},
			{
				id: "preserveTok",
				label: "Preserve tail context:",
				currentValue: `${fmt(extKeepRecent(this.ctx))} tok of the context tail kept verbatim` + (cfg.scaling.preserveTokens !== extKeepRecent(this.ctx) ? ` (you set ${fmt(cfg.scaling.preserveTokens)}, this window keeps ${fmt(extKeepRecent(this.ctx))})` : ""),
				values: [EDIT_NUMBER],
				description:
					"How much of the window is kept free for the conversation that stays word for word: the tail " +
					"compaction keeps, and the recent tool results elision refuses to stub. This is the extension's " +
					"equivalent of pi's keepRecentTokens. Also a token count now, with the old min and max limit rows " +
					"gone. Enter opens a free numeric entry.",
			},
		// pi's own compaction settings are read for information only. This extension never writes
		// that file: it compacts on its own trigger with its own models, which works whether pi's
		// flag is on or off, and leaves pi's own summary path out of the picture while it is off.
			numberRow("retries", "Retries per model", String(cfg.retries), [], "How many times the next model in the list is tried when one fails."),
			numberRow("retryWait", "Wait between retries", `${cfg.retryDelaySeconds} s`, [], "Seconds to wait before trying the next model."),
			{
				id: "nothinkTag",
				label: "No-think tag for local models:",
				currentValue: cfg.noThinkMarker || "(empty)",
				values: [EDIT_TEXT, "(empty)"],
				description: "The tag a local model's summary request carries when its thinking is off, so the chat template does not re-open reasoning. A model's own No-think tag row overrides it.",
			},
			// 2026-10-06: these settings used to sit behind a model list. They are ONE setting for every
			// model in the try order, so they live here in the open. The models list keeps only what really
			// differs per model: the thinking level, the no-think tag, the summary timeout, the sampling flags.
			// Enter cycles a choice; a row showing EDIT opens a field to type the number in.
			{
				id: "summaryPct",
				label: "Summary aim:",
				// The percent alone told nobody how big that is: the token number rides next to it, the
				// way the other size rows show their tokens (2026-10-07).
				currentValue: `${o.summaryPercent}% (${fmt(Math.round((o.summaryPercent / 100) * ((this.ctx as any)?.model?.contextWindow ?? 0)))} tok)`,
				values: [EDIT_NUMBER],
				description:
					"What percent of the context that gets summarized the summary should aim at. One setting for every model in the try order.",
			},
			{
				id: "summaryMin",
				label: "Summary minimum (all models):",
				currentValue: `${fmt(o.summaryMinTokens)} tok`,
				values: [EDIT_NUMBER],
				description:
					"A summary shorter than this is thrown away and the next model in the try order is tried. One setting for every model.",
			},
			{
				id: "summaryMax",
				label: "Summary maximum (all models):",
				currentValue: `${fmt(o.summaryMaxTokens)} tok`,
				values: [EDIT_NUMBER],
				description:
					"A summary longer than this is thrown away too. One setting for every model.",
			},
			{
				id: "inputstubs",
				label: "Summarizer input (all models):",
				currentValue: `${o.inputStubs ? "stubs" : "raw"}`,
				values: ["stubs", "raw"],
				description:
					"What the summarizer is shown for the older tool results: informative stubs, or the raw text.",
			},
			{
				id: "argcap",
				label: "Tool-call argument cap (all models):",
				currentValue: `${charCapLabel(o.argCap)}`,
				values: [EDIT_NUMBER, "full"],
				description:
					"How much of a tool call argument values text the summarizer sees. The tool RESULTS are separate, see the preserve rows.",
			},
			{
				id: "preservestubs",
				label: "Tool stubs in summary (all models):",
				currentValue: `${o.preserveStubs}`,
				values: withCurrent(PRESERVE_CYCLE as string[], o.preserveStubs),
				description:
					"How the stubbed tool results are carried into the summary.",
			},
			{
				id: "preserveuser",
				label: "User prompts in summary (all models):",
				currentValue: `${o.preserveUser}`,
				values: withCurrent(PRESERVE_CYCLE as string[], o.preserveUser),
				description:
					"How your own prompts are carried into the summary.",
			},
			{
				id: "preservereply",
				label: "Assistant replies in summary (all models):",
				currentValue: `${o.preserveReplies}`,
				values: withCurrent(PRESERVE_CYCLE as string[], o.preserveReplies),
				description:
					"How the assistant replies are carried into the summary.",
			},
			{
				id: "preservethinking",
				label: "Assistant thinking in summary (all models):",
				currentValue: `${o.preserveThinking}`,
				values: withCurrent(PRESERVE_CYCLE as string[], o.preserveThinking),
				description:
					"How the assistant thinking is carried into the summary.",
			},
			{
				id: "chain",
				label: "Previous summary (all models):",
				currentValue: `${o.chainMode}`,
				values: ["fold", "attach", "skip"],
				description:
					"What the previous compaction summary becomes in the new one: folded into its own turn, attached whole, or dropped.",
			},
			// 2026-10-07: moved here from the main board on request - both are compaction sizing, they
			// belong next to the compaction switches, not on the top level.
			{
				id: "piReserve",
				label: `pi's own compaction numbers (read-only):`,
				// Two numbers the operator has to keep in step, shown side by side (2026-10-21): where pi asks,
				// and where this extension is ready. While pi asks early, every held turn re-reads the whole
				// context before a summary is written, so the row says what to set instead of leaving a puzzle.
				currentValue: (() => {
					const win = (this.ctx as any)?.model?.contextWindow ?? 0;
					const piReserve = readPiReserveTokens();
					if (!(win > 0)) return `${fmt(piReserve)} tok reserve`;
					const ours = win - extReserve(this.ctx);
					const theirs = Math.max(0, win - piReserve);
					// What is wanted from pi is now the opposite of what it used to be: its alarm should ring
					// ABOVE our line, that is, with a SMALLER reserve than ours. Equal is fine, earlier is not.
					const alarmOk = theirs >= ours - Math.max(1_000, Math.round(win * 0.01));
					return alarmOk
						? `alarm at ${fmt(theirs)} tok, above our line ${fmt(ours)} - it rings only if our own timer fails`
						: `alarm at ${fmt(theirs)} tok, BELOW our line ${fmt(ours)} by ${fmt(ours - theirs)} - lower pi's reserveTokens to about ${fmt(Math.max(2_048, Math.round(win * 0.07)))} and /reload, or every held ask re-reads the context`
				})(),
				values: [],
				description:
				"pi's settings.json, read only - this extension never writes it. pi asks when the context " +
				"passes window minus reserveTokens, and since 2026-10-22 this extension's own timer is the " +
				"trigger, so pi is the alarm: its reserveTokens should be SMALLER than the reserve above, " +
				"which puts the alarm above the line where compaction actually happens. If an alarm still " +
				"arrives below the line, the gate holds it (the context is not full yet), the hold is logged, " +
				"and pi reads the file again on /reload.",
			},
			numberRow("chatCap", "Chat max_tokens cap", cfg.chatMaxTokensCap > 0 ? fmt(cfg.chatMaxTokensCap) : "no cap", ["no cap"], "Caps the generation permission (max_tokens) on every CHAT request. pi sizes it as window minus its context estimate minus 4,096 - on big-window models that asks for absurd room (943k on glm-flash) and the whole request is rejected whenever the estimate undercounts by more than 4,096 tokens (the overflow that forced the emergency compactions). With a cap the request always fits until the real context reaches window minus cap. Enter opens a free numeric entry (0 = no cap); cycling reaches 'no cap'. Default 65,536."),
		];
	}

	private rebuild(): void {
		const items = this.rows();
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 10),
			getSettingsListTheme(),
			(id: string, value: string) => {
				if (id === "compactionOn") {
					const on = value === "on";
					// ONE COMPACTOR (2026-10-07): refused while another context manager is present; the row is
					// redrawn from the config, which still says off.
					if (on && enableBlockedByOtherCompactor()) {
						(this as unknown as { rebuild?: () => void }).rebuild?.();
						return;
					}
					update((c) => void (c.enabled = on));
					// pi's own compaction is a separate thing in pi's own settings file, and this switch never
					// touches it. The automatic trigger here is this extension's own check, which works with
					// pi's flag either way - so switching pi's compaction off means off, and it stays off.
					this.onAction(
						"__exit:notice:" +
							(on
								? `COMPACTION IS ON. WHO TIMERS DEPENDS ON PI'S OWN SWITCH: with pi's compaction enabled, pi's threshold is the doorbell and this extension's trigger point is the gate - the two are computed from the same window so they agree, and a hold only happens if pi's reserveTokens drift out of step; with pi's switched off, this extension's own check is the trigger. Either way THIS EXTENSION WRITES EVERY SUMMARY. Your start point is on the settings board. pi's settings are read, never written.`
								: "THIS EXTENSION'S COMPACTION IS OFF, and nothing of it touches pi's settings. IF PI'S OWN COMPACTION WAS OFF TOO, NOTHING COMPACTS UNTIL ONE OF THE TWO IS ON."),
					);
					return;
				}
				// The settings that are one setting for every model: the first model in the try order stands
				// for all of them, because setModelOptions writes these keys to every model in the order.
				const first = Object.keys(loadConfig().models)[0] ?? "";
				if (!first) {
					// 2026-10-07 BUG FIX: with nothing in the Compaction models list there is no model to write
					// these "all models" settings to. It used to write them onto an empty ref, the row kept
					// showing the old number and the setting looked permanently stuck.
					this.ctx.ui?.notify?.("Add a model to the Compaction models list first - these settings are stored on the models that write the summary.", "warning");
					return;
				}
				if (id === "nothinkTag") {
					this.onAction("__exit:edit-nothinktag:menu");
					return;
				}
				if (id === "summaryPct" || id === "summaryMin" || id === "summaryMax" || id === "argcap") {
					// A number: the box runs over the plain chat. This is the ALL-MODELS copy of the setting, so
					// it asks for the first model in the order WITHOUT saying "open that model's options" - the
					// per-model action used here is what threw the operator into the 4-item model screen after a
					// compaction setting (2026-10-07); its own action restores this submenu.
					this.onAction(`__exit:edit-allmodels-number:${id}`);
					return;
				}
				else if (id === "inputstubs") setModelOptions(first, { inputStubs: value === "stubs" });
				else if (id === "preservestubs") setModelOptions(first, { preserveStubs: value as never });
				else if (id === "preserveuser") setModelOptions(first, { preserveUser: value as never });
				else if (id === "preservereply") setModelOptions(first, { preserveReplies: value as never });
				else if (id === "preservethinking") setModelOptions(first, { preserveThinking: value as never });
				else if (id === "chain") setModelOptions(first, { chainMode: isChainMode(value) ? value : "fold" });
				else this.onAction(`__exit:edit-number:${id}`);
			},
			() => this.onAction("__close"),
		);
	}

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	render(width: number): string[] {
		const inner = Math.max(40, width);
		const cfg = loadConfig();
		const lines: string[] = [
			"",
			`Compaction settings - this extension ${cfg.enabled ? "ON" : "off"}, pi's built-in ${readPiCompactionEnabled() ? "ON (needs /reload)" : "off"} - never both on at once`,
		];
		lines.push(...(this.list?.render(inner) ?? []));
		return lines.map((l) => truncateToWidth(l, inner));
	}
}

// ---------------------------------------------------------------- TUI: the settings board
// ---------------------------------------------------------------- TUI: the settings board

interface BoardResult {
	action?: string;
}

/** Where the board's loop should land after an edit: the same board row, or - for a model's
 *  options edit - the Compaction models list with that model's options screen open, on the edited
 *  row. (2026-10-01: the loop used to reopen the board at the top level after every edit.) */
interface BoardRestore {
	/** After the board reopens a model's options screen, open this row's submenu as well (2026-10-21,
	 *  defect #52). A sampling flag is edited over the plain chat, and coming back on the row that
	 *  opens the flags page still left the operator one menu up from where they had been working. */
	subKey?: string;
	rowId?: string;
	orderRef?: string;
	optionKey?: string;
	catDesc?: { kind: string; level?: string };
	compactionMenu?: boolean;
	elisionMenu?: boolean;
	/** Re-open the Summary template submenu (the previews and the template actions live in it, so the
	 *  board has to come back there, not on the row that opened it - that was the previous menu). */
	templateMenu?: boolean;
	/** With templateMenu: step into that submenu's own screen ("userTemplates" = the template library). */
	templateKey?: string;
}

/**
 * The one settings screen. Rows: Extension active, Compaction models (the order list), Retries per
 * model, Wait between retries, pi's two compaction numbers (Reserve tokens / Keep recent tokens -
 * written to pi's settings.json, /reload needed), the elision block (on/off, start %, protected
 * recent, stub tail, min batch savings, min stub batch, stop gap), Additional instruction, Log.
 * Numeric rows are FREE-ENTRY: Enter opens pi's input box, type any number in range. Every
 * description states the default, so forgetting it costs nothing. Everything saves as it is
 * toggled.
 */

/** Enter-sentinel on a free-entry numeric row: onChange receives it and opens the number input. */
const EDIT_NUMBER = "type a number...";

/** Enter-sentinel on a free-text row: onChange receives it and opens the text input. */
const EDIT_TEXT = "type text...";

/** A free-entry numeric row: Enter opens the input; other cycle values (like "full") still work. */
function numberRow(id: string, label: string, value: string, extraValues: string[], description: string): SettingItem {
	return { id, label, currentValue: value, values: [EDIT_NUMBER, ...extraValues], description };
}
export class CompactionBoard implements Component, Focusable {
	private list: SettingsList | null = null;
	private items: SettingItem[] = [];
	private byId = new Map<string, SettingItem>();
	private isFocused = false;

	constructor(
		private tui: TUI,
		private theme: Theme,
		private ctx: Ctx,
		private finish: (result: BoardResult | undefined) => void,
		private restore?: BoardRestore,
	) {
		this.build();
		this.applyRestore();
	}

	/** Land where the last edit happened: highlight the edited board row, or re-open the submenu
	 *  chain (Compaction models -> that model's options screen) on the edited row. */
	/** The Summary template row's value: which layer is in effect right now. */
	private templateStateLabel(): string {
		const cfg = loadConfig();
		const ext = DEFAULT_TEMPLATE_TEXT.trim();
		if (cfg.additionalInstruction !== null && cfg.additionalInstruction !== undefined) {
			const t = cfg.additionalInstruction.trim();
			if (!t) return "extension default (pinned)";
			if (t === ext) return "extension default";
			const usr = cfg.templateUserDefaults?.trim();
			if (usr && t === usr) return "user default";
			return `${cfg.additionalInstruction.split("\n").length} lines (custom)`;
		}
		if (cfg.templateUserDefaults?.trim()) return cfg.templateUserDefaults!.trim() === ext ? "extension default (via the user default)" : "user default";
		return "extension default";
	}

	private applyRestore(): void {
		const r = this.restore;
		if (!r || !this.list) return;
		const select = (id: string): boolean => {
			const i = this.items.findIndex((it) => it.id === id);
			if (i < 0) return false;
			(this.list as unknown as { selectedIndex: number }).selectedIndex = i;
			return true;
		};
		if (r.rowId) {
			select(r.rowId);
			return;
		}
		if (r.catDesc && select("catDesc")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
			return;
		}
		if (r.compactionMenu && select("compactionMenu")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
			return;
		}
		if (r.templateMenu && select("instruction")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
			return;
		}
		if (r.elisionMenu && select("elisionMenu")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
			return;
		}
		// 2026-10-07 BUG FIX: the branches above had no `return`, so ONE restore request opened the
		// requested submenu and then kept going: after a compaction setting (retries, wait) the board
		// opened the compaction submenu, then the elision submenu, then the models list - which is the
		// "it throws me into the 4-item per-model menu" complaint. Exactly one screen is opened now.
		if (r.orderRef && select("order")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
		}
	}

	// -- Component -------------------------------------------------------------

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
		this.list?.handleInput(data);
		this.tui.requestRender();
	}

	render(width: number): string[] {
		const inner = Math.max(30, width);
		const lines: string[] = [];
		lines.push(this.theme.fg("border", "─".repeat(inner)));
		lines.push(this.theme.fg("accent", this.theme.bold("Compaction settings — Enter opens · Esc leaves · type to filter")));
		lines.push("");
		if (this.list) lines.push(...this.list.render(inner));
		return lines.map((l) => truncateToWidth(l, inner));
	}

	// -- Focusable -------------------------------------------------------------

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	// -- internals --------------------------------------------------------------

	private patch(id: string, value: string): void {
		const item = this.byId.get(id);
		if (item) item.currentValue = value;
	}

	private refreshDerived(): void {
		const cfg = loadConfig();

		this.patch("order", rotationText(cfg, this.ctx));


		this.patch("elision", cfg.elision.enabled ? "on" : "off");
		this.patch("elisionProtect", `${fmt(cfg.elision.protectRecentTokens)} tok`);
		this.patch("elisionTail", cfg.elision.stubTailChars === "full" ? "full" : `${cfg.elision.stubTailChars} chars`);
		this.patch("elisionSavings", `${fmt(cfg.elision.minSavingsTokens)} tok`);
		this.patch("elisionResults", `${cfg.elision.minResultsToStub} results`);
		this.patch("elisionStop", `${fmt(cfg.elision.stopGapTokens)} tok`);
		this.patch("elisionProtect", `${fmt(cfg.elision.protectRecentTokens)} tok`);
		this.patch("elisionTail", cfg.elision.stubTailChars === "full" ? "full" : `${cfg.elision.stubTailChars} chars`);
		this.patch("retryWait", `${cfg.retryDelaySeconds} s`);
		this.patch("extReserve", fmt(extReserve(this.ctx)));
		this.patch("extKeep", fmt(extKeepRecent(this.ctx)));
		this.patch("autoCompact", cfg.autoCompact ? "on" : "off");
		this.patch("chatCap", cfg.chatMaxTokensCap > 0 ? `${fmt(cfg.chatMaxTokensCap)} tok` : "no cap");
		this.patch("instruction", this.templateStateLabel());
		this.tui.requestRender();
	}

	private buildItems(): SettingItem[] {
		const cfg = loadConfig();
		const rows: SettingItem[] = [];
		rows.push({
			id: "compactionMenu",
			label: "Compaction settings",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new CompactionMenu(this.tui, this.ctx, (action: string) => {
					if (action === "__close") done();
					else done(action);
				}) as unknown as Component,
			description:
				"THE COMPACTION SWITCHES. ONE compactor is ever on: turning this extension's compaction on turns pi's own compaction off, and the other way round; both can be off, then nothing compacts. Elision has its own submenu and runs whichever compactor is picked. pi's switch is a file on disk, so changing it does nothing until you /reload.",
		});
			rows.push({
			id: "order",
			label: "Compaction models",
			currentValue: "",
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new OrderList(
					this.tui,
					this.ctx,
					done,
					this.restore?.orderRef ? { optionsRef: this.restore.orderRef, optionKey: this.restore.optionKey, subKey: this.restore.subKey } : undefined,
					MODEL_KEEP_ROWS,
				) as unknown as Component,
			description:
				"The models that write the summary, first one first. In the list: type to filter, digits 1-9 move a model to that place (0 = the end), Space removes it from the order, Enter opens the options of a model that is IN the order. Unselected models show the default options and cannot be opened.",
		});
		rows.push({
			id: "elisionMenu",
			label: "Elision settings",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new ElisionMenu(this.tui, this.ctx, (action: string) => {
					if (action === "__close") done();
					done(action);
				}) as unknown as Component,
			description:
				"Old tool results are replaced by short stubs in the requests the model receives, so the context stays much lower for much longer; your screen and the session file always keep the originals. The submenu holds the on/off toggle and all six settings. Enter opens it; Enter on a setting opens its editor (over the plain chat, with the position restored afterwards).",
		});
		rows.push({
			id: "elisionNow",
			label: "Run elision now",
			currentValue: "Run the elision now",
			values: ["Run the elision now"],
			description:
			"Runs an elision sweep RIGHT NOW, regardless of the start percentage and the batch thresholds: every tool result outside the preserved tail becomes a stub (the board closes, the stubs persist in the session state, and the next request carries them). Use it when you can see the context climbing and do not want to wait for the sweep to accumulate candidates.",
		});
		rows.push({
			id: "compactNow",
			label: "Run compaction now",
			currentValue: "Run the compaction now",
			values: ["Run the compaction now"],
			description:
				"Starts a compaction RIGHT NOW: pi prepares the session and the Compaction models rotation answers it (the board closes so you can watch the progress). The automatic starter lives in the Auto-compaction submenu's trigger point - when the ctx crosses it while pi is idle, the same compaction starts by itself.",
		});
		rows.push({
			id: "compactNowResume",
			label: "Run compaction now and resume the session",
			currentValue: "Run the compaction, keep the turn going",
			values: ["Run the compaction, keep the turn going"],
			description:
				"Same rotation as the row above, but the summary is written at a safe point WITHOUT stopping the " +
				"turn: the session goes on by itself on the short history, so you can leave the machine and come " +
				"back to a running session. Use this when you want the room cleared and the work to continue; use " +
				"the row above when you want the session to stop and wait for you. If no safe point can be used it " +
				"says so and writes nothing - the ordinary row is right above it.",
		});
		rows.push({
			id: "compactStock",
			label: "Run pi's built-in compaction now",
			currentValue: `Run pi's built-in compaction (pi aims for summary size of 0.8 x reserveTokens = ${fmt(Math.floor(readPiReserveTokens() * 0.8))} tok)`,
			values: [`Run pi's built-in compaction (pi aims for summary size of 0.8 x reserveTokens = ${fmt(Math.floor(readPiReserveTokens() * 0.8))} tok)`],
			description:
				"Runs pi's OWN compaction once, bypassing the rotation: pi prepares the session and summarizes with the active model, aiming at 0.8 x its reserveTokens (the number on the right). Use it to compare pi's stock summary with the extension's. Everything else (the rotation, the gates) stays in place for the next compaction.",
		});
		rows.push({
			id: "instruction",
			label: "Summary template settings",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new TemplateMenu(
					this.tui,
					this.ctx,
					(action: string) => {
						if (action === "__close") done();
						else done(`__exit:${action}`);
					},
					this.restore?.templateKey,
				) as unknown as Component,
			description:
				"THE SUMMARY TEMPLATE - the exact text the summarizer reads (after the size block). It contains {PLACEHOLDER} tokens the extension substitutes per model: {TOOL CALLS SUMMARY}, {USER PROMPT SUMMARY}, {ASSISTANT REPLIES SUMMARY}, {ASSISTANT THINKING SUMMARY}, {STUB EXAMPLE RULE}, {PREVIOUS SUMMARY RULE}, {PREVIOUS SUMMARY SECTION}. DO NOT reword or delete the {TOKENS} - each is replaced by the per-model option sentence (the editable wording of every token lives in the Category templates screen just below); a deleted token is auto-added at the end with a note. Everything else you write is sent word for word. Enter opens the template submenu: your template library (any number of saved templates, one of them made active), read-only previews of the active / user / extension template, and the resets (each asks first).",
		});
		rows.push({
			id: "catDesc",
			label: "Category templates",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new CategoryPick(
					this.tui,
					this.ctx,
					() => done(),
					(kind: string, level: string) => done(`__exit:edit-category-sentence:${kind}:${level}`),
					(kind: string, target: "ext" | "user") => done(`__exit:preview-category-default:${kind}:${target}`),
					this.restore?.catDesc?.kind,
					this.restore?.catDesc?.level,
				) as unknown as Component,
			description:
				"THE 4 x 4 CATEGORY TEMPLATES - each content category (Tool calls / User prompts / Assistant replies / Assistant thinking) has its OWN editable sentence for EACH level (verbatim / detailed / summary / brief). Enter opens the categories; Enter on a category opens its four sentences; Enter on a sentence opens the editor; Ctrl+R on a sentence resets it to the built-in default (asks here in the screen: Enter confirms, Escape cancels). These substitute into the template's {TOOL CALLS SUMMARY} / {USER PROMPT SUMMARY} / {ASSISTANT REPLIES SUMMARY} / {ASSISTANT THINKING SUMMARY} tokens for EVERY model - each model's combination is visible in its own Preview final description row.",
		});
		rows.push({
			id: "log",
			label: "Log",
			currentValue: "",
			submenu: (_current: string, done: (selectedValue?: string) => void) => new LogView(this.tui, done),
			description: `One line per try: status, model, time, tok in and out, and why something failed. File: ${LOG_PATH}`,
		});
		return rows;
	}

	private build(): void {
		this.items = this.buildItems();
		this.byId.clear();
		for (const item of this.items) this.byId.set(item.id, item);
		this.list = new SettingsList(
			this.items,
			Math.min(this.items.length + 2, 14),
			getSettingsListTheme(),
			(id: string, value: string) => this.onChange(id, value),
			() => this.finish(undefined),
			{ enableSearch: true },
		);
	}

	private onChange(id: string, value: string): void {
		if (typeof value === "string" && value.startsWith("__exit:")) {
			// A submenu screen asked for something that must happen over the plain chat (a free-number
			// edit, a category-sentence editor, a read-only preview): it returned the action through
			// this row's submenu. Finish with it - the command loop performs it over the plain chat
			// (where pi's overlays are safe) and reopens this board, restoring the position.
			this.finish({ action: value.slice("__exit:".length) });
			return;
		}
		if (value === EDIT_NUMBER) {
			this.finish({ action: `edit-number:${id}` });
			return;
		}
		if (value === EDIT_TEXT) {
			this.finish({ action: "edit-nothinktag" });
			return;
		}
		if (id === "enabled") {
			// ONE COMPACTOR (2026-10-07): the switch is refused while another manager is present.
			if (value === "on" && enableBlockedByOtherCompactor()) {
				(this as unknown as { rebuild?: () => void }).rebuild?.();
				return;
			}
		} else if (id === "directRequest") {
			update((c) => void (c.directRequest = value === "on"));
		} else if (id === "retries") {
			update((c) => void (c.retries = clamp(Number(value) || 0, 0, 5)));
		} else if (id === "retryWait") {
			update((c) => void (c.retryDelaySeconds = parseDuration(value) / 1000));
		} else if (id === "elision") {
			update((c) => void (c.elision = { ...c.elision, enabled: value === "on" }));
		} else if (id === "chatCap") {
			update((c) => void (c.chatMaxTokensCap = value === "no cap" ? 0 : Number(String(value).replace(/[^0-9]/g, "")) || 0));
		} else if (id === "elisionNow") {
			// A manual sweep: the board closes, the sweep runs against the current session, and the
			// stubs persist in the state (the next request carries them).
			this.finish({ action: "elision-now" });
			return;
		} else if (id === "compactNow") {
			// Manual compaction through pi (same as /compact): it fires session_before_compact and the
			// fork's rotation answers. Close the board first so the progress is visible.
			const ctx = this.ctx;
			// A manual run is what re-arms the proactive trigger after a pause (see pauseAutoCompaction).
			lastCompactionTrigger = "manual";
			this.finish({ action: "compact-now" });
			ctx?.compact?.({
				onComplete: () => ctx?.ui?.notify?.("Manual compaction finished.", "info"),
				onError: (err: any) => ctx?.ui?.notify?.(`Manual compaction failed: ${String(err?.message ?? err).slice(0, 160)}`, "error"),
			});
			return;
		} else if (id === "compactNowResume") {
			// Manual compaction that does NOT stop the session: the summary lands at a safe point and the
			// running loop reads the short history on its next round. Pressing this row is the instruction, so
			// it runs whatever the landWithoutAbort setting happens to say.
			const ctx2 = this.ctx;
			lastCompactionTrigger = "manual";
			this.finish({ action: "compact-resume-now" });
			if (!landNoAbortHook) {
				ctx2?.ui?.notify?.("Compaction with resume is not available with this version of pi - use 'Run compaction now'.", "warning");
			} else {
				void landNoAbortHook(ctx2, "menu")
					.then((r) => {
						if (!r?.ok) ctx2?.ui?.notify?.(`COMPACTION NOT LANDED · the summary could not be written without stopping the turn${r?.note ? ` (${String(r.note).slice(0, 140)})` : ""}. Nothing was written - use 'Run compaction now'.`, "warning");
					})
					.catch((err: any) => ctx2?.ui?.notify?.(`COMPACTION NOT LANDED · ${String(err?.message ?? err).slice(0, 160)}. Nothing was written.`, "error"));
			}
			return;
		} else if (id === "compactStock") {
			// A one-shot stock run: the rotation sees the flag and steps aside, so pi's own
			// compaction (0.8 x its reserveTokens aim) handles this one.
			skipNextStockCompaction = true;
			this.finish({ action: "compact-stock-now" });
			const c = this.ctx;
			c?.compact?.({
				onComplete: () => c.ui?.notify?.("pi's built-in compaction finished.", "info"),
				onError: (err: any) => c.ui?.notify?.(`pi's built-in compaction failed: ${String(err?.message ?? err).slice(0, 160)}`, "error"),
			});
			return;
		} else if (id === "elisionStart") {
			update((c) => void (c.elision = { ...c.elision, softPercent: Number(String(value).replace("%", "")) }));
		} else if (id === "elisionProtect") {
			update((c) => void (c.elision = { ...c.elision, protectRecentTokens: Number(String(value).replace(/[^0-9]/g, "")) }));
		} else if (id === "elisionTail") {
			update((c) => void (c.elision = { ...c.elision, stubTailChars: value === "full" ? "full" : Number(String(value).replace(/[^0-9]/g, "")) }));
		} else if (id === "elisionStart" || id === "elisionProtect" || id === "elisionSavings" || id === "elisionResults" || id === "elisionStop") {
			// cycle rows were converted to free entry; the only non-EDIT value left is their current
			// display - re-open the input (safety net):
			this.finish({ action: `edit-number:${id}` });
			return;
		} else if (id === "elisionSavings") {
			update((c) => void (c.elision = { ...c.elision, minSavingsTokens: Number(String(value).replace(/[^0-9]/g, "")) }));
		} else if (id === "elisionResults") {
			update((c) => void (c.elision = { ...c.elision, minResultsToStub: Number(String(value).replace(/[^0-9]/g, "")) || 1 }));
		} else if (id === "elisionStop") {
			update((c) => void (c.elision = { ...c.elision, stopGapTokens: Number(String(value).replace(/[^0-9]/g, "")) }));
		} else if (id === "marker") {
			update((c) => void (c.noThinkMarker = value === "(empty)" ? "" : value));
		} else if (id === "instruction") {
			// The row is a submenu now; everything routes through __exit (handled above).
		}
		this.refreshDerived();
	}
}

// ---------------------------------------------------------------- TUI: log view

/** Read-only view of the last log lines; Esc closes. */
export class LogView implements Component {
	constructor(private tui: any, private done: () => void) {}
	handleInput(data: string): void {
		const k = keyOf(data);
		if (k === "escape" || k === "enter" || k === "q") this.done();
		else if (k === "v") { this.verbose = !this.verbose; this.offset = -1; this.tui?.requestRender?.(); }
		else if (k === "up") { this.offset = Math.max(0, this.offset - 1); this.tui?.requestRender?.(); }
		else if (k === "down") { this.offset += 1; this.tui?.requestRender?.(); }
		else if (k === "pgup") { this.offset = Math.max(0, this.offset - this.perPage); this.tui?.requestRender?.(); }
		else if (k === "pgdn") { this.offset += this.perPage; this.tui?.requestRender?.(); }
	}
	private offset = -1; // -1 = start at the newest page
	/** false = the result lines only; true = every line, bookkeeping included (v toggles it) */
	private verbose = false;
	/** Rows wrap now, so one result is often two or three screen rows: fourteen results is about a full
	 *  screen of text (2026-10-23, on request - the rows were being cut at the right edge). */
	private perPage = 14;

	invalidate(): void {}

	render(width: number): string[] {
		const all = tailLogPretty(200, this.verbose).split("\n");
		const perPage = this.perPage;
		const maxOffset = Math.max(0, all.length - perPage);
		if (this.offset < 0) this.offset = maxOffset; // the first open shows the NEWEST entries
		this.offset = Math.min(this.offset, maxOffset);
		const shown = all.slice(this.offset, this.offset + perPage);
		const lines = [
			`Log — ${this.verbose ? "EVERY line, bookkeeping included" : "results only"} — showing ${all.length - this.offset - shown.length + 1}-${all.length - this.offset} of ${all.length} — v shows ${this.verbose ? "just the results" : "every line"}, Up/Down or PgUp/PgDn to scroll, Esc to go back`,
			"",
			...shown,
		];
		// WRAP, DO NOT CUT (2026-10-23): every row of this log is a result the operator has to read in full
		// - which model, how many tokens in and out, what failed - and a row longer than the terminal simply
		// stopped existing past the right edge. Continuation rows are indented, so a wrapped entry still
		// reads as one entry.
		const out: string[] = [];
		for (const l of lines) {
			const parts = wrapToWidth(l, Math.max(60, width));
			out.push(parts[0]);
			for (const p of parts.slice(1)) out.push(`  ${p}`);
		}
		return out;
	}
}

// ---------------------------------------------------------------- template menu + editor

function templateMenu(done: (selectedValue?: string) => void): Component {
	const cfg = loadConfig();
	const own = cfg.additionalInstruction;
	const rows: SettingItem[] = [
		{
			id: "edit",
			label: "Edit in the text editor:",
			currentValue: own ? "custom text saved" : "default text in use",
			values: ["open the editor"],
			description:
				"The text is added after pi's own compaction prompt, for every model (local and online alike). 'default' is the built-in text below - resetting always gets you back to it, exactly as shipped.",
		},
		{
			id: "reset",
			label: "Reset to the default text:",
			currentValue: own ? own.split("\n")[0].slice(0, 44) : "already default",
			values: ["reset now"],
			description: "Throws away your custom text and puts the default Additional instruction back.",
		},
	];
	return new SettingsList(
		rows,
		rows.length + 2,
		getSettingsListTheme(),
		(id: string, value: string) => {
			if (id === "edit" && value === "open the editor") done("edit");
			else if (id === "reset" && value === "reset now") done("reset");
		},
		() => done(),
	);
}

/** Break one line into screen rows of at most `width` columns, cutting at a space where there is one.
 *  2026-10-21, on request: a template line is written as one long line, and the read-only preview cut it
 *  at the right edge, so the middle of every sentence was simply invisible. A word longer than the row is
 *  split rather than dropped - nothing in a preview may be unreachable. */
function wrapToWidth(line: string, width: number): string[] {
	const w = Math.max(10, width);
	if (!line) return [""];
	if (line.length <= w) return [line];
	const out: string[] = [];
	let rest = line;
	while (rest.length > w) {
		const cut = rest.lastIndexOf(" ", w) > 0 ? rest.lastIndexOf(" ", w) : w;
		out.push(rest.slice(0, cut));
		rest = rest.slice(cut).replace(/^ +/, "");
		if (!rest) break;
	}
	if (rest) out.push(rest);
	return out.length ? out : [""];
}

/**
 * A text screen of this extension's own (2026-10-06). pi's extension UI context carries only select,
 * confirm, input and notify - there is NO editor in it - so every editor and preview this extension
 * opened through it silently did nothing: picking a row closed the menu and left the user back at the
 * prompt with nothing shown. The previews and the editors are therefore drawn by this extension itself,
 * in the same kind of overlay the settings board uses.
 *
 * Keys: Up/Down move (in a one-line field they do nothing), Left/Right move the caret, printable keys
 * type, Backspace deletes before the caret, Enter starts a new line (in a one-line field it saves),
 * Ctrl+S saves, Escape closes without saving.
 */
/** How many screen rows a read-only view shows at once - the same window the editable view uses. */
const READONLY_PAGE = 19;

export class TextScreen implements Component, Focusable {
	private lines: string[];
	private line = 0;
	private col = 0;
	private top = 0;
	/** Where the read-only view starts, counted in screen rows after wrapping. */
	private vrow = 0;
	private dirty = false;
	private isFocused = false;

	constructor(
		private tui: TUI,
		private theme: Theme,
		private title: string,
		text: string,
		private editable: boolean,
		private done: (value?: string) => void,
	) {
		this.lines = String(text ?? '').split('\n');
	}

	invalidate(): void {
		/* the screen is drawn from the lines array on every pass */
	}

	get focused(): boolean {
		return this.isFocused;
	}

	set focused(value: boolean) {
		this.isFocused = value;
	}

	private move(line: number, col: number): void {
		this.line = Math.max(0, Math.min(this.lines.length - 1, line));
		this.col = Math.max(0, Math.min((this.lines[this.line] ?? '').length, col));
	}

	handleInput(data: string): void {
		const k = keyOf(data);
		if (k === 'escape') {
			// Escape means drop it - the header says so. Ctrl+S (or Enter in a one-line field) is the save key.
			this.done(undefined);
			return;
		}
		if (chord(data, 'ctrl+s')) {
			this.done(this.lines.join('\n'));
			return;
		}
		if (!this.editable) {
			// Read-only scrolling moves by SCREEN rows, not source lines (2026-10-21, the same report as the
			// word wrap): one wrapped paragraph is several rows, so a line at a time jumped past whatever was
			// actually on screen. 'end' asks for more than exists and render() clamps it to the last page.
			if (k === 'up') this.vrow = Math.max(0, this.vrow - 1);
			else if (k === 'down') this.vrow += 1;
			else if (k === 'pgup') this.vrow = Math.max(0, this.vrow - READONLY_PAGE);
			else if (k === 'pgdn') this.vrow += READONLY_PAGE;
			else if (k === 'home') this.vrow = 0;
			else if (k === 'end') this.vrow = Number.MAX_SAFE_INTEGER;
			this.tui.requestRender();
			return;
		}
		if (k === 'up') this.move(this.line - 1, this.col);
		else if (k === 'down') this.move(this.line + 1, this.col);
		else if (k === 'left') this.move(this.line, this.col - 1);
		else if (k === 'right') this.move(this.line, this.col + 1);
		else if (k === 'home') this.move(this.line, 0);
		else if (k === 'end') this.move(this.line, (this.lines[this.line] ?? '').length);
		else if (k === 'enter' && !this.multiline) {
			this.done(this.lines.join('\n'));
			return;
		} else if (k === 'enter') {
			const cur = this.lines[this.line] ?? '';
			this.lines.splice(this.line, 1, cur.slice(0, this.col), cur.slice(this.col));
			this.move(this.line + 1, 0);
			this.dirty = true;
		} else if (k === 'backspace') {
			const cur = this.lines[this.line] ?? '';
			if (this.col > 0) this.lines[this.line] = cur.slice(0, this.col - 1) + cur.slice(this.col);
			else if (this.line > 0) {
				const prev = this.lines[this.line - 1];
				this.lines.splice(this.line - 1, 2, prev + cur);
				this.move(this.line - 1, prev.length);
			}
			this.dirty = true;
		} else if (k === 'space') {
			const cur = this.lines[this.line] ?? '';
			this.lines[this.line] = cur.slice(0, this.col) + ' ' + cur.slice(this.col);
			this.col += 1;
			this.dirty = true;
		} else if (data && data.length === 1 && data >= ' ' && data !== ' ' && data !== '\t') {
			const cur = this.lines[this.line] ?? '';
			this.lines[this.line] = cur.slice(0, this.col) + data + cur.slice(this.col);
			this.col += 1;
			this.dirty = true;
		} else if (data === '\t') {
			const cur = this.lines[this.line] ?? '';
			this.lines[this.line] = cur.slice(0, this.col) + '\t' + cur.slice(this.col);
			this.col += 1;
			this.dirty = true;
		}
		this.tui.requestRender();
	}

	/** The template is many lines, so Enter is a newline there and Ctrl+S is the save key. */
	private get multiline(): boolean {
		return this.lines.length > 1 || this.title.toLowerCase().includes('template') || this.title.toLowerCase().includes('sentence');
	}

	render(width: number): string[] {
		const inner = Math.max(40, width);
		const theme = this.theme as any;
		const head = typeof theme?.bold === 'function' ? theme.bold(this.title) : this.title;
		const keys = this.editable
			? (this.multiline
				? 'type to edit - Up/Down line, Left/Right caret, Enter breaks the line, Ctrl+S saves, Esc closes without saving'
				: 'type to edit - Enter saves, Esc closes without saving')
			: 'Up/Down or PgUp/PgDn to scroll - Esc closes';
		const out: string[] = ['', head, typeof theme?.dim === 'function' ? theme.dim(keys) : keys, ''];
		if (!this.editable) {
			// WORD WRAP, read-only only (2026-10-21, the operator: "the preview does not word wrap, so i
			// cannot see what it says"). The editor below deliberately keeps one long line per line: the caret
			// position, the arrow keys and the saved text all speak in real lines, and wrapping the editing
			// view would break the first two to fix the third.
			const rows: string[] = [];
			for (const l of this.lines) rows.push(...wrapToWidth(l.replace(/\t/g, "  "), inner - 2));
			const maxTop = Math.max(0, rows.length - READONLY_PAGE);
			this.vrow = Math.max(0, Math.min(this.vrow, maxTop));
			for (let i = this.vrow; i < Math.min(rows.length, this.vrow + READONLY_PAGE); i++) out.push("  " + rows[i]);
			if (!rows.length) out.push("  (nothing to show)");
			const where = `  row ${String(this.vrow + 1)}-${String(Math.min(rows.length, this.vrow + READONLY_PAGE))} of ${String(rows.length)}`;
			out.push(typeof theme?.dim === 'function' ? theme.dim(where) : where);
			return out.map((l) => truncateToWidth(l, inner));
		}
		if (this.line - this.top > 17) this.top = this.line - 17;
		if (this.line < this.top) this.top = this.line;
		const last = Math.min(this.lines.length, this.top + 19);
		for (let i = this.top; i < last; i++) {
			const mark = this.editable && i === this.line ? '>' : ' ';
			const text = this.editable && i === this.line ? (this.lines[i].slice(0, this.col) + '|' + this.lines[i].slice(this.col)) : this.lines[i];
			out.push(truncateToWidth(mark + ' ' + text, inner));
		}
		return out.map((l) => truncateToWidth(l, inner));
	}
}

/** Show text read-only, or edit it, in this extension's own screen. Resolves with the new text, or
 *  with undefined when it was closed without saving. */
async function showText(ctx: Ctx, title: string, text: string, editable: boolean): Promise<string | undefined> {
	if ((ctx as any)?.mode !== 'tui' || typeof (ctx as any)?.ui?.custom !== 'function') {
		// No terminal UI to draw in (a script run): the text goes to the notice instead.
		ctx?.ui?.notify?.(title + '\n\n' + (text.length > 3000 ? text.slice(0, 3000) + '\n... (cut here)' : text), 'info');
		return undefined;
	}
	return await (ctx as any).ui.custom((tui: TUI, theme: Theme, _kb: KeybindingsManager, done: (value?: string) => void) => new TextScreen(tui, theme, title, text, editable, done) as unknown as Component & { dispose?: () => void }) as string | undefined;
}

/** Show one of the templates, read-only, in this extension's own screen. */
async function uiPreview(ctx: Ctx, title: string, text: string): Promise<void> {
	await showText(ctx, title, text, false);
}

/**
 * Edit a piece of text - a template or a category sentence - in PI'S OWN multi-line editor (2026-10-07,
 * on request: "every setting where I need to enter characters shows some weird vertical line cursor, it
 * used to be a normal editing box with a normal cursor"). The extension's own screen draws its own caret,
 * which does not read as an editor; pi's editor is the box the operator knows. The old screen stays as the
 * fallback for a host that has no editor, and it is still used for read-only previews.
 */
async function editText(ctx: Ctx, title: string, prefill: string): Promise<string | undefined> {
	const ed = (ctx.ui as any)?.editor;
	if (typeof ed === "function") return await ed.call(ctx.ui, title, prefill);
	return await showText(ctx, title, prefill, true);
}

/** A one-line value (a number, a tag) in pi's own one-line box, same reason. */
async function editField(ctx: Ctx, title: string, value: string): Promise<string | undefined> {
	const inp = (ctx.ui as any)?.input;
	if (typeof inp === "function") return await inp.call(ctx.ui, title, value);
	return await showText(ctx, title, value, true);
}

/**
 * What a summarizer would actually be sent for one model: the template with its placeholders already
 * filled in for this session, plus the block order and the system prompt. This is what the Summary
 * template menu previews - the point of a preview is to see the request, not the template source.
 */
async function previewRequest(ctx: Ctx, ref: string, full = false): Promise<void> {
	const cfg = loadConfig();
	const model = lookupRef(ref, ctx);
	// 2026-10-21, on request: "if the current compaction model number 1 is not available, the preview
	// refuses to open. but that should not happen, because the preview does not depend on a model". It was
	// right - a preview sends nothing, and what it shows (the active template with its tags filled, the
	// size block, the stubbing, the caps) is read from the saved settings for that ref, not measured from
	// a live model. So it previews, and says plainly which one figure came out of the session's own window
	// because the model itself was not there to be asked.
	if (!model) {
		ctx.ui?.notify?.(`Previewing ${ref} from its saved settings - that model is not available right now, so nothing about it was measured live.`, "info");
	}
	const profile = profileFor(cfg, model, ref);
	const projection: any = (ctx as any)?.sessionManager?.buildSessionProjection?.();
	const msgs: any[] = ((projection?.messages ?? []) as any[]).filter((m: any) => m?.role !== "system");
	if (msgs.length < 4) {
		ctx.ui?.notify?.("Not enough conversation yet to preview a real request.", "info");
		return;
	}
	const kept = msgs.slice(-2);
	const region = msgs.slice(0, -2);
	const prevTurn = readPrevTurnInfo(ctx, {});
	const main = serializeRegion(region, { startTurn: prevTurn.lastTurn, inputStubs: profile.inputStubs, argCap: profile.argCap });
	const keptText = kept.map((m: any) => `[${String(m.role)} - kept]: ${contentTextOf(m.content).slice(0, 400)}`).join("\n\n");
	const inputTokens = Math.ceil((main.text.length * 1.3) / 4) + SUMMARISER_FRAMING_TOKENS;
	const sizes = sizesFor(profile, { inputTokens, historyTokens: inputTokens, headroom: 0, overflow: false, windowTokens: model?.contextWindow ?? (ctx as any)?.model?.contextWindow ?? 0 }, 0);
	const instructions = buildInstructions({ sizes, inputTokens, segmentTokens: inputTokens - SUMMARISER_FRAMING_TOKENS, kind: model ? (isLocalModel(model) ? "local" : "online") : (/localhost|127\.0\.0\.1/i.test(ref) ? "local" : "online"), options: profile, additionalInstruction: cfg.additionalInstruction ?? null, userDefault: cfg.templateUserDefaults, customInstructions: undefined });
	// 2026-10-07 ORDER FIX ("it shows random code, not the request"): the conversation region used to come
	// FIRST, and in a session full of scripts and JSON that is a wall of code, with the filled template
	// pushed off the bottom. The request is presented in the order a reader wants it: the INSTRUCTIONS (the
	// active template with every tag filled in), then the SYSTEM PROMPT, then the conversation region - and
	// the region is DIGESTED by default (counts, turns, a peek at the ends), because it is the user's own
	// session, not something the template preview is for. The extra row shows it unabridged.
	const peek = (m: any) => `[${String(m?.role)}]: ${contentTextOf(m?.content).replace(/\s+/g, " ").slice(0, 180)}`;
	const conv = full
		? ['<conversation>', main.text, '</conversation>']
		: [
			`<conversation>  (${region.length} messages, Turns ${String(prevTurn.lastTurn + 1)}-${String(main.lastTurn)}, about ${fmt(inputTokens)} tok - YOUR OWN SESSION, left out of this preview so the template is readable. "View the summary request with the full conversation" shows it all.)`,
			`  first message -> ${peek(region[0])}`,
			`  last message  -> ${peek(region[region.length - 1])}`,
			`</conversation>`,
		];
	const text = [
		`REQUEST PREVIEW - everything ${ref} would receive for a compaction of this session (nothing is sent).`,
		`PART 1 OF 3 - THE INSTRUCTIONS: the ACTIVE template with every tag filled in for this session - the size block, the preserve rules, the previous-summary rule, your extra instruction.`,
		instructions,
		`PART 2 OF 3 - THE SYSTEM PROMPT that rides above everything:`,
		COMPACTOR_SYSTEM_PROMPT,
		`PART 3 OF 3 - THE CONVERSATION REGION to summarize. ${labeledPreviousSummary(undefined, prevTurn.lastTurn, prevTurn.lastTurn + 1)}`,
		...conv,
		`[kept tail, for the summarizer's context - NOT part of the summarized region:]`,
		full ? keptText : `${peek(kept[0])}
  ... ${kept.length} kept messages in the real request`,
	].filter(Boolean).join("\n\n");
	await showText(ctx, `What ${ref} would receive - the real request (${full ? "full" : "digest"}, ${fmt(text.length)} chars)`, text, false);
}

/** One of the stored templates, as stored. */
async function previewTemplate(ctx: Ctx, which: string): Promise<void> {
	const cfg = loadConfig();
	const ext = DEFAULT_TEMPLATE_TEXT;
	// One template (2026-10-06): the text the template menu edits, or the built-in one when nothing is
	// saved. A stored "additional instruction" is an extra paragraph after it, never a second template.
	const usr = cfg.templateUserDefaults?.trim() ? cfg.templateUserDefaults! : null;
	const extra = cfg.additionalInstruction?.trim();
	const active = (usr ?? ext) + (extra && !looksLikeTemplate(extra) ? String.fromCharCode(10) + String.fromCharCode(10) + extra : "");
	const text =
		which === "ext"
			? ext
			: which === "user"
				? usr ?? "(no user default saved)\n\nWhen the active template is empty, this slot follows the extension default. Save one with 'Save current as user default' in the template submenu."
				: active;
	await uiPreview(ctx,
		`READ-ONLY PREVIEW — the summary template (${which === "ext" ? "extension default" : which === "user" ? "user default" : "active"}). Nothing is sent; changes are DISCARDED when you close it.`,
		text,
	);
}

/** The user-default actions for the main template, each confirmed with pi's select (the board is
 *  closed here, so overlays are safe). */
async function templateDefaultAction(ctx: Ctx, action: string): Promise<void> {
	const cfg = loadConfig();
	if (action === "save-template-user-default") {
		// One template: a copy is a copy of the text that is actually in use.
		const active = cfg.templateUserDefaults?.trim() || DEFAULT_TEMPLATE_TEXT;
		const ok = await ctx.ui?.select?.("Save the CURRENT template as your user default?", ["Confirm - save as user default", "Cancel"]);
		if (ok !== "Confirm - save as user default") return;
		update((c) => void (c.templateUserDefaults = active));
		ctx.ui?.notify?.("Saved as the user default. The active template is unchanged; 'Reset to user default' restores it.", "info");
		return;
	}
	if (action === "reset-template-user-default") {
		if (!cfg.templateUserDefaults?.trim()) {
			ctx.ui?.notify?.("No user default is saved - nothing to reset to. Save one with 'Save current as user default'.", "info");
			return;
		}
		const ok = await ctx.ui?.select?.("Reset the template to your USER default? (the active custom text is dropped)", ["Confirm - reset to user default", "Cancel"]);
		if (ok !== "Confirm - reset to user default") return;
		update((c) => void (c.templateUserDefaults = c.templatePreviousCopy));
		ctx.ui?.notify?.("The kept copy is the template again.", "info");
		return;
	}
	const ok = await ctx.ui?.select?.("Reset the template to the EXTENSION default? (the active custom text is dropped; your user default is kept)", ["Confirm - reset to extension default", "Cancel"]);
	if (ok !== "Confirm - reset to extension default") return;
	update((c) => void (c.templateUserDefaults = null));
	ctx.ui?.notify?.("The template is pinned to the extension default.", "info");
}

/** Read-only: one category's four default sentences (user or extension layer). */
async function previewCategoryDefaults(ctx: Ctx, kind: string, target: string): Promise<void> {
	const cfg = loadConfig();
	const meta = CATEGORY_KINDS.find((k) => k.kind === kind);
	const label = meta?.label ?? kind;
	const lines = CATEGORY_LEVELS.map((level) => {
		const kk = kind as "stubs" | "user" | "replies" | "thinking";
		if (target === "ext") return `${label} — ${level} (extension default):\n${PRESERVE_SENTENCES[level][kk]}`;
		const u = cfg.categoryUserDefaults?.[kind]?.[level]?.trim();
		if (!u) return `${label} — ${level} (no user default saved):\n(falls back to the extension default)`;
		const tag = userDefaultLayer(kk, level) === "ext" ? "same as the extension default" : "custom";
		return `${label} — ${level} (${tag}):\n${u}`;
	}).join("\n\n");
	await uiPreview(ctx,
		`READ-ONLY PREVIEW — ${label}: ${target === "ext" ? "extension" : "user"} defaults. Nothing is sent; changes are DISCARDED when you close it.`,
		lines,
	);
}

async function editInstruction(ctx: Ctx): Promise<void> {
	const cfg = loadConfig();
	const ext = DEFAULT_ADDITIONAL_INSTRUCTION;
	let prefill =
		cfg.additionalInstruction !== null && cfg.additionalInstruction !== undefined
			? cfg.additionalInstruction.trim() || ext
			: cfg.templateUserDefaults?.trim() || ext;
	const title = `Summary template — the exact text the summarizer reads (the {TOKENS} substitute per model). Submit = save as the ACTIVE template; empty = follow the user default; submit the lone word :extension or :user to restart from that default.`;
	for (;;) {
		const text = await editText(ctx, title, prefill);
		if (text === undefined) return;
		const t = String(text).trim();
		if (t === ":extension") {
			prefill = ext;
			ctx.ui?.notify?.("Restarted from the extension default. Edit and submit to save it as the active template.", "info");
			continue;
		}
		if (t === ":user") {
			const usr = cfg.templateUserDefaults?.trim();
			prefill = usr ?? ext;
			ctx.ui?.notify?.(usr ? "Restarted from your user default." : "No user default saved - restarted from the extension default instead.", "info");
			continue;
		}
		// One template (2026-10-06): what is edited here is what is sent. Empty clears it, and with it
		// the extension default rides again.
		update((c) => void (c.templateUserDefaults = t ? String(text) : null));
		ctx.ui?.notify?.(
			t
				? t === ext.trim()
					? "Saved: the template is pinned to the extension default."
					: "Saved as the active template. Preview per model: /compact-plus preview <ref>."
				: "Empty - the template follows the user default (or the extension default when none is saved).",
			"info",
		);
		return;
	}
}

// ---------------------------------------------------------------- the menu loop + non-TUI fallback

async function runBoard(ctx: Ctx): Promise<void> {
	let restore: BoardRestore | undefined;
	for (;;) {
		const cfg = loadConfig();
		if (ctx.mode !== "tui") {
			await fallbackMenu(ctx);
			return;
		}
		const factory = (tui: TUI, theme: Theme, _kb: KeybindingsManager, done: (r: BoardResult | undefined) => void): Component =>
			new CompactionBoard(tui, theme, ctx, done, restore);
		const result = (await ctx.ui.custom(factory)) as BoardResult | undefined;
		if (!result?.action) return;
		// pi's compaction flag is read for information only and never written by this extension.
		// The automatic trigger here is this extension's own: it measures the live context and starts
		// the compaction itself, so with pi's flag off automatic compaction still works - and pi never
		// writes a summary of its own. The coupling that armed pi's flag is deleted for good.
	if (result.action?.startsWith("__exit:")) result.action = result.action.slice("__exit:".length);
	if (result.action?.startsWith("notice:")) {
			// 2026-10-07 BUG FIX: these are the two compaction switches, and this used to QUIT the settings
			// after the notice - the "selecting this item throws me out of the menu completely" complaint.
			// The notice now rides on top of the reopened Compaction settings submenu, on the same row.
			ctx.ui.notify(result.action.slice("notice:".length), "info");
			restore = { compactionMenu: true };
			continue;
		}
		if (result.action === "edit-instruction") {
			await editInstruction(ctx);
			restore = { templateMenu: true };
			continue;
		}
		// The request preview: the filled-in template, not the template source. It used to sit inside the
		// edit-instruction branch above, where it could never be reached - that is why picking a model in
		// the template menu closed the menu and showed nothing.
		if (result.action?.startsWith("preview-request-full:")) {
			await previewRequest(ctx, decodeURIComponent(result.action.slice("preview-request-full:".length)), true);
			restore = { templateMenu: true };
			continue;
		}
		// The digest version (the row on top). Self-check 2026-10-07: adding the -full variant above first
		// OVERWROTE this branch - the row would have opened nothing. The two prefixes are distinct
		// ("preview-request:" is not a prefix of "preview-request-full:"), so order does not matter here.
		if (result.action?.startsWith("preview-request:")) {
			await previewRequest(ctx, decodeURIComponent(result.action.slice("preview-request:".length)));
			restore = { templateMenu: true };
			continue;
		}
		if (result.action?.startsWith("preview-template:")) {
			await previewTemplate(ctx, result.action.slice("preview-template:".length));
			restore = { templateMenu: true };
			continue;
		}
		if (result.action === "elision-now") {
			const projection: any = ctx?.sessionManager?.buildSessionProjection?.();
			const msgs: any[] = (projection?.messages ?? []).filter((m: any) => m?.role !== "system");
			const r = applyElision(structuredClone(msgs), ctx, { force: true });
			ctx?.ui?.notify?.(
				r
					? `Elision done - ${fmt(r.messages.length)} messages processed; the stubs are live from the next request.`
					: "Nothing to elide - no eligible tool results outside the preserved tail.",
				"info",
			);
			restore = { rowId: "elisionNow" };
			continue;
		}
		if (result.action === "compact-now") {
			// The board started it before finishing; the board must stay closed so the progress is visible.
			return;
		}
		if (result.action === "compact-resume-now") {
			// The row started it itself, without stopping the turn. The board stays closed so the session is
			// visible while the summariser works, and there is nothing to resume afterwards.
			return;
		}
		if (result.action === "compact-stock-now") {
			// The compaction was started by the dispatch (ctx.compact with the skip flag set); the
			// board stays closed while it runs.
			return;
		}
		if (result.action === "save-template-user-default" || result.action === "reset-template-user-default" || result.action === "reset-template-extension-default") {
			await templateDefaultAction(ctx, result.action);
			restore = { templateMenu: true };
			continue;
		}
		if (result.action?.startsWith("preview-category-default:")) {
			const body = result.action.slice("preview-category-default:".length);
			const at = body.lastIndexOf(":");
			if (at > 0) await previewCategoryDefaults(ctx, body.slice(0, at), body.slice(at + 1));
			restore = at > 0 ? { catDesc: { kind: body.slice(0, at) } } : undefined;
			continue;
		}
		if (result.action?.startsWith("edit-number:")) {
			const id = result.action.slice("edit-number:".length);
			await editNumber(ctx, id);
			// Return to the submenu the row lives in (the scaling/elision rows moved into submenus).
			if (id === "reserveTok" || id === "preserveTok") restore = { compactionMenu: true };
			else if (id.startsWith("elision")) restore = { elisionMenu: true };
			else if (id === "retries" || id === "retryWait") restore = { compactionMenu: true };
			else restore = { rowId: id };
			continue;
		}
		if (result.action?.startsWith("edit-allmodels-number:")) {
			// The "one setting for every model" numbers, stored on the first model of the order. The position
			// goes back to the Compaction settings submenu - NOT to that model's own options screen, which is
			// where the per-model action landed and what looked like the menu jumping away (2026-10-07).
			const only = result.action.slice("edit-allmodels-number:".length);
			const firstRef = Object.keys(loadConfig().models)[0] ?? "";
			if (firstRef) await editModelNumber(ctx, firstRef, only);
			else ctx.ui?.notify?.("Add a model to the Compaction models list first - these settings are stored on the models that write the summary.", "warning");
			restore = { compactionMenu: true };
			continue;
		}
		if (result.action?.startsWith("edit-model-number:")) {
			// The ref may itself contain ':' (llama-server URLs), so split from the RIGHT.
			const body = result.action.slice("edit-model-number:".length);
			const at = body.lastIndexOf(":");
			if (at > 0) await editModelNumber(ctx, body.slice(0, at), body.slice(at + 1));
			// A sampling flag is not a row of its own on the model screen - it lives behind the "Sampling
			// flags" page - so the position is restored on that row, not on a row that is not there.
			const mKey = at > 0 ? body.slice(at + 1) : "";
			restore = at > 0
				? { orderRef: body.slice(0, at), subKey: (SAMPLING_KEYS as readonly string[]).includes(mKey) ? "sampling" : undefined, optionKey: (SAMPLING_KEYS as readonly string[]).includes(mKey) ? "sampling" : mKey } : undefined;
			continue;
		}
		if (result.action?.startsWith("edit-category-sentence:")) {
			const body = result.action.slice("edit-category-sentence:".length);
			const at = body.lastIndexOf(":");
			if (at > 0) await editCategorySentence(ctx, body.slice(0, at), body.slice(at + 1));
			restore = at > 0 ? { catDesc: { kind: body.slice(0, at), level: body.slice(at + 1) } } : undefined;
			continue;
		}
		if (result.action?.startsWith("preview-description:")) {
			const ref = result.action.slice("preview-description:".length);
			await previewDescription(ctx, ref);
			restore = { orderRef: ref, optionKey: "previewDesc" };
			continue;
		}
		if (result.action === "edit-nothinktag" || result.action === "edit-nothinktag:menu") {
			await editNoThinkTag(ctx);
			// 2026-10-07 BUG FIX: this restored to { rowId: "marker" }, a row id that does not exist, so the
			// board came back on its FIRST row - the "it takes me to the previous menu" report. The row is
			// "nothinkTag"; the all-models copy of it lives in the Compaction settings submenu and says so.
			// The tag's row lives in the Compaction settings submenu (id "nothinkTag"), so that is where both
			// paths come back - the all-models copy and the board-level text-row path, which is the same row.
			restore = { compactionMenu: true };
			continue;
		}
		if (result.action?.startsWith("edit-sampling:")) {
			const ref = result.action.slice("edit-sampling:".length);
			await editSamplingFlags(ctx, ref);
			restore = { orderRef: ref, optionKey: "sampling" };
			continue;
		}
		// 2026-10-07 BUG FIX: anything the dispatch above did not recognise used to END the settings and
		// drop the operator in the chat (and the next /compact-plus opened at the top level, which is what
		// felt like "exiting a submenu lands me in a different menu"). An action the loop does not know is
		// now reported and the board reopens, instead of the settings quitting.
		if (result.action?.startsWith("__exit:")) {
			ctx.ui?.notify?.(`Unknown menu request: ${String(result.action).slice(0, 100)}`, "warning");
			restore = undefined;
			continue;
		}
		// The template library's dialogs (2026-10-07). The library is a mounted screen; an editor, a copy
		// or a confirmation opened over a mounted screen leaves the TUI with no focused input. The library
		// asks, the work happens HERE, over the plain chat, and the board comes back on the library list.
		if (result.action === "tpl-new" || result.action === "tpl-previewext" || result.action.startsWith("tpl-edit") || result.action.startsWith("tpl-dup:") || result.action.startsWith("tpl-del:")) {
			restore = await templateLibraryAction(ctx, String(result.action));
			continue;
		}
		ctx.ui?.notify?.(`Unknown menu request: ${String(result.action).slice(0, 100)} - the menu stayed open so nothing is lost.`, "warning");
		restore = undefined;
		continue;
	}
}

/** Free-text editor for the pi-wide no-think tag; opens with the current tag as the start value. */
async function editNoThinkTag(ctx: Ctx): Promise<void> {
	const cfg = loadConfig();
	const res = await ctx.ui.input("No-think tag — default <|think_off|> (empty = write nothing)", cfg.noThinkMarker || "");
	if (res === undefined) return;
	update((c) => void (c.noThinkMarker = String(res).trim()));
}

/**
 * The sampling-flags editor (CLI + fallback menus): a loop over the six flags; each opens pi's
 * input with the current value prefilled - type a number, or "ignore" / empty to clear the flag.
 * Values land in range automatically; everything saves the moment it is entered.
 */
async function editSamplingFlags(ctx: Ctx, ref: string): Promise<void> {
	for (;;) {
		const cfg = loadConfig();
		const model = lookupRef(ref, ctx);
		const o = optionsFor(cfg, ref, model);
		const rows: Row[] = SAMPLING_KEYS.map((key) => {
			const cur = o.sampling[key];
			return {
				label: `${key} - now ${typeof cur === "number" ? cur : "ignore"}`,
				hint: SAMPLING_HELP[key],
				run: async () => {
					const res = await ctx.ui.input(`${key} — a number, or "ignore" (empty = ignore)`, typeof cur === "number" ? String(cur) : "ignore");
					if (res === undefined) return;
					const t = String(res).trim();
					let next: number | null;
					if (!t || t.toLowerCase() === "ignore" || t.toLowerCase() === "none" || t.toLowerCase() === "off") {
						next = null;
					} else {
						const parsed = Number(t);
						if (!Number.isFinite(parsed)) {
							ctx.ui.notify(`Not a number: ${t}`, "error");
							return;
						}
						const range = SAMPLING_RANGES[key];
						next = clamp(parsed, range[0], range[1]);
					}
					setModelOptions(ref, { sampling: { ...o.sampling, [key]: next } });
				},
			};
		});
		rows.push({ label: "Reset all to ignore:", hint: "Clears every flag for this model - nothing is sent.", run: () => setModelOptions(ref, { sampling: { ...SAMPLING_DEFAULTS } }) });
		rows.push({ label: "Back:", hint: "Changes above are already saved.", run: () => {} });
		const row = await pickRow(ctx, `Sampling flags — ${ref}`, rows);
		if (!row || row.label === "Back") return;
		await row.run();
	}
}

/** A whole-number prompt for the fallback menus. Escape = go back. */
/**
 * A number row, edited in this extension's own one-line field (2026-10-06: it used to open pi's input
 * box, which closed the settings screen and dropped the user back at the prompt to type a number).
 */
async function askNumber(ctx: Ctx, title: string, current: number, lo: number, hi: number, unit: string, decimal = false): Promise<number | undefined> {
	// 2026-10-07 BUG FIX: this opened the extension's own text screen, which draws its own caret and
	// behaves nothing like a normal edit field. Every number here is a single value, so it goes through
	// pi's standard one-line input box (normal caret, Enter accepts, Escape cancels). These rows are only
	// ever dispatched over the plain chat, which is where pi's box is safe to open.
	const words = decimal ? "a number" : "a whole number";
	const res = await ctx.ui?.input?.(`${title} - ${words} from ${fmt(lo)} to ${fmt(hi)} ${unit}`, String(current));
	if (res === undefined) return undefined;
	const typed = String(res).trim();
	if (typed === "") return undefined;
	const n = decimal ? Number(typed) : Math.round(Number(typed));
	if (!Number.isFinite(n) || n < lo || n > hi) {
		ctx.ui?.notify?.(`Not changed: type ${words} between ${fmt(lo)} and ${fmt(hi)} ${unit}.`, "error");
		return undefined;
	}
	return n;
}

/** Free-entry editor for a MODEL's number options (summary % / floor / ceiling). Runs over the
 *  plain chat - never while a submenu screen is mounted - and saves straight into the model's
 *  options (live config; read again on the next event). */
async function editModelNumber(ctx: Ctx, ref: string, key: string): Promise<void> {
	const model = lookupRef(ref, ctx);
	const o = optionsFor(loadConfig(), ref, model);
	let title = "", cur = 0, lo = 0, hi = 0, unit = "";
	let save: (v: number) => void = () => {};
	let shown: (v: number) => string = (v) => fmt(v);
	if (key === "summaryPct") {
		title = "Summary aim — % of the ctx window — default 10";
		cur = o.summaryPercent; lo = 1; hi = 99; unit = "percent of the ctx window";
		save = (v) => setModelOptions(ref, { summaryPercent: v });
		shown = (v) => `${v}%`;
	} else if (key === "summaryMin") {
		title = "Summary min — absolute tokens (the acceptance floor) — default 4096";
		cur = o.summaryMinTokens; lo = 0; hi = 1_000_000; unit = "tokens (absolute)";
		save = (v) => setModelOptions(ref, { summaryMinTokens: v });
		shown = (v) => `${fmt(v)} tok`;
	} else if (key === "summaryMax") {
		title = "Summary max — absolute tokens (the acceptance ceiling) — default 30000";
		cur = o.summaryMaxTokens; lo = 1; hi = 1_000_000; unit = "tokens (absolute)";
		save = (v) => setModelOptions(ref, { summaryMaxTokens: v });
		shown = (v) => `${fmt(v)} tok`;
	} else if (key === "argcap") {
		title = "Tool-call arg cap — default 500 chars";
		cur = o.argCap === "full" ? 0 : Number(o.argCap); lo = 0; hi = 1_000_000; unit = "chars (0 = full)";
		save = (v) => setModelOptions(ref, { argCap: v === 0 ? "full" : v });
		shown = (v) => (v === 0 ? "full" : `${fmt(v)} chars`);
	} else if (key === "timeout") {
		title = "Timeout period — default 10 min";
		cur = Math.round((o.timeoutMs || 0) / 60_000); lo = 0; hi = 720; unit = "minutes (0 = unlimited)";
		save = (v) => setModelOptions(ref, { timeoutMs: v === 0 ? 0 : v * 60_000 });
		shown = (v) => (v === 0 ? "unlimited" : `${v} min`);
	} else if ((SAMPLING_KEYS as readonly string[]).includes(key)) {
		// The six sampling flags used to have no box at all: the row asked for free entry, this editor
		// did not know the key and returned without touching anything, so the value looked stuck.
		const sk = key as SamplingKey;
		const rng = SAMPLING_RANGES[sk];
		const cur0 = (o.sampling as any)[sk];
		title = `Sampling flag ${key} - this model's compaction call only, your chat is never touched`;
		cur = typeof cur0 === "number" ? cur0 : 0; lo = rng[0]; hi = rng[1];
		unit = `a number (${rng[0]} to ${rng[1]}) - 0 is a real value; the word "ignore" (or an empty box) is the only way to clear the flag`;
		// 2026-10-07 BUG FIX (min_p): the box was a number field whose contract read "0 = ignore", and the
		// save wrote `v === 0 ? null : v` - so min_p 0 (the value Qwen3.8-Flash-Next recommends), top_k 0
		// and presence_penalty 0 were stored as "ignore" and the row snapped back: "I change it and it
		// doesn't change". A number is a number now, 0 included; ignore is a word, typed on purpose.
		const res0 = await ctx.ui.input(`Sampling flag ${key} - a number (${rng[0]} to ${rng[1]}), or "ignore"`, typeof cur0 === "number" ? String(cur0) : "ignore");
		if (res0 === undefined) return;
		const txt = String(res0).trim();
		if (!txt || ["ignore", "none", "off"].includes(txt.toLowerCase())) {
			setModelOptions(ref, { sampling: { ...o.sampling, [sk]: null } });
			ctx.ui?.notify?.(`${key} = ignore for ${ref.split("/").pop()} - nothing is sent, the server default applies.`, "info");
			return;
		}
		const parsed0 = Number(txt);
		if (!Number.isFinite(parsed0)) { ctx.ui?.notify?.(`Not a number: ${txt} - nothing saved.`, "error"); return; }
		const fv = clamp(parsed0, rng[0], rng[1]);
		setModelOptions(ref, { sampling: { ...o.sampling, [sk]: fv } });
		ctx.ui?.notify?.(`${key} = ${fv} for ${ref.split("/").pop()} - saved${fv === 0 ? " (0 is a real value: it is sent to the server)" : ""}.`, "info");
		// And stop here. This branch used to fall through to the shared tail below, which called the generic
		// save (a no-op for a flag, so harmless) and then announced the same change a SECOND time in generic
		// words - the reason a sampling flag looked like it had been edited twice, or bounced.

		return;
	} else {
		// An unknown key used to return in silence, which is exactly the "I press Enter and nothing ever
		// changes" complaint. Say what happened instead of swallowing the key.
		ctx.ui?.notify?.(`"${key}" has no numeric entry - this setting is changed by choosing from its list, not by typing.`, "warning");
		return;
	}
	const v = await askNumber(ctx, `${title} ${unit}`, cur, lo, hi, unit);
	if (v === undefined) return;
	save(v);
	ctx.ui?.notify?.(`Saved: ${shown(v)} for ${ref.split("/").pop()}.`, "info");
}

/**
 * Free-entry numeric settings: Enter on a numeric row opens pi's input box with the DEFAULT in the
 * title, so forgetting the default costs nothing. Everything writes the extension config (live,
 * read again on the next event - no /reload for any of it).
 */
const NUMBER_EDITORS: Record<string, { title: string; lo: number; hi: number; unit: string; def: number; apply: (v: number) => string | undefined }> = {
	retryWait: { title: "Wait between retries", lo: 0, hi: 300, unit: "seconds", def: 5, apply: (v) => { update((c) => void (c.retryDelaySeconds = v)); return undefined; } },
	reserveTok: { title: "Autocompaction trigger point", lo: 0, hi: 10_000_000, unit: "tokens before the full context window size (0 = pi's own reserve)", def: 0, apply: (v) => { update((c) => void (c.scaling.reserveTokens = v)); return undefined; } },
	preserveTok: { title: 'Preserve tail context', lo: 1024, hi: 10_000_000, unit: 'tokens of the context tail preserved verbatim', def: 20_000, apply: (v) => { update((c) => void (c.scaling.preserveTokens = v)); return undefined; } },
	elisionStart: { title: "Elision start", lo: 1, hi: 99, unit: "percent of window", def: 70, apply: (v) => { update((c) => void (c.elision = { ...c.elision, softPercent: v })); return undefined; } },
	elisionProtect: { title: "Elision protect (tool results)", lo: 0, hi: 1_000_000, unit: "tokens", def: 20_000, apply: (v) => { update((c) => void (c.elision = { ...c.elision, protectRecentTokens: v })); return undefined; } },
	elisionTail: { title: "Stub tail", lo: 0, hi: 100_000, unit: "chars (0 = none; 'full' is a separate row value)", def: 200, apply: (v) => { update((c) => void (c.elision = { ...c.elision, stubTailChars: v })); return undefined; } },
	elisionSavings: { title: "Min batch savings", lo: 0, hi: 100_000, unit: "tokens (0 = fire immediately)", def: 2_000, apply: (v) => { update((c) => void (c.elision = { ...c.elision, minSavingsTokens: v })); return undefined; } },
	elisionResults: { title: "Min stub batch", lo: 1, hi: 50, unit: "results", def: 5, apply: (v) => { update((c) => void (c.elision = { ...c.elision, minResultsToStub: v })); return undefined; } },
	elisionStop: { title: "Elision stop gap", lo: 0, hi: 100_000, unit: "tokens", def: 4_000, apply: (v) => { update((c) => void (c.elision = { ...c.elision, stopGapTokens: v })); return undefined; } },
	elisionBatchGap: { title: "Elision batch gap", lo: 0, hi: 86_400, unit: "seconds between sweeps", def: 300, apply: (v) => { update((c) => void (c.elision = { ...c.elision, minBatchGapSec: v })); return undefined; } },
	elisionCap: { title: "Max tool results per batch", lo: 1, hi: 500, unit: "results", def: 40, apply: (v) => { update((c) => void (c.elision = { ...c.elision, maxResultsPerBatch: v })); return undefined; } },
	chatCap: { title: "Chat max_tokens cap", lo: 0, hi: 2_000_000, unit: "tokens (0 = no cap)", def: 65_536, apply: (v) => { update((c) => void (c.chatMaxTokensCap = v)); return undefined; } },
	retries: { title: "Retries per model", lo: 0, hi: 99, unit: "tries (0 = no retries)", def: 2, apply: (v) => { update((c) => void (c.retries = v)); return undefined; } },
};

async function editNumber(ctx: Ctx, id: string): Promise<void> {
	const d = NUMBER_EDITORS[id];
	if (!d) return;
	const cfg = loadConfig();
	const current =
		id === "retryWait" ? cfg.retryDelaySeconds
		: id === "reserveTok" ? cfg.scaling.reserveTokens
		: id === 'preserveTok' ? cfg.scaling.preserveTokens
		: id === "piReserve" ? readPiReserveTokens()
		: id === "elisionStart" ? cfg.elision.softPercent
		: id === "elisionProtect" ? cfg.elision.protectRecentTokens
		: id === "elisionTail" ? (typeof cfg.elision.stubTailChars === "number" ? cfg.elision.stubTailChars : 0)
		: id === "elisionSavings" ? cfg.elision.minSavingsTokens
		: id === "elisionResults" ? cfg.elision.minResultsToStub
		: id === "elisionBatchGap" ? cfg.elision.minBatchGapSec
		: id === "elisionCap" ? cfg.elision.maxResultsPerBatch
		: cfg.elision.stopGapTokens;
	const v = await askNumber(ctx, `${d.title} — default ${fmt(d.def)} ${d.unit}`, current, d.lo, d.hi, d.unit);
	if (v === undefined) return;
	const err = d.apply(v);
	if (err) ctx.ui?.notify?.(`Not saved: ${err}`, "error");
}

/**
 * One select screen for terminals without custom-component UI (RPC mode): returns the row the user
 * picked, or undefined on Escape. pi's own selector draws every option, so a list taller than the
 * terminal is shown a page at a time.
 */
async function pickRow(ctx: Ctx, title: string, rows: Row[]): Promise<Row | undefined> {
	if (rows.length === 0) return undefined;
	if (rows.length <= 8) {
		const items = rows.map((r) => (r.hint ? `${r.label}\n${r.hint}` : r.label));
		const choice = await ctx.ui.select(title, items);
		if (choice === undefined) return undefined;
		const i = items.indexOf(choice);
		return i >= 0 ? rows[i] : undefined;
	}
	const PAGE = 10;
	let offset = 0;
	for (;;) {
		const page = rows.slice(offset, offset + PAGE);
		const items = page.map((r) => r.label);
		if (offset > 0) items.unshift("... previous page");
		if (offset + PAGE < rows.length) items.push(`... next page (${rows.length - offset - PAGE} more)`);
		const choice = await ctx.ui.select(title, items);
		if (choice === undefined) return undefined;
		if (choice === "... previous page") {
			offset = Math.max(0, offset - PAGE);
			continue;
		}
		if (choice.startsWith("... next page")) {
			offset += PAGE;
			continue;
		}
		const ci = items.indexOf(choice) - (offset > 0 ? 1 : 0);
		if (ci >= 0 && ci < page.length) return page[ci];
		return undefined;
	}
}

async function orderMenuFallback(ctx: Ctx): Promise<void> {
	for (;;) {
		const cfg = loadConfig();
		const reserveTokens = extReserve(ctx);
		const rows: Row[] = [];
		for (const ref of Object.keys(cfg.models)) {
			const model = lookupRef(ref, ctx);
			rows.push({
				label: `[${orderPosition(cfg, ref) + 1}] ${slotLabel(ref, model)}`,
				hint: model ? modelRowValue(cfg, ref, model, reserveTokens) : "pi cannot see this model right now",
				run: () => modelOptionsFallback(ctx, ref, model),
			});
		}
		rows.push({
			label: "Add a model from the list pi has available:",
			hint: "Defaults are applied (thinking off, 10 min, aim 0.8, min 0.1, max 0.9, tag as on the board). Open it afterwards to tune.",
			run: async () => {
				const pool = poolOf(ctx).filter((p) => !loadConfig().models[p.ref]);
				if (!pool.length) {
					ctx.ui.notify("Every reachable model is already in the order.", "info");
					return;
				}
				const chosen = await pickRow(
					ctx,
					"Which model should join the Compaction models list?",
					pool.sort((a, b) => a.ref.localeCompare(b.ref)).map((p) => ({
						label: p.model ? (isLocalModel(p.model) ? shortName(String(p.model.id)) : String(p.model.id)) : p.ref,
						run: () => rotateAdd(p.ref),
					})),
				);
				void chosen;
			},
		});
		rows.push({ label: "Back:", hint: "Changes above are already saved.", run: () => {} });
		const row = await pickRow(ctx, `Compaction models - ${rotationText(cfg, ctx)}`, rows);
		if (!row || row.label === "Back") return;
		await row.run();
	}
}

async function modelOptionsFallback(ctx: Ctx, ref: string, model: any): Promise<void> {
	for (;;) {
		const cfg = loadConfig();
		const o = optionsFor(cfg, ref, model);
		const reserveTokens = extReserve(ctx);
		const th = effectiveThinking(o.thinking, model);
		const rows: Row[] = [
			{
				label: `Thinking method - now ${th.level}`,
				hint: `Cycles off → minimal → low → medium → high → xhigh → max. This model accepts: ${supportedThinkingLevels(model).join(", ")}.`,
				run: async () => {
					const levels = supportedThinkingLevels(model);
					const chosen = await ctx.ui.select("Thinking method", withCurrent(levels, o.thinking));
					if (chosen !== undefined) setModelOptions(ref, { thinking: chosen as Level });
				},
			},
			{
			},
			{
				label: `Summary timeout period - now ${timeoutLabel(o.timeoutMs)}`,
				hint: "2 → 3 → 5 → 7 → 10 → 15 → 20 → 30 → 45 → 60 min → unlimited. How long to wait with nothing arriving; output restarts the clock.",
				run: async () => {
					const labels = TIMEOUT_STEPS.map(timeoutLabel);
					const chosen = await ctx.ui.select("Timeout period", withCurrent(labels, timeoutLabel(o.timeoutMs)));
					if (chosen !== undefined) setModelOptions(ref, { timeoutMs: parseDuration(chosen) });
				},
			},
			{
				label: `Summary % of region - now ${o.summaryPercent}%`,
				hint: "THE SUMMARY AIM: this percent of the region being folded (measured after the stubbing pass). Free numeric entry, 1-99.",
				run: async () => {
					const v = await askNumber(ctx, "Summary aim — % of the ctx window — default 10", o.summaryPercent, 1, 99, "percent");
					if (v !== undefined) setModelOptions(ref, { summaryPercent: v });
				},
			},
			{
				label: `Summary min - now ${fmt(o.summaryMinTokens)} tok (absolute)`,
				hint: "THE ACCEPTANCE FLOOR: a shorter summary is thrown away and the next model in the rotation is asked. Absolute tokens, whatever the model's window (default 4,096). Free numeric entry.",
				run: async () => {
					const v = await askNumber(ctx, "Summary min \u2014 absolute tokens (default 4096)", o.summaryMinTokens, 0, 1_000_000, "tokens");
					if (v !== undefined) setModelOptions(ref, { summaryMinTokens: v });
				},
			},
			{
				label: `Summary max - now ${fmt(o.summaryMaxTokens)} tok (absolute)`,
				hint: "THE ACCEPTANCE CEILING: a longer summary is thrown away too. Absolute tokens - a model cannot write a 100k-token summary however big its window, which is why these two limits are absolute tokens and not percents of the window (default 30,000). Free numeric entry.",
				run: async () => {
					const v = await askNumber(ctx, "Summary max \u2014 absolute tokens (default 30000)", o.summaryMaxTokens, 1, 1_000_000, "tokens");
					if (v !== undefined) setModelOptions(ref, { summaryMaxTokens: v });
				},
			},
			{
				label: `Summarizer input - now ${o.inputStubs ? "stubs" : "raw"}`,
				hint: '"stubs" = every tool result becomes an informative stub (tool, size, exit/tail message, re-run hint) in the summarizer input; "raw" = pi\'s own input with the 2,000-char head-only clip.',
				run: async () => {
					const chosen = await ctx.ui.select("Summarizer input", withCurrent(["stubs", "raw"], o.inputStubs ? "stubs" : "raw"));
					if (chosen !== undefined) setModelOptions(ref, { inputStubs: chosen === "stubs" });
				},
			},
			{
				label: `Tool-call arg cap - now ${charCapLabel(o.argCap)}${o.argCap === "full" ? "" : " chars"}`,
				hint: "Caps tool-call argument VALUES in the summarizer input (whole files inside write/edit). Default 500 (OMP's value).",
				run: async () => {
					const labels = CHAR_CAP_STEPS.map((c) => (c === "full" ? "full" : `${c} chars`));
					const chosen = await ctx.ui.select("Tool-call arg cap", withCurrent(labels, `${charCapLabel(o.argCap)}${o.argCap === "full" ? "" : " chars"}`));
					if (chosen !== undefined) setModelOptions(ref, { argCap: chosen === "full" ? "full" : Number(String(chosen).replace(/[^0-9]/g, "")) });
				},
			},
			{
				label: `Tool stubs in summary - now ${o.preserveStubs}`,
				hint: `verbatim | detailed | summary | brief. ${PRESERVE_SENTENCES[o.preserveStubs].stubs}`,
				run: async () => {
					const chosen = await ctx.ui.select("Tool stubs in summary", withCurrent(PRESERVE_CYCLE, o.preserveStubs));
					if (chosen !== undefined) setModelOptions(ref, { preserveStubs: chosen as Preserve });
				},
			},
			{
				label: `User prompts - now ${o.preserveUser}`,
				hint: PRESERVE_SENTENCES[o.preserveUser].user,
				run: async () => {
					const chosen = await ctx.ui.select("User prompts in the summary", withCurrent(PRESERVE_CYCLE, o.preserveUser));
					if (chosen !== undefined) setModelOptions(ref, { preserveUser: chosen as Preserve });
				},
			},
			{
				label: `Assistant replies - now ${o.preserveReplies}`,
				hint: `The [Assistant] lines (separate from thinking). ${PRESERVE_SENTENCES[o.preserveReplies].replies}`,
				run: async () => {
					const chosen = await ctx.ui.select("Assistant replies in the summary", withCurrent(PRESERVE_CYCLE, o.preserveReplies));
					if (chosen !== undefined) setModelOptions(ref, { preserveReplies: chosen as Preserve });
				},
			},
			{
				label: `Assistant thinking - now ${o.preserveThinking}`,
				hint: `The [Assistant thinking] lines (separate from replies). ${PRESERVE_SENTENCES[o.preserveThinking].thinking}`,
				run: async () => {
					const chosen = await ctx.ui.select("Assistant thinking in the summary", withCurrent(PRESERVE_CYCLE, o.preserveThinking));
					if (chosen !== undefined) setModelOptions(ref, { preserveThinking: chosen as Preserve });
				},
			},
			{
				label: `No-think tag - now ${o.noThinkMarker || "(empty)"}`,
				hint: 'Written into the request when thinking is off. Enter to type the tag (default <|think_off|>); "ignore"/empty writes nothing.',
				run: async () => {
					const res = await ctx.ui.input("No-think tag for this model — default <|think_off|> (empty = write nothing)", o.noThinkMarker || "");
					if (res === undefined) return;
					setModelOptions(ref, { noThinkMarker: String(res).trim() });
				},
			},
			{
				label: `Sampling flags - now ${samplingLabel(o.sampling)}`,
				hint: "Temperature / top_p / top_k / min_p / presence / repetition for THIS model's compaction request; each a number or ignore.",
				run: () => editSamplingFlags(ctx, ref),
			},
			{ label: "Back:", hint: "Changes above are already saved.", run: () => {} },
		];
		const row = await pickRow(ctx, `${slotLabel(ref, model)} — options`, rows);
		if (!row || row.label === "Back") return;
		await row.run();
	}
}

async function fallbackMenu(ctx: Ctx): Promise<void> {
	for (;;) {
		const cfg = loadConfig();
		const rows: Row[] = [
			{
				label: `Compaction (this extension) - ${cfg.enabled ? "ON" : "OFF"}`,
				hint: `Tried in this order until one writes a usable summary: ${rotationText(cfg, ctx)}. The automatic trigger is this extension's own check of the live context, and /compact works while this is off. pi running its own compaction is a separate setting in pi's own settings.json, which this menu never writes; see the row below it.`,
				run: () => {
					const on = !loadConfig().enabled;
					if (on && enableBlockedByOtherCompactor()) return; // another manager present: one at a time
					update((c) => void (c.enabled = on));
					ctx.ui?.notify?.(
						on
							? "COMPACTION IS ON. This extension compacts on its own trigger, with its own models, and /compact is answered by this extension. pi's own compaction is untouched by this switch - it lives in " + join(AGENT_DIR, "settings.json") + ", and while it stays on pi can compact by itself with pi's own summariser."
							: "COMPACTION IS OFF. Nothing of this extension compacts or elides any more; pi is left exactly as its own settings have it.",
					"info",
					);
				},
			},
			{
				label: `pi's own compaction (read-only) - ${readPiCompactionEnabled() ? "ENABLED in pi's settings" : "disabled in pi's settings"}`,
				hint: "This menu never writes pi's settings. That flag only decides whether pi runs its own compaction with its own summariser on pi's own threshold check; it does not gate /compact, and it does not gate this extension's automatic compaction, which runs on this extension's own trigger with its own models. To change it, edit compaction.enabled in " + join(AGENT_DIR, "settings.json") + " by hand and /reload.",
				run: () => {
					const f = join(AGENT_DIR, "settings.json");
					ctx.ui?.notify?.(
						readPiCompactionEnabled()
							? `pi's OWN compaction is ENABLED in ${f} (compaction.enabled). This extension never writes that file. While it stays on, pi can compact by itself with pi's own summariser, and it will if this extension is ever stopped by an error. Set the flag to false there and /reload to keep pi's own checks off for good.`
							: `pi's own compaction is disabled in ${f}: pi will not compact by itself, on the threshold check or after an overflow. This extension still compacts on its own trigger, automatically and on /compact.`,
					"info",
					);
				},
			},
			{ label: "Compaction models", hint: "Pick, place (digits), remove (space) and tune models. Number 1 is asked first.", run: () => orderMenuFallback(ctx) },
			{
				label: `Retries per model - now ${cfg.retries}`,
				hint: "Only for busy / rate-limited / dropped connections. A too-thin or too-long summary moves to the next model instead.",
				run: async () => {
					const n = await askNumber(ctx, "Retries per model (0-5)", cfg.retries, 0, 5, "retries");
					if (n === undefined) return;
					const s = await askNumber(ctx, "Seconds to wait before a retry (3-60)", cfg.retryDelaySeconds, 0, 300, "seconds");
					if (s === undefined) return;
					update((c) => {
						c.retries = n;
						c.retryDelaySeconds = s;
					});
				},
			},
			{
				label: `No-think tag - now ${cfg.noThinkMarker || "(empty)"}`,
				hint: 'The default tag for local models when their thinking is off. Each model can override it in its options. Enter to type it; "ignore"/empty writes nothing.',
				run: () => editNoThinkTag(ctx),
			},
			{
				label: `Elision - now ${cfg.elision.enabled ? "ON" : "OFF"} (start ${cfg.elision.softPercent}%, protect ${fmt(cfg.elision.protectRecentTokens)} tok, stub tail ${charCapLabel(cfg.elision.stubTailChars)})`,
				hint: "Old tool results become informative stubs in live requests; calls stay; originals stay on disk. Tune with: /compact-plus elision ...",
				run: () => {
					update((c) => void (c.elision = { ...c.elision, enabled: !c.elision.enabled }));
					resetElisionBreaker();
				},
			},
			{ label: "Additional instruction:", hint: cfg.additionalInstruction ? "Custom text saved - Enter to edit or reset." : "Default text in use - Enter to edit a copy.", run: () => editInstruction(ctx) },
			{ label: "Log", hint: `Status, model, time, tok in/out for every try. File: ${LOG_PATH}`, run: () => ctx.ui.notify(tailLogPretty(16), "info") },
		];
		const row = await pickRow(ctx, "Compaction models", rows);
		if (!row) return;
		await row.run();
	}
}

// ---------------------------------------------------------------- status text

export function describe(cfg: Config, ctx: { modelRegistry: any; model?: any }): string {
	if (!cfg.enabled || Object.keys(cfg.models).length === 0) return "compaction: pi default (session model) — extension on but no models in the list";
	const reserve = extReserve(ctx as Ctx);
	const lines: string[] = [
		`Compaction models, in try order (if all fail, pi's own compaction runs):`,
		`  trigger: estimate crosses window - ${fmt(reserve)} tok · auto-compact while idle: ${cfg.autoCompact ? (parkedThisSession(cfg, ctx) ? "PARKED FOR THIS SESSION (a manual compaction arms it again)" : "ON") : "off"} · chat max_tokens cap: ${cfg.chatMaxTokensCap > 0 ? fmt(cfg.chatMaxTokensCap) : "off"}`,
	];
	// pi's threshold is the trigger while pi's own switch is on, so its ask point belongs in the status
	// next to this extension's line: when the two disagree, the operator can read the cost here too.
	if (readPiCompactionEnabled() && (ctx as any)?.model?.contextWindow) {
		const win = (ctx as any).model.contextWindow as number;
		const piAsk = Math.max(0, win - readPiReserveTokens());
		const ours = win - reserve;
		const gap = ours - piAsk;
		// A one percent band counts as agreement: pi and this extension measure the same context two ways.
		const agree = gap <= Math.max(1_000, Math.round(win * 0.01));
		lines.push(
			`  pi asks at ${fmt(piAsk)} tok; this extension is ready at ${fmt(ours)} tok` +
			(agree
				? " - agreed, so no context is re-read waiting for the line."
				: ` - pi's alarm is ${fmt(gap)} tok BELOW our line, so it would be held and the context re-read; lower pi's reserveTokens (its settings.json) and /reload.`),
	);
}
	const refs = Object.keys(cfg.models);
	refs.forEach((ref, i) => {
		const model = lookupRef(ref, ctx);
		const o = optionsFor(cfg, ref, model);
		const th = model ? effectiveThinking(o.thinking, model) : { level: o.thinking, note: "" };
		lines.push(
			`  ${i + 1}. ${slotLabel(ref, model)} — summary aim ${o.summaryPercent}% of the ctx window, limits min ${fmt(o.summaryMinTokens)} / max ${fmt(o.summaryMaxTokens)} tok, thinking ${th.level}, timeout ${timeoutLabel(o.timeoutMs)}, sampling ${samplingLabel(o.sampling)}`,
		);
	});
	lines.push(`retries: ${cfg.retries} per model, ${cfg.retryDelaySeconds}s apart · no-think tag: ${cfg.noThinkMarker || "(empty)"} · additional instruction: ${cfg.additionalInstruction ? "custom" : "default (turn ledger)"} · transcript pointer: ${cfg.transcriptPointer ? "on" : "off"}`);
	const e = cfg.elision;
	lines.push(`direct request: ${cfg.directRequest ? "ON" : "off"} (a direct failure moves to the next candidate; pi's compact() runs only after all fail) · shape gate: ${cfg.shapeGate ? "on" : "off"} · chat max_tokens cap: ${cfg.chatMaxTokensCap > 0 ? fmt(cfg.chatMaxTokensCap) : "off"}`);
	lines.push(`elision: ${e.enabled ? "ON" : "off"} · start ${e.softPercent}% of window · protect ${fmt(e.protectRecentTokens)} tok · stub tail ${charCapLabel(e.stubTailChars)} chars · min batch ${e.minResultsToStub} results / ${fmt(e.minSavingsTokens)} tok · stop gap ${fmt(e.stopGapTokens)} tok (CLI: /compact-plus elision)`);
	return lines.join("\n");
}

// ---------------------------------------------------------------- auto-compact watcher

/** PROACTIVE TRIGGER (2026-10-01): pi's own threshold check compares its estimate against
 *  window - ITS static reserveTokens (16,384). On a 1M window that line sits ABOVE the request
 *  death line (window - chatMaxTokensCap), so pi would only ever compact via overflow recovery -
 *  which first DELETES session entries. The extension instead watches the context estimate and,
 *  while the agent is IDLE, starts pi's manual compaction (ctx.compact -> session_before_compact
 *  -> this fork's rotation) the moment the estimate crosses window - extReserve. The line is
 *  recomputed every tick, so a mid-session model switch moves it with zero /reload. */
let watcherCtx: Ctx | null = null;
/**
 * Set when this extension's own watcher fires while the loop is mid-turn. pi's compact() begins by
 * aborting what is running and pi never resumes an interrupted turn on its own, so a mid-turn trigger
 * would otherwise end the session's work until the user types something (2026-10-18: that is exactly
 * what it did, on a 55-minute turn, with nothing left to defer to). Armed here, spent by the
 * session_compact handler below once the summary is really in the journal, and expired by age so a
 * cancelled compaction cannot resume a turn much later.
 */
/**
 * Tell pi-stamp-plus that THIS compaction was cancelled on purpose (2026-10-21, on request).
 *
 * pi reports every cancelled compaction the same way: `session_compact_failed` with `aborted: true`
 * and no error message, so a hold from the gate above and an Escape key pressed during a summary look
 * identical to anything listening. pi-stamp-plus stamps that as `Compaction . ABORTED`, which reads
 * like a fault when it is a Tuesday. The two extensions cannot read each other's mind, so the moment
 * of cancelling is written down next to the session file; pi-stamp-plus drops the stamp for that note
 * and for nothing else. This extension never reads the file back, and a session file pi did not tell
 * us about means no note and no change.
 */
function markCompactionHeld(): void {
	const sf = (process.env.PI_SESSION_FILE ?? "").trim();
	if (!sf || writesBlocked()) return;
	try {
		writeFileSync(`${sf}.pcp-held.json`, JSON.stringify({ at: Date.now(), by: "pi-compact-plus", why: "not time yet" }), "utf8");
	} catch {
		/* A note nobody can write is a stamp nobody suppresses: the honest ABORTED line still appears. */
	}
}
/**
 * THE GATE: should a compaction that PI started be held back for now?
 *
 * pi's threshold and this extension's trigger point are two different numbers on purpose. pi's is set
 * low (its settings.json `compaction.reserveTokens`) so that pi asks early - pi's request-time check is
 * the one trigger that reliably fires while a session is mid-turn, which no timer of ours can promise.
 * This extension's line is `window - extReserve(ctx)`, the percentage on the settings board, and it is
 * the number that decides when a compaction is actually worth writing.
 *
 * Between the two, the answer is a cancel: nothing is summarised, nothing is cut, pi asks again on the
 * next request, and the same request gets a real summary once our line is reached.
 *
 * A HOLD IS NOT FREE, and the operator found the price on 2026-10-21: "autocompaction now initially
 * triggers full context re-read before it starts creating the summary. this wasn' happening before.
 * Correct, and this rule is the mechanism. Cancelling pi's ask lets the session carry on, so every turn
 * in that gap goes to the model with the context UNCOMPACTED - a pass over the whole of it, minutes on
 * a local model, before the summary is finally written. With pi's reserveTokens at 50,000 and this line
 * at window-30,600 the gap was 19,400 tokens of re-reading, which is exactly what was reported.
 *
 * So the two numbers have to agree. When they do not, the log line, the notice and the settings board
 * all name the reserveTokens value that ends it. The gate stays for what it is genuinely for: a
 * compaction that arrives materially early, which once the numbers are aligned means a mis-set
 * number or a session on a different window - not the everyday path.
 *
 * Returns a sentence for the log while it should be held, and null when the compaction may proceed.
 * Pure numbers by design - no files, no pi objects - so the rule is testable with a rig.
 */
export function holdUntilOurLine(args: {
	reason: string;
	fill: number;
	window: number;
	line: number;
	/** How far below the line a compaction is still accepted. pi counts from the server's own usage
	 *  figures and this extension counts its own way, so the two differ by a little; a hold is not free
	 *  (see above), so a hair's-breadth miss must not cause one. Callers pass about 1% of the window. */
	slack?: number;
}): string | null {
	// Only pi's threshold is a candidate. "manual" is the operator's own /compact, and an overflow is pi
	// asking because a request was already refused for room - holding either would be a refusal.
	if (args.reason !== "threshold") return null;
	// Nothing measured, or no line to compare against: never invent a veto out of a zero.
	if (!(args.window > 0) || !(args.fill > 0) || !(args.line > 0)) return null;
	if (args.fill >= args.line - (args.slack ?? 0)) return null;
	// One line of the log view, no more: the view truncates. It must still name both numbers, and it
	// ends with the setting that stops this happening on the next request.
		// The advice is the same every time now that this extension owns the trigger: an ask arrives
		// below our line only when pi's alarm is set too low, and pi's settings.json is where that is fixed.
		const fix = "; lower pi's reserveTokens to end the wait";
	return `pi asked at ${fmt(args.fill)}/${fmt(args.window)} tok, our point is ${fmt(args.line)} tok - held, nothing written${fix}`;
}

/** When the gate last said "held" on the screen. The log records every hold on its own throttle below. */
let lastGateHoldNoticeAtMs = 0;

// NO AUTOMATIC CONTINUATION (deleted 2026-10-21, on the operator's instruction). This extension used
// to send its own message into the session after a mid-turn compaction, worded to read as the
// operator's own prompt, with a timer and retries behind it to push it through pi's guards. That put
// words in the operator's mouth inside their own session. A mid-turn compaction still interrupts the
// turn - pi's compact() has to abort it to write the summary - and now the operator decides how to
// pick the work back up. The notices below say exactly that, and nothing else.

let lastAutoTriggerMs = 0;
/** The watcher writes its "nothing to do" decision at most once every five minutes. */
let lastWatchLogMs = 0;
/** True while this extension's rotation is running: the watcher must not fire into it (pi's
 *  compact() is single-flight - a re-trigger KILLS the in-flight attempt, which is exactly the
 *  minute-by-minute kill storm of 2026-10-02 17:01-17:05). */
let compactionInFlight = false;
/** The one-shot skip for "Start pi's built-in compaction now": the next session_before_compact is
 *  left to pi itself. */
let skipNextStockCompaction = false;
const WATCHER_INTERVAL_MS = 5_000;

/**
 * WHAT THE AGENT LOOP IS DOING RIGHT NOW, kept current from pi's own events.
 *
 * WHY THIS EXISTS (2026-10-14, twice in one session). The watcher used to start a compaction whenever the
 * size crossed the line, at any moment, because 2026-10-07 had removed the idle gate on the theory that a
 * long working turn must not be allowed to overflow. What that actually bought was this: ctx.compact() took
 * pi's abort signal with it, so the request in flight died with "This operation was aborted" - a bash tool
 * cut off 0.5 s into running, a turn whose only output was an abort - and the turn never restarted, so the
 * session sat there while the summary ran and after it. Then the summary, asked to compress an already
 * compacted conversation, answered "Nothing to compact (session too small)" and read as a second error.
 * The trigger also mistook the on-disk history for the live context and believed it was 234,591 tokens in
 * a 20,300-token conversation (see measureContext), which is what made it knock at all.
 *
 *
 * WHAT THAT BOUGHT, AND WHY IT IS NOW HALF-REVERSED (2026-10-15, written down because the pendulum has swung
 * twice). Started only at a settled point, the trigger fired 10,000 tokens LATE in a session that had already
 * crossed its line, and the summary it finally ran at 178,039 tokens was too late to cover the newest material:
 * the ledger it wrote got as far as the seventeenth turn and stopped mid-sentence, so the eighteenth prompt was
 * neither summarised nor kept, which is what the user read as "all the newer prompts are missing". Meanwhile
 * pi's own between-rounds check - the one non-aborting mid-run trigger there is - was silent the whole time.
 * Why is still open: the guess written into this file the same evening, that pi's measure sits 20,000-30,000
 * tokens under the real fill because a projection holds no system prompt and no tool schemas, was taken out by
 * its own regression test hours later, because the newest usage figure already counts the whole request it
 * answered - system prompt, tool schemas and all - and pi only adds a size guess on top of it, so its measure
 * is at least as big as ours. The padding was never written to pi's settings; what is left of that theory is a
 * log field, piMeasureDelta, and the answer will come from those numbers rather than from this paragraph.
 * What is not in doubt is that a settled-only rule left the session walking from the line to the wall by itself.
 *
 * The rule now is: a compaction may start WHILE THE LOOP WORKS, at a round boundary. Measuring happens at any
 * moment; starting happens when nothing is answering, no tool is running, nothing is queued for the next round,
 * and every tool call the newest answer asked for has its result back - that is the boundary between rounds,
 * and it recurs many times in a long turn. It is still not free: ctx.compact() is pi's manual path and pi's own
 * source begins it with `await this.abort()`, so a start mid-run interrupts the turn, and threshold compaction
 * does not resume it. That is the price, and it is now accepted deliberately, because the alternative measured
 * out worse: one interrupted turn against a session that reaches 200,000 tokens and is refused, and a refused
 * session cannot be compacted at all. On top of that, within MIDRUN_DANGER_TOKENS of the window the niceties go
 * out of the window with them and the compaction starts in front of running tools rather than not at all.
 * Elision, the capped max_tokens and the engine's own fit_max_tokens clamp guard the quiet part of a turn
 * exactly as before.
 */
const agentLoop = {
	running: false,
	sinceMs: 0,
	tools: 0,
	promptOpen: false,
	/** Requests this process has sent and has not seen answered yet, recorded as a PAIR OF TIMESTAMPS rather
	 *  than a counter, and only trusted while the evidence is young. The reason: the watcher's timer and the
	 *  settle check work from a context object picked up on an earlier event, and ctx.signal captured mid-run
	 *  stays set on that object afterwards - reading it blindly would have meant a trigger that silently never
	 *  fires again after one long turn, the invisible dead-switch this file warns about elsewhere. A counter
	 *  alone would have the same flaw in a subtler form (one response event missing - a request killed before
	 *  headers, a proxy that never logged it - and the trigger is blocked for the rest of the session), so the
	 *  evidence expires: an unanswered request holds the trigger for at most ten minutes, and a healthy pair
	 *  of events releases it the moment the answer lands. */
	requestsOut: 0,
	requestStartedAtMs: 0,
	requestAnsweredAtMs: 0,
	lastToolEndAtMs: 0,
	lastSettledAtMs: 0,
	lastAbortAtMs: 0,
};
/** A park has no clock (2026-10-17): only a manual compaction re-arms the trigger, which is what logs
 *  `auto_compaction_resumed`. Three no-op strikes inside a quarter of an hour mean the trigger is
 *  firing somewhere it cannot possibly find work, so park it. */
const noOpStrikes = { n: 0, windowStartMs: 0 };
let settleCheckTimer: ReturnType<typeof setTimeout> | undefined;
/** "Over the line but not at a safe point" is written down at most once a minute. */
let lastHoldLogMs = 0;
/** ... and the one alarm that deserves the screen - the live wire really is near the window - once per 5 min. */
let lastHoldNoticeAtMs = 0;

/** Start a fresh 15 minute window every time one expires. */
function countNoOpStrike(nowMs: number): number {
	if (nowMs - noOpStrikes.windowStartMs > 900_000) {
		noOpStrikes.windowStartMs = nowMs;
		noOpStrikes.n = 0;
	}
	noOpStrikes.n += 1;
	return noOpStrikes.n;
}

/**
 * Can a compaction start right now without disturbing anything? Returns the reason it cannot, so the log
 * says which door was shut. Everything here is an API pi really exports (types.d.ts: isIdle(), signal,
 * hasPendingMessages()) or a fact this file observes itself; the 2026-10-07 gates read ctx.compaction.*,
 * which is not a member of ExtensionContext at all, so those three checks were dead code and silently
 * never gated anything.
 */
// How long after an answer lands, or a tool finishes, the mid-run gate still waits before it will start a
// compaction. The race it buys off is the one the user was bitten by on 2026-10-14 at 14:53:34: an answer
// arrives, its tool calls have not begun to execute yet, and a start in that gap aborts a tool 0.5 s into
// running it. One second is longer than pi needs to commit to the next round or to its tools.
const MIDRUN_DWELL_MS = 1200;
/** Within this many tokens of the whole window the mid-run gates stop holding: a session that gets refused
 *  for want of room cannot be compacted at all, so an interrupted turn is the better of the two outcomes. */
const MIDRUN_DANGER_TOKENS = 12_000;

/**
 * THE ROUND BOUNDARY (2026-10-15). Whether the newest answer's tool calls have all come back with their
 * results, read off the projection rather than off event timing, because event timing has a gap in it: a
 * response event can land a tick before the first tool_execution_start of the same round, and a start inside
 * that gap is what used to abort a tool that had only just begun. Returns 0 when the round is closed (the
 * newest answer asked for nothing, or every call it asked for has its result, or a newer answer has already
 * followed it), otherwise the number of calls still owed a result.
 */
function pendingToolCalls(ctx: any): number {
	try {
		const projection: any = (ctx as any)?.sessionManager?.buildSessionProjection?.();
		if (!projection?.entries) return 0;
		const all: any[] = [];
		for (const e of projection.entries) for (const m of e?.messages ?? []) all.push(m);
		for (let i = all.length - 1; i >= 0; i--) {
			const m: any = all[i];
			if (m?.role !== 'assistant') continue;
			const want: string[] = [];
			for (const b of Array.isArray(m.content) ? m.content : []) {
				if (!b) continue;
				if (b.type === 'toolCall' || b.type === 'tool-call' || b.type === 'tool_use' || b.type === 'function_call') {
					const id = b.id ?? b.toolCallId ?? b.callId;
					if (id) want.push(String(id));
				}
			}
			if (!want.length) return 0; // the newest answer asked for no tools: this round is closed
			const got = new Set<string>();
			for (let j = i + 1; j < all.length; j++) {
				const r: any = all[j];
				if (r?.role === 'assistant') break; // a newer answer already followed: this round is closed too
				if (r?.role !== 'toolResult') continue;
				if (r.toolCallId) got.add(String(r.toolCallId));
				for (const part of Array.isArray(r.content) ? r.content : []) if (part?.toolCallId) got.add(String(part.toolCallId));
			}
			return want.filter((id) => !got.has(id)).length;
		}
		return 0;
	} catch {
		return 0;
	}
}

export function compactSafePoint(ctx: any, freshCtx = false, urgent = false): { ok: boolean; why: string; midRun: boolean } {
	// A park is honoured here as well as in the watcher (2026-10-17), so a future caller cannot
	// slip past it: until a manual compaction re-arms the trigger, no gate here may green-light one.
	if (parkedThisSession(loadConfig(), ctx)) return { ok: false, why: "parked for this session until a manual compaction", midRun: false };
	const midRun = agentLoop.running === true;
	const hold = (why: string) => ({ ok: false, why, midRun });
	if (compactionInFlight) return hold('a compaction is already in flight');
	if (agentLoop.requestStartedAtMs > agentLoop.requestAnsweredAtMs && Date.now() - agentLoop.requestStartedAtMs < 600_000) {
		return hold(`a provider request has gone unanswered for ${Math.round((Date.now() - agentLoop.requestStartedAtMs) / 1000)} s`);
	}
	// A signal is only evidence when it came with the event now being handled; the caller that holds an old
	// context object is answered by the request counter above.
	if (freshCtx && (ctx as any)?.signal) return hold('a provider request is in flight');
	if (agentLoop.promptOpen) return hold('a dialog is open on screen');
	if (urgent) {
		// Past the danger line the mid-run gates below are set aside rather than waited out: with a request
		// answered and nothing unanswered, tools may well still be running, and a turn that is interrupted is
		// the cheaper loss next to a session the engine refuses to answer. A dialog on screen and a compaction
		// already running still hold, because those two cannot be recovered by sending another message.
		return {
			ok: true,
			why: midRun
				? agentLoop.tools > 0
					? `the danger line - starting in front of ${agentLoop.tools} running tool(s) and ${pendingToolCalls(ctx)} tool call(s) still owed a result, because waiting for them would let the context get refused`
					: `past the danger line mid-run, between rounds (${agentLoop.sinceMs ? `${Math.round((Date.now() - agentLoop.sinceMs) / 1000)} s into the turn` : 'mid-run, the loop has not been timed this turn'})`
				: 'the loop is settled and the context is at the danger line',
			midRun,
		};
	}
	if (agentLoop.tools > 0) return hold(`${agentLoop.tools} tool(s) are executing`);
	// A tool result that just landed may not be in the transcript yet; pi's own rule waited a second after
	// the last tool call for exactly this reason.
	if (Date.now() - agentLoop.lastToolEndAtMs < MIDRUN_DWELL_MS) return hold('a tool only just finished');
	if (midRun) {
		// The turn is live. Start only in the pause between rounds, and only when nothing is queued for the
		// round that is about to start: a steering message the user typed would otherwise die with the abort.
		const owed = pendingToolCalls(ctx);
		if (owed > 0) return hold(`${owed} tool call(s) from the newest answer have no result yet`);
		if (agentLoop.requestAnsweredAtMs > 0 && Date.now() - agentLoop.requestAnsweredAtMs < MIDRUN_DWELL_MS) return hold('the newest answer only just landed');
		if ((ctx as any)?.hasPendingMessages?.() === true) return hold('a message is queued for the next round');
		return { ok: true, why: `between rounds, mid-run (${agentLoop.sinceMs ? `${Math.round((Date.now() - agentLoop.sinceMs) / 1000)} s into the turn` : 'mid-run, the loop has not been timed this turn'})`, midRun: true };
	}
	const idle = (ctx as any)?.isIdle?.();
	if (idle === false) return hold('pi reports the agent is not idle');
	return { ok: true, why: 'the loop is settled', midRun: false };
}

/**
 * WHAT PI'S OWN BETWEEN-ROUNDS CHECK MEASURES, AS A SIGNED DIFFERENCE AGAINST THE WIRE (2026-10-15, written
 * down because the first theory about it was wrong and the log is what will settle it). pi arms its mid-run
 * trigger from `settings.compaction`, and its `_exceedsCompactionThreshold` asks
 * `estimateProjectedContextTokens`, which is the newest usage figure in the projection plus a chars-per-3.5
 * size guess over the entries that came after it. The obvious guess was that a projection holds no system
 * prompt, no tool schema and none of the guidance this file appends, so pi's figure would sit some 20,000
 * tokens under the fill the server really holds, and that a line written into pi's settings ought therefore to
 * be padded by the shortfall. The regression harness says the arithmetic runs the other way: the newest usage
 * figure already counts the whole request it answered - system prompt, tool schemas and all - and pi only adds
 * to it, so its measure is at least as big as ours, and padding it would have made pi's line EARLIER than
 * ours, not later. So this number is recorded and nothing more is done with it: it is the measurement that
 * will show, scenario by scenario, why pi's check crosses or does not cross while this file's watchdog holds.
 * Positive means pi sees more than the wire; negative means it sees less; a zero means there was no usage
 * figure after the newest summary to go on, which is itself a reason a check cannot fire.
 */
function piMeasureDelta(ctx: any): number {
	try {
		const wire = measureContext(ctx).wire;
		if (!(wire > 0)) return 0;
		const projection: any = (ctx as any)?.sessionManager?.buildSessionProjection?.();
		const msgs: any[] = projection?.messages ?? [];
		if (!msgs.length) return 0;
		let like = 0;
		let usageAt: number | null = null;
		for (let i = msgs.length - 1; i >= 0; i--) {
			const m: any = msgs[i];
			if (m?.role !== 'assistant' || !m.usage) continue;
			// The same rule as wireTokensOf: an answer that was aborted or errored describes the context as it
			// was, so pi passes it over when it looks for the usage figure to build its estimate on.
			if (m.stopReason === 'aborted' || m.stopReason === 'error') continue;
			const u = m.usage;
			like = u.totalTokens || (u.input ?? 0) + (u.output ?? 0) + (u.cacheRead ?? 0) + (u.cacheWrite ?? 0);
			usageAt = i;
			break;
		}
		if (usageAt === null) return 0; // no usage to go on: pi falls back to estimating the lot, we cannot say
		for (let i = usageAt + 1; i < msgs.length; i++) like += estimateTokens(msgs[i] as never);
		return Math.max(-60_000, Math.min(60_000, Math.round(like) - wire));
	} catch {
		return 0;
	}
}

/**
 * Keep pi's own compaction settings in step with this extension's trigger (2026-10-06). pi decides by
 * `contextTokens > window - reserveTokens`, and its contextTokens is the usage figure the server
 * reported - the same elided fill this extension measures - so writing our reserve into it makes pi's
 * line exactly our line. Nothing else changes hands: pi's stock summarizer can never answer, because
 * the session_before_compact hook of this extension replies to every compaction request with its own
 * summary, or cancels.
 */
/**
 * FULLY OFF (2026-10-06): the compaction switch is off AND the elision switch is off. In that state this
 * extension must behave as if it were not installed at all - it must not change a request, not answer or
 * cancel a compaction, not start a timer, and not write pi's settings.json - so that pi, or another
 * compaction extension working in a completely different way, runs exactly as designed.
 * The command /compact-plus still exists: a command nobody calls does nothing.
 */
/**
 * pi's own compaction flag, reported to the user once per process. Every line of this extension
 * READS pi's settings and not one writes them: the user switches pi's compaction on or off in pi's
 * own file, and this extension's automatic trigger works with pi's flag either way, because pi's
 * compact() and /compact ignore it - only pi's own threshold and overflow checks consult it.
 * (2026-10-18: the coupling that armed pi's flag on every tick is deleted, after the user found
 * pi's compaction enabled again without ever having enabled it, twice in as many days.)
 */
let piFlagAdvisoryShown = false;
function adviseOnPiFlag(ctx: any, cfg: Config): void {
	if (piFlagAdvisoryShown) return;
	piFlagAdvisoryShown = true;
	if (!readPiCompactionEnabled()) return;
	logLine(cfg, {
		event: "info",
		note: `pi's own compaction is enabled in ${join(AGENT_DIR, "settings.json")} (compaction.enabled): pi's threshold is the trigger, this extension answers it with its own summary; nothing here writes that file`,
	});
	// The flag is logged, never announced. An explanation of pi's compaction flag is on the settings
	// board in one line, and popping a paragraph over the operator's own prompt was the wrong place for
	// it (2026-10-21, on request: 'get rid of this').

}

function fullyOff(cfg: Config): boolean {
	return cfg.enabled !== true && cfg.elision.enabled !== true;
}


// 2026-10-07: THE START FLOOR, not a measuring floor. The context itself is measured on every tick and
// on every request; this stamp only says how soon a compaction may be STARTED after the last one that
// cleared the safe-point gates.
let nextCheckAtMs = 0;

/** How recently the gate said "held" in the log - the log is the record, so every hold is written at most
 *  once a minute rather than once per request. */
let lastGateHoldLogMs = 0;

function watcherTick(source: "timer" | "context" | "settled" = "timer", ctxArg?: any): void {
	// The `context` event hands its own context down, so its gate reads the signal of the request it is
	// standing in front of; the timer and the settle check work from whichever context an event last left
	// behind and are answered by the counters on agentLoop instead (see compactSafePoint).
	const ctx = ctxArg ?? watcherCtx;
	if (!ctx) return;
	let cfg: Config;
	try {
		cfg = loadConfig();
	} catch {
		return;
	}
	// NOTHING COMPACTION-WORTHY HAPPENS WHILE COMPACTION IS OFF (2026-10-06): the timer is stopped
	// rather than left running on a guard, and pi's settings file is handed back, so an off extension
	// leaves no timer, no writes and no trigger behind for anybody else to trip over.
	if (!cfg.enabled || !cfg.autoCompact) {
		stopWatcherTimer();
		return;
	}
	// pi's compaction flag is read for information only and never written by this extension.
	// The automatic trigger here is this extension's own: it measures the live context and starts
	// the compaction itself, so with pi's flag off automatic compaction still works - and pi never
	// writes a summary of its own. The coupling that armed pi's flag is deleted for good.
	if (parkedThisSession(cfg, ctx)) {
		// PARKED MEANS PARKED (2026-10-17, reversing a decision from 2026-10-06). A park used to expire by
		// itself after half an hour and arm the trigger again; that half-hour existed because a session once sat
		// parked for eight hours and filled its window while nobody was watching. The answer to that was never
		// to resume over your head. Stopping a compaction is a stop: the proactive trigger stays down until a
		// manual compaction runs, and running one is what re-arms it. What a park gets instead of an override is
		// a loud warning - once the room runs out you are told, and told again every ten minutes, because a
		// session that gets refused cannot be compacted at all and silence is how the eight-hour accident, which
		// is no longer allowed to happen, once passed unnoticed.
		const win = (ctx as any)?.model?.contextWindow ?? 0;
		const parked = measureContext(ctx);
		if (win > 0 && parked.wire > win - MIDRUN_DANGER_TOKENS * 2 && Date.now() - lastParkedWarnAtMs > 600_000) {
			lastParkedWarnAtMs = Date.now();
			logLine(loadConfig(), { event: "auto_compaction_parked_warning", wire: parked.wire, window: win, note: "still parked until a manual compaction, and the window is nearly gone" });
			ctx.ui?.notify?.(`Auto-compaction is parked because you stopped it, and it stays parked until you run a manual compaction. The context is at ${fmt(parked.wire)} of ${fmt(win)} tokens.`, "warning");
		}
		return;
	}
	if (compactionInFlight) return; // never re-trigger while a compaction is running
	try {
		const window = (ctx as any)?.model?.contextWindow ?? 0;
		if (!(window > 0)) return;
		// MEASURING IS FREE, STARTING NEEDS A SAFE POINT (2026-10-14). The gates that decide whether a
		// compaction may start moved to compactSafePoint(), just below the trigger line check. The
		// 2026-10-07 gates that used to sit here read ctx.compaction.*, which pi does not expose on
		// ExtensionContext, so they never once engaged: the watcher was starting compactions in the middle
		// of live requests and tools from that day until this one, and every one of those aborted the work
		// it was racing. Measuring still runs on every request and every five seconds.
	// WHAT THE SERVER HOLDS IS THE MEASURE: the elided request the model actually sees, the same figure
	// the status line shows first. The raw transcript on disk is bigger and is never the measure.
	const m = measureContext(ctx);
	const wireTokens = m.wire;
	const rawTokens = m.pi || m.sum;
	const tokens = m.tokens;
	const line = window - extReserve(ctx);
	// THE ONE STATE IN WHICH THERE IS NOTHING TO MEASURE (2026-10-14): the newest entry on the branch is a
	// summary and no answer has come since it. The wire is empty, pi reports its own estimate as unknown
	// until the next answer, and the only figure left is a guess over the summary plus the tail that stayed.
	// Starting a compaction there cannot find anything above that tail to summarise, so pi answered
	// "Nothing to compact (session too small)" and the user read it as a failure 0.9 s after a summary had
	// landed. The trigger stands down; the tick comes back in 5 s and will have a real number by then.
	if (m.justCompacted) {
		if (Date.now() - lastWatchLogMs > 300_000) {
			lastWatchLogMs = Date.now();
			// (2026-10-18: this watcher record is no longer written, as asked. The tick itself still
			//   runs - it measures the fill and decides what to do - and lastWatchLogMs still throttles it.)
		}
		return;
	}
	// 2026-10-06: a trigger that stays quiet says nothing, which is how a parked trigger sat out a whole
	// session. Once every five minutes the watcher writes down what it measured and why it did nothing.
	if (tokens < line) {
		if (Date.now() - lastWatchLogMs > 300_000) {
			lastWatchLogMs = Date.now();
			if (m.how === 'unmeasurable') {
				// A zero that is written down is far better than a zero that looks like an empty context:
				// this is the state in which a trigger silently never fires.
				// (2026-10-18: this watcher record is no longer written, as asked. The tick itself still
				//   runs - it measures the fill and decides what to do - and lastWatchLogMs still throttles it.)
				return;
			}
			// (2026-10-18: this watcher record is no longer written, as asked. The tick itself still
			//   runs - it measures the fill and decides what to do - and lastWatchLogMs still throttles it.)
		}
		return;
	}
		// PAST THE LINE IS A REASON TO ACT (2026-10-15, undoing 2026-10-14). Started only at a settled point, the
		// trigger was 10,000 tokens late and too late to be useful; see the note above agentLoop. So the loop is
		// not a veto any more: compactSafePoint now looks for the pause between rounds instead. The one case that
		// drops even that is the danger line, MIDRUN_DANGER_TOKENS under the whole window, where a refusal is the
		// alternative and a refused session cannot be compacted at all.
		const urgent = wireTokens > 0 && wireTokens >= window - MIDRUN_DANGER_TOKENS;
	// THE TRIGGER IS OUR OWN TIMER (2026-10-22, option A, the operator's choice: "if you can have your
	// own trigger and we don't need pi's trigger then great"). Until today this block stood the timer
	// down whenever pi's compaction.enabled was on, because pi's threshold was made the trigger and a
	// second scheduler would have raced it. The race was the wrong thing to fear: what the stand-down
	// bought was a line that belonged to pi's reserveTokens, so a smaller-context model compacted at
	// pi's number. Now the timer decides WHEN, from a measurement that costs nothing (the usage
	// figures the server already wrote into the session), and pi's threshold - with its reserveTokens
	// set far smaller than our reserve - is the alarm that rings only if this timer failed. The hook
	// still answers pi's ask with our own summary, and pi's own summariser is still the last resort.
	// What has NOT changed: the safe point below (no compaction in the middle of a write-up or a
	// pending tool call, exactly like pi waiting for the current step), the once-a-minute floor, the
	// refusal to fire twice on an unmeasured guess, and `urgent`, which drops the safe point when the
	// request already in flight is within MIDRUN_DANGER_TOKENS of the whole window.
		const sp = compactSafePoint(ctx, source === "context", urgent);
		if (!sp.ok) {
			const nowMs = Date.now();
			if (nowMs - lastHoldLogMs > 60_000) {
				lastHoldLogMs = nowMs;
				logLine(cfg, { event: "safe_point_held", note: `past the line but ${sp.why} - holding; nothing was started and nothing was aborted`, source, tokens: m.tokens, wire: wireTokens, raw: rawTokens, all: m.all, how: m.how, piDelta: piMeasureDelta(ctx), line, window, reserve: extReserve(ctx) });
			}
			// The one hold that deserves the screen: the request ALREADY SENT is close to the whole window,
			// so the next one in this turn is heading for the wall. Elision and the engine's clamp stand in
			// front of it; this only says so, at most once every five minutes.
			if (wireTokens > 0 && wireTokens >= window - 12_000 && nowMs - lastHoldNoticeAtMs > 300_000) {
				lastHoldNoticeAtMs = nowMs;
				ctx.ui?.notify?.(`Context is close to the whole window (${fmt(wireTokens)} of ${fmt(window)} tokens) but ${sp.why}, so no compaction was started - cutting one short would abort the work in progress. Elision is holding the request down and the engine clamps what is left of the room; compaction starts the moment the loop settles.`, "warning");
			}
			return;
		}
		const now = Date.now();
		if (now - lastAutoTriggerMs < 60_000) return; // one trigger per minute, no hammering
		// A second compaction on a GUESS, so soon after one landed, is how a session gets compacted twice for
		// nothing: allow it only on a measured wire figure, never on an estimate (2026-10-14).
		if (m.how !== 'wire' && lastCompactionDoneAtMs > 0 && now - lastCompactionDoneAtMs < 45_000) {
			if (now - lastWatchLogMs > 300_000) {
				lastWatchLogMs = now;
				// (2026-10-18: this watcher record is no longer written, as asked. The tick itself still
				//   runs - it measures the fill and decides what to do - and lastWatchLogMs still throttles it.)
			}
			return;
		}
		// THE START FLOOR: over the line is checked on every request and every five-second tick, but a
		// compaction is started at most once every 30 s (the 60 s floor above and the circuit breaker sit
		// on top of this).
		if (now < nextCheckAtMs) return;
		nextCheckAtMs = now + 30_000;
		// A failed auto-compaction is retried ONCE, 3 minutes later; the second failure stops it (see
		// pauseAutoCompaction), so a broken compaction cannot re-upload the whole context all night.
		if (autoFailure.retryAtMs > 0) {
			if (now < autoFailure.retryAtMs) return;
			autoFailure.retryAtMs = 0;
			logLine(cfg, { event: "auto_trigger_retry", estimate: tokens, line, window, note: "the one allowed retry after a failed auto-compaction" });
		}
		lastAutoTriggerMs = now;
		lastWatchLogMs = now;
		logLine(cfg, { event: "auto_trigger", estimate: tokens, wire: wireTokens, raw: rawTokens, line, window, reserve: extReserve(ctx), source, safePoint: sp.why, midRun: sp.midRun === true, urgent });
		lastCompactionTrigger = "auto";
		// Nothing is armed here any more. The interruption below is real - pi's compact() aborts the running
		// turn to write the summary - but the continuation is the operator's to send (2026-10-21).

		ctx.ui?.notify?.(
			sp.midRun === true
				? `Context at ${fmt(tokens)} tokens (trigger line ${fmt(line)}) - compacting now, mid-turn: ${sp.why}. pi's compact() interrupts this turn to let the summary land; when it finishes, send any message and the session continues from the summary.`
				: `Context at ${fmt(tokens)} tokens (trigger line ${fmt(line)}) - compacting now.`,
			"info",
		);
		// LAND IT WITHOUT STOPPING THE TURN when the operator has switched that on and we are not at the
		// danger line. The call runs detached, exactly like `ctx.compact()` (which is fire-and-forget by
		// contract and takes minutes here): the timer must not wait, and `compactionInFlight` inside keeps
		// a second compaction from starting meanwhile. A failure says so and falls back to pi's path, so
		// the session still gets compacted - by the interrupting route, which is the behaviour this
		// switch exists to move away from, but never a session left to overflow.
		if (cfg.landWithoutAbort && !urgent && landNoAbortHook) {
			void landNoAbortHook(ctx, source)
				.then((r) => {
					if (r.ok) {
						ctx.ui?.notify?.("COMPACTION · landed without stopping the turn - the next request carries the summary", "info");
						return;
					}
					const note = String(r.note ?? "").slice(0, 160);
					logLine(loadConfig(), { event: "no_abort_landing_failed", note, source, line, window });
					ctx.ui?.notify?.(`Compaction could not land without an interrupt (${note.slice(0, 90)}) - using pi's own path, which stops the turn.`, "warning");
					(ctx as any).compact?.({
						onComplete: () => ctx.ui?.notify?.("Automatic compaction landed. The turn was interrupted to run it - send any message and the session continues from the summary.", "info"),
						onError: (err: any) => ctx.ui?.notify?.(`Automatic compaction failed: ${String(err?.message ?? err).slice(0, 160)}`, "error"),
					});
				})
				.catch((err: any) => logLine(loadConfig(), { event: "no_abort_landing_failed", note: `unhandled: ${String(err?.message ?? err).slice(0, 140)}`, source }));
			return;
		}

		(ctx as any).compact?.({
			onComplete: () => ctx.ui?.notify?.(sp.midRun === true ? "Automatic compaction landed. The turn was interrupted to run it - send any message and the session continues from the summary." : "Automatic compaction landed. Nothing was running, so the session simply carries on from the summary.", "info"),
			onError: (err: any) => ctx.ui?.notify?.(`Automatic compaction failed: ${String(err?.message ?? err).slice(0, 160)}`, "error"),
		});
	} catch {
		/* the watcher must never break the session */
	}
}

/** THE AUTO-COMPACTION CIRCUIT BREAKER (2026-10-05). An auto-compaction that fails used to leave the
 *  context above the trigger line, so the watcher fired again 60 s later - ~100 times in one session,
 *  each attempt uploading the whole context. Now: a user cancel stops the auto-compactor at once; a
 *  failure is retried ONCE after 3 minutes and, if it fails again, the auto-compactor is switched off
 *  and says so on screen in capitals. It stays off until the user runs a manual compaction. */
// Exported so the rig can arrive at the moment AFTER a failure without a rig having to fail twice for
// three minutes; the code reads and writes the same object it always did.
const autoFailure = { fails: 0, retryAtMs: 0, pausedAtMs: 0 };
/** "auto" while the watcher's own compaction is in flight, "manual" for a user-run one. */
let lastCompactionTrigger = "manual";
/** When a compaction LANDED, whoever asked for it. While that number is young the context size cannot be
 *  measured (the server has not answered since), so the watcher refuses to start a compaction on a guess:
 *  2026-10-14, a session triggered again 0.9 s after a summary had landed, on the size of the whole
 *  on-disk history, and pi answered it with "Nothing to compact (session too small)". */
/** The models tried and lost by the compaction now in flight (see `summarizeWithRotation`). Read when
 *  the report is printed; empty when the first candidate worked. */
let lastRotationFailures: string[] = [];

/** What the winning attempt knew, kept for the one closing row. The `done` record used to be a row of
 *  its own, one second before COMPACTED; the operator asked for a single line at the end (2026-10-23), so
 *  these fields ride on that row instead. */
let lastLandingInfo: Record<string, any> = {};

/** ONE GLANCE AT THE SCREEN (2026-10-22, the operator's request): what a finished compaction cost, who
 *  wrote it, and how full the session was - printed as rows, because reading those facts meant opening
 *  the settings and the log before. pi prints one row per notice, so the lines go out one at a time.
 *  A pure builder, so a rig can read exactly what the screen would have said. */
export function compactionNoticeLines(
	r: any,
	opts: { before: number; window: number; keepRecent: number; failures: string[]; trigger: string },
): string[] {
	const out: string[] = [];
	const pct = opts.window > 0 ? Math.round((opts.before / opts.window) * 100) : 0;
	out.push(
		`COMPACTION · ${opts.trigger === "manual" ? "you asked for it" : "automatic"} · summary by ${slotLabel(r.ref, r.model)}` +
			(r.via === "direct" ? " (direct request)" : "") +
			(typeof r.seconds === "number" ? ` · ${r.seconds}s` : ""),
	);
	out.push(
		`COMPACTION · read ${fmt(r.input ?? r.material ?? 0)} tok -> wrote ${fmt(r.measured)} tok` +
			(typeof r.words === "number" ? ` · ${fmt(r.words)} words` : "") +
			` (accepted ${fmt(r.min)}-${fmt(r.max)})`,
	);
	out.push(
		`COMPACTION · context was ${fmt(opts.before)} of ${fmt(opts.window)} tok (${pct}% full) · newest ${fmt(opts.keepRecent)} tok kept word for word`,
	);
	for (const f of opts.failures) out.push(`COMPACTION · tried and lost: ${String(f).slice(0, 150)}`);
	return out;
}

/** Every line on its own row. A screen is not worth losing a compaction over, so a UI that refuses a
 *  line is ignored rather than allowed to throw between the summary and the session file. */
function announceCompaction(ctx: any, out: string[], kind: "info" | "warning" | "error" = "info"): void {
	for (const line of out) {
		try {
			ctx?.ui?.notify?.(line, kind);
		} catch {
			/* ignore */
		}
	}
}

/** The same for a compaction that produced no summary: the rotation's failure list is one string per
 *  model joined with ' | ', which becomes one row per model, plus what happens next. */
/** A short, readable name for a model out of its reference alone - the failure rows have the reference
 *  and not the model object. `http://127.0.0.1:9931/qwen3.8-flash…` becomes the file it is serving, and
 *  `openrouter/z-ai/glm-5.3-flash` keeps its family, because dropping `z-ai` would make it ambiguous. */
export function failureRowLabel(ref: string): string {
	const bare = String(ref).replace(/^https?:\/\//, "").replace(/\.gguf$/i, "");
	const parts = bare.split("/").filter(Boolean);
	if (parts.length <= 1) return bare || String(ref);
	const host = /^(\d{1,3}\.){3}\d{1,3}/i.test(parts[0]) || /^localhost/i.test(parts[0]);
	return host ? parts[parts.length - 1] : parts.slice(-2).join("/");
}

/** The same report for a compaction that produced no summary. The rotation's failure list is one string
 *  per model joined with ' | ', and in the ordinary case - the model wrote far too few or far too many
 *  tokens - each of those strings already carries the number that was measured and the limit it broke,
 *  which is exactly the information that used to be buried in the log. So: a header, one row per model,
/** A failed compaction names the models that refused and says why, then what the trigger does next. */
export function compactionFailureNoticeLines(error: string, note: string): string[] {
	const parts = String(error).split(" | ").map((p) => p.trim()).filter(Boolean);
	const named = parts.filter((p) => p.includes(": "));
	const out: string[] = [];
	out.push(
		named.length
			? `COMPACTION FAILED · no summary written · ${named.length} model${named.length > 1 ? "s" : ""} tried`
			: `COMPACTION FAILED · ${String(error).slice(0, 180) || "unknown reason"}`,
	);
	for (const p of named.slice(0, 6)) {
		const i = p.indexOf(": ");
		out.push(`COMPACTION FAILED · ${failureRowLabel(p.slice(0, i))} · ${p.slice(i + 2).slice(0, 140)}`);
	}
	if (named.length > 6) out.push(`COMPACTION FAILED · …and ${named.length - 6} more, in the compaction log`);
	if (note) out.push(`COMPACTION FAILED · ${note}`);
	return out;
}
let lastCompactionDoneAtMs = 0;
/** The no-abort landing of #43, which lives inside install() because `handleCompaction` does. The
 *  watcher is a top-level function and cannot see into that closure, so the landing hands itself to the
 *  module here - and if nothing ever assigned it (a rig that loads the file without installing it, or a
 *  future build where the code moved), the watcher simply uses pi's own path. */
let landNoAbortHook: ((ctx: any, why: string) => Promise<{ ok: boolean; note?: string }>) | undefined;
/** The "nothing left to summarise" notice is worth reading once per 15 minutes, not once a minute. */
let lastNoOpNoticeAtMs = 0;
/** The parked-and-the-room-is-running-out warning, throttled to once every ten minutes. */
let lastParkedWarnAtMs = 0;

/** Whether the proactive trigger is parked FOR THE SESSION THIS CALL IS ABOUT (2026-10-17). A stop belongs
 *  to the session it happened in: a new or resumed session must find the trigger armed, and the stopped
 *  session must stay parked until a manual compaction in it re-arms it. A legacy `true` from an older file
 *  still parks everywhere - silently un-parking a session somebody stopped would be its own betrayal, and
 *  the overlay's switch or a manual compaction is what clears it. */
export function parkedThisSession(cfg: any, ctx: any): boolean {
	const v = cfg?.autoCompactPaused;
	if (!v) return false;
	if (v === true) return true;
	const id = String(ctx?.sessionManager?.getSessionId?.() ?? "session");
	return v === id || v === "session";
}

/** Switch the proactive trigger off and say so loudly. A manual compaction is the only thing that switches
 *  it back on: there is no clock on a park (2026-10-17), and pi's stock summariser is never substituted. */
function pauseAutoCompaction(ctx: any, reason: string, detail: string): void {
	autoFailure.fails = 0;
	autoFailure.retryAtMs = 0;
	autoFailure.pausedAtMs = Date.now();
	lastParkedWarnAtMs = 0;
	const sid = String(ctx?.sessionManager?.getSessionId?.() ?? "session");
	update((c) => void (c.autoCompactPaused = sid));
	logLine(loadConfig(), { event: "auto_compaction_paused", reason, detail: String(detail).slice(0, 300), session: sid.slice(-8), at: new Date().toISOString() });
	ctx?.ui?.notify?.(`AUTO-COMPACTION PARKED (${reason}): ${String(detail).slice(0, 160)} - IT STAYS PARKED FOR THIS SESSION UNTIL A MANUAL COMPACTION ARMS IT AGAIN. ANY OTHER SESSION, NEW OR RESUMED, STARTS ARMED.`, "warning");
}

let watcherTimer: ReturnType<typeof setInterval> | undefined;

/** pi's docs: do not start timers in the factory, because some invocations load extensions without
 *  a session (print mode, one-shot tries). Start the watcher lazily on the first live request
 *  instead of at module load. */
/** Elision's own switch, read fresh, for the request hook: off means the message list is untouched. */
function cfgForHookElision(): boolean {
	try {
		const c = loadConfig();
		return c.elision.enabled === true;
	} catch {
		return false;
	}
}

/** Whether the idle watcher is worth running at all: elision on, or auto-compaction on. Off = no timer. */
function cfgForHookTimer(): boolean {
	try {
		const c = loadConfig();
		return c.elision.enabled === true || (c.enabled === true && c.autoCompact === true);
	} catch {
		return false;
	}
}

function stopWatcherTimer(): void {
	if (!watcherTimer) return;
	clearInterval(watcherTimer);
	watcherTimer = undefined;
}

/** pi retires a captured ctx the moment the session behind it is replaced or the extension reloads, and
 *  it announces the fact by THROWING from every property of that dead ctx: "This extension ctx is stale
 *  after session replacement or reload...". Every timer in this extension holds a ctx, or the api, or
 *  both, across that line - and on 2026-10-20 the watchdog tick reached for a session id through a ctx
 *  the settings menu had already retired. The throw happened inside a setInterval callback with no one
 *  above it to catch anything, so it surfaced as an uncaughtException and pi exited.
 *
 *  A timer that finds its context dead therefore stops itself, drops what it was holding and waits:
 *  the next agent_start leaves a live ctx behind, and the first live request starts the clock again.
 *  A timer error of any other kind is still swallowed, because a timer callback has nowhere else for an
 *  error to go - but now it is swallowed INTO THE LOG, where it can be read, instead of into the exit
 *  of the program or into silence.
 */
export function ctxIsStaleError(e: any): boolean {
	return /extension ctx is stale/i.test(String(e?.message ?? e ?? ""));
}
function noteDeadTimer(where: string, e: any): void {
	const stale = ctxIsStaleError(e);
	try {
		logLine(loadConfig(), { event: stale ? "ctx_stale" : "timer_error", where, error: String(e?.message ?? e).slice(0, 200), note: stale ? "a timer was holding a context pi had already replaced; it stopped itself and waits for the next agent_start to hand it a live one" : "a timer callback threw; pi stayed up, because a timer error has nowhere else to be caught" });
	} catch {
		/* a logger that cannot write must never become the second problem */
	}
	if (!stale) return;
	stopWatcherTimer();
	watcherCtx = null;
}
/** Run the body of a timer, sync or async, and route anything it throws - including a rejection - to
 *  noteDeadTimer rather than letting it reach pi's process-level handler, where an uncaughtException
 *  ends the session. */
export function runTimerBody(where: string, run: () => void | Promise<unknown>): void {
	try {
		const r = run();
		if (r && typeof (r as any).then === "function") void (r as Promise<unknown>).catch((e) => noteDeadTimer(where, e));
	} catch (e) {
		noteDeadTimer(where, e);
	}
}

function ensureWatcherTimer(): void {
	if (watcherTimer) return;
	watcherTimer = setInterval(() => runTimerBody("watchdog", () => watcherTick("timer")), WATCHER_INTERVAL_MS);
	(watcherTimer as unknown as { unref?: () => void })?.unref?.();
}

export default function piCompactPlusExtension(pi: ExtensionAPI): void {
	// A setting the loader does not copy is worth saying something about: the operator sees a switch
	// that appears to work and reverts over a restart. Say it once, in the log, and write the missing
	// keys back so the file is complete from now on (2026-10-21, defect #41).
	{
		const cfg = loadConfig();
		const missing = missingConfigKeys(cfg);
		if (missing.length) {
			logLine(cfg, { event: "config_repaired", missing: missing.join(",") });
			if (!writesBlocked()) saveConfig(cfg);
		}
	}
	// WHAT PI DID WITH A COMPACTION AFTER THIS EXTENSION'S OWN HOOK WAS DONE (2026-10-14). The hook sees a
	// compaction it takes part in; these two see every one, including the ones pi raises on its own side -
	// and until now those were invisible: a refusal for want of anything to summarise appeared on screen as an
	// ERROR with nothing in the log to explain it, and a genuine failure from that path was never counted by
	// the circuit breaker, so the trigger could knock on the same door once a minute for the rest of the day.
	// WHERE THE AGENT LOOP IS (2026-10-14). These handlers exist for one reason: to be able to say, at any
	// instant, whether starting a compaction would interrupt something. pi emits them all (types.d.ts); the
	// counters are kept defensively (a start without its end, or an end without its start, cannot drift the
	// count below zero or leave the loop stuck as running) and every one of them also refreshes the watcher's
	// context, so the trigger works before the first request of a session has ever gone out.
	pi.on("agent_start", (_event: any, ctx: any) => {
		watcherCtx = ctx;
		agentLoop.running = true;
		agentLoop.sinceMs = Date.now();
		return undefined;
	});
	pi.on("agent_end", (_event: any, ctx: any) => {
		watcherCtx = ctx;
		agentLoop.running = false;
		return undefined;
	});
	pi.on("agent_settled", (event: any, ctx: any) => {
		watcherCtx = ctx;
		agentLoop.running = false;
		agentLoop.lastSettledAtMs = Date.now();
		if ((event as any)?.aborted) agentLoop.lastAbortAtMs = Date.now();
		// THE BEST MOMENT TO COMPACT IS THE SECOND THE TURN ENDS, and the five-second timer can miss it: a
		// queued message starts the next turn inside that gap and the gate shuts again. So look 1.2 s after
		// the loop settles - long enough for a follow-on turn to declare itself and be waited out, early
		// enough to shrink the context before the next request is ever built. Nothing is started by the settle
		// itself: it asks the watcher to look, and the watcher still applies every gate.
		if (settleCheckTimer) clearTimeout(settleCheckTimer);
		settleCheckTimer = setTimeout(() => {
			settleCheckTimer = undefined;
			// Guarded rather than wrapped in a silent catch (2026-10-20): the swallow here used to hide the
			// one error class that could not be survived elsewhere - a ctx pi had already retired - and a
			// silent catch on a dead ctx means the watcher stops looking and nobody is told why.
			runTimerBody("settle-check", () => watcherTick("settled"));
		}, 1200);
		(settleCheckTimer as unknown as { unref?: () => void })?.unref?.();
		return undefined;
	});
	pi.on("tool_execution_start", (event: any, ctx: any) => {
		watcherCtx = ctx;
		if ((event as any)?.parentToolCallId) return undefined; // a nested call lives inside its parent
		agentLoop.tools += 1;
		return undefined;
	});
	pi.on("tool_execution_end", (event: any, ctx: any) => {
		watcherCtx = ctx;
		if (!(event as any)?.parentToolCallId) agentLoop.tools = Math.max(0, agentLoop.tools - 1);
		agentLoop.lastToolEndAtMs = Date.now();
		return undefined;
	});
	pi.on("ui_prompt_start", (_event: any, ctx: any) => {
		watcherCtx = ctx;
		agentLoop.promptOpen = true;
		return undefined;
	});
	pi.on("ui_prompt_end", (_event: any, ctx: any) => {
		watcherCtx = ctx;
		agentLoop.promptOpen = false;
		return undefined;
	});

	pi.on("session_compact", (event: any) => {
		lastCompactionDoneAtMs = Date.now();

		const entry: any = (event as any)?.compactionEntry ?? {};
		logLine(loadConfig(), {
			event: "compaction_landed",
			reason: (event as any)?.reason,
			fromExtension: Boolean((event as any)?.fromExtension),
			// One figure, not two: when the engine told us what the summary cost, that is the size on the row.
			// The old character estimate of the stored text sat beside it as a second, different number
			// (13,090 against an engine count of 12,355) and both looked authoritative.
			summaryTokens: landedSummarySize(lastLandingInfo, String(entry?.summary ?? c?.summary ?? "")),
			handoff: true,
			// No note: the 45-second floor this used to repeat is the trigger's own arithmetic, and this row
			// is a result the operator reads for one number - the size of the summary that landed.
			...lastLandingInfo,
		});
		return undefined;
	});
	pi.on("session_compact_failed", (event: any, ctx: any) => {
		const cfg = loadConfig();
		if (fullyOff(cfg)) return undefined;
		const failure: any = (event as any)?.errorMessage ?? ((event as any)?.aborted ? 'cancelled' : '');
		const error = String(failure ?? '').trim() || 'unknown';
		// Cancel and the extension's own failures are already logged and counted by the hook above.
		if ((event as any)?.aborted) return undefined;
		if ((event as any)?.fromExtension) return undefined;
		// Nothing above the kept tail to summarise is a NO-OP, not a failure: it must not spend the one
		// allowed retry, must not park the trigger, and must not read like a breakdown.
		if (/nothing to compact|already compacted|too small/i.test(error)) {
			const strikes = countNoOpStrike(Date.now());
			logLine(cfg, { event: "compaction_no_op", reason: (event as any)?.reason, error: error.slice(0, 200), trigger: lastCompactionTrigger, strikes, windowMin: 15, note: "the summary plus the tail that stayed already cover the conversation - NOT counted as a failure, no retry spent" + (strikes >= 3 && lastCompactionTrigger === "auto" ? "; THIRD ONE IN 15 MINUTES - the trigger parks itself instead of knocking once a minute forever" : ", the trigger stays armed") });
			if (strikes >= 3 && lastCompactionTrigger === "auto") {
				pauseAutoCompaction(ctx, "nothing to summarise three times", `${error.slice(0, 90)} - three times in 15 minutes, which means the trigger is firing somewhere it cannot find any work; the aim and the trigger line are worth checking (/compact-plus menu)`);
				return undefined;
			}
			if (Date.now() - lastNoOpNoticeAtMs > 900_000) {
				lastNoOpNoticeAtMs = Date.now();
				ctx?.ui?.notify?.(`Compaction found nothing to summarise (${error.slice(0, 60)}) - the conversation is already down to its summary plus the recent tail, and the full transcript is on disk. Nothing was retried and the automatic trigger is still armed.`, "info");
			}
			return undefined;
		}
		// A real failure pi raised outside this extension's path: same breaker as our own.
		autoFailure.fails += 1;
		if (lastCompactionTrigger === "auto" && autoFailure.fails === 1) {
			autoFailure.retryAtMs = Date.now() + 180_000;
			logLine(cfg, { event: "auto_compaction_failed", error: error.slice(0, 200), trigger: lastCompactionTrigger, failure: autoFailure.fails, from: "pi", retryInMs: 180_000, note: "raised by pi itself, not by this extension's compaction path" });
			ctx?.ui?.notify?.(`COMPACTION FAILED (1 of 2): ${error.slice(0, 140)} - IT WAITS 3 MINUTES AND TRIES ONCE MORE; IF THAT FAILS TOO IT SWITCHES ITSELF OFF.`, "warning");
			return undefined;
		}
		if (lastCompactionTrigger === "auto") pauseAutoCompaction(ctx, "pi refused the compaction twice", error);
		else logLine(cfg, { event: "compaction_failed", reason: (event as any)?.reason, error: error.slice(0, 200), trigger: lastCompactionTrigger, from: "pi" });
		return undefined;
	});

	// ELISION: rule-based stubbing of old tool results in every live request (see applyElision).
	// The handler's returned message list is what actually gets sent; stored history is untouched,
	// so pi's own compaction machinery always sees the raw originals. The handler also captures the
	// freshest extension ctx for the auto-compact watcher above.
	pi.on("context", (event: any, ctx: any) => {
		// The same moment is used to say the quiet part out loud, once: if pi's own compaction flag is on,
		// two compactors would be live in this session (adviseOnPiFlag, which nags at most once an hour and
		// writes nothing anywhere). This is NOT where pi's line used to be kept equal to ours - the pre-request
		// coupling that did that came out on 2026-10-18, and pi's settings.json is read here and never written.
		// A comment claiming a write survived three days longer than the code that justified it did.
		try {
			const cc = loadConfig();
			// FULLY OFF = invisible (2026-10-06): no settings write, no timer, no elision, no log line.
			if (fullyOff(cc)) return undefined;
			adviseOnPiFlag(ctx, cc);
			// pi's compaction flag is read for information only and never written by this extension.
			// The automatic trigger here is this extension's own: it measures the live context and starts
			// the compaction itself, so with pi's flag off automatic compaction still works - and pi never
			// writes a summary of its own. The coupling that armed pi's flag is deleted for good.
		} catch {
			/* never blocks a request */
		}
		// The elision stage must NEVER break a live request: it runs on every request, so an error
		// here would take the whole session down with it. First failure: a detailed log line, an
		// ALL-CAPS notice on screen and one retry allowed 3 minutes later; second failure: elision is
		// off for this session and says so (it is a context helper, not something allowed to eat the
		// session). /compact-plus elision on (or a /reload) switches it back on.
		watcherCtx = ctx;
		// ONE COMPACTOR (2026-10-07): the startup check, once per session, over the freshest ctx.
		warnOnceAboutTwoCompactors(ctx);
		// The watcher runs when either side could need it; with both switches off it is never started.
		if (cfgForHookTimer()) ensureWatcherTimer();
		// 2026-10-07: THE TRIGGER IS DRIVEN BY THIS EVENT, not only by the timer. The `context` event
		// fires before EVERY request - after every tool batch, after every assistant block - which makes it
		// the one hook in pi that runs in the middle of a long job; that is how the other context manager
		// triggers with pi's own compaction switched off. Running the tick here means the trigger line is
		// checked at the exact moment the next request is built, not up to five seconds later. Safe to call
		// on every request: measuring is cheap, and STARTING a compaction is gated by pi's safe-point rules
		// plus the 30 s start floor, the one-per-minute floor and the circuit breaker. The five-second
		// timer stays as the backstop for one very long generation with no requests in between.
		if (cfgForHookTimer()) {
			try {
				watcherTick("context", ctx);
			} catch {
				/* the trigger must never block a live request */
			}
		}
		if (!cfgForHookElision()) {
			// Returning straight here used to skip the stubber entirely, which quietly lost two
			// things: the record that the switch was ever thrown, and the clearing of the applied-
			// stub state. That state remembers WHICH results are already stubbed on the wire and
			// where the prompt-cache break point sits, so stubs left in it after a switch-on saw
			// raw originals as already stubbed (they are not - an off request goes out untouched)
			// and it placed the break point where nothing had been changed. applyElision has its
			// own off branch for exactly this bookkeeping and returns undefined, so the request
			// is still handed on exactly as it arrived.
			return applyElision(event?.messages, ctx);
		}
		if (elisionFailure.off) return undefined;
		if (elisionFailure.retryAtMs > 0) {
			if (Date.now() < elisionFailure.retryAtMs) return undefined; // waiting out the one retry
			elisionFailure.retryAtMs = 0;
		}
		try {
			const out = applyElision(event?.messages, ctx);
			if (elisionFailure.fails) elisionFailure.fails = 0;
			return out;
		} catch (err: any) {
			elisionFailure.fails += 1;
			const msg = String(err?.message ?? err).slice(0, 300);
			const stack = String(err?.stack ?? "").split("\n").slice(1, 4).join(" | ").slice(0, 300);
			logLine(loadConfig(), {
				event: "elision_error",
				failure: elisionFailure.fails,
				error: msg,
				stack,
				messages: Array.isArray(event?.messages) ? event.messages.length : -1,
				session: String(ctx?.sessionManager?.getSessionId?.() ?? "session").slice(-8),
				note: elisionFailure.fails >= 2 ? "ELISION OFF for this session - /compact-plus elision on to switch it back" : "elision paused, one retry allowed in 3 minutes",
			});
			ctx?.ui?.notify?.(elisionFailure.fails >= 2
				? `ELISION IS OFF FOR THIS SESSION - IT FAILED TWICE AND WILL NOT TRY AGAIN UNTIL YOU TURN IT BACK ON (/compact-plus elision on). LAST ERROR: ${msg.slice(0, 120)}`
				: `ELISION FAILED (attempt ${elisionFailure.fails}): ${msg.slice(0, 120)} - IT PAUSES NOW AND RETRIES ONCE IN 3 MINUTES. FULL DETAIL IN ~/.pi/agent/pi-compact-plus.log`, "warning");
			if (elisionFailure.fails >= 2) elisionFailure.off = true;
			else elisionFailure.retryAtMs = Date.now() + 180_000;
			return undefined; // the request goes out unstubbed - elision never blocks a turn
		}	});

	// CHAT max_tokens cap: pi sizes each chat request's generation permission as
	// window - context estimate - 4,096, which balloons on big-window models (943,718 on
	// glm-flash-latest) and then gets the whole request rejected whenever pi's estimate
	// undercounts the real wire by more than the 4,096 cushion - the overflow class behind the
	// 630k and 295k emergency compactions (the estimate undercounted by ~69k on 2026-10-01
	// 00:25). With the cap the request always fits until real ~ window - cap. The fork's own
	// compaction requests size themselves via genCap and never pass this hook (only the agent
	// loop carries onPayload).
	pi.on("before_provider_request", (event: any, ctx: any) => {
		// A request this process sent and has not seen answered yet (see the two timestamps on
		// agentLoop for why the gate trusts this only while it is young). The hook also REWRITES the outgoing
		// payload below, which is why an extension that is switched off stays out of it entirely (2026-10-06).
		watcherCtx = ctx;
		agentLoop.requestsOut += 1;
		agentLoop.requestStartedAtMs = Date.now();
		// CHAT max_tokens cap: pi sizes each chat request's generation permission as
		// window - context estimate - 4,096, which balloons on big-window models (943,718 on
		// glm-flash-latest) and then gets the whole request rejected whenever pi's estimate
		// undercounts the real wire by more than the 4,096 cushion - the overflow class behind the
		// 630k and 295k emergency compactions (the estimate undercounted by ~69k on 2026-10-01
		// 00:25). With the cap the request always fits until real ~ window - cap. The fork's own
		// compaction requests size themselves via genCap and never pass this hook (only the agent
		// loop carries onPayload).
		const cfgHook = loadConfig();
		if (!cfgHook.enabled) return undefined;
		const cap = cfgHook.chatMaxTokensCap;
		if (!cap || cap <= 0) return undefined;
		const p: any = event?.payload;
		if (!p || typeof p !== "object") return undefined;
		let changed = false;
		for (const field of ["max_tokens", "max_completion_tokens"] as const) {
			if (typeof p[field] === "number" && p[field] > cap) {
				p[field] = cap;
				changed = true;
			}
		}
		const gc = p?.generationConfig;
		if (gc && typeof gc === "object" && typeof gc.maxOutputTokens === "number" && gc.maxOutputTokens > cap) {
			gc.maxOutputTokens = cap;
			changed = true;
		}
		if (!changed) return undefined;
		/* no log line: the cap fires on most big-window requests - pure noise in the log */
		return p;
	});

	pi.on("after_provider_response", (_event: any, ctx: any) => {
		watcherCtx = ctx;
		agentLoop.requestsOut = Math.max(0, agentLoop.requestsOut - 1);
		agentLoop.requestAnsweredAtMs = Date.now();
		return undefined;
	});

	pi.on("session_before_compact", async (event: any, ctx: any) => {
		// FIRST THING (2026-10-06): with this extension's compaction switched off, this hook answers
		// NOTHING - not a summary, not a cancel - so pi's own compaction, or another extension's, runs
		// exactly as designed. Everything below, including the cancels that protect pi from its own
		// summarizer while we ARE the compactor, is unreachable when off.
		try {
			const c0 = loadConfig();
			if (!c0.enabled) {
				return undefined;
			}
		} catch {
			return undefined;
		}
		// The file is read again here every time, so an edit to it takes effect without /reload.
		if (skipNextStockCompaction) {
			skipNextStockCompaction = false;
			return undefined;
		}
		// PI RINGS THE DOORBELL, THIS EXTENSION OPENS THE DOOR (2026-10-21, the operator's design). pi's
		// threshold is deliberately set earlier than our own trigger point; between the two, the compaction
		// is held. The cancel answers pi so plainly that pi does not fall back to its own summariser.
		{
			const win = ctx.model?.contextWindow ?? Number(event.windowSize ?? 0) ?? 0;
			const note = holdUntilOurLine({
				reason: String(event.reason ?? ""),
				fill: realContextTokens(ctx),
				window: win,
				line: win - extReserve(ctx),
				// One percent of the window, floor 1,000 tokens. pi measures the context from the server's own
				// usage figures and this extension measures its own way; refusing a compaction over that
				// difference would cost a full pass over the context for nothing.
				slack: Math.max(1_000, Math.round(win * 0.01)),
			});
			if (note !== null) {
				const nowMs = Date.now();
				if (nowMs - lastGateHoldLogMs > 60_000) {
					lastGateHoldLogMs = nowMs;
					// `loadConfig()` here, not the config from the try block above: that one is block-scoped and
					// does not reach this line, which is how a hold would have thrown instead of cancelling.
					logLine(loadConfig(), { event: "auto_compaction_held", reason: event.reason, note });
				}
				// pi-stamp-plus is told before the cancel goes back to pi, because pi stamps the
				// failed compaction the instant it receives the answer.
				markCompactionHeld();
				if (nowMs - lastGateHoldNoticeAtMs > 600_000) {
					lastGateHoldNoticeAtMs = nowMs;
					ctx.ui?.notify?.(`Compaction held: ${note}`, "info");
				}
				return { cancel: true };
			}
		}
		compactionInFlight = true;
		try {
			return await handleCompaction(event, ctx);
		} catch (err: any) {
			// A crash inside our own compaction must never turn into "pi writes a summary instead". On
			// 2026-10-22 a plain coding error escaped from here, pi fell back to its own summariser, and it
			// produced ~7k-token summaries while the operator was trying to stop the whole thing. Cancel,
			// say so on screen, and leave the session's context alone.
			const why = String(err?.stack ?? err?.message ?? err).split("\n").slice(0, 3).join(" | ").slice(0, 300);
			logLine(loadConfig(), { event: "compaction_failed", error: why, note: "the extension's own compaction threw; pi's summarizer was NOT allowed to substitute", trigger: lastCompactionTrigger });
			ctx?.ui?.notify?.(`COMPACTION FAILED · the extension's own compaction hit an internal error and was cancelled; pi's summarizer was not substituted: ${why.slice(0, 140)}`, "error");
			return { cancel: true };
		} finally {
			compactionInFlight = false;
		}
	});

	/**
	 * LAND THE SUMMARY WITHOUT STOPPING THE TURN (2026-10-22, #43), when `landWithoutAbort` is on.
	 *
	 * Today this extension starts its compactions with `ctx.compact()`, which is pi's MANUAL path: pi
	 * stops the running loop, writes the summary, and waits for the operator to type something. That is
	 * the pair of messages the operator reported - "operation aborted" and the ABORTED stamp - and it is
	 * not a pi limitation: pi's own threshold compaction happens inside the request pipeline and never
	 * stops anything, because a compaction that is merely APPENDED to the session is picked up by the
	 * loop's next request on its own (pi rebuilds `context.messages` from `buildSessionProjection()` in
	 * `prepareRequest`, agent-session.js:434-446).
	 *
	 * So: build pi's own preparation, hand it to the same `summarizeWithRotation` the event handler
	 * uses (identical model rotation, shape gate, retries and log rows), and append the result with the
	 * same six arguments pi passes at agent-session.js:2268. What is NOT reproduced is pi's
	 * `session_compact` event, which an extension cannot emit: the on-screen transcript may keep the
	 * pre-compaction view until a reload, and pi-stamp-plus writes no stamp for it. Both are recorded
	 * here in the log instead, and the switch stays off until that has been watched on a real run.
	 *
	 * The danger line never takes this path: inside MIDRUN_DANGER_TOKENS a request is about to be
	 * refused, and a refused session cannot be compacted at all, so an interrupted turn is the lesser
	 * failure and pi's interrupting path stays in charge.
	 */
	async function landCompactionNoAbort(ctx: any, why = "timer"): Promise<{ ok: boolean; note?: string }> {
		const cfg = loadConfig();
		const sm: any = ctx?.sessionManager;
		// `ctx` types the session manager as read-only; at runtime pi passes the real object
		// (agent-session-services.js:136). If a future build really does hand out a read-only view, this
		// check says so and the caller uses pi's path instead of half-landing a summary.
		if (typeof sm?.appendCompaction !== "function") return { ok: false, note: "this pi gives extensions a read-only session manager" };
		let mod: any;
		try {
			mod = await piCompactionModLoad();
		} catch (err: any) {
			return { ok: false, note: `pi's compaction module could not be loaded: ${String(err?.message ?? err).slice(0, 110)}` };
		}
		if (typeof mod?.prepareCompaction !== "function") return { ok: false, note: "pi's prepareCompaction is not exported in this build" };
		const pathEntries: any[] = sm.getBranch?.() ?? [];
		if (pathEntries.length === 0) return { ok: false, note: "the session branch is empty" };
		const piSettings = readPiCompactionSettings();
		let preparation: any;
		try {
			preparation = mod.prepareCompaction(pathEntries, {
				...(mod.DEFAULT_COMPACTION_SETTINGS ?? {}),
				reserveTokens: piSettings.reserveTokens,
				keepRecentTokens: piSettings.keepRecentTokens,
			});
		} catch (err: any) {
			return { ok: false, note: `pi's preparation could not be built: ${String(err?.message ?? err).slice(0, 130)}` };
		}
		// Undefined is pi's own "there is nothing to compact" answer (it returns it when the newest entry
		// is already a compaction). Falling back to `ctx.compact()` here would ask pi to summarise a
		// session it calls too small, which is the "Nothing to compact" message the operator has seen.
		if (!preparation) return { ok: false, note: "pi's preparation found nothing to compact" };
		compactionInFlight = true;
		let answer: any;
		try {
			answer = await handleCompaction(
				{ reason: "threshold", windowSize: ctx?.model?.contextWindow ?? 0, customInstructions: undefined, signal: undefined },
				ctx,
				preparation,
			);
		} catch (err: any) {
			return { ok: false, note: `the summary could not be written: ${String(err?.message ?? err).slice(0, 150)}` };
		} finally {
			compactionInFlight = false;
		}
		const c: any = answer?.compaction;
		if (!c?.summary || !c.firstKeptEntryId) {
			return {
				ok: false,
				note: answer?.cancel ? "the compaction was skipped or held, nothing was written" : "the summary came back without a cut point",
			};
		}
		let entryId: string | undefined;
		try {
			entryId = sm.appendCompaction(c.summary, c.firstKeptEntryId, c.tokensBefore ?? 0, c.details ?? {}, true, c.usage);
		} catch (err: any) {
			return { ok: false, note: `the summary could not be appended: ${String(err?.message ?? err).slice(0, 150)}` };
		}
		// The two things pi's `session_compact` event did for us, done here: the watcher must know a
		// compaction just landed (it refuses to start another one while the size is unmeasured), and the
		// log must carry the COMPACTED row with the size that landed.
		lastCompactionDoneAtMs = Date.now();
		logLine(cfg, {
			event: "compaction_landed",
			reason: "threshold",
			fromExtension: true,
			// One figure, not two: when the engine told us what the summary cost, that is the size on the row.
			// The old character estimate of the stored text sat beside it as a second, different number
			// (13,090 against an engine count of 12,355) and both looked authoritative.
			summaryTokens: landedSummarySize(lastLandingInfo, String(entry?.summary ?? c?.summary ?? "")),
			handoff: false,
			noAbort: true,
			source: why,
			entry: String(entryId ?? "").slice(0, 12),
			...lastLandingInfo,
		});
// The closing block on screen, the same one the interrupting route prints. Both routes announce the
		// opening row; until now only that one route announced the end, so a compaction that landed
		// without an interrupt left the operator reading a start row and then silence (2026-10-23).
		announceCompaction(
			ctx,
			compactionNoticeLines({ ...lastLandingInfo, summary: c.summary, firstKeptEntryId: c.firstKeptEntryId }, {
				before: realContextTokens(ctx),
				window: ctx?.model?.contextWindow ?? 0,
				keepRecent: extKeepRecent(ctx),
				failures: lastRotationFailures,
				trigger: lastLandingInfo?.trigger ?? "auto",
			}),);
		return { ok: true };
	}
	// The top-level watcher reaches the nested routine through this single assignment.
	landNoAbortHook = landCompactionNoAbort;


		async function handleCompaction(event: any, ctx: any, preparationOverride?: any) {
		const cfg = loadConfig();
		// Who asked for this compaction: the watcher (auto) or the user (manual). A manual run always
		// clears the pause, successful or not - it is the user proving the trigger works again.
		// pi names its own reasons: the pre-request threshold check and the overflow recovery are both
		// automatic; only /compact, a button or the menu arrives as "manual".
		const reason = String(event?.reason ?? "");
		const trigger = reason.includes("manual")
			? "manual"
			: reason.includes("threshold") || reason.includes("overflow")
				? "auto"
				// A reason pi does not name today is still not a manual run: only /compact, a button or the
				// settings menu arrive as "manual". The old fallback here was the label the LAST trigger had
				// left behind, so an unnamed reason inherited whatever that happened to be - and the label is
				// what decides whether a refusal gets the one allowed retry, a park, or just a log line.
			: "auto";
		if (trigger === "auto") {
			// An automatic trigger is answered HERE while the breaker is cooling down or parked, and it is
			// cancelled: falling through at any point below would let pi write a summary with the session
			// model, which is the double compaction this setup must never do.
			const cooling = autoFailure.retryAtMs > 0 && Date.now() < autoFailure.retryAtMs;
			if (parkedThisSession(cfg, ctx) || cooling) {
				logLine(cfg, {
					event: "auto_compaction_skipped",
					note: parkedThisSession(cfg, ctx)
						? "parked after repeated failures - pi's stock summarizer was NOT allowed to run"
						: `waiting for the one allowed retry - pi's stock summarizer was NOT allowed to run`,
					secondsLeft: autoFailure.retryAtMs > 0 ? Math.max(0, Math.round((autoFailure.retryAtMs - Date.now()) / 1000)) : undefined,
					window: (event as any)?.windowSize,
					actual: (event as any)?.actualTokens,
				});
				return { cancel: true };
			}
		}
		// Remember WHO asked, for the failure handlers below: the nothing-to-summarise park, the one
		// allowed retry and the two-refusals-stop-it park all key off this label. It used to be written
		// as "manual" here whatever had actually asked, which is how an automatic compaction that pi
		// refused got logged as a manual one - reported on screen, never retried, never parked, and the
		// park the operator had put in place was cleared by an automatic ask that looked manual.
		lastCompactionTrigger = trigger;
		if (trigger === "manual" && parkedThisSession(cfg, ctx)) {
			update((c) => void (c.autoCompactPaused = false));
			logLine(cfg, { event: "auto_compaction_resumed", session: String(ctx?.sessionManager?.getSessionId?.() ?? "session").slice(-8), note: "a manual compaction ran in this session - the proactive trigger is armed again, for this session" });
		}
		const refs = Object.keys(cfg.models);
		if (!cfg.enabled || refs.length === 0) {
		// pi's compaction flag is read for information only and never written by this extension.
		// The automatic trigger here is this extension's own: it measures the live context and starts
		// the compaction itself, so with pi's flag off automatic compaction still works - and pi never
		// writes a summary of its own. The coupling that armed pi's flag is deleted for good.
		return trigger === "auto" ? { cancel: true } : undefined;
	} // the compaction switch; elision has its own

		// The arithmetic of this compaction, recorded so a wrong choice can be traced afterwards.
		// Either pi handed us the preparation (its own threshold or overflow path) or this extension built
		// one to land a summary without stopping the turn (#43); everything below treats the two alike.
		const prep = preparationOverride ?? event.preparation;
		const history = historyTokensOnly(prep);
		const overhead = fixedRequestTokens(ctx);
		const fill = realContextTokens(ctx);
		// THE SAME OPENING ROW ON SCREEN FOR BOTH ROUTES (2026-10-23): a compaction you started and one the
		// trigger started tell you the same thing at the same moment - who asked, how full the session is,
		// how much of the tail survives. The result block at the end already read the same.
		{
			const win0 = ctx?.model?.contextWindow ?? 0;
			const at0 = realContextTokens(ctx);
			announceCompaction(
				ctx,
				[
					`COMPACTION START · ${lastCompactionTrigger === "manual" ? "manual" : "automatic"} · context ${fmt(at0)} of ${fmt(win0)} tok` +
						(win0 > 0 ? ` (${Math.round((at0 / win0) * 100)}% full)` : "") +
						` · newest ${fmt(extKeepRecent(ctx))} tok kept word for word`,
				],
				"info",
			);
		}
		ctx.ui?.setStatus?.("compact-plus", `compacting via ${rotationText(cfg, ctx)} ...`);
		let outcome;
		try {
			outcome = await summarizeWithRotation(prep, event.customInstructions, cfg, ctx, event.signal);
		} finally {
			ctx.ui?.setStatus?.("compact-plus", undefined);
		}

		if (!outcome.ok) {
			if (outcome.error === "cancelled") {
				// pi's compaction flag is read for information only and never written by this extension.
				// The automatic trigger here is this extension's own: it measures the live context and starts
				// the compaction itself, so with pi's flag off automatic compaction still works - and pi never
				// writes a summary of its own. The coupling that armed pi's flag is deleted for good.
				// A STOP MEANS STOP, BUT ONLY IN THE SESSION AND THE MODE IT HAPPENED IN. Stopping a compaction you
				// started yourself (/compact, the menu) says nothing about the automatic trigger - you were testing, or
				// you changed your mind about that one run. Stopping one the extension started on its own is a stop to
				// the automation, and parks it until a manual compaction re-arms it (2026-10-22: the manual case
				// parked it as well, and the operator watched an automatic run arrive after saying "not now").
				if (trigger === "manual") {
					logLine(cfg, { event: "compaction_cancelled", trigger, note: "a manual compaction was stopped; the automatic trigger is untouched" });
					ctx.ui?.notify?.("COMPACTION CANCELLED - the manual compaction was stopped.", "info");
					return { cancel: true };
				}
				pauseAutoCompaction(ctx, "you stopped the automatic compaction", outcome.error);
				return { cancel: true };
			}
			autoFailure.fails += 1;
			if (autoFailure.fails === 1) {
				// First failure: keep the trigger armed but wait 3 minutes for the single retry.
				autoFailure.retryAtMs = Date.now() + 180_000;
				logLine(cfg, { event: "auto_compaction_failed", error: outcome.error, trigger, failure: autoFailure.fails, retryInMs: 180_000, note: "one retry allowed, 3 minutes from now" });
			announceCompaction(
				ctx,
				compactionFailureNoticeLines(
					outcome.error,
					"first failure: the trigger stays armed, one retry in 3 minutes, pi's own summarizer is not used",
				),
				"warning",
			);
				return { cancel: true };
			}
			pauseAutoCompaction(ctx, "compaction failed twice", outcome.error);
			logLine(cfg, { event: "give_up", error: outcome.error, trigger, note: "the automatic trigger is parked for this session until a manual compaction re-arms it; pi's stock summarizer was NOT allowed to run" });
			announceCompaction(
				ctx,
				compactionFailureNoticeLines(
					outcome.error,
					"every candidate failed: the automatic trigger is now PARKED for this session and will not try again by itself. pi's own summarizer is deliberately never substituted. Run /compact - it works right away and re-arms the automatic trigger.",
				),
				"error",
			);
			// And that is the whole rule (2026-10-17, after a session lost two hours of work this way): failing
			// is not the same as handing over. The extension rests, the trigger stays ours, and pi's summary -
			// written by the session model, in a shape this session's settings exist to avoid - is never the
			// fallback for a failure. If pi's compaction is wanted, that is a switch the user flips.
			return { cancel: true };
		}
		autoFailure.fails = 0;
		autoFailure.retryAtMs = 0;
		// A summary that landed is also the end of any rest period (2026-10-17): the 30-minute pause exists to
		// stop a broken model list from knocking every round, not to outlast a good compaction that came after it.
		autoFailure.pausedAtMs = 0;

		const r = outcome.result;
		// The chained-compaction counter rides in the entry's details (read by the next compaction).
		try {
			r.compaction.details = { ...(r.compaction.details ?? {}), lastTurnNumber: maxTurnInSummary(r.compaction.summary) };
		} catch {
			/* counter is best-effort */
		}
		r.compaction.summary = appendTranscriptPointer(cfg, ctx, r.compaction.summary);
		lastLandingInfo = {
			ref: r.ref, trigger, seconds: r.seconds, measured: r.measured, min: r.min, target: (r as any).target,
			max: r.max, words: r.words, via: (r as any).via, before: realContextTokens(ctx),
			window: (ctx as any)?.model?.contextWindow ?? 0, reason: event.reason,
			// Every candidate that was tried and lost, on the row that says the compaction landed.
			failures: lastRotationFailures, quoted: lastSummaryQuote.quoted, given: lastSummaryQuote.given,
		};
		announceCompaction(
			ctx,
			compactionNoticeLines(r, {
				before: realContextTokens(ctx),
				window: ctx?.model?.contextWindow ?? 0,
				keepRecent: extKeepRecent(ctx),
				failures: lastRotationFailures,
				trigger,
			}),
		);
		return { compaction: r.compaction };
	}

	// -------------------------------------------------------------- command line verbs
	pi.registerCommand("compact-plus", {
		description: "Choose which models write compaction summaries (no argument opens the menu)",
		handler: async (args: string, ctx: Ctx) => {
			const parts = String(args ?? "").trim().split(/\s+/).filter(Boolean);
			const verb = (parts[0] ?? "").toLowerCase();
			const cfg = loadConfig();
			const say = (t: string) => ctx.ui?.notify?.(t, "info");

			if (!verb) {
				await runBoard(ctx);
				return;
			}
			switch (verb) {
				case "status":
					return say(describe(cfg, ctx));
				case "on":
					// One compactor at a time: while another manager is installed, enabling this is refused (2026-10-07).
					// pi's own compaction is not part of that bargain and is never touched from here.
					const blocked = enableBlockedByOtherCompactor();
					if (blocked) return say(`NOT ENABLED: ${blocked} is present. Only one compaction extension can own the context at a time.`);
					update((c) => void (c.enabled = true));
					return say(
						"COMPACTION IS ON. This extension writes every summary and answers /compact. Who times it depends on pi's own switch: with pi's compaction enabled, pi fires first and this extension holds the compaction until its own start point; with pi's off, this extension's own check is the trigger. " +
						"pi's own compaction is untouched by this command: nothing here writes pi's settings, so while compaction.enabled is true there, pi can still compact by itself with pi's own summariser. Read-only state of that flag: /compact-plus pi.",
					);
				case "off":
					update((c) => void (c.enabled = false));
					return say("THIS EXTENSION'S COMPACTION IS OFF. pi's own compaction stays as it is - if it was off too, NOTHING COMPACTS ANY MORE until one of the two is on.");
				case "pi":
					return say(
						`pi's own compaction is ${readPiCompactionEnabled() ? "ENABLED" : "DISABLED"} in ${join(AGENT_DIR, "settings.json")} (compaction.enabled). THIS EXTENSION NEVER WRITES THAT FILE, and no command here switches pi's compaction on or off: that flag is pi's, for pi's own threshold and overflow checks. This extension's own automatic trigger and /compact work with it either way. Edit the number in that file yourself, then /reload.` +
						(loadConfig().enabled
							? " This extension: compaction ON, automatic trigger " + (loadConfig().autoCompact ? "armed." : "PARKED or OFF, and a manual /compact re-arms it when it is parked.")
							: " This extension: compaction OFF, so nothing compacts on its own until it is switched back on."),
					);
				case "models": {
					if (!parts[1]) {
						const reserve = extReserve(ctx);
						const refs = Object.keys(loadConfig().models);
						if (!refs.length) return say("no models in the list — pi uses the session model");
						return say(
							refs
								.map((ref, i) => {
									const model = lookupRef(ref, ctx);
									const o = optionsFor(loadConfig(), ref, model);
									return `${i + 1}. ${ref} — summary aim ${o.summaryPercent}% of the ctx window, limits min ${fmt(o.summaryMinTokens)} / max ${fmt(o.summaryMaxTokens)} tok, ${timeoutLabel(o.timeoutMs)}, ${o.thinking}`;
								})
								.join("\n"),
						);
					}
					if (parts[1] === "remove" && parts[2]) {
						rotateRemove(parts.slice(2).join("/"));
						return say(`Removed. List: ${rotationText(loadConfig(), ctx)}`);
					}
					const ref = parts.slice(1).join("/");
					rotateAdd(ref);
					return say(`Added at the end with default options: ${rotationText(loadConfig(), ctx)}`);
				}
				case "move": {
					if (!parts[1]) return say("Use: move <provider/model> <up|down|number>");
					const ref = parts[1];
					if (/^(up|down)$/i.test(parts[2] ?? "")) rotateMove(ref, parts[2].toLowerCase() === "up" ? -1 : 1);
					else if (Number.isInteger(Number(parts[2]))) rotateTo(ref, clamp(Number(parts[2]) - 1, 0, MAX_SLOTS - 1));
					else return say("Use: move <provider/model> <up|down|n>");
					return say(`Order now: ${rotationText(loadConfig(), ctx)}`);
				}
				case "options": {
					const ref = refArg(parts);
					if (!ref) return say("Use: options <provider/model> [thinking <lvl>] [timeout <min|0>] [aim <1-99%>] [min <tokens>] [max <tokens>] [inputstubs on|off] [argcap <0-1000|full>] [stubs <verbatim|detailed|summary|brief>] [user <..>] [replies <..>] [thought <..>] [chain <fold|attach|skip>] [tag on|off]");
					if (parts.length === 2) {
						const model = lookupRef(ref, ctx);
						const o = optionsFor(loadConfig(), ref, model);
						return say(`${ref}: thinking ${o.thinking}, timeout ${timeoutLabel(o.timeoutMs)}, summary aim ${o.summaryPercent}% of the ctx window, limits min ${fmt(o.summaryMinTokens)} / max ${fmt(o.summaryMaxTokens)} tok, input ${o.inputStubs ? "stubs" : "raw"} (arg cap ${charCapLabel(o.argCap)}), keeps: stubs ${o.preserveStubs} / user ${o.preserveUser} / replies ${o.preserveReplies} / thought ${o.preserveThinking}, chain ${o.chainMode}, tag ${o.noThinkMarker || "(empty)"}`);
					}
					const patch: Partial<ModelOptions> = {};
					for (let i = 2; i < parts.length; i += 2) {
						const key = parts[i]?.toLowerCase();
						const value = parts[i + 1];
						if (!key || value === undefined) break;
						if (key === "thinking" && ALL_LEVELS.includes(value as Level)) patch.thinking = value as Level;
						else if (key === "timeout") patch.timeoutMs = clamp(Math.round(Number(value) * 60_000) || 0, 0, 7_200_000);
						else if (key === "aim" || key === "pct") patch.summaryPercent = clamp(Math.round(Number(value)), 1, 99);
						else if (key === "min") patch.summaryMinTokens = clamp(Math.round(Number(value)), 0, 1_000_000);
						else if (key === "max") patch.summaryMaxTokens = clamp(Math.round(Number(value)), 1, 1_000_000);
						
						else if (key === "inputstubs") patch.inputStubs = value === "on";
						else if (key === "argcap") patch.argCap = parseCharCap(value, 500);
						else if (key === "stubs" && isPreserve(value)) patch.preserveStubs = value;
						else if (key === "user" && isPreserve(value)) patch.preserveUser = value;
						else if (key === "replies" && isPreserve(value)) patch.preserveReplies = value;
						else if (key === "thinking2" && isPreserve(value)) patch.preserveThinking = value;
						else if (key === "chain" && isChainMode(value)) patch.chainMode = value;
						else if (key === "tag") patch.noThinkMarker = value === "on" ? DEFAULT_NO_THINK_TAG : "";
					}
					if (!Object.keys(patch).length) return say("Nothing to change. See: /compact-plus options <provider/model>");
					// Adding options for a model not yet in the list selects it as well.
					if (!loadConfig().models[ref]) rotateAdd(ref);
					setModelOptions(ref, patch);
					const o2 = optionsFor(loadConfig(), ref, lookupRef(ref, ctx));
					return say(`${ref}: thinking ${o2.thinking}, timeout ${timeoutLabel(o2.timeoutMs)}, summary aim ${o2.summaryPercent}% of the ctx window, limits min ${fmt(o2.summaryMinTokens)} / max ${fmt(o2.summaryMaxTokens)} tok, input ${o2.inputStubs ? "stubs" : "raw"} (arg cap ${charCapLabel(o2.argCap)}), keeps: stubs ${o2.preserveStubs} / user ${o2.preserveUser} / replies ${o2.preserveReplies} / thought ${o2.preserveThinking}, chain ${o2.chainMode}, tag ${o2.noThinkMarker || "(empty)"}`);
				}
				case "elision": {
					const sub = (parts[1] ?? "").toLowerCase();
					const setEl = (patch: Partial<ElisionConfig>) => update((c) => void (c.elision = { ...c.elision, ...patch }));
					if (!sub) {
						const e = loadConfig().elision;
						return say(`Elision: ${e.enabled ? "ON" : "OFF"} · start ${e.softPercent}% of window · protect ${fmt(e.protectRecentTokens)} tok · stub tail ${charCapLabel(e.stubTailChars)} chars · min batch savings ${fmt(e.minSavingsTokens)} tok / ${e.minResultsToStub} results · stop gap ${fmt(e.stopGapTokens)} tok.  (elision on|off|start <1-99>|tail <0..1000|full>|protect <tokens>|savings <tokens>|stop <tokens>|results <1-50>)`);
					}
					if (sub === "on") { setEl({ enabled: true }); resetElisionBreaker(); }
					else if (sub === "off") { setEl({ enabled: false }); resetElisionBreaker(); }
					else if (sub === "start" && Number.isFinite(Number(parts[2]))) setEl({ softPercent: clamp(Math.round(Number(parts[2])), 1, 99) });
					else if (sub === "tail") setEl({ stubTailChars: parseCharCap(parts[2] ?? "", ELISION_DEFAULTS.stubTailChars) });
					else if (sub === "protect" && Number.isFinite(Number(parts[2]))) setEl({ protectRecentTokens: clamp(Math.round(Number(parts[2])), 0, 1_000_000) });
					else if (sub === "savings" && Number.isFinite(Number(parts[2]))) setEl({ minSavingsTokens: clamp(Math.round(Number(parts[2])), 0, 100_000) });
					else if (sub === "stop" && Number.isFinite(Number(parts[2]))) setEl({ stopGapTokens: clamp(Math.round(Number(parts[2])), 0, 100_000) });
					else if (sub === "results" && Number.isFinite(Number(parts[2]))) setEl({ minResultsToStub: clamp(Math.round(Number(parts[2])), 1, 50) });
					else return say("Use: elision [on|off] [start <1-99>] [tail <0..1000|full>] [protect <tokens>] [savings <tokens>] [stop <tokens>] [results <1-50>]");
					const e = loadConfig().elision;
					return say(`Elision: ${e.enabled ? "ON" : "OFF"} · start ${e.softPercent}% · protect ${fmt(e.protectRecentTokens)} tok · stub tail ${charCapLabel(e.stubTailChars)} · min savings ${fmt(e.minSavingsTokens)} tok · min batch ${e.minResultsToStub} results · stop gap ${fmt(e.stopGapTokens)} tok`);
				}
				case "retries": {
					const n = Number(parts[1]);
					if (Number.isFinite(n)) update((c) => void (c.retries = clamp(Math.round(n), 0, 5)));
					const s = Number(parts[2]);
					if (Number.isFinite(s)) update((c) => void (c.retryDelaySeconds = clamp(Math.round(s), 0, 300)));
					return say(`Retries: ${loadConfig().retries} per model, ${loadConfig().retryDelaySeconds}s apart.`);
				}
				case "marker":
					if (parts[1] === "clear") {
						update((c) => void (c.noThinkMarker = ""));
						return say("No-think tag: (empty)");
					}
					if (!parts[1]) return say(`No-think tag: ${loadConfig().noThinkMarker || "(empty)"}  (set with: marker <text> | marker clear)`);
					update((c) => void (c.noThinkMarker = parts.slice(1).join(" ")));
					return say(`No-think tag saved: ${loadConfig().noThinkMarker}`);
				case "sampling": {
					const cfgS = loadConfig();
					const refS = refArg(parts) ?? Object.keys(cfgS.models)[0];
					if (!refS) return say("No models configured - nothing to tune.");
					if (!cfgS.models[refS]) return say(`Model not in the list: ${refS}`);
					await editSamplingFlags(ctx, refS);
					return say("Sampling flags updated.");
				}
				case "instruction": {
					if (parts[1] === "reset") {
						update((c) => void (c.additionalInstruction = null));
						return say("Cleared - the template follows the user default (or the extension default when none is saved).");
					}
					if (parts[1] === "clear") {
						update((c) => void (c.additionalInstruction = null));
						return say("Cleared — the template follows the user default (or the extension default when none is saved).");
					}
					if (parts.length < 2) {
						const own = loadConfig().additionalInstruction;
						return say(own ? `Current summary template source:\n${own}` : `The template follows the user default (or the extension default when none is saved):\n${DEFAULT_ADDITIONAL_INSTRUCTION}`);
					}
					update((c) => void (c.additionalInstruction = parts.slice(1).join(" ")));
					return say("Summary template saved (one line here; the menu editor takes several). Keep the {PLACEHOLDER} tokens - they substitute per model.");
				}
				case "preview": {
					// Dry run: build exactly what the direct request would send, for the first (or the
					// named) candidate, and show it - nothing is called, nothing is sent.
					const cfg2 = loadConfig();
					const ref = refArg(parts) ?? Object.keys(cfg2.models)[0];
					if (!ref) return say("No models configured - nothing to preview.");
					const model = lookupRef(ref, ctx);
					if (!model) return say(`Model not found: ${ref}`);
					const profile = profileFor(cfg2, model, ref);
					try {
						const projection: any = ctx?.sessionManager?.buildSessionProjection?.();
						const msgs: any[] = (projection?.messages ?? []).filter((m: any) => m?.role !== "system");
						if (msgs.length < 4) return say("Not enough conversation yet for a preview.");
						const kept = msgs.slice(-2);
						const region = msgs.slice(0, -2);
						const prevTurn = readPrevTurnInfo(ctx, {});
						const main = serializeRegion(region, { startTurn: prevTurn.lastTurn, inputStubs: profile.inputStubs, argCap: profile.argCap });
						const keptText = kept.map((m: any) => `[${String(m.role)} - kept]: ${contentTextOf(m.content).slice(0, 200)}`).join("\n");
						const inputTokens = Math.ceil((main.text.length * 1.3) / 4) + SUMMARISER_FRAMING_TOKENS;
						const sizes = sizesFor(profile, { inputTokens, historyTokens: inputTokens, headroom: 0, overflow: false, windowTokens: ctx?.model?.contextWindow ?? 0 }, 0);
						const instructions = buildInstructions({ sizes, inputTokens, segmentTokens: inputTokens - SUMMARISER_FRAMING_TOKENS, kind: isLocalModel(model) ? "local" : "online", options: profile, additionalInstruction: cfg2.additionalInstruction, customInstructions: undefined });
						const text = [
							"PREVIEW - what the direct summarization request would contain (approximation: the real region is chosen by pi's cut logic at compaction time; the last two messages stand in for the kept tail).",
							"",
							labeledPreviousSummary(undefined, prevTurn.lastTurn, prevTurn.lastTurn + 1),
							`The conversation segment below covers Turns ${prevTurn.lastTurn + 1}-${main.lastTurn} (already numbered - use those numbers).`,
							`<conversation>`,
							main.text,
							`</conversation>`,
							`[kept tail, for reference - NOT part of the summarized region:]`,
							keptText,
							"",
							instructions,
							"",
							`[system prompt that would ride above all this:]`,
							COMPACTOR_SYSTEM_PROMPT,
						].filter(Boolean).join("\n\n");
						await uiPreview(ctx,`compact-plus preview for ${ref} (nothing was sent)`, text);
						return say(`Preview opened in the editor (${fmt(text.length)} chars, region ~${fmt(inputTokens)} tok).`);
					} catch (err: any) {
						return say(`Preview failed: ${String(err?.message ?? err).slice(0, 160)}`);
					}
				}
				case "log":
					return say(tailLogPretty(16));
				case "cap":
				case "extra":
				case "mandate":
				case "size":
				case "timeout":
				case "thinking":
				case "preserve":
				case "second":
				case "template":
				case "prompt":
				case "oprompt":
					return say("The old words are gone. Use: /compact-plus options <provider/model> [...] — or open the menu with no argument. New: elision [on|off] [start <50-95>] [tail <0..1000|full>] [protect <tokens>] [savings <tokens>].");
				default:
					return say(
						"Use /compact-plus with no argument for the menu, or:\nstatus | on | off | models [add <ref> | remove <ref>] | move <ref> <up|down|n>\noptions <ref> [thinking <lvl>] [timeout <min|0>] [aim <1-99%>] [min <tokens>] [max <tokens>] [inputstubs on|off] [argcap <0-1000|full>] [stubs|user|replies|thought <verbatim|detailed|summary|brief>] [chain <fold|attach|skip>] [tag on|off]\nelision [on|off] [start <50-95>] [tail <0..1000|full>] [protect <tokens>] [savings <tokens>]\nretries <0-5> [sec] | marker <text|clear> | instruction [text | reset] | log",
					);
			}
		},
		getArgumentCompletions: (prefix: string) =>
			["status", "on", "off", "models", "move", "options", "elision", "retries", "marker", "instruction", "sampling", "preview", "log"]
				.filter((v) => v.startsWith(prefix))
				.map((v) => ({ value: v, label: v })),
	});
}
