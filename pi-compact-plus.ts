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
 *      After the call: draft stripping, size gate, SHAPE gate (sections + turn-label sanity),
 *      pointer line; the turn counter stored in details.
 *      FALLBACKS: direct request fails → pi's compact() with the extension instructions (the old
 *      path) → fails → next model in the rotation → all failed → pi's own compaction.
 *   3. PER-MODEL OPTIONS: thinking, timeout, aim/min/max shares of pi's reserveTokens, draft block
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
import { SettingsList, truncateToWidth } from "@earendil-works/pi-tui";
import type { Component, Focusable, KeybindingsManager, TUI } from "@earendil-works/pi-tui";

/** The context object pi hands to handlers and commands. Typed loosely on purpose: this file is
 *  checked on its own, without the package type declarations on the module path. */
type Ctx = any;
import { appendFileSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

const AGENT_DIR = join(homedir(), ".pi", "agent");
/** Test isolation: point COMPACTION_PLUS_CONFIG at a scratch config and the extension reads that
 *  instead of the live one - unit tests can then never touch the user's real settings. */
const CONFIG_PATH = process.env.COMPACTION_PLUS_CONFIG?.trim()
	? resolve(process.env.COMPACTION_PLUS_CONFIG.trim())
	: join(AGENT_DIR, "pi-compact-plus.json");
const LOG_PATH = join(AGENT_DIR, "pi-compact-plus.log");
const LOG_MAX_BYTES = 1_000_000;

/** Pseudo model reference meaning "the model pi is using right now". */
const SESSION_REF = "session";

type Level = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
type Kind = "local" | "online";
type Draft = "off" | "mini" | "full";
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
type ScalingConfig = {
	/** keepRecent = this percent of the window, clamped to [keepRecentMin, keepRecentMax]. */
	keepRecentPercent: number;
	keepRecentMin: number;
	keepRecentMax: number;
	/** reserveTokens = this percent of the window, clamped to [reserveMin, reserveMax]. Drives the
	 *  compaction trigger line (window - reserve) and every summary size share (aim 0.8 x reserve). */
	/** The auto-compaction trigger point as a percent of the window (2026-10-02: the old
	 *  reservePercent inverted - 18 kept free became 82 used). The reserve = window - point,
	 *  clamped by reserveMin/reserveMax. */
	startPercent: number;
	reserveMin: number;
	reserveMax: number;
};

const SCALING_DEFAULTS: ScalingConfig = {
	keepRecentPercent: 20,
	keepRecentMin: 16_384,
	keepRecentMax: 100_000,
	startPercent: 82,
	reserveMin: 12_288,
	reserveMax: 80_000,
};

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
	/** Abort the request after this long. 0 = unlimited. */
	timeoutMs: number;
	/** THE SUMMARY AIM: this percent of the model's ctx window (2026-10-03: everything - aim, min,
	 *  max - is a window percent; the numbers the settings show are the numbers enforced). */
	summaryPercent: number;
	/** THE LIMITS (2026-10-03): percent of the model's WINDOW, used DIRECTLY as the acceptance
	 *  bounds - a summary below min or above max is thrown away. The aim is clamped inside them,
	 *  so at a 95k window 23% means 21,850 tokens, and the same settings scale on a 1M model. */
	summaryMinPercent: number;
	summaryMaxPercent: number;
	/** Hidden analysis written before the summary in the same call, stripped before storing:
	 *  "full" = a chronological per-turn analysis; "mini" = at most ten lines; "off" = none. */
	draft: Draft;
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
	/** The elision stage (extension-wide: fires in the live session for every model alike). */
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
};

/** The default no-think tag written into a local model's request when its thinking is off. */
const DEFAULT_NO_THINK_TAG = "<|think_off|>";


/** Timeout periods in the exact order they toggle. 0 = unlimited. */
const TIMEOUT_STEPS = [120_000, 180_000, 300_000, 420_000, 600_000, 900_000, 1_200_000, 1_800_000, 2_700_000, 3_600_000, 0];

const THINKING_CYCLE: Level[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
const DRAFT_CYCLE: Draft[] = ["off", "mini", "full"];
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
	};
}

const DEFAULT_CONFIG: Config = {
	enabled: false,
	models: {},
	retries: 2,
	retryDelaySeconds: 5,
	log: true,
	transcriptPointer: true,
	directRequest: true,
	shapeGate: true,
	chatMaxTokensCap: 65_536,
	scaling: { ...SCALING_DEFAULTS },
	autoCompact: true,
	elision: ELISION_DEFAULTS,
	noThinkMarker: DEFAULT_NO_THINK_TAG,
	additionalInstruction: null,
	categorySentences: null,
	categoryUserDefaults: null,
	templateUserDefaults: null,
};

/** pi's default reserveTokens (docs/compaction.md), used only to turn old absolute token caps into
 *  shares and to show approximate token figures in the menu. The real value is read from
 *  preparation.settings at compaction time. */
export const DEFAULT_RESERVE_TOKENS = 16384;

/** pi's OWN settings.json compaction.reserveTokens - read for display, written by the board's
 *  stock-fallback row. This extension never USES the value; it only matters to pi's built-in
 *  compaction (the fallback that runs when the extension is disabled or every candidate failed). */
function readPiReserveTokens(): number {
	try {
		const raw = JSON.parse(readFileSync(join(AGENT_DIR, "settings.json"), "utf8"));
		const v = raw?.compaction?.reserveTokens;
		return typeof v === "number" && Number.isFinite(v) && v >= 0 ? Math.round(v) : DEFAULT_RESERVE_TOKENS;
	} catch {
		return DEFAULT_RESERVE_TOKENS;
	}
}

function writePiReserveTokens(v: number): void {
	try {
		const file = join(AGENT_DIR, "settings.json");
		const raw = JSON.parse(readFileSync(file, "utf8"));
		raw.compaction = { ...(raw.compaction ?? {}), reserveTokens: Math.max(0, Math.round(v)) };
		writeFileSync(file, JSON.stringify(raw, null, 2) + "\n", "utf8");
	} catch {
		/* leave pi's settings untouched on any error */
	}
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
function lengthBlock(sizes: Sizes, inputTokens: number): string {
	const window =
		sizes.min > 0
			? `your summary must measure between ${fmt(sizes.min)} and ${fmt(sizes.max)} TOKENS (aim for about ${fmt(sizes.target)} tokens, roughly ${fmt(wordsFor(sizes.target))} words). A summary much shorter than ${fmt(sizes.min)} tokens is too thin: it will be REJECTED as a failure.`
			: `aim for about ${fmt(sizes.target)} tokens (roughly ${fmt(wordsFor(sizes.target))} words); there is no hard minimum, but a very thin summary leaves the next request blind, so write freely and do not stop early. Your text is cut off at ${fmt(sizes.max)} tokens.`;
	return [
		`This is a summary request. The conversation segment below is about ${fmt(inputTokens)} tokens and is the memory being replaced - the moment you stop writing, the original messages are deleted and the next model continues the work from your summary alone.`,
		`IMPORTANT - length: ${window} Do not stop early and do not save tokens - stopping before the segment is fully covered is the one failure to avoid. If you approach the ${fmt(sizes.max)}-token ceiling, tighten the least important details and land on a finished list, never mid-item.`,
		`Override: the line "Keep each section concise" in the instructions above does not apply to this job. It is written for a summary that sits beside the full transcript; here the transcript is deleted and your text is the only memory the next request has. Completeness beats brevity. Long bullet lists are good. Repeating an exact value twice is good. Splitting one vague sentence into five specific ones is the point. Vagueness and omission are the only failures.`,
	].join("\n");
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
		stubs: `TOOL CALLS: copy EVERY tool-call stub from the input VERBATIM, EXACTLY as written, word for word, one per line, in input order, into that turn's "tools:" lines. Do NOT modify text, do NOT skip any call. Stubs are already compact - copying them is the requirement; changing them in any way is a failure.`,
		user: `USER PROMPTS: quote every user prompt VERBATIM, in FULL, word for word, inside quotation marks, in that turn's "user asked:" line. Do NOT change the text in any way, do NOT omit any sentence.`,
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
 * stripped before storing (stripDraft). Anthropic's Claude Code compaction prompt runs the same
 * trick in one call (an analysis scratchpad before the summary, removed afterwards); it organizes
 * the summary and costs no extra request. It costs OUTPUT tokens, so it is a per-model switch:
 * online models default to full, local ones to off (a full draft roughly doubles a 10-20
 * tokens/second summary).
 */
const DRAFT_FULL = `First write a scratchpad between <draft> and </draft>: go through the conversation chronologically, turn by turn - for each user turn note what was asked, what you concluded and did, every tool call with its key arguments and what the result showed, every error and how it was fixed, and every point where the user corrected, narrowed or reversed an instruction. Then check that every section of the OUTPUT FORMAT below is fully coverable from what you noted, and note anything that is not. After the closing draft tag, write the summary itself. The scratchpad is working space: it is discarded before the summary is stored and it does not count as part of the summary.`;

const DRAFT_MINI = `First write a scratchpad between <draft> and </draft>: at most ten lines, one line per user turn in order - what was asked, what changed, what it led to. Then check that every section of the OUTPUT FORMAT below is fully coverable. After the closing draft tag, write the summary itself; the scratchpad is discarded and does not count as part of the summary.`;

function draftBlock(mode: Draft): string {
	if (mode === "full") return DRAFT_FULL;
	if (mode === "mini") return DRAFT_MINI;
	return "";
}

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
	`OUTPUT FORMAT - the format block above is replaced by this one. Use only the sections below, in this order, no preamble, no closing remarks.`,
	``,
	`## CURRENT STATE (the newest truth - where anything below conflicts with this section, THIS section wins)`,
	`- Goal now: what is being built or fixed, why, plus every constraint and preference.`,
	`- Where things stand: done / in progress / blocked, with file paths.`,
	`- Next step: the exact next action, with its parameters.`,
	``,
	`## TURN LEDGER (chronological; newer turns get more detail, oldest may tighten to one line)`,
	`Every user turn gets one entry, in order, numbered with the word Turn and its number - exactly "Turn 12", never "12." or "1." alone.`,
	`Turn 12 - user asked: "<the user's words, per the USER PROMPTS rule in Rules below>"`,
	`    did: what the assistant concluded and did.`,
	`    tools: <per the TOOL CALLS rule in Rules below; when that rule says VERBATIM, copy each stub whole onto its own indented line>`,
	`    errors & fixes: what broke and how it was fixed.`,
	`    decided: X over Y because ...`,
	`(older turns tighten but are never dropped; leave a sub-line out when it has nothing to say)`,
	``,
	`## FILES & DATA`,
	`- every file read, created or edited: full path, what it is, why it matters.`,
	``,
	`## KEY DECISIONS`,
	`- each choice and why it was picked over the alternatives, cross-referenced like "(Turn 12)".`,
	``,
	`## NEXT STEPS`,
	`1. the exact next action, with the parameters needed to take it.`,
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
	`- One concrete fact per bullet; never drop a file path, a number, or an error message to save space - when a heading looks thin, list the concrete items you saw instead of summarising them away.`,
	`- If a later turn changes an earlier one, KEEP the earlier line and tag it: "(superseded by Turn N)" for a full reversal, "(amended by Turn N)" for a partial change. Never delete history.`,
	`- Several user prompts can arrive while the assistant is still working - each is its own Turn, even a short one. In a chain of refinements, the newest wording is the current instruction; earlier ones stay, tagged against the turn that changed them.`,
	`{STUB EXAMPLE RULE}`,
	`- No wall-clock times exist in the input; order is expressed ONLY by the turn numbers.`,
	`{PREVIOUS SUMMARY RULE}`,
	``,
	`Overrides - these lines in the instructions above do not apply here:`,
	`- The format block and "Keep each section concise": the transcript is deleted the moment you stop writing and your text is the only memory the next request has - completeness beats brevity; long lists are good; splitting one vague sentence into five specific ones is the point; vagueness and omission are the only failures.`,
	`- "If something is no longer relevant, you may remove it" and "UPDATE the Progress section": history is tagged, never deleted; CURRENT STATE is the progress section.`,
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
	return t || PRESERVE_SENTENCES[level][kind];
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

/** Substitute the {TOKENS} in a template source for one model's options. Tokens that the source
 *  lost are auto-added at the end under a visible note - a user edit can never silently drop a
 *  preserve rule, and a hand-written source still yields a complete instruction. */
export function applyTemplate(src: string, o: ModelOptions, inputStubs: boolean): string {
	let out = String(src ?? "");
	const missing: string[] = [];
	const sub = (token: string, value: string): void => {
		if (out.includes(token)) out = out.split(token).join(value);
		else if (value.trim()) missing.push(value);
	};
	sub("{TOOL CALLS SUMMARY}", `- ${sentenceFor("stubs", o.preserveStubs)}`);
	sub("{USER PROMPT SUMMARY}", `- ${sentenceFor("user", o.preserveUser)}`);
	sub("{ASSISTANT REPLIES SUMMARY}", `- ${sentenceFor("replies", o.preserveReplies)}`);
	sub("{ASSISTANT THINKING SUMMARY}", `- ${sentenceFor("thinking", o.preserveThinking)}`);
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
	if (missing.length) out += `\n\nRules the template lost (auto-added here; restore their {PLACEHOLDER} tokens to place them properly):\n${missing.join("\n")}`;
	return out;
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
	/** The main template's USER default: used when the active is null (empty = follow it). */
	userDefault?: string | null;
	customInstructions?: string;
}): string {
	const { sizes, inputTokens, kind, options } = args;
	const parts = [lengthBlock(sizes, inputTokens)];
	if (options.draft !== "off") parts.push(draftBlock(options.draft));
	// THE SUMMARY TEMPLATE: the stored text (or the default) is the SOURCE; the per-model options
	// substitute its {PLACEHOLDER} tokens. The template is always sent; the user may reword
	// anything except the tokens (lost tokens are auto-added with a visible note).
	// The template source: an explicit active wins ("" pins the extension default); a null active
	// follows the USER default; with neither, the built-in extension default rides.
	const tplSource =
		args.additionalInstruction !== undefined && args.additionalInstruction !== null
			? args.additionalInstruction.trim() || DEFAULT_TEMPLATE_TEXT
			: args.userDefault?.trim() || DEFAULT_TEMPLATE_TEXT;
	parts.push(applyTemplate(tplSource, options, options.inputStubs));
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

/** Guard for a stored draft value of unknown shape. */
function isDraft(v: unknown): v is Draft {
	return typeof v === "string" && (DRAFT_CYCLE as string[]).includes(v);
}
/** Migrate one stored per-model profile (config v2 "overrides"/"defaults") to the new options. */
/** Migrate one stored per-model profile (config v2 "overrides"/"defaults") to the new options.
 *  fallbackDraft covers configs from before the draft switch: local models defaulted to off,
 *  online ones to full - the same kind defaults optionsFor applies today. */
function modelOptionsFrom(src: any, fallbackMarker: string, fallbackDraft: Draft): ModelOptions {
	const levels = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
	const o: ModelOptions = {
		thinking: (levels.includes(src?.thinking) ? src.thinking : "off") as Level,
		timeoutMs: clamp(Math.round(numOr(src?.timeoutMs, 600_000)), 0, 7_200_000),
		summaryPercent: pctField(src?.summaryPercent, DEFAULT_OPTIONS.summaryPercent),
		summaryMinPercent: pctField(src?.summaryMinPercent, DEFAULT_OPTIONS.summaryMinPercent),
		summaryMaxPercent: pctField(src?.summaryMaxPercent, DEFAULT_OPTIONS.summaryMaxPercent),
		draft: isDraft(src?.draft) ? src.draft : fallbackDraft,
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
		const kindDraft: Draft = looksLocal ? "off" : "full";
		const o2 = overridesV2[ref];
		if (o2 && typeof o2 === "object" && (o2.thinking !== undefined || o2.timeoutMs !== undefined || o2.targetFrac !== undefined || o2.summaryBudgetTokens !== undefined)) {
			// An override rides ON TOP of the kind defaults, so an entry that only set the timeout
			// still inherits the kind's thinking and sizes.
			models[ref] = modelOptionsFrom({ ...kindDefault, ...o2 }, kindMarker, kindDraft);
			continue;
		}
		if (kindDefault && (kindDefault.thinking !== undefined || kindDefault.timeoutMs !== undefined || kindDefault.targetFrac !== undefined || kindDefault.summaryBudgetTokens !== undefined)) {
			models[ref] = modelOptionsFrom(kindDefault, kindMarker, kindDraft);
		}
	}
	return { models, retries, retryDelaySeconds, noThinkMarker, additionalInstruction };
}

/** Percent field: any number 1-99, else the default. */
function pctField(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 1 && value <= 99 ? Math.round(value) : fallback;
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
		directRequest: typeof raw.directRequest === "boolean" ? raw.directRequest : base.directRequest,
		shapeGate: typeof raw.shapeGate === "boolean" ? raw.shapeGate : base.shapeGate,
		chatMaxTokensCap: typeof raw.chatMaxTokensCap === "number" && Number.isFinite(raw.chatMaxTokensCap) && raw.chatMaxTokensCap >= 0 ? Math.round(raw.chatMaxTokensCap) : base.chatMaxTokensCap,
		scaling: {
			keepRecentPercent: pctField(raw?.scaling?.keepRecentPercent, base.scaling.keepRecentPercent),
			keepRecentMin: nonNegField(raw?.scaling?.keepRecentMin, base.scaling.keepRecentMin),
			keepRecentMax: nonNegField(raw?.scaling?.keepRecentMax, base.scaling.keepRecentMax),
			// The inversion (2026-10-02): a stored reservePercent migrates to startPercent = 100 - it.
			startPercent: (() => {
				const sp = raw?.scaling?.startPercent;
				if (typeof sp === "number" && Number.isFinite(sp) && sp >= 1 && sp <= 99) return Math.round(sp);
				const rp = raw?.scaling?.reservePercent;
				if (typeof rp === "number" && Number.isFinite(rp) && rp >= 1 && rp <= 99) return 100 - Math.round(rp);
				return base.scaling.startPercent;
			})(),
			reserveMin: nonNegField(raw?.scaling?.reserveMin, base.scaling.reserveMin),
			reserveMax: nonNegField(raw?.scaling?.reserveMax, base.scaling.reserveMax),
		},
		autoCompact: typeof raw.autoCompact === "boolean" ? raw.autoCompact : base.autoCompact,
		elision: sanitizeElision(raw?.elision),
		noThinkMarker: m.noThinkMarker,
		additionalInstruction: m.additionalInstruction,
		categorySentences: sanitizeCategorySentences(raw?.categorySentences),
		categoryUserDefaults: sanitizeCategorySentences(raw?.categoryUserDefaults),
		templateUserDefaults: typeof raw?.templateUserDefaults === "string" && raw.templateUserDefaults.trim() ? raw.templateUserDefaults : null,
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

function saveConfig(cfg: Config): void {
	mkdirSync(AGENT_DIR, { recursive: true });
	writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2) + "\n", "utf8");
}

function update(fn: (cfg: Config) => void): void {
	const cfg = loadConfig();
	fn(cfg);
	saveConfig(cfg);
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
		appendFileSync(LOG_PATH, JSON.stringify({ ts: localStamp(), ...record }) + "\n", "utf8");
	} catch {
		/* logging must never break compaction */
	}
}

/**
 * The log, human-readable: date and time first, then status, model, elapsed time and the transfer
 * figures in tokens (words are not logged - they were the old sizing flaw, and tokens are what the
 * acceptance window is written in). Unknown fields are simply left out.
 */
function tailLogPretty(lines = 16): string {
	try {
		const all = readFileSync(LOG_PATH, "utf8").split("\n").filter(Boolean).slice(-lines);
		if (!all.length) return "(the log is empty so far)";
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
					const window = typeof d.min === "number" && typeof d.max === "number" ? `accept ${fmt(d.min)}-${fmt(d.max)} tok (min ${d.minPct ?? "?"}% / max ${d.maxPct ?? "?"}% of the ${fmt(d.window ?? 0)} tok window)` : "";
					const status =
						d.event === "ok"
							? "OK"
							: d.event === "fail"
								? "FAIL"
								: d.event === "attempt"
									? "TRY"
									: d.event === "give_up"
										? "GIVE UP"
										: d.event === "done"
											? "DONE"
											: d.event === "start"
												? "START"
												: String(d.event ?? "").toUpperCase();
					const why = d.reason ?? (d.error ? String(d.error).split("\n")[0].slice(0, 90) : "");
					return [stamp, status, ref ?? "", seconds, transfer, window, why].filter(Boolean).join(" · ");
				} catch {
					return l.slice(0, 160);
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
const TOKENS_PER_WORD = 2;

function wordsFor(tokens: number): number {
	return Math.round(tokens / TOKENS_PER_WORD);
}

/**
 * The size window one summarisation attempt is governed by, all in tokens.
 *
 * THE LIMITS (2026-10-03): summaryMin%/summaryMax% of the model's WINDOW are the ACCEPTANCE
 * BOUNDS, used directly - 23% of a 95k window means 21,850 tok; the same settings scale on a 1M
 * model. The AIM (what the instructions request) = summaryPercent% of the region BEING FOLDED
 * (the session ctx minus the preserved tail, after the stubbing pass), clamped into [min, max]:
 *   - max    = min(summaryMax% × window, the region itself) — a summary longer than the source is
 *             pointless; the generation request is capped a little above max (GEN_MARGIN) so a
 *             summary landing exactly on max is not killed by pi's length-stop;
 *   - min    = min(summaryMin% × window, max) — a thinner summary is thrown away.
 * Then trimmed down where reality demands it: an overflow compaction (pi refuses a summary longer
 * than the history it replaces) and the room actually left in the session window after compaction.
 */
type Sizes = { min: number; target: number; max: number; genCap: number; notes: string[] };

/** Slack above `max` on the generation cap only, so a summary landing exactly on max is not cut. */
const GEN_MARGIN = 512;

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

/** Output-token headroom a draft block needs on top of the summary ceiling, so a long draft cannot
 *  push the summary itself into pi's length-stop. The draft is stripped before measuring, but it
 *  rides inside the model's output and the generation cap governs draft + summary together. */
function draftAllowance(draft: Draft | undefined): number {
	if (draft === "full") return 4096;
	if (draft === "mini") return 1024;
	return 0;
}

export function sizesFor(profile: ModelOptions, opts: { inputTokens: number; historyTokens: number; headroom: number; overflow: boolean; draft?: Draft; windowTokens?: number }, prevTokens = 0): Sizes {
	const notes: string[] = [];
	// The REGION: the material being folded (the session ctx minus the preserved tail, measured
	// after the stubbing pass) - kept only for the sanity cap and the unknown-window fallback.
	const region = Math.max(0, Math.round(opts.inputTokens));
	const window = Math.max(0, Math.round(opts.windowTokens ?? 0));
	// THE AIM AND THE LIMITS (2026-10-03): ALL percents of the model's WINDOW, used directly - the
	// numbers the settings show are the numbers the checks enforce, and the same settings scale
	// across models. A summary longer than the source is pointless, so the max is capped by the
	// region; the min never rises above the max (and yields to it on tiny regions).
	let max = window > 0 ? Math.round((profile.summaryMaxPercent * window) / 100) : Math.max(4_096, region);
	if (max > Math.max(region, 1_024)) {
		max = Math.max(region, 1_024);
		notes.push("max capped to the region itself (a summary longer than the source is pointless)");
	}
	let min = window > 0 ? Math.round((profile.summaryMinPercent * window) / 100) : 1_024;
	min = Math.min(min, max);
	let target = clamp(Math.round((profile.summaryPercent * (window > 0 ? window : region)) / 100), min, max);
	if (prevTokens > 0 && profile.chainMode === "attach") {
		// The verbatim previous summary rides INSIDE the summary: raise the ceiling by its size and
		// demand it in the minimum too - a lazy model cannot pass by returning only the previous
		// text, because the minimum still includes the region-derived quarter.
		max += prevTokens;
		min = Math.min(max, min + prevTokens);
		target = clamp(target, min, max);
		notes.push(`chain attach: +${fmt(prevTokens)} tok for the verbatim previous summary`);
	}
	if (opts.overflow && opts.historyTokens > 0 && max > opts.historyTokens) {
		max = opts.historyTokens;
		notes.push("overflow compaction: pi refuses a summary longer than the history it replaces");
	}
	if (opts.headroom > 0 && max > opts.headroom) {
		max = Math.max(1024, opts.headroom - 256);
		notes.push("trimmed to the room left in the session window after compaction");
	}
	if (target > max) target = max;
	if (min > target) min = target;
	// min 0 = "no minimum": any real output is accepted (the empty-summary check still applies).
	if (min > 0 && min < 128) min = 128;
	return { min, target, max, genCap: max + GEN_MARGIN + draftAllowance(opts.draft), notes };
}

/**
 * pi sizes the summary as min(floor(0.8 × reserveTokens), model.maxTokens) (compaction.js), so the
 * generation cap is turned back into a reserve value here: floor(0.8 × reserve) = genCap.
 */
export function withBudget(preparation: any, genCap: number): any {
	const reserve = clamp(Math.ceil(genCap / 0.8), 320, 250_000);
	return { ...preparation, settings: { ...preparation.settings, reserveTokens: reserve } };
}

/**
 * How big the delivered summary really is. The provider's own output count is the truth (reasoning
 * tokens are a subset of `output` in pi's Usage, so they are subtracted); without it the text is
 * estimated slightly above pi's chars/4, because bullet summaries with numbers and paths tokenize
 * denser than prose.
 */
export function measuredTokens(summary: string, usage: any): { tokens: number; by: "usage" | "estimate" } {
	if (usage && typeof usage.output === "number" && usage.output > 0) {
		const reasoning = typeof usage.reasoning === "number" ? usage.reasoning : 0;
		return { tokens: Math.max(0, usage.output - reasoning), by: "usage" };
	}
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
export function stripDraft(text: string): { summary: string; strippedChars: number } {
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
export function preprocessForSummariser(prep: any, options: { inputStubs: boolean; argCap: CharCap; stubTailChars?: CharCap }): { prep: any; stubbed: number; argsCapped: number } {
	const tail = options.stubTailChars ?? ELISION_DEFAULTS.stubTailChars;
	let stubbed = 0;
	let argsCapped = 0;
	const out: any = { ...prep };
	for (const key of ["messagesToSummarize", "turnPrefixMessages"]) {
		const list: any[] = prep?.[key] ?? [];
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
	}
	return { prep: out, stubbed, argsCapped };
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

export function applyElision(messages: any[], ctx: Ctx): { messages: any[] } | undefined {
	const cfg = loadConfig();
	const el = cfg.elision;
	const sessionId = String(ctx?.sessionManager?.getSessionId?.() ?? "session");
	let state = elisionStateBySession.get(sessionId);
	if (!el.enabled) {
		if (state?.applied.size) state.applied.clear();
		return undefined;
	}
	const window = ctx?.model?.contextWindow ?? 0;
	if (!(window > 0) || !Array.isArray(messages) || messages.length === 0) return undefined;

	// Context size of the INCOMING messages (always unstubbed - pi rebuilds them from storage).
	let effTokens = 0;
	for (const m of messages) effTokens += msgTokens(m);
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
		state = { applied: new Map() };
		elisionStateBySession.set(sessionId, state);
	}
	// Prune applied stubs whose result no longer exists (after a compaction or /clear).
	const present = new Set<string>();
	for (const m of messages) if (isToolResult(m)) present.add(String(m.toolCallId));
	for (const key of [...state.applied.keys()]) if (!present.has(key)) state.applied.delete(key);

	// Eligible = tool results before the protected window, oldest first, not yet applied, big
	// enough for a stub to pay for itself (the stub itself costs ~60-100 tokens).
	const candidates: Array<{ index: number; key: string; stub: string; savedTokens: number }> = [];
	if (effTokens >= softTokens) {
		for (let i = 0; i < messages.length && i < protectedFrom; i++) {
			const m = messages[i];
			if (!isToolResult(m)) continue;
			const key = String(m.toolCallId);
			if (state.applied.has(key)) continue;
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
	const newCandidates = candidates.filter((c) => !state!.applied.has(c.key));
	const newSavings = newCandidates.reduce((sum, c) => sum + c.savedTokens, 0);
	const compactionPoint = Math.max(0, window - extReserve(ctx));
	const stopTokens = Math.max(softTokens, compactionPoint - el.stopGapTokens);
	const appliedSaved = [...state!.applied.values()].reduce((s, e) => s + e.savedTokens, 0);
	const postBatchWire = effTokens - appliedSaved - newSavings;
	if (newCandidates.length >= el.minResultsToStub && newSavings >= el.minSavingsTokens && postBatchWire < stopTokens) {
		for (const c of newCandidates) state!.applied.set(c.key, { stub: c.stub, savedTokens: c.savedTokens });
		logLine(cfg, { event: "elision", applied: newCandidates.length, savedTokens: newSavings, effTokens, postBatchWire, softTokens, stopTokens, window, protectRecentTokens: el.protectRecentTokens });
		ctx?.ui?.notify?.(`elision: ${newCandidates.length} tool result${newCandidates.length === 1 ? "" : "s"} stubbed · ${fmt(effTokens)} → ${fmt(postBatchWire)} tokens`, "info");
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

/** Per-request chat-template kwargs that make the THINKING EFFORT real for a local llama server:
 *  pi-ai sends nothing thinking-related for a plain openai-format model (its reasoning branches do
 *  not cover it), so low/medium/xhigh would otherwise be silently dropped. ik_llama.cpp merges
 *  chat_template_kwargs over the server's --chat-template-kwargs defaults. "off" keeps using the
 *  no-think tag instead - deliberately not this. */
export function chatTemplateKwargsFor(thinking: Level): Record<string, unknown> | undefined {
	const level = thinking === "minimal" ? "low" : thinking === "high" ? "xhigh" : thinking;
	if (level === "low" || level === "medium" || level === "xhigh") return { reasoning_effort: level };
	return undefined;
}

/** The streamFn handed to pi's compact() so a FALLBACK request carries the same sampling flags and
 *  effort kwargs as the direct request. undefined = nothing to inject - call compact() as before. */
function samplingStreamFn(ref: string, sampling: SamplingFlags | undefined, thinking: Level): ((model: any, context: any, opts: any) => any) | undefined {
	const params = samplingParamsFor(ref, sampling);
	const kwargs = refLooksLocal(ref) ? chatTemplateKwargsFor(thinking) : undefined;
	if (!params && !kwargs) return undefined;
	return (model, context, opts) => piStreamSimple(model, context, { ...opts, samplingParams: { ...(params ?? {}), ...(kwargs ?? {}) } });
}

/** Board label for the state of a model's flags. */
function samplingLabel(s: SamplingFlags): string {
	const n = SAMPLING_KEYS.filter((k) => typeof s[k] === "number").length;
	return n ? `custom (${n} set)` : "ignore";
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
	for (const m of s.matchAll(/(?:superseded|amended) by Turn (\d+)/g)) {
		if (Number(m[1]) > maxTurn) return { ok: false, reason: `tag references Turn ${m[1]} which is not in the summary` };
	}
	return { ok: true };
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
}): Promise<{ text: string; usage: any; lastTurn: number; inputText: string }> {
	const { preparation, model, registry, sizes, options } = args;
	const startTurn = args.prevTurn > 0 ? args.prevTurn : 0;
	const main = serializeRegion(preparation?.messagesToSummarize ?? [], { startTurn, inputStubs: options.inputStubs, argCap: options.argCap });
	const prefix = serializeRegion(preparation?.turnPrefixMessages ?? [], { startTurn: main.lastTurn, inputStubs: options.inputStubs, argCap: options.argCap, partial: true });
	const nextTurn = main.lastTurn + (prefix.hadTurns ? 1 : 0);
	const prevBlock = labeledPreviousSummary(preparation?.previousSummary, args.prevTurn, nextTurn);
	const regionText = [main.text, prefix.text].filter(Boolean).join("\n\n");
	const coverLine = `The conversation segment below covers Turns ${startTurn + 1}-${Math.max(main.lastTurn, prefix.lastTurn)} (already numbered - use those numbers).`;
	const inputTokens = Math.ceil((regionText.length * 1.3) / 4) + SUMMARISER_FRAMING_TOKENS;
	const instructions = buildInstructions({ sizes, inputTokens, kind: args.kind, options, additionalInstruction: null, customInstructions: args.customInstructions });
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
	const requestOptions: Record<string, unknown> = { maxTokens: sizes.genCap, signal: args.signal, cacheRetention: "none" };
	if (model?.reasoning && args.thinking !== "off") requestOptions.reasoning = args.thinking;
	// The per-model sampling flags + (local) effort kwargs ride into the request body here; the
	// adapter merges samplingParams over the named fields, and llama-server reads them per request.
	const dParams = samplingParamsFor(args.ref, options.sampling);
	const dKwargs = refLooksLocal(args.ref) ? chatTemplateKwargsFor(args.thinking) : undefined;
	if (dParams || dKwargs) requestOptions.samplingParams = { ...(dParams ?? {}), ...(dKwargs ?? {}) };
	const response: any = await registry.streamSimple(model, context, requestOptions).result();
	if (response?.stopReason === "length") throw new Error(`generation hit the token cap (${sizes.genCap} tok: acceptance max ${sizes.max} tok + margins) and the summary is incomplete - the model wrote ~${fmt(response?.usage?.output ?? 0)} tok. Raise Summary max, or set the draft to off/mini.`);
	if (response?.stopReason === "error") throw new Error(response?.errorMessage || "provider error");
	if (response?.stopReason === "aborted") throw new Error("aborted");
	if ((response?.content ?? []).some((b: any) => b?.type === "toolCall")) throw new Error("the model tried to call a tool");
	const text = (response?.content ?? [])
		.filter((b: any) => b?.type === "text")
		.map((b: any) => String(b.text ?? ""))
		.join("")
		.trim();
	if (!text) throw new Error("empty response");
	// The exact conversation text the request carried - the stub-coverage gate counts the stub
	// markers in it against the copies in the summary.
	return { text, usage: response.usage, lastTurn: Math.max(main.lastTurn, prefix.lastTurn), inputText: `${main.text}\n${prefix.text}` };
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
	for (const ref of missing) logLine(cfg, { event: "skip", ref, reason: "not in the model list" });
	if (!candidates.length) {
		return { ok: false, error: missing.length ? `no model found for ${missing.join(", ")}` : "no models configured" };
	}

	// What the summariser will really be sent: the flattened region (tool results clipped to 2,000
	// characters by pi) plus the previous summary. No system prompt, no tool definitions.
	const needTokens = summariserInputTokens(preparation);
	const prevTurn = readPrevTurnInfo(ctx, preparation);
	const failures: string[] = [];

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
		if (pre.stubbed || pre.argsCapped) logLine(cfg, { event: "preprocess", ref: c.ref, stubbed: pre.stubbed, argsCapped: pre.argsCapped });
		const inputTokens = summariserInputTokens(prep);
		const sizes = sizesFor(profile, {
			inputTokens,
			historyTokens: historyTokensOnly(preparation),
			headroom: overflow ? 0 : postCompactionHeadroom(ctx, preparation),
			overflow,
			draft: profile.draft,
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
		const realInput = summariserInputTokens(prep);
		const realSizes = prep === preparation ? sizes : sizesFor(profile, { inputTokens: realInput, historyTokens: historyTokensOnly(prep), headroom: overflow ? 0 : postCompactionHeadroom(ctx, prep), overflow, draft: profile.draft, windowTokens: ctx?.model?.contextWindow ?? 0 }, prevSummaryTokens(prep));
		const genCapFinal = realSizes.genCap + (THINKING_ALLOWANCE[th.level] ?? 0);
		let instructions = buildInstructions({
			sizes: realSizes,
			inputTokens: realInput,
			kind: c.kind,
			options: profile,
			additionalInstruction: cfg.additionalInstruction,
			userDefault: cfg.templateUserDefaults,
			customInstructions,
		});
		const started = Date.now();
		logLine(cfg, {
			event: "attempt",
			ref: c.ref,
			kind: c.kind,
			inputTokens: realInput,
			min: realSizes.min,
			target: realSizes.target,
			max: realSizes.max,
			minPct: profile.summaryMinPercent,
			maxPct: profile.summaryMaxPercent,
			window: ctx?.model?.contextWindow ?? 0,
			genCap: genCapFinal,
			draft: profile.draft,
			thinking: th.level,
			timeoutMs: profile.timeoutMs,
			extraTries: cfg.retries,
		});
		let timedOut = false;
		const combined = combineSignals(signal, profile.timeoutMs, () => void (timedOut = true));
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
					customInstructions,
					ref: c.ref,
					kind: c.kind,
				});
				const dSeconds = Math.round(((Date.now() - started) / 1000) * 10) / 10;
				const cleaned = stripDraft(raw.text);
				const summary: string = cleaned.summary;
				const words = countWords(summary);
				const u: any = raw.usage ?? {};
				let measured = measuredTokens(summary, u);
			if (th.level !== "off") measured = { tokens: Math.ceil((summary.length * 1.15) / 4), by: "estimate" };
				if (cleaned.strippedChars > 0 && measured.by === "usage") {
					measured = { tokens: Math.max(0, measured.tokens - Math.ceil((cleaned.strippedChars * 1.15) / 4)), by: "usage" };
				}
				const split = { output: u.output, reasoning: u.reasoning };
				const dFail = (reason: string): Error => {
					logLine(cfg, { event: "fail", ref: c.ref, via: "direct", seconds: dSeconds, measured: measured.tokens, min: realSizes.min, max: realSizes.max, words, reason, ...split });
					return new Error(reason);
				};
				if (timedOut) throw dFail(`timed out with a half written summary`);
				if (!summary || summary.trim().length < 80) throw dFail("empty summary");
				if (measured.tokens < realSizes.min) throw dFail(`too thin - the summary is ~${fmt(measured.tokens)} tok vs the acceptance min ${fmt(realSizes.min)} tok (summary min ${profile.summaryMinPercent}% of the ${fmt(ctx?.model?.contextWindow ?? 0)} tok window). Raise Summary min, or lower the aim %.`);
				if (measured.tokens > realSizes.max) throw dFail(`too long - the summary is ~${fmt(measured.tokens)} tok vs the acceptance max ${fmt(realSizes.max)} tok (summary max ${profile.summaryMaxPercent}% of the ${fmt(ctx?.model?.contextWindow ?? 0)} tok window). Raise Summary max, or lower the aim %.`);
				if (cfg.shapeGate) {
					const sh = shapeCheck(summary);
					if (!sh.ok) throw dFail(`shape: ${sh.reason}`);
				}
				// The STUB-COVERAGE GATE: with preserveStubs=verbatim the summary must carry the tool
				// calls, not narrate them. Count the stub markers the request carried vs the copies
				// in the summary; a model that paraphrases instead of copying fails and the next
				// candidate is asked.
				const inStubs = (String(raw.inputText ?? "").match(/output elided:/g) ?? []).length;
				const gotStubs = (summary.match(/output elided:/g) ?? []).length;
				if (profile.preserveStubs === "verbatim" && inStubs >= 4 && gotStubs < Math.ceil(inStubs * 0.8)) {
					throw dFail(`stub coverage ${gotStubs}/${inStubs} below 80% with preserveStubs=verbatim (the model paraphrased the tool calls)`);
				}
				// The CHAIN-ATTACH GATE: with chainMode=attach the previous summary must appear,
				// verbatim, in its own section.
				if (profile.chainMode === "attach" && preparation.previousSummary?.trim() && !/^##\s*PREVIOUS SUMMARY\b/im.test(summary)) {
					throw dFail("chain attach: the PREVIOUS SUMMARY section is missing");
				}
				logLine(cfg, { event: "ok", ref: c.ref, via: "direct", kind: c.kind, seconds: dSeconds, measured: measured.tokens, measuredBy: measured.by, draft: profile.draft, draftStripped: cleaned.strippedChars || undefined, min: realSizes.min, target: realSizes.target, max: realSizes.max, words, stubsIn: inStubs || undefined, stubsCopied: gotStubs || undefined, input: u.input, output: u.output, reasoning: u.reasoning });
				const compaction = {
					summary: summary + fileOpsAppendix(preparation.fileOps),
					firstKeptEntryId: prep.firstKeptEntryId,
					tokensBefore: prep.tokensBefore,
					usage: u,
					details: { lastTurnNumber: Math.max(raw.lastTurn, maxTurnInSummary(summary)) },
				};
				combined.dispose();
				return { ok: true, result: { ok: true, compaction, ref: c.ref, model: c.model, kind: c.kind, measured: measured.tokens, target: realSizes.target, min: realSizes.min, max: realSizes.max, words, seconds: dSeconds } };
			} catch (dErr: any) {
				if (signal?.aborted) {
					logLine(cfg, { event: "note", note: "cancelled", ref: c.ref });
					return { ok: false, error: "cancelled" };
				}
				logLine(cfg, { event: "direct_fail", ref: c.ref, error: String(dErr?.message ?? dErr).slice(0, 220), note: "moving to the next candidate; pi's compact() runs only after every candidate failed" });
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
			// The draft scratchpad (when the model has one) is discarded here, before measuring or
			// storing: the acceptance window judges the summary the next request will actually see.
			const cleaned = stripDraft(compaction?.summary ?? "");
			const summary: string = cleaned.summary;
			const words = countWords(summary);
			const u: any = compaction?.usage ?? {};
			let measured = measuredTokens(summary, u);
			if (th.level !== "off") measured = { tokens: Math.ceil((summary.length * 1.15) / 4), by: "estimate" };
			// The draft rode inside the provider's output count; take its estimate back out so the
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
				logLine(cfg, { event: "fail", ref: c.ref, seconds, measured: measured.tokens, reason: "timed out with a half written summary", timeoutMs: profile.timeoutMs, ...split });
				failures.push(`${c.ref}: nothing finished within ${secs(profile.timeoutMs)}`);
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
			if (measured.tokens > realSizes.max) {
				logLine(cfg, { event: "fail", ref: c.ref, seconds, measured: measured.tokens, min: realSizes.min, max: realSizes.max, words, reason: "too long - over the accepted ceiling", ...split });
				failures.push(`${c.ref}: ~${fmt(measured.tokens)} tok, ceiling is ${fmt(realSizes.max)}`);
				continue;
			}
			logLine(cfg, {
				event: "ok",
				ref: c.ref,
				kind: c.kind,
				seconds,
				measured: measured.tokens,
				measuredBy: measured.by,
				draft: profile.draft,
				draftStripped: cleaned.strippedChars || undefined,
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
				result: { ok: true, compaction: { ...compaction, summary }, ref: c.ref, model: c.model, kind: c.kind, measured: measured.tokens, target: realSizes.target, min: realSizes.min, max: realSizes.max, words, seconds },
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

function keyOf(data: string): string {
	if (!data) return "";
	if (data === "\r" || data === "\n") return "enter";
	if (data === "\x1b") return "escape";
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

function cut(text: string, width: number): string {
	if (width <= 0) return "";
	return text.length <= width ? text : text.slice(0, Math.max(0, width - 1)) + "…";
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
export const DEFAULT_OPTIONS: ModelOptions = {
	sampling: { ...SAMPLING_DEFAULTS },
	thinking: "off",
	timeoutMs: 600_000, // 10 min
	summaryPercent: 10,
	summaryMinPercent: 5,
	summaryMaxPercent: 50,
	draft: "off",
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
	const stored = cfg.models[ref];
	const local = isLocalModel(model ?? {});
	// The marker defaults to the pi-wide setting for local models; anything stored wins.
	const markerDefault = local ? cfg.noThinkMarker : "";
	// The draft default is the kind's: online models get the full hidden analysis (Anthropic's own
	// compaction does the same), local ones off - a full draft roughly doubles a 10-20 tok/s summary.
	const draftDefault: Draft = local ? "off" : "full";
	const samplingDefault = stored?.sampling ?? SAMPLING_DEFAULTS;
	return {
		thinking: stored?.thinking ?? DEFAULT_OPTIONS.thinking,
		timeoutMs: clamp(Math.round(numOr(stored?.timeoutMs, DEFAULT_OPTIONS.timeoutMs)), 0, 7_200_000),
		summaryPercent: pctField(stored?.summaryPercent, DEFAULT_OPTIONS.summaryPercent),
		summaryMinPercent: pctField(stored?.summaryMinPercent, DEFAULT_OPTIONS.summaryMinPercent),
		summaryMaxPercent: pctField(stored?.summaryMaxPercent, DEFAULT_OPTIONS.summaryMaxPercent),
		draft: isDraft(stored?.draft) ? stored.draft : draftDefault,
		inputStubs: typeof stored?.inputStubs === "boolean" ? stored.inputStubs : DEFAULT_OPTIONS.inputStubs,
		argCap: isCharCap(stored?.argCap) ? stored.argCap : DEFAULT_OPTIONS.argCap,
		preserveStubs: isPreserve(stored?.preserveStubs) ? stored.preserveStubs : DEFAULT_OPTIONS.preserveStubs,
		preserveUser: isPreserve(stored?.preserveUser) ? stored.preserveUser : DEFAULT_OPTIONS.preserveUser,
		preserveReplies: isPreserve(stored?.preserveReplies) ? stored.preserveReplies : DEFAULT_OPTIONS.preserveReplies,
		preserveThinking: isPreserve(stored?.preserveThinking) ? stored.preserveThinking : DEFAULT_OPTIONS.preserveThinking,
		chainMode: isChainMode(stored?.chainMode) ? stored.chainMode : DEFAULT_OPTIONS.chainMode,
		noThinkMarker: stored?.noThinkMarker ?? markerDefault,
		sampling: sanitizeSampling(samplingDefault),
	};
}

/** Selected models get their stored options; the rest display (and would use) the defaults. */
function optionsOfAnyModel(cfg: Config, ref: string, model: any): ModelOptions {
	if (cfg.models[ref]) return optionsFor(cfg, ref, model);
	return { ...DEFAULT_OPTIONS, draft: isLocalModel(model ?? {}) ? "off" : "full" };
}

function setModelOptions(ref: string, patch: Partial<ModelOptions>): void {
	update((c) => {
		const prev = c.models[ref] ?? {};
		c.models[ref] = { ...prev, ...patch };
	});
}

function rotateAdd(ref: string, options?: ModelOptions): void {
	update((c) => {
		if (c.models[ref]) return;
		c.models[ref] = options ?? { ...DEFAULT_OPTIONS };
	});
}

function rotateRemove(ref: string): void {
	update((c) => {
		delete c.models[ref];
	});
}

function toggleEnabled(): void {
	update((c) => void (c.enabled = !c.enabled));
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
/** THE UNLINK (2026-10-01): pi's compaction.reserveTokens/keepRecentTokens are absolute token
 *  counts tuned for one window size; the extension no longer reads them. Instead the two anchors
 *  are derived from the ACTIVE model's context window - percent first, then clamped to absolute
 *  min/max - so a 95k local model gets ~today's tuned values, a 200k model 4x that, a 1M model the
 *  clamped maximum. Recomputed at EVERY use: a mid-session model switch changes everything on the
 *  next request, with no /reload and nothing written to pi's settings.json. */
export function extReserve(ctx: Ctx): number {
	const window = (ctx as any)?.model?.contextWindow ?? 0;
	if (!(window > 0)) return DEFAULT_RESERVE_TOKENS;
	const s = loadConfig().scaling;
	// The reserve is what the trigger point leaves free: window - (startPercent% of window),
	// clamped by the min/max reserve limits.
	const lo = Math.min(s.reserveMin, s.reserveMax);
	const hi = Math.max(s.reserveMin, s.reserveMax);
	return clamp(Math.round((window * (100 - s.startPercent)) / 100), lo, hi);
}

/** The same scheme for the verbatim tail: what compaction keeps and what the elision protects. */
export function extKeepRecent(ctx: Ctx): number {
	const window = (ctx as any)?.model?.contextWindow ?? 0;
	if (!(window > 0)) return DEFAULT_KEEP_RECENT_TOKENS;
	const s = loadConfig().scaling;
	const lo = Math.min(s.keepRecentMin, s.keepRecentMax);
	const hi = Math.max(s.keepRecentMin, s.keepRecentMax);
	return clamp(Math.round((window * s.keepRecentPercent) / 100), lo, hi);
}

/** Short effective-settings line for a model row. */
function modelRowValue(cfg: Config, ref: string, model: any, reserveTokens: number): string {
	const o = optionsOfAnyModel(cfg, ref, model);
	const place = orderPosition(cfg, ref);
	const pos = place >= 0 ? `#${place + 1}` : "—";
	const th = model ? effectiveThinking(o.thinking, model) : { level: o.thinking, note: "" };
	return `${pos} · ${timeoutLabel(o.timeoutMs)} · summary aim ${o.summaryPercent}% of region, limits min ${o.summaryMinPercent}% / max ${o.summaryMaxPercent}% of window · ${th.level} · draft ${o.draft}`;
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
		private restore?: { optionsRef?: string; optionKey?: string },
	) {
		this.rebuild();
		if (this.restore?.optionsRef) {
			this.highlightRef(this.restore.optionsRef);
			this.openHighlighted(this.restore.optionKey);
		}
	}

	// -- Component -------------------------------------------------------------

	invalidate(): void {
		this.list?.invalidate();
	}

	handleInput(data: string): void {
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
	private openHighlighted(initialKey?: string): void {
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
				? "Enter opens its options (thinking, draft block, timeout, summary sizes, no-think tag). Space removes it from the order. Digits move it to that place."
				: "Not in the Compaction models order - it shows the default options. Press a digit (or 0) to add it at that place; Space on a selected model removes it. Enter opens options only for models in the order.",
		};
	}
}

// ---------------------------------------------------------------- TUI: per-model options submenu

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
export function modelOptionsScreen(ctx: Ctx, ref: string, model: any, onDone: () => void, onExit?: (action: string) => void, initialKey?: string): Component {
	const buildItems = (): SettingItem[] => {
		const cfg = loadConfig();
		const o = optionsFor(cfg, ref, model);
		const reserveTokens = extReserve(ctx);
		const levels = model ? supportedThinkingLevels(model) : [...THINKING_CYCLE];
		const th = model ? effectiveThinking(o.thinking, model) : { level: o.thinking, note: "" };
		const rows: SettingItem[] = [];
		rows.push({
			id: "thinking",
			label: "Thinking method:",
			currentValue: o.thinking,
			values: withCurrent(levels, o.thinking),
			description: `Enter/Space cycles: ${levels.join(" → ")}. Private reasoning before it writes - it costs time, it does not make the summary longer. "off" also writes the no-think tag (row below) into the request for local models. A level the model does not have falls back to the nearest it has.`,
		});
		rows.push({
			id: "draft",
			label: "Draft block:",
			currentValue: o.draft,
			values: withCurrent(DRAFT_CYCLE, o.draft),
			description:
				'A hidden analysis written before the summary in the SAME call, then stripped off before storing. "full" = go through every user turn (what was asked, what you did, tools, errors, user corrections) and check the sections are coverable; "mini" = at most ten lines; "off" = start the summary directly. Its tokens do not count against the acceptance window, but they do cost time - online models default to full (Anthropic\'s own compaction does this), local models to off, because at 10-20 tokens/s a full draft roughly doubles the summary time.',
		});
		rows.push({
			id: "timeout",
			label: "Summary timeout period:",
			currentValue: timeoutLabel(o.timeoutMs),
			values: [EDIT_NUMBER, "unlimited"],
			description:
				"Enter opens a free numeric entry in minutes (0 = unlimited); cycling reaches 'unlimited'. After this the try is given up and the next model in the Compaction models list is asked. A local model writes 10-20 tokens a second, so a 13,000-token summary needs 11-22 minutes - a 10-minute timeout is only for fast models.",
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
			label: "Summary min (% of ctx window):",
			currentValue: `${o.summaryMinPercent}% (now: ${fmt(Math.round((o.summaryMinPercent * (ctx?.model?.contextWindow ?? 0)) / 100))} tok)`,
			values: [EDIT_NUMBER],
			description:
				"THE ACCEPTANCE MINIMUM: a summary shorter than this is thrown away and the next model is asked. Percent of the model's ctx window, so it scales when you switch models - the value on the right is what it is right now. Never rises above the max, and yields to it on tiny regions. Enter opens a free numeric entry (1-99). Default 5.",
		});
		rows.push({
			id: "summaryMax",
			label: "Summary max (% of ctx window):",
			currentValue: `${o.summaryMaxPercent}% (now: ${fmt(Math.round((o.summaryMaxPercent * (ctx?.model?.contextWindow ?? 0)) / 100))} tok)`,
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
			label: "Preview final description:",
			currentValue: "Enter shows it",
			values: ["show"],
			description:
				"The COMPLETE description THIS model would receive on the next compaction: the size block, the draft instruction, and the summary template with this model's {TOKENS} substituted (its thinking level, its preserve levels, its chain mode, the category sentences as customized on the board). Read-only preview - nothing is sent; edits in the editor are discarded.",
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
			values: ["ignore all", "open flags"],
			submenu: (_current: string, done: (selectedValue?: string) => void) => samplingScreen(ref, done) as unknown as Component,
			description:
				'Extra sampling parameters sent with THIS model\'s compaction request only (your chat requests are never touched): temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty - each a number or "ignore". Enter opens the flags page (Esc comes back); Space cycles here, "ignore all" clears every flag. All flags default to ignore; the Qwen3.8-Flash-Next entry ships with its recommended non-thinking values (temp 0.7, top_p 0.8, top_k 20, min_p 0, presence 1.5, repeat 1.0). Free numeric entry: /compact-plus sampling',
		});
		rows.push({
			id: "remove",
			label: "Remove from the Compaction models order:",
			currentValue: "remove",
			values: ["remove now"],
			description: "Same as Space on the list: the model drops to its alphabetical place among the unselected ones and its stored options are deleted. Nothing is deleted from pi itself.",
		});
		return rows;
	};

	const items = buildItems();
	const list = new SettingsList(
		items,
		Math.min(items.length + 2, 12),
		getSettingsListTheme(),
		(id: string, value: string) => {
			const o = optionsFor(loadConfig(), ref, model);
			const reserveTokens = extReserve(ctx);
			const patchRow = (text: string) => {
				const row = items.find((i) => i.id === id);
				if (row) row.currentValue = text;
			};
			if (id === "thinking") {
				setModelOptions(ref, { thinking: value as Level });
			} else if (id === "draft") {
				setModelOptions(ref, { draft: (DRAFT_CYCLE as string[]).includes(value) ? (value as Draft) : "off" });
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
	return list;
}

/**
 * The sampling flags sub-page of a model's options: one row per flag, Enter/Space cycles through a
 * value ladder (the Qwen instruct value first) plus "ignore"; the current value is always in the
 * cycle even when it was typed freely via the CLI. Esc comes back to the model's options.
 */
function samplingScreen(ref: string, onDone: () => void): Component {
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
				values: withCurrent(["ignore", ...SAMPLING_LADDERS[key]], curLabel),
				description: SAMPLING_HELP[key],
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
		if (data === "\x1b" && this.pending) {
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

/** The summary template's submenu: edit, read-only previews (active / user / extension), and the
 *  user-default actions. Every row exits through the board loop - editors and pi's select live
 *  over the plain chat, never over a mounted screen. */
class TemplateMenu implements Component, Focusable {
	private list: SettingsList | null = null;
	private isFocused = false;

	constructor(
		private tui: TUI,
		private ctx: Ctx,
		private onAction: (action: string) => void,
	) {
		this.rebuild();
	}

	private stateLabel(): string {
		const cfg = loadConfig();
		if (cfg.additionalInstruction && cfg.additionalInstruction.trim()) return `${cfg.additionalInstruction.split("\n").length} lines (custom)`;
		if (cfg.additionalInstruction !== null && cfg.additionalInstruction !== undefined) return "extension default (pinned)";
		return cfg.templateUserDefaults?.trim() ? "follows the user default" : "extension default";
	}

	private rows(): SettingItem[] {
		const cfg = loadConfig();
		const hasUser = Boolean(cfg.templateUserDefaults?.trim());
		return [
			{
				id: "edit",
				label: "Edit the template",
				currentValue: this.stateLabel(),
				values: ["open"],
				description: "Opens the editor with the ACTIVE text. Submit saves it as the active template; empty follows the user default; submit the lone word :extension or :user to restart the editor from that default.",
			},
			{
				id: "previewActive",
				label: "Preview active template:",
				currentValue: "read-only",
				values: ["preview"],
				description: "The template as currently in effect (before the per-model {TOKEN} substitution). This is the text every model's request starts from.",
			},
			{
				id: "previewUser",
				label: "Preview user default:",
				currentValue: !hasUser ? "(none saved)" : loadConfig().templateUserDefaults!.trim() === DEFAULT_TEMPLATE_TEXT.trim() ? "saved - same as extension default" : "saved (custom)",
				values: ["preview"],
				description: "Your saved USER default. When the active template is empty, this is what rides instead. Save one with 'Save current as user default'.",
			},
			{
				id: "previewExt",
				label: "Preview extension default:",
				currentValue: "read-only",
				values: ["preview"],
				description: "The built-in template as shipped - what 'Reset to extension default' pins the active to.",
			},
			{
				id: "saveUser",
				label: "Save current as user default",
				currentValue: hasUser ? "replaces the saved one" : "not saved yet",
				values: ["go"],
				description: "The ACTIVE template becomes your USER default (a checkpoint). The active template itself does not change. Asks for confirmation.",
			},
			{
				id: "resetUser",
				label: "Reset to user default",
				currentValue: hasUser ? "ready" : "(none saved)",
				values: ["go"],
				description: "Drops the active custom text; the template then follows the USER default (falls back to the extension default when none is saved). Asks for confirmation.",
			},
			{
				id: "resetExt",
				label: "Reset to extension default",
				currentValue: "ready",
				values: ["go"],
				description: "Pins the template to the built-in extension default (ignores the user default until you edit again). Asks for confirmation.",
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
				if (value === "open") this.onAction("edit-instruction");
				else if (value === "preview") this.onAction(`preview-template:${id === "previewActive" ? "active" : id === "previewUser" ? "user" : "ext"}`);
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
		const lines: string[] = ["", "Summary template - edit, preview the three layers, manage the user default:", ""];
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
		const text = await ctx.ui?.editor?.(title, prefill);
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
		kind: isLocalModel(model) ? "local" : "online",
		options: profile,
		additionalInstruction: cfg.additionalInstruction,
		userDefault: cfg.templateUserDefaults,
		customInstructions: undefined,
	});
	await ctx.ui?.editor?.(
		`FINAL summary description for ${ref} — READ-ONLY PREVIEW`,
		[
			`READ-ONLY PREVIEW - this is what ${ref} would receive. Nothing is sent; ANY changes made in this editor are DISCARDED when you close it (the submit/newline/external-editor keys below belong to the TEMPLATE editors, not to this preview).`,
			`The real request adds the conversation region (numbered turns + stubs + the previous summary) below this description.`,
			``,
			text,
		].join("\n"),
	);
}

// ------------------------------------------------- TUI: the three settings submenus

/** The auto-compaction submenu: the live trigger point (read-only) and the three settings that
 *  define it. The editor rows exit through the board loop (pi's overlays are safe only there). */
class AutoCompactionMenu implements Component, Focusable {
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
		const window = this.ctx?.model?.contextWindow ?? 0;
		const point = window - extReserve(this.ctx);
		return [
			{
				id: "startPct",
				label: "Start auto-compaction at % of ctx:",
				currentValue: `${cfg.scaling.startPercent}% (now: ${fmt(point)} tok, based on min/max limits and current ctx window of ${fmt(window)} tok)`,
				values: ["edit"],
				description: "The compaction starts when the ctx reaches this percent of the model's window (the value on the right is what that is right now, based on the min/max limits below). Enter opens a free numeric entry (1-99). Default 82.",
			},
			{
				id: "resMin",
				label: "Min reserve tok before auto-comp.:",
				currentValue: `${fmt(cfg.scaling.reserveMin)} tok (Overrides the auto-compaction point if the % value is lower than this min limit)`,
				values: ["edit"],
				description: "The minimum space kept free at the top of the window. If the % above would leave less room than this, this minimum wins. Enter opens a free numeric entry. Default 12,288.",
			},
			{
				id: "resMax",
				label: "Max reserve tok before auto-comp.:",
				currentValue: `${fmt(cfg.scaling.reserveMax)} tok (Overrides the auto-compaction point if the % value is higher than this max limit)`,
				values: ["edit"],
				description: "The maximum space kept free at the top of the window. If the % above would leave more room than this, this maximum wins (on a 1M window this is what caps the reserve). Enter opens a free numeric entry. Default 80,000.",
			},
		];
	}

	private rebuild(): void {
		const items = this.rows();
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 8),
			getSettingsListTheme(),
			(id: string, value: string) => {
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
		const lines: string[] = ["", "Auto-compaction - when the context crosses the trigger point, compaction starts by itself:", ""];
		lines.push(...(this.list?.render(inner) ?? []));
		return lines.map((l) => truncateToWidth(l, inner));
	}
}

/** The preserve-recent submenu: the live preserved tail (read-only) and the three settings that
 *  define it. */
class PreserveMenu implements Component, Focusable {
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
		const window = this.ctx?.model?.contextWindow ?? 0;
		const keep = extKeepRecent(this.ctx);
		return [
			{
				id: "keepPct",
				label: "Preserve recent % of ctx window:",
				currentValue: `${cfg.scaling.keepRecentPercent}% (now: ${fmt(keep)} tok, based on min/max limits and current ctx window of ${fmt(window)} tok)`,
				values: ["edit"],
				description: "The preserved tail as a percent of the model's window (the value on the right is what that is right now). Enter opens a free numeric entry (1-99). Default 20.",
			},
			{
				id: "keepMin",
				label: "Min preserve window limit in tok:",
				currentValue: `${fmt(cfg.scaling.keepRecentMin)} tok (Overrides the Preserve recent if the above value is lower than this min limit)`,
				values: ["edit"],
				description: "If the % above would preserve less than this, this minimum wins. Enter opens a free numeric entry. Default 16,384.",
			},
			{
				id: "keepMax",
				label: "Max preserve window limit in tok:",
				currentValue: `${fmt(cfg.scaling.keepRecentMax)} tok (Overrides the Preserve recent if the above value is higher than this max limit)`,
				values: ["edit"],
				description: "If the % above would preserve more than this, this maximum wins. Enter opens a free numeric entry. Default 100,000.",
			},
		];
	}

	private rebuild(): void {
		const items = this.rows();
		this.list = new SettingsList(
			items,
			Math.min(items.length + 2, 8),
			getSettingsListTheme(),
			(id: string, value: string) => {
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
		const lines: string[] = ["", "Preserve recent tokens - the most recent part of the ctx that will not be compacted:", ""];
		lines.push(...(this.list?.render(inner) ?? []));
		return lines.map((l) => truncateToWidth(l, inner));
	}
}

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
				label: "Auto-elide only at x results:",
				currentValue: `${e.minResultsToStub} results`,
				values: ["edit"],
				description: "A sweep waits until at least this many tool results qualify - no churn for one lonely old test run. Enter opens a free numeric entry. Default 4.",
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

// ---------------------------------------------------------------- TUI: the settings board

interface BoardResult {
	action?: string;
}

/** Where the board's loop should land after an edit: the same board row, or - for a model's
 *  options edit - the Compaction models list with that model's options screen open, on the edited
 *  row. (2026-10-01: the loop used to reopen the board at the top level after every edit.) */
interface BoardRestore {
	rowId?: string;
	orderRef?: string;
	optionKey?: string;
	catDesc?: { kind: string; level?: string };
	autoMenu?: boolean;
	preserveMenu?: boolean;
	elisionMenu?: boolean;
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
		if (r.autoMenu && select("autoMenu")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
			return;
		}
		if (r.preserveMenu && select("preserveMenu")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
			return;
		}
		if (r.elisionMenu && select("elisionMenu")) {
			(this.list as unknown as { activateItem?: () => void }).activateItem?.();
			return;
		}
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
		lines.push(this.theme.fg("accent", this.theme.bold("Compaction models — type to filter · Enter changes or opens · Esc leaves")));
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
		this.patch("enabled", cfg.enabled ? "on" : "off");
		this.patch("directRequest", cfg.directRequest ? "on" : "off");
		this.patch("order", rotationText(cfg, this.ctx));
		this.patch("retries", String(cfg.retries));
		this.patch("retryWait", `${cfg.retryDelaySeconds} s`);
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
			id: "enabled",
			label: "Extension active:",
			currentValue: cfg.enabled ? "on" : "off",
			values: ["on", "off"],
			description:
				"Off = pi compacts with the session model, like it always did. On = the summary is written by the first model in the Compaction models list that comes back with a usable length; if all of them fail, pi's own compaction runs.",
		});
		rows.push({
			id: "order",
			label: "Compaction models",
			currentValue: "",
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new OrderList(this.tui, this.ctx, done, this.restore?.orderRef ? { optionsRef: this.restore.orderRef, optionKey: this.restore.optionKey } : undefined) as unknown as Component,
			description:
				"The models that write the summary, first one first. In the list: type to filter, digits 1-9 move a model to that place (0 = the end), Space removes it from the order, Enter opens the options of a model that is IN the order. Unselected models show the default options and cannot be opened.",
		});
		rows.push(numberRow("retries", "Retries per model", String(cfg.retries), [], "Only used when the service says it is busy, rate limited, or the connection dropped. A model that returns a summary too thin or too long is NOT retried - the next model in the list is asked instead. Enter opens a free numeric entry (0 = no retries). Default 2."));
		rows.push(numberRow("retryWait", "Wait between retries", `${cfg.retryDelaySeconds} s`, [], "Seconds before each retry (the next wait doubles). Free entry: type any number of seconds. Default 5 s."));
		rows.push({
			id: "autoMenu",
			label: "Auto-compaction",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new AutoCompactionMenu(this.tui, this.ctx, (action: string) => {
					if (action === "__close") done();
					else done(action);
				}) as unknown as Component,
			description:
				"WHEN COMPACTION STARTS BY ITSELF: the trigger point (read-only, computed live from the settings below and the current model's window), the start percent, and the min/max reserve limits. Enter opens the submenu; Enter on a setting opens its editor (over the plain chat, with the position restored afterwards).",
		});
		rows.push({
			id: "preserveMenu",
			label: "Preserve recent tok",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new PreserveMenu(this.tui, this.ctx, (action: string) => {
					if (action === "__close") done();
					else done(action);
				}) as unknown as Component,
			description:
				"WHAT STAYS WORD FOR WORD: the newest tail of the conversation kept outside every summary (and protected from elision), so resuming feels continuous. The submenu shows the live preserved window (read-only) and its three settings. Enter opens it; Enter on a setting opens its editor.",
		});
		rows.push({
			id: "compactNow",
			label: "Start compaction now",
			currentValue: "Start the compaction now",
			values: ["Start the compaction now"],
			description:
				"Starts a compaction RIGHT NOW: pi prepares the session and the Compaction models rotation answers it (the board closes so you can watch the progress). The automatic starter lives in the Auto-compaction submenu's trigger point - when the ctx crosses it while pi is idle, the same compaction starts by itself.",
		});
		rows.push({
			id: "compactStock",
			label: "Start pi's built-in compaction now",
			currentValue: `Start pi's built-in compaction (pi aims for summary size of 0.8 x reserveTokens = ${fmt(Math.floor(readPiReserveTokens() * 0.8))} tok)`,
			values: [`Start pi's built-in compaction (pi aims for summary size of 0.8 x reserveTokens = ${fmt(Math.floor(readPiReserveTokens() * 0.8))} tok)`],
			description:
				"Runs pi's OWN compaction once, bypassing the rotation: pi prepares the session and summarizes with the active model, aiming at 0.8 x its reserveTokens (the number on the right). Use it to compare pi's stock summary with the extension's. Everything else (the rotation, the gates) stays in place for the next compaction.",
		});
		rows.push({
			id: "piReserve",
			label: "pi's reserveTokens (stock fallback):",
			currentValue: `${fmt(readPiReserveTokens())} tok`,
			values: [EDIT_NUMBER],
			description:
				"Writes pi's OWN settings.json (compaction.reserveTokens) - the number the row above aims with. Applies ONLY to pi's built-in compaction (the row above, or the fallback when this extension is disabled or EVERY model failed). This extension itself never reads it (its own trigger lives in the Auto-compaction submenu above). pi caches its settings, so /reload is needed for the change to take effect.",
		});
		rows.push(numberRow("chatCap", "Chat max_tokens cap", cfg.chatMaxTokensCap > 0 ? fmt(cfg.chatMaxTokensCap) : "no cap", ["no cap"], "Caps the generation permission (max_tokens) on every CHAT request. pi sizes it as window minus its context estimate minus 4,096 - on big-window models that asks for absurd room (943k on glm-flash) and the whole request is rejected whenever the estimate undercounts by more than 4,096 tokens (the overflow that forced the emergency compactions). With a cap the request always fits until the real context reaches window minus cap. Enter opens a free numeric entry (0 = no cap); cycling reaches 'no cap'. Default 65,536."));
		rows.push({
			id: "directRequest",
			label: "Direct request:",
			currentValue: cfg.directRequest ? "on" : "off",
			values: ["on", "off"],
			description:
				"ON = the extension builds the whole summarization request itself: turn-numbered input, labelled previous summary, ONE instruction source (no pi format block, no override sentences); the model is called through the registry; a shape gate checks the result. A direct failure moves to the NEXT candidate; pi's compact() runs only after every candidate failed. OFF = the old layered path. Default on. (The shape gate also applies to fallback results; off with \"shapeGate\": false in the config.)",
		});
		rows.push({
			id: "nothinkTag",
			label: "No-think tag for local models:",
			currentValue: cfg.noThinkMarker || "(empty)",
			values: [EDIT_TEXT, "(empty)"],
			description:
				'The default tag written into a LOCAL model\'s summary request when its thinking is "off" (each model\'s own No-think tag row overrides it). Enter on "type text..." opens the editor with the current tag as the starting text; "(empty)" writes nothing. Default <|think_off|>. Free text also via: /compact-plus marker <text>',
		});
		rows.push({
			id: "elisionMenu",
			label: "Elision",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new ElisionMenu(this.tui, this.ctx, (action: string) => {
					if (action === "__close") done();
					else done(action);
				}) as unknown as Component,
			description:
				"Old tool results are replaced by short stubs in the requests the model receives, so the context stays much lower for much longer; your screen and the session file always keep the originals. The submenu holds the on/off toggle and all six settings. Enter opens it; Enter on a setting opens its editor (over the plain chat, with the position restored afterwards).",
		});
		rows.push({
			id: "instruction",
			label: "Summary template",
			currentValue: "",
			values: [],
			submenu: (_current: string, done: (selectedValue?: string) => void) =>
				new TemplateMenu(this.tui, this.ctx, (action: string) => {
					if (action === "__close") done();
					else done(`__exit:${action}`);
				}) as unknown as Component,
			description:
				"THE SUMMARY TEMPLATE - the exact text the summarizer reads (after the size/draft blocks). It contains {PLACEHOLDER} tokens the extension substitutes per model: {TOOL CALLS SUMMARY}, {USER PROMPT SUMMARY}, {ASSISTANT REPLIES SUMMARY}, {ASSISTANT THINKING SUMMARY}, {STUB EXAMPLE RULE}, {PREVIOUS SUMMARY RULE}, {PREVIOUS SUMMARY SECTION}. DO NOT reword or delete the {TOKENS} - each is replaced by the per-model option sentence (the editable wording of every token lives in the Category templates screen just below); a deleted token is auto-added at the end with a note. Everything else you write is sent word for word. Enter opens the template submenu: edit, read-only previews of the active / user / extension template, save the current as the user default, and both resets (each asks first).",
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
			submenu: (_current: string, done: (selectedValue?: string) => void) => new LogView(done),
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
			update((c) => void (c.enabled = value === "on"));
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
		} else if (id === "compactNow") {
			// Manual compaction through pi (same as /compact): it fires session_before_compact and the
			// fork's rotation answers. Close the board first so the progress is visible.
			const ctx = this.ctx;
			this.finish({ action: "compact-now" });
			ctx?.compact?.({
				onComplete: () => ctx?.ui?.notify?.("Manual compaction finished.", "info"),
				onError: (err: any) => ctx?.ui?.notify?.(`Manual compaction failed: ${String(err?.message ?? err).slice(0, 160)}`, "error"),
			});
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
	constructor(private done: () => void) {}
	invalidate(): void {}
	handleInput(data: string): void {
		const k = keyOf(data);
		if (k === "escape" || k === "enter" || k === "q") this.done();
	}
	render(width: number): string[] {
		const lines = ["Log — Esc to go back", "", ...tailLogPretty(16).split("\n")];
		return lines.map((l) => truncateToWidth(l, Math.max(60, width)));
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

/** Edit the Additional instruction in pi's multi-line editor; empty text = default again. */
/** Read-only: one of the three template layers (active / user default / extension default). */
async function previewTemplate(ctx: Ctx, which: string): Promise<void> {
	const cfg = loadConfig();
	const ext = DEFAULT_TEMPLATE_TEXT;
	const usr = cfg.templateUserDefaults?.trim() ? cfg.templateUserDefaults! : null;
	const active =
		cfg.additionalInstruction !== null && cfg.additionalInstruction !== undefined
			? cfg.additionalInstruction.trim() || ext
			: usr ?? ext;
	const text =
		which === "ext"
			? ext
			: which === "user"
				? usr ?? "(no user default saved)\n\nWhen the active template is empty, this slot follows the extension default. Save one with 'Save current as user default' in the template submenu."
				: active;
	await ctx.ui?.editor?.(
		`READ-ONLY PREVIEW — summary template (${which === "ext" ? "extension default" : which === "user" ? "user default" : "active"})`,
		[`READ-ONLY PREVIEW - changes are DISCARDED; nothing is sent.`, ``, text].join("\n"),
	);
}

/** The user-default actions for the main template, each confirmed with pi's select (the board is
 *  closed here, so overlays are safe). */
async function templateDefaultAction(ctx: Ctx, action: string): Promise<void> {
	const cfg = loadConfig();
	if (action === "save-template-user-default") {
		const active =
			cfg.additionalInstruction !== null && cfg.additionalInstruction !== undefined
				? cfg.additionalInstruction.trim() || DEFAULT_TEMPLATE_TEXT
				: cfg.templateUserDefaults?.trim() || DEFAULT_TEMPLATE_TEXT;
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
		update((c) => void (c.additionalInstruction = null));
		ctx.ui?.notify?.("The template now follows the user default.", "info");
		return;
	}
	const ok = await ctx.ui?.select?.("Reset the template to the EXTENSION default? (the active custom text is dropped; your user default is kept)", ["Confirm - reset to extension default", "Cancel"]);
	if (ok !== "Confirm - reset to extension default") return;
	update((c) => void (c.additionalInstruction = ""));
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
	await ctx.ui?.editor?.(
		`READ-ONLY PREVIEW — ${label}: ${target === "ext" ? "extension" : "user"} defaults`,
		[`READ-ONLY PREVIEW - changes are DISCARDED; nothing is sent.`, ``, lines].join("\n"),
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
		const text = await ctx.ui?.editor?.(title, prefill);
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
		update((c) => {
			if (!t) c.additionalInstruction = null;
			else if (t === ext.trim()) c.additionalInstruction = "";
			else if (c.templateUserDefaults && t === c.templateUserDefaults.trim()) c.additionalInstruction = null;
			else c.additionalInstruction = String(text);
		});
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
		if (result.action === "edit-instruction") {
			await editInstruction(ctx);
			restore = { rowId: "instruction" };
			continue;
		}
		if (result.action?.startsWith("preview-template:")) {
			await previewTemplate(ctx, result.action.slice("preview-template:".length));
			restore = { rowId: "instruction" };
			continue;
		}
		if (result.action === "compact-stock-now") {
			// The compaction was started by the dispatch (ctx.compact with the skip flag set); the
			// board stays closed while it runs.
			return;
		}
		if (result.action === "save-template-user-default" || result.action === "reset-template-user-default" || result.action === "reset-template-extension-default") {
			await templateDefaultAction(ctx, result.action);
			restore = { rowId: "instruction" };
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
			if (id === "startPct" || id === "resMin" || id === "resMax") restore = { autoMenu: true };
			else if (id === "keepPct" || id === "keepMin" || id === "keepMax") restore = { preserveMenu: true };
			else if (id.startsWith("elision")) restore = { elisionMenu: true };
			else restore = { rowId: id };
			continue;
		}
		if (result.action?.startsWith("edit-model-number:")) {
			// The ref may itself contain ':' (llama-server URLs), so split from the RIGHT.
			const body = result.action.slice("edit-model-number:".length);
			const at = body.lastIndexOf(":");
			if (at > 0) await editModelNumber(ctx, body.slice(0, at), body.slice(at + 1));
			restore = at > 0 ? { orderRef: body.slice(0, at), optionKey: body.slice(at + 1) } : undefined;
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
		if (result.action === "edit-nothinktag") {
			await editNoThinkTag(ctx);
			restore = { rowId: "marker" };
			continue;
		}
		if (result.action?.startsWith("edit-sampling:")) {
			const ref = result.action.slice("edit-sampling:".length);
			await editSamplingFlags(ctx, ref);
			restore = { orderRef: ref, optionKey: "sampling" };
			continue;
		}
		restore = undefined;
		return;
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
async function askNumber(ctx: Ctx, title: string, current: number, lo: number, hi: number, unit: string): Promise<number | undefined> {
	const res = await ctx.ui.input(title, String(current));
	if (res === undefined) return undefined;
	const n = Math.round(Number(String(res).trim()));
	if (!Number.isFinite(n) || n < lo || n > hi) {
		ctx.ui.notify(`Type a whole number between ${fmt(lo)} and ${fmt(hi)} ${unit}.`, "error");
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
		title = "Summary min — % of the model's window (the acceptance minimum)";
		cur = o.summaryMinPercent; lo = 1; hi = 99; unit = "percent";
		save = (v) => setModelOptions(ref, { summaryMinPercent: v });
		shown = (v) => `${v}%`;
	} else if (key === "summaryMax") {
		title = "Summary max — % of the model's window (the acceptance maximum)";
		cur = o.summaryMaxPercent; lo = 1; hi = 99; unit = "percent";
		save = (v) => setModelOptions(ref, { summaryMaxPercent: v });
		shown = (v) => `${v}%`;
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
	} else return;
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
	startPct: { title: "Start auto-compaction at % of context window (1-99)", lo: 1, hi: 99, unit: "percent", def: 82, apply: (v) => { update((c) => void (c.scaling.startPercent = v)); return undefined; } },
	resMin: { title: "Reserve min tokens", lo: 0, hi: 10_000_000, unit: "tokens", def: 12_288, apply: (v) => { update((c) => void (c.scaling.reserveMin = v)); return undefined; } },
	resMax: { title: "Reserve max tokens", lo: 0, hi: 10_000_000, unit: "tokens", def: 80_000, apply: (v) => { update((c) => void (c.scaling.reserveMax = v)); return undefined; } },
	keepPct: { title: "Keep recent % of window (1-99)", lo: 1, hi: 99, unit: "percent", def: 20, apply: (v) => { update((c) => void (c.scaling.keepRecentPercent = v)); return undefined; } },
	keepMin: { title: "Keep recent min tokens", lo: 0, hi: 10_000_000, unit: "tokens", def: 16_384, apply: (v) => { update((c) => void (c.scaling.keepRecentMin = v)); return undefined; } },
	keepMax: { title: "Keep recent max tokens", lo: 0, hi: 10_000_000, unit: "tokens", def: 100_000, apply: (v) => { update((c) => void (c.scaling.keepRecentMax = v)); return undefined; } },
	elisionStart: { title: "Elision start", lo: 1, hi: 99, unit: "percent of window", def: 70, apply: (v) => { update((c) => void (c.elision = { ...c.elision, softPercent: v })); return undefined; } },
	elisionProtect: { title: "Elision protect (tool results)", lo: 0, hi: 1_000_000, unit: "tokens", def: 20_000, apply: (v) => { update((c) => void (c.elision = { ...c.elision, protectRecentTokens: v })); return undefined; } },
	elisionTail: { title: "Stub tail", lo: 0, hi: 100_000, unit: "chars (0 = none; 'full' is a separate row value)", def: 200, apply: (v) => { update((c) => void (c.elision = { ...c.elision, stubTailChars: v })); return undefined; } },
	elisionSavings: { title: "Min batch savings", lo: 0, hi: 100_000, unit: "tokens (0 = fire immediately)", def: 2_000, apply: (v) => { update((c) => void (c.elision = { ...c.elision, minSavingsTokens: v })); return undefined; } },
	elisionResults: { title: "Min stub batch", lo: 1, hi: 50, unit: "results", def: 5, apply: (v) => { update((c) => void (c.elision = { ...c.elision, minResultsToStub: v })); return undefined; } },
	elisionStop: { title: "Elision stop gap", lo: 0, hi: 100_000, unit: "tokens", def: 4_000, apply: (v) => { update((c) => void (c.elision = { ...c.elision, stopGapTokens: v })); return undefined; } },
	chatCap: { title: "Chat max_tokens cap", lo: 0, hi: 2_000_000, unit: "tokens (0 = no cap)", def: 65_536, apply: (v) => { update((c) => void (c.chatMaxTokensCap = v)); return undefined; } },
	retries: { title: "Retries per model", lo: 0, hi: 99, unit: "tries (0 = no retries)", def: 2, apply: (v) => { update((c) => void (c.retries = v)); return undefined; } },
	piReserve: { title: "pi's reserveTokens (stock compaction fallback)", lo: 0, hi: 10_000_000, unit: "tokens", def: 16_384, apply: (v) => { writePiReserveTokens(v); return undefined; } },
};

async function editNumber(ctx: Ctx, id: string): Promise<void> {
	const d = NUMBER_EDITORS[id];
	if (!d) return;
	const cfg = loadConfig();
	const current =
		id === "retryWait" ? cfg.retryDelaySeconds
		: id === "startPct" ? cfg.scaling.startPercent
		: id === "resMin" ? cfg.scaling.reserveMin
		: id === "resMax" ? cfg.scaling.reserveMax
		: id === "keepPct" ? cfg.scaling.keepRecentPercent
		: id === "keepMin" ? cfg.scaling.keepRecentMin
		: id === "keepMax" ? cfg.scaling.keepRecentMax
		: id === "piReserve" ? readPiReserveTokens()
		: id === "elisionStart" ? cfg.elision.softPercent
		: id === "elisionProtect" ? cfg.elision.protectRecentTokens
		: id === "elisionTail" ? (typeof cfg.elision.stubTailChars === "number" ? cfg.elision.stubTailChars : 0)
		: id === "elisionSavings" ? cfg.elision.minSavingsTokens
		: id === "elisionResults" ? cfg.elision.minResultsToStub
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
				label: `Draft block - now ${o.draft}`,
				hint: "A hidden analysis before the summary in the same call, stripped before storing. off / mini / full. Online defaults full, local off (a full draft doubles a slow summary).",
				run: async () => {
					const chosen = await ctx.ui.select("Draft block", withCurrent(DRAFT_CYCLE, o.draft));
					if (chosen !== undefined) setModelOptions(ref, { draft: (DRAFT_CYCLE as string[]).includes(chosen) ? (chosen as Draft) : "off" });
				},
			},
			{
				label: `Summary timeout period - now ${timeoutLabel(o.timeoutMs)}`,
				hint: "2 → 3 → 5 → 7 → 10 → 15 → 20 → 30 → 45 → 60 min → unlimited. After this the next model is asked.",
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
				label: `Summary min - now ${o.summaryMinPercent}% of the ctx window`,
				hint: "The acceptance minimum: a summary shorter than this is thrown away. Percent of the model's ctx window. Free numeric entry.",
				run: async () => {
					const v = await askNumber(ctx, "Summary min — % of the ctx window (default 5)", o.summaryMinPercent, 1, 99, "percent");
					if (v !== undefined) setModelOptions(ref, { summaryMinPercent: v });
				},
			},
			{
				label: `Summary max - now ${o.summaryMaxPercent}% of the ctx window`,
				hint: "The acceptance maximum: a summary longer than this is thrown away. Percent of the model's ctx window. Free numeric entry.",
				run: async () => {
					const v = await askNumber(ctx, "Summary max — % of the ctx window (default 50)", o.summaryMaxPercent, 1, 99, "percent");
					if (v !== undefined) setModelOptions(ref, { summaryMaxPercent: v });
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
				label: `Extension active - ${cfg.enabled ? "ON" : "OFF"}`,
				hint: `Tried in this order until one writes a usable summary: ${rotationText(cfg, ctx)}.`,
				run: () => toggleEnabled(),
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
				run: () => update((c) => void (c.elision = { ...c.elision, enabled: !c.elision.enabled })),
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
		`  trigger: estimate crosses window - ${fmt(reserve)} tok · auto-compact while idle: ${cfg.autoCompact ? "ON" : "off"} · chat max_tokens cap: ${cfg.chatMaxTokensCap > 0 ? fmt(cfg.chatMaxTokensCap) : "off"}`,
	];
	const refs = Object.keys(cfg.models);
	refs.forEach((ref, i) => {
		const model = lookupRef(ref, ctx);
		const o = optionsFor(cfg, ref, model);
		const th = model ? effectiveThinking(o.thinking, model) : { level: o.thinking, note: "" };
		lines.push(
			`  ${i + 1}. ${slotLabel(ref, model)} — summary aim ${o.summaryPercent}% of the ctx window, limits min ${o.summaryMinPercent}% / max ${o.summaryMaxPercent}% of it, thinking ${th.level}, draft ${o.draft}, timeout ${timeoutLabel(o.timeoutMs)}, sampling ${samplingLabel(o.sampling)}`,
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
let lastAutoTriggerMs = 0;
/** True while this extension's rotation is running: the watcher must not fire into it (pi's
 *  compact() is single-flight - a re-trigger KILLS the in-flight attempt, which is exactly the
 *  minute-by-minute kill storm of 2026-10-02 17:01-17:05). */
let compactionInFlight = false;
/** The one-shot skip for "Start pi's built-in compaction now": the next session_before_compact is
 *  left to pi itself. */
let skipNextStockCompaction = false;
const WATCHER_INTERVAL_MS = 5_000;

function watcherTick(): void {
	const ctx = watcherCtx;
	if (!ctx) return;
	let cfg: Config;
	try {
		cfg = loadConfig();
	} catch {
		return;
	}
	if (!cfg.enabled || !cfg.autoCompact) return;
	if (compactionInFlight) return; // never re-trigger while a compaction is running
	try {
		const window = (ctx as any)?.model?.contextWindow ?? 0;
		if (!(window > 0)) return;
		if ((ctx as any)?.isIdle?.() === false) return; // never abort a running generation
		const usage = (ctx as any)?.getContextUsage?.();
		const tokens = usage?.tokens;
		if (typeof tokens !== "number" || !(tokens > 0)) return; // unknown until the next response
		const line = window - extReserve(ctx);
		if (tokens < line) return;
		const now = Date.now();
		if (now - lastAutoTriggerMs < 60_000) return; // one trigger per minute, no hammering
		lastAutoTriggerMs = now;
		logLine(cfg, { event: "auto_trigger", estimate: tokens, line, window, reserve: extReserve(ctx) });
		ctx.ui?.notify?.(`Context at ${fmt(tokens)} tokens (trigger line ${fmt(line)}) - compacting now.`, "info");
		(ctx as any).compact?.({
			onComplete: () => ctx.ui?.notify?.("Automatic compaction finished.", "info"),
			onError: (err: any) => ctx.ui?.notify?.(`Automatic compaction failed: ${String(err?.message ?? err).slice(0, 160)}`, "error"),
		});
	} catch {
		/* the watcher must never break the session */
	}
}

let watcherTimer: ReturnType<typeof setInterval> | undefined;

/** pi's docs: do not start timers in the factory, because some invocations load extensions without
 *  a session (print mode, one-shot tries). Start the watcher lazily on the first live request
 *  instead of at module load. */
function ensureWatcherTimer(): void {
	if (watcherTimer) return;
	watcherTimer = setInterval(watcherTick, WATCHER_INTERVAL_MS);
	(watcherTimer as unknown as { unref?: () => void })?.unref?.();
}

export default function piCompactPlusExtension(pi: ExtensionAPI): void {
	// ELISION: rule-based stubbing of old tool results in every live request (see applyElision).
	// The handler's returned message list is what actually gets sent; stored history is untouched,
	// so pi's own compaction machinery always sees the raw originals. The handler also captures the
	// freshest extension ctx for the auto-compact watcher above.
	pi.on("context", (event: any, ctx: any) => {
		watcherCtx = ctx;
		ensureWatcherTimer();
		return applyElision(event.messages, ctx);
	});

	// CHAT max_tokens cap: pi sizes each chat request's generation permission as
	// window - context estimate - 4,096, which balloons on big-window models (943,718 on
	// glm-flash-latest) and then gets the whole request rejected whenever pi's estimate
	// undercounts the real wire by more than the 4,096 cushion - the overflow class behind the
	// 630k and 295k emergency compactions (the estimate undercounted by ~69k on 2026-10-01
	// 00:25). With the cap the request always fits until real ~ window - cap. The fork's own
	// compaction requests size themselves via genCap and never pass this hook (only the agent
	// loop carries onPayload).
	pi.on("before_provider_request", (event: any) => {
		const cap = loadConfig().chatMaxTokensCap;
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
		logLine(loadConfig(), { event: "cap", model: String(p.model ?? "").slice(0, 80), capped: cap });
		return p;
	});

	pi.on("session_before_compact", async (event: any, ctx: any) => {
		// The file is read again here every time, so an edit to it takes effect without /reload.
		if (skipNextStockCompaction) {
			skipNextStockCompaction = false;
			return undefined;
		}
		compactionInFlight = true;
		try {
			return await handleCompaction(event, ctx);
		} finally {
			compactionInFlight = false;
		}
	});

	async function handleCompaction(event: any, ctx: any) {
		const cfg = loadConfig();
		const refs = Object.keys(cfg.models);
		if (!cfg.enabled || refs.length === 0) return undefined;

		// The arithmetic of this compaction, recorded so a wrong choice can be traced afterwards.
		const history = historyTokensOnly(event.preparation);
		const overhead = fixedRequestTokens(ctx);
		const fill = realContextTokens(ctx);
		logLine(cfg, {
			event: "start",
			reason: event.reason,
			historyTokens: history,
			overheadTokens: overhead,
			contextFill: fill,
			sessionWindow: ctx.model?.contextWindow ?? 0,
			reserveTokens: numOr(event.preparation?.settings?.reserveTokens, DEFAULT_RESERVE_TOKENS),
			keepRecentTokens: numOr(event.preparation?.settings?.keepRecentTokens, DEFAULT_KEEP_RECENT_TOKENS),
			models: refs,
			sessionModel: ctx.model ? refOf(ctx.model) : "?",
		});
		ctx.ui?.setStatus?.("compact-plus", `compacting via ${rotationText(cfg, ctx)} ...`);
		let outcome;
		try {
			outcome = await summarizeWithRotation(event.preparation, event.customInstructions, cfg, ctx, event.signal);
		} finally {
			ctx.ui?.setStatus?.("compact-plus", undefined);
		}

		if (!outcome.ok) {
			if (outcome.error === "cancelled") return undefined;
			logLine(cfg, { event: "give_up", error: outcome.error, note: "pi does this compaction itself" });
			ctx.ui?.notify?.(`Compaction models all failed (${outcome.error.slice(0, 220)}). Letting pi do it with the session model.`, "warning");
			return undefined;
		}

		const r = outcome.result;
		// The chained-compaction counter rides in the entry's details (read by the next compaction).
		try {
			r.compaction.details = { ...(r.compaction.details ?? {}), lastTurnNumber: maxTurnInSummary(r.compaction.summary) };
		} catch {
			/* counter is best-effort */
		}
		r.compaction.summary = appendTranscriptPointer(cfg, ctx, r.compaction.summary);
		logLine(cfg, { event: "done", ref: r.ref, status: "ok", measured: r.measured, min: r.min, max: r.max, seconds: r.seconds, reason: event.reason });
		ctx.ui?.notify?.(
			`Summary written by ${slotLabel(r.ref, r.model)} in ${r.seconds}s — ${fmt(r.measured)} tok (accepted ${fmt(r.min)}–${fmt(r.max)}).`,
			"info",
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
					update((c) => void (c.enabled = true));
					return say("Extension active: ON.");
				case "off":
					update((c) => void (c.enabled = false));
					return say("Extension active: OFF — pi uses the session model.");
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
									return `${i + 1}. ${ref} — summary aim ${o.summaryPercent}% of the ctx window, limits min ${o.summaryMinPercent}% / max ${o.summaryMaxPercent}% of it, ${timeoutLabel(o.timeoutMs)}, ${o.thinking}`;
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
					if (!ref) return say("Use: options <provider/model> [thinking <lvl>] [timeout <min|0>] [aim <0.05-0.95>] [min <0.05-0.95>] [max <0.05-0.95>] [draft <off|mini|full>] [inputstubs on|off] [argcap <0-1000|full>] [stubs <verbatim|detailed|summary|brief>] [user <..>] [replies <..>] [thought <..>] [chain <fold|attach|skip>] [tag on|off]");
					if (parts.length === 2) {
						const model = lookupRef(ref, ctx);
						const o = optionsFor(loadConfig(), ref, model);
						return say(`${ref}: thinking ${o.thinking}, timeout ${timeoutLabel(o.timeoutMs)}, summary aim ${o.summaryPercent}% of the ctx window, limits min ${o.summaryMinPercent}% / max ${o.summaryMaxPercent}% of it, draft ${o.draft}, input ${o.inputStubs ? "stubs" : "raw"} (arg cap ${charCapLabel(o.argCap)}), keeps: stubs ${o.preserveStubs} / user ${o.preserveUser} / replies ${o.preserveReplies} / thought ${o.preserveThinking}, chain ${o.chainMode}, tag ${o.noThinkMarker || "(empty)"}`);
					}
					const patch: Partial<ModelOptions> = {};
					for (let i = 2; i < parts.length; i += 2) {
						const key = parts[i]?.toLowerCase();
						const value = parts[i + 1];
						if (!key || value === undefined) break;
						if (key === "thinking" && ALL_LEVELS.includes(value as Level)) patch.thinking = value as Level;
						else if (key === "timeout") patch.timeoutMs = clamp(Math.round(Number(value) * 60_000) || 0, 0, 7_200_000);
						else if (key === "pct") patch.summaryPercent = clamp(Math.round(Number(value)), 1, 99);
						else if (key === "smin") patch.summaryMinPercent = clamp(Math.round(Number(value)), 1, 99);
						else if (key === "smax") patch.summaryMaxPercent = clamp(Math.round(Number(value)), 1, 99);
						else if (key === "draft" && DRAFT_CYCLE.includes(value as Draft)) patch.draft = value as Draft;
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
					return say(`${ref}: thinking ${o2.thinking}, timeout ${timeoutLabel(o2.timeoutMs)}, summary aim ${o2.summaryPercent}% of the ctx window, limits min ${o2.summaryMinPercent}% / max ${o2.summaryMaxPercent}% of it, draft ${o2.draft}, input ${o2.inputStubs ? "stubs" : "raw"} (arg cap ${charCapLabel(o2.argCap)}), keeps: stubs ${o2.preserveStubs} / user ${o2.preserveUser} / replies ${o2.preserveReplies} / thought ${o2.preserveThinking}, chain ${o2.chainMode}, tag ${o2.noThinkMarker || "(empty)"}`);
				}
				case "elision": {
					const sub = (parts[1] ?? "").toLowerCase();
					const setEl = (patch: Partial<ElisionConfig>) => update((c) => void (c.elision = { ...c.elision, ...patch }));
					if (!sub) {
						const e = loadConfig().elision;
						return say(`Elision: ${e.enabled ? "ON" : "OFF"} · start ${e.softPercent}% of window · protect ${fmt(e.protectRecentTokens)} tok · stub tail ${charCapLabel(e.stubTailChars)} chars · min batch savings ${fmt(e.minSavingsTokens)} tok / ${e.minResultsToStub} results · stop gap ${fmt(e.stopGapTokens)} tok.  (elision on|off|start <1-99>|tail <0..1000|full>|protect <tokens>|savings <tokens>|stop <tokens>|results <1-50>)`);
					}
					if (sub === "on") setEl({ enabled: true });
					else if (sub === "off") setEl({ enabled: false });
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
						const instructions = buildInstructions({ sizes, inputTokens, kind: isLocalModel(model) ? "local" : "online", options: profile, additionalInstruction: cfg2.additionalInstruction, customInstructions: undefined });
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
						await ctx.ui?.editor?.(`compact-plus preview for ${ref} (nothing was sent)`, text);
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
						"Use /compact-plus with no argument for the menu, or:\nstatus | on | off | models [add <ref> | remove <ref>] | move <ref> <up|down|n>\noptions <ref> [thinking <lvl>] [timeout <min|0>] [aim <0.05-0.95>] [min <..>] [max <..>] [draft <off|mini|full>] [inputstubs on|off] [argcap <0-1000|full>] [stubs|user|replies|thought <verbatim|detailed|summary|brief>] [chain <fold|attach|skip>] [tag on|off]\nelision [on|off] [start <50-95>] [tail <0..1000|full>] [protect <tokens>] [savings <tokens>]\nretries <0-5> [sec] | marker <text|clear> | instruction [text | reset] | log",
					);
			}
		},
		getArgumentCompletions: (prefix: string) =>
			["status", "on", "off", "models", "move", "options", "elision", "retries", "marker", "instruction", "sampling", "preview", "log"]
				.filter((v) => v.startsWith(prefix))
				.map((v) => ({ value: v, label: v })),
	});
}
