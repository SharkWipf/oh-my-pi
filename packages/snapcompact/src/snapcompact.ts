/**
 * Snapcompact compaction: archive conversation history as dense bitmap images.
 *
 * Instead of asking an LLM to summarize discarded history, the serialized
 * conversation is rendered into PNG frames of pixel-font text that vision
 * models read back directly, like an archivist at a snapcompact frame
 * reader. Frames are `frameSize` wide; their height hugs the text rows
 * actually printed, so a partially filled frame never bills blank rows.
 *
 * The frame shape is provider-aware. Original choices came from the SQuAD
 * prose evals (`packages/snapcompact`, 200k-token monolithic runs); the
 * spacing choices below come from the tool-result legibility bench
 * (`research/toolbench.py`, real search/read/find output with structure QA),
 * which exposed that the prose-tuned dense cells erase the line numbers and
 * indentation that code/search output depends on:
 *
 * - **Anthropic** (`11on16-bw`): 8x13 glyphs on an 11px advance (extra
 *   letter-spacing), black ink. On the tool-result bench, tracking the
 *   readable cell beat plain `8on16-bw` (opus-4.8 f1 .806 vs .755) and far
 *   beat the prior dense `6x12-dim` (.351, which fell below the OCR ~16px/char
 *   floor and abstained). Opus 4.7+/Fable/Mythos ingest high-res natively
 *   (2576px edge, 4,784 visual-token cap), so those lines get 1932px frames:
 *   same bill, fewer frames. Older Claude lines downscale past 1568px.
 * - **Google** (`8on22-bw` @2048): 8x13 glyphs on a 22px pitch (extra line
 *   spacing), black ink. Leading lifted gemini-3.5-flash to f1 .934 vs .807
 *   for `8on16-bw` and .287 for the prior `doc-8on16-sent-dim`. Gemini 3.x
 *   bills a fixed `media_resolution` budget per image (default 1,120 tokens)
 *   regardless of pixels, so the 2048px frame carries more chars at the same
 *   bill.
 * - **OpenAI** (`8on22-bw`): same leading win (gpt-5.5/gpt-5.4-mini). Patch
 *   billing (32px × 1.2, 10k-patch budget at `detail: "original"`) is
 *   area-proportional, so resolution cannot improve chars/$ — 1568 stays.
 *   `detail: "high"` would downgrade (2,500-patch cap); `original` is sent.
 * - **Unknown providers** default to `8on22-bw` with Anthropic-style
 *   visual-token area billing. `providerImageBudget` still caps per-request
 *   images per provider so inline imaging cannot flood a request with
 *   attachments, but the old OpenRouter-specific 8-image cap is gone; routers
 *   now use the same permissive budget as direct Anthropic/Claude lines unless
 *   configured otherwise upstream.
 *
 * The whole pass is local and deterministic — no LLM call, no API key, no
 * latency beyond rendering. Rasterization and PNG encoding happen in native
 * code (`renderSnapcompactPng` in `crates/pi-natives/src/snapcompact.rs`).
 * Frames persist in the compaction entry's `preserveData` and are
 * re-attached to the compaction summary message on every context rebuild.
 */

import type { Api, ImageContent, Message, TextContent } from "@oh-my-pi/pi-ai";
import { compactionSourceKey, type SourceBlockRange, type SourceCoverageRun, type SourceLayoutPart, type SourceMessage, type SourceRange, type SourceRepresentation } from "@oh-my-pi/pi-ai/compaction-source";
import { classifyModel, compareRevision, parseRevision } from "@oh-my-pi/pi-catalog/identity";
import { renderSnapcompactPng, snapcompactSupportedChars } from "@oh-my-pi/pi-natives";
import { formatGroupedPaths, prompt } from "@oh-my-pi/pi-utils";
import { INTENT_FIELD } from "@oh-my-pi/pi-wire";
import fileOperationsTemplate from "./prompts/file-operations.md" with { type: "text" };
import snapcompactSummaryPrompt from "./prompts/snapcompact-summary.md" with { type: "text" };

// ============================================================================
// Shapes
// ============================================================================

/** One eval-validated frame shape: font, cell, ink, repetition, and size. */
export interface Shape {
	/** Bundled font in the native renderer. */
	font: "5x8" | "8x8" | "6x12" | "8x13" | "silver";
	/** Target cell advance in pixels; differing from the font's natural cell
	 *  renders via Lanczos stretch (anti-aliased RGB frame). */
	cellWidth: number;
	/** Target cell pitch in pixels. */
	cellHeight: number;
	/** `false` → glyphs drawn at natural size on the cell pitch (8on16);
	 *  `true`/`undefined` → legacy auto Lanczos stretch when cell ≠ natural. */
	stretch?: boolean;
	/** Ink: `sent` cycles six hues at sentence boundaries; `bw` is black. */
	variant: "sent" | "bw";
	/** Print stopwords in dim ink (research `dim`/`sent-dim` variants). */
	stopwordDim?: boolean;
	/** 1/undefined = row-major grid; 2 = two word-wrapped newspaper columns
	 *  (research `doc`). */
	columns?: number;
	/** Each text line is printed this many times; copies after the first sit
	 *  on a pale highlight band (redundancy coding). */
	lineRepeat: number;
	/** Frame edge in pixels. */
	frameSize: number;
	/** Per-frame billed-token estimate for the shape's target provider. */
	frameTokenEstimate: number;
	/** Resolution hint attached to frame images (OpenAI-only). */
	imageDetail?: ImageContent["detail"];
}

/** Geometry half of a {@link Shape}: everything except provider billing. */
export type ShapeGeometry = Omit<Shape, "frameTokenEstimate" | "imageDetail">;

/**
 * Frame variants exercised by the SQuAD evals in `research/` that the native
 * renderer reproduces faithfully, keyed by their research names. Font codes:
 * `8x8u` unscii square cell, `8x8r` unscii with every line printed twice
 * (redundancy coding), `6x6u` unscii Lanczos-squeezed to 6x6 (densest
 * readable cell), `5x8` the X.org legacy font on its 2576px frame, `6x12`
 * and `8x13` the X.org misc fonts, `8on16` 8x13 glyphs on an 8x16 cell pitch
 * (no stretch, extra leading), `8on22` the same glyphs on a 22px pitch (more
 * leading), `11on16` the same glyphs on an 11px advance (more tracking),
 * `silver16` the embedded Silver TrueType font on a 16px grid for CJK and
 * other non-Latin text, and `doc-` prefixed shapes a two-column word-wrapped
 * newspaper layout. Ink: `sent` cycles six hues at sentence boundaries, `bw`
 * is plain black, `-dim` suffix prints stopwords in gray.
 */
export const SHAPE_VARIANTS = {
	"8x8r-bw": { font: "8x8", cellWidth: 8, cellHeight: 8, variant: "bw", lineRepeat: 2, frameSize: 1568 },
	"8x8r-sent": { font: "8x8", cellWidth: 8, cellHeight: 8, variant: "sent", lineRepeat: 2, frameSize: 1568 },
	"8x8u-bw": { font: "8x8", cellWidth: 8, cellHeight: 8, variant: "bw", lineRepeat: 1, frameSize: 1568 },
	"8x8u-sent": { font: "8x8", cellWidth: 8, cellHeight: 8, variant: "sent", lineRepeat: 1, frameSize: 1568 },
	"6x6u-bw": { font: "8x8", cellWidth: 6, cellHeight: 6, variant: "bw", lineRepeat: 1, frameSize: 1568 },
	"6x6u-sent": { font: "8x8", cellWidth: 6, cellHeight: 6, variant: "sent", lineRepeat: 1, frameSize: 1568 },
	"5x8-bw": { font: "5x8", cellWidth: 5, cellHeight: 8, variant: "bw", lineRepeat: 1, frameSize: 2576 },
	"5x8-sent": { font: "5x8", cellWidth: 5, cellHeight: 8, variant: "sent", lineRepeat: 1, frameSize: 2576 },
	"6x12-dim": {
		font: "6x12",
		cellWidth: 6,
		cellHeight: 12,
		variant: "bw",
		stopwordDim: true,
		lineRepeat: 1,
		frameSize: 1568,
	},
	"8x13-bw": { font: "8x13", cellWidth: 8, cellHeight: 13, variant: "bw", lineRepeat: 1, frameSize: 1568 },
	"8on16-bw": {
		font: "8x13",
		cellWidth: 8,
		cellHeight: 16,
		stretch: false,
		variant: "bw",
		lineRepeat: 1,
		frameSize: 1568,
	},
	"8on22-bw": {
		font: "8x13",
		cellWidth: 8,
		cellHeight: 22,
		stretch: false,
		variant: "bw",
		lineRepeat: 1,
		frameSize: 1568,
	},
	"11on16-bw": {
		font: "8x13",
		cellWidth: 11,
		cellHeight: 16,
		stretch: false,
		variant: "bw",
		lineRepeat: 1,
		frameSize: 1568,
	},
	"silver16-bw": {
		font: "silver",
		cellWidth: 16,
		cellHeight: 16,
		variant: "bw",
		lineRepeat: 1,
		frameSize: 1568,
	},
	"doc-8on16-bw": {
		font: "8x13",
		cellWidth: 8,
		cellHeight: 16,
		stretch: false,
		variant: "bw",
		columns: 2,
		lineRepeat: 1,
		frameSize: 1568,
	},
	"doc-8on16-sent": {
		font: "8x13",
		cellWidth: 8,
		cellHeight: 16,
		stretch: false,
		variant: "sent",
		columns: 2,
		lineRepeat: 1,
		frameSize: 1568,
	},
	"doc-8on16-sent-dim": {
		font: "8x13",
		cellWidth: 8,
		cellHeight: 16,
		stretch: false,
		variant: "sent",
		stopwordDim: true,
		columns: 2,
		lineRepeat: 1,
		frameSize: 1568,
	},
} as const satisfies Record<string, ShapeGeometry>;

/** Research name of one renderable frame variant. */
export type ShapeVariantName = keyof typeof SHAPE_VARIANTS;

/** All variant names, in declaration order (for settings enums). */
export const SHAPE_VARIANT_NAMES = Object.keys(SHAPE_VARIANTS) as readonly ShapeVariantName[];

/** Runtime guard for variant names loaded from config. */
export function isShapeVariantName(value: unknown): value is ShapeVariantName {
	return typeof value === "string" && value in SHAPE_VARIANTS;
}

/** Provider families with distinct image billing. */
type BillingFamily = "anthropic" | "google" | "openai" | "unknown";

function billingFamily(api?: Api): BillingFamily {
	switch (api) {
		case "anthropic-messages":
		case "bedrock-converse-stream":
			return "anthropic";
		case "openai-completions":
		case "openai-responses":
		case "openai-codex-responses":
		case "azure-openai-responses":
			return "openai";
		case "google-generative-ai":
		case "google-gemini-cli":
		case "google-vertex":
			return "google";
		default:
			// Unknown APIs share Anthropic's pixel-area pricing as the safe ceiling.
			return "unknown";
	}
}

/**
 * Per-frame billing for a square frame of edge `frameSize`, by family.
 * Formulas verified against live bills in the resolution benchmarks:
 * - Anthropic: 28px patches, capped at 4,784 visual tokens (the API
 *   downscales past the cap; 1568 → 3,136 measured) + 5% margin.
 * - Google: Gemini 3.x bills a fixed `media_resolution` budget per image —
 *   default HIGH = 1,120 tokens — regardless of pixel size.
 * - OpenAI: 32px patches × 1.2 flagship multiplier, 10,000-patch budget at
 *   `detail: "original"` (1568 → 2,881 measured).
 */
function familyBilling(family: BillingFamily, frameSize: number): Pick<Shape, "frameTokenEstimate" | "imageDetail"> {
	switch (family) {
		case "google":
			return { frameTokenEstimate: 1120 };
		case "openai": {
			const patches = Math.min(Math.ceil(frameSize / 32) ** 2, 10_000);
			return { frameTokenEstimate: Math.ceil(patches * 1.2), imageDetail: "original" };
		}
		default: {
			const patches = Math.min(Math.ceil(frameSize / 28) ** 2, 4784);
			return { frameTokenEstimate: Math.ceil(patches * 1.05) };
		}
	}
}

/** Attach a provider family's billing to a variant geometry. */
function priceShape(base: ShapeGeometry, family: BillingFamily): Shape {
	return { ...base, ...familyBilling(family, base.frameSize) };
}

/** Eval-validated shapes, keyed by the provider family they won on. */
export const SHAPES = {
	/** `11on16-bw`: 8x13 glyphs on an 11px advance (extra tracking), black ink.
	 *  Tool-result legibility bench (real search/read/find output, structure QA)
	 *  on opus-4.8: f1 .806 vs .755 for plain `8on16-bw` and .351 for the prior
	 *  `6x12-dim` default — letter-spacing the readable cell wins; the dense
	 *  6x12 was below the OCR ~16px/char floor and abstained. */
	anthropic: priceShape(SHAPE_VARIANTS["11on16-bw"], "anthropic"),
	/** `8on22-bw`: 8x13 glyphs on a 22px pitch (extra leading), black ink.
	 *  Tool-result legibility bench on gemini-3.5-flash: f1 .934 vs .807 for
	 *  plain `8on16-bw` and .287 for the prior `doc-8on16-sent-dim`; the
	 *  line-spacing reduces row crowding so line numbers stay legible. */
	google: priceShape(SHAPE_VARIANTS["8on22-bw"], "google"),
	/** `8on22-bw`: 8x13 glyphs on a 22px pitch (extra leading), black ink.
	 *  Same line-spacing win for OpenAI; bench on gpt-5.5/gpt-5.4-mini showed
	 *  leading lifts recall on the readable cell over plain `8on16-bw`. */
	openai: priceShape(SHAPE_VARIANTS["8on22-bw"], "openai"),
	/** Original 5x8 X.org shape (pre-shape-table sessions rendered this). */
	legacy: priceShape(SHAPE_VARIANTS["5x8-sent"], "anthropic"),
} satisfies Record<string, Shape>;

/** Runtime guard for shape overrides loaded from config or preserve data. */
export function isShape(value: unknown): value is Shape {
	if (!value || typeof value !== "object") return false;
	const shape = value as Record<string, unknown>;
	const font = shape.font;
	const variant = shape.variant;
	const detail = shape.imageDetail;
	return (
		(font === "5x8" || font === "8x8" || font === "6x12" || font === "8x13" || font === "silver") &&
		typeof shape.cellWidth === "number" &&
		shape.cellWidth > 0 &&
		typeof shape.cellHeight === "number" &&
		shape.cellHeight > 0 &&
		(shape.stretch === undefined || typeof shape.stretch === "boolean") &&
		(variant === "sent" || variant === "bw") &&
		(shape.stopwordDim === undefined || typeof shape.stopwordDim === "boolean") &&
		(shape.columns === undefined || shape.columns === 1 || shape.columns === 2) &&
		typeof shape.lineRepeat === "number" &&
		shape.lineRepeat > 0 &&
		typeof shape.frameSize === "number" &&
		shape.frameSize > 0 &&
		typeof shape.frameTokenEstimate === "number" &&
		shape.frameTokenEstimate > 0 &&
		(detail === undefined || detail === "auto" || detail === "low" || detail === "high" || detail === "original")
	);
}

/** Eval-winning variant per provider family (billing fallback when the
 *  model id matches no known reader line). */
const FAMILY_VARIANT: Record<BillingFamily, ShapeVariantName> = {
	anthropic: "11on16-bw",
	google: "8on22-bw",
	openai: "8on22-bw",
	unknown: "8on22-bw",
};

/** Denser companion variant per family for the foveated archive middle: same
 *  pixels (identical per-frame bill) but a tighter 8px cell, trading some
 *  legibility for ~40% more chars per frame so the least-important middle of a
 *  long archive compresses into fewer frames. */
const FAMILY_VARIANT_LOW: Record<BillingFamily, ShapeVariantName> = {
	anthropic: "8on16-bw",
	google: "8on16-bw",
	openai: "8on16-bw",
	unknown: "8on16-bw",
};

const FAMILY_SHAPE: Record<BillingFamily, Shape> = {
	anthropic: SHAPES.anthropic,
	google: SHAPES.google,
	openai: SHAPES.openai,
	unknown: priceShape(SHAPE_VARIANTS["8on22-bw"], "unknown"),
};

/** One model line's ideal format: variant plus an optional frame-size
 *  override when the line reads larger frames at no extra cost. */
export interface IdealShape {
	variant: ShapeVariantName;
	frameSize?: number;
}

/** Eval-winning format per model line. The wire API only identifies the
 *  gateway — a Claude served through Vertex or OpenRouter still reads best
 *  with its own shape. Classified Anthropic models use the shared catalog
 *  identity parser; remaining model lines use first-match regex rules and fall
 *  back to the API family's winner at the standard 1568px frame. */
const HIGH_RES_ANTHROPIC_VARIANT = { variant: "11on16-bw", frameSize: 1932 } as const satisfies IdealShape;
const MODEL_VARIANTS: readonly (readonly [RegExp, IdealShape])[] = [
	// Versionless Fable/Mythos aliases (e.g. `claude-fable-latest`) never parse
	// a numeric version, so keep them on the high-res tier by name — every
	// Fable/Mythos line reads it natively.
	[/claude.*(fable|mythos)/i, HIGH_RES_ANTHROPIC_VARIANT],
	// Older Claude lines downscale past 1568px — keep the safe size.
	[/claude/i, { variant: "11on16-bw" }],
	// Gemini 3.x bills a fixed 1,120-token budget per image regardless of
	// pixels: 2048px packs more chars per frame at the same bill.
	[/gemini/i, { variant: "8on22-bw", frameSize: 2048 }],
	// gpt-5.5 patch billing is area-proportional; 1568 is already optimal.
	[/gpt|codex/i, { variant: "8on22-bw" }],
	// kimi-k3 chunked bench: `8on22-bw` scored f1 .915 @ $0.66 vs .813 on `8on16-bw` ($0.70);
	// 1568 wins on chars/$ (image processor downscales past 1792px).
	[/kimi/i, { variant: "8on22-bw" }],
	// glm-4.6v .780 mono via direct vendor routing.
	[/glm/i, { variant: "8on16-bw" }],
];

/** Eval-ideal format for a model id, or undefined when unmeasured. */
export function idealShapeVariant(modelId: string): IdealShape | undefined {
	const identity = classifyModel("", modelId, { lenient: true });
	const revision = identity.revision === undefined ? undefined : parseRevision(identity.revision);
	const opusFloor = parseRevision("4.7");
	if (
		identity.class === "anthropic" &&
		(identity.family === "fable" ||
			identity.family === "mythos" ||
			(identity.family === "opus" &&
				revision !== undefined &&
				opusFloor !== undefined &&
				compareRevision(revision, opusFloor) >= 0))
	) {
		// Opus 4.7+ and Fable/Mythos read high-res natively: same recall and
		// cost as 1568, a third fewer frames. 1932 is the largest *square* not
		// downscaled under Anthropic's 4,784 visual-token cap ((1932/28)² =
		// 69² = 4,761 ≤ 4,784 28px patches), and staying below 2000px clears
		// the stricter ≤2000px limit for requests with more than 20 images.
		return HIGH_RES_ANTHROPIC_VARIANT;
	}
	return MODEL_VARIANTS.find(([pattern]) => pattern.test(modelId))?.[1];
}

/** What will read the frames: the wire API (billing) and model id (shape). */
export interface ShapeTarget {
	api?: Api;
	id?: string;
}

/**
 * Pick the frame shape for a reader. An explicit `variant` (anything but
 * `"auto"`) forces that geometry; otherwise the model id selects the
 * eval-winning shape — and frame size — for its model line, falling back to
 * the API family's winner when the model is unmeasured. Billing (token
 * estimate, detail hint) always follows the API family actually carrying
 * the request, computed for the resolved frame size. Accepts a full pi-ai
 * `Model` or any `{ api, id }` subset.
 */
export function resolveShape(model?: ShapeTarget, variant?: ShapeVariantName | "auto"): Shape {
	const family = billingFamily(model?.api);
	if (variant && variant !== "auto") return priceShape(SHAPE_VARIANTS[variant], family);
	const ideal = model?.id ? idealShapeVariant(model.id) : undefined;
	const name = ideal?.variant ?? FAMILY_VARIANT[family];
	if (name === FAMILY_VARIANT[family] && ideal?.frameSize === undefined) return FAMILY_SHAPE[family];
	const base = SHAPE_VARIANTS[name];
	return priceShape(ideal?.frameSize ? { ...base, frameSize: ideal.frameSize } : base, family);
}

const CJK_HEAVY_MIN_WIDE_CHARS = 8;
const CJK_HEAVY_WIDE_RATIO = 0.25;

function isCjkHeavyText(text: string): boolean {
	const chars = normalizedInputChars(text);
	let graphicChars = 0;
	let wideChars = 0;
	for (const ch of chars) {
		if (ch === " " || ch === DIM_ON || ch === DIM_OFF || ch === NEWLINE_GLYPH) continue;
		const cp = ch.codePointAt(0);
		if (cp === undefined || UNRENDERABLE.test(ch)) continue;
		graphicChars++;
		if (isWideCodePoint(cp)) wideChars++;
	}
	return wideChars >= CJK_HEAVY_MIN_WIDE_CHARS && wideChars / graphicChars >= CJK_HEAVY_WIDE_RATIO;
}

/**
 * Pick the frame shape for `text`. Explicit variants remain forced. Auto first
 * resolves the model/provider default, then selects the Silver CJK grid when
 * the default font cannot safely render the text or wide CJK glyphs dominate
 * the transcript and Silver can render it safely.
 */
export function resolveShapeForText(text: string, model?: ShapeTarget, variant?: ShapeVariantName | "auto"): Shape {
	const shape = resolveShape(model, variant);
	if (variant && variant !== "auto") return shape;
	const silver = resolveShape(model, "silver16-bw");
	if (!scanRenderability(text, { shape }).isSafe) {
		return scanRenderability(text, { shape: silver }).isSafe ? silver : shape;
	}
	return shape.font !== "silver" && isCjkHeavyText(text) && scanRenderability(text, { shape: silver }).isSafe
		? silver
		: shape;
}

// ============================================================================
// Constants
// ============================================================================

/** Legacy frame edge in pixels (the 5x8 shape's eval-validated size). New
 *  shapes carry their own `frameSize`. */
export const FRAME_SIZE = 2576;

/** Default upper bound on archive frames carried per compaction. Sized to hold
 *  ~400k tokens of the high-res Anthropic frame Opus reads (1932px ≈ 5,000
 *  billed tokens each → 80 frames) while staying under the ~100-image
 *  per-request wire cap. Oldest frames are dropped first once the budget is
 *  exceeded (mirrors how iterative text summaries fade the oldest detail); a
 *  caller may pass a lower `maxFrames` upper limit, and per-model context
 *  fitting is handled by the caller's overflow guard. */
export const MAX_FRAMES_DEFAULT = 80;

/** High-quality (legible) frames rendered at each chronological edge of a
 *  foveated archive — the session head (oldest) and the slice just before the
 *  text region (newest) — with the denser low-quality tier filling the middle. */
export const HQ_EDGE_FRAMES = 3;

/** Conservative per-frame token estimate used for context budgeting — the
 *  upper bound across shapes: high-res Claude frames hit the 4,784 visual-token
 *  cap, billed at +5% margin (ceil(4784 * 1.05)). Keeps the overflow guard from
 *  undercounting a high-res archive at the raised {@link MAX_FRAMES_DEFAULT}. */
export const FRAME_TOKEN_ESTIMATE = 5024;

/** Conservative upper bound for one persisted frame's base64 payload. The
 *  measured high-res Anthropic `8x13`/`11on16` PNG frames sit around 159 KB;
 *  170 KB leaves margin for denser glyph pages without permitting multi-MB
 *  standing request bodies at large context windows. */
export const FRAME_DATA_BYTES_ESTIMATE = 170_000;

/** Maximum snapcompact image base64 carried in every rebuilt provider request.
 *  Above this, provider backends can accept the HTTP body but fail mid-stream
 *  with opaque 5xx errors. Keep this independent from visual-token budgeting:
 *  a 1M-token model can afford 70 images on paper, but not the resulting
 *  ~11 MB JSON payload on every turn. */
export const FRAME_DATA_BYTES_BUDGET = 3_000_000;

/** Frame-count cap implied by {@link FRAME_DATA_BYTES_BUDGET}. */
export function maxFramesForDataBudget(maxFrameDataBytes: number = FRAME_DATA_BYTES_BUDGET): number {
	return Math.max(1, Math.floor(maxFrameDataBytes / FRAME_DATA_BYTES_ESTIMATE));
}

/** Base64 byte length for persisted snapcompact frames. */
export function frameDataBytes(frames: readonly Pick<Frame, "data">[]): number {
	return frames.reduce((sum, frame) => sum + frame.data.length, 0);
}

/**
 * Per-request image-count budgets by provider id. These cap how many images an
 * entire request may carry (archive/system-prompt/tool-result imaging combined).
 * The values are conservative policy caps under the vendor hard limits
 * (Anthropic 100, OpenAI 500, Gemini ~2500); unknown providers fall to a safe
 * floor rather than sending unbounded attachments.
 */
export const PROVIDER_IMAGE_BUDGETS: Record<string, number> = {
	anthropic: 90,
	"amazon-bedrock": 90,
	openai: 200,
	"openai-codex": 200,
	google: 200,
	"google-vertex": 200,
	"google-gemini-cli": 200,
	openrouter: 90,
	umans: 10,
};

/** Safe floor for unknown providers (strictest mainstream measured: Groq ~5). */
export const DEFAULT_PROVIDER_IMAGE_BUDGET = 5;

/** Per-request image budget for `provider`; unknown providers get the floor. */
export function providerImageBudget(provider: string | undefined): number {
	return (provider !== undefined ? PROVIDER_IMAGE_BUDGETS[provider] : undefined) ?? DEFAULT_PROVIDER_IMAGE_BUDGET;
}

/** Archive frame cap for `provider`: image budget, never above {@link MAX_FRAMES_DEFAULT}. */
export function providerFrameBudget(provider: string | undefined): number {
	return Math.min(providerImageBudget(provider), MAX_FRAMES_DEFAULT);
}

/** Key under `CompactionEntry.preserveData` holding the frame archive. */
export const PRESERVE_KEY = "snapcompact";

// ============================================================================
// Types
// ============================================================================

/** One developed snapcompact frame: a base64 PNG plus its reading geometry. */
export interface Frame {
	/** Base64-encoded PNG. */
	data: string;
	mimeType: string;
	/** Characters per row in the frame grid (per-column width on doc frames). */
	cols: number;
	/** Text rows in the frame grid (unique lines, not repeated copies). */
	rows: number;
	/** Characters actually printed onto this frame. */
	chars: number;
	/** Shape metadata (absent on legacy frames, which are 5x8 `sent`). */
	font?: Shape["font"];
	variant?: Shape["variant"];
	lineRepeat?: number;
	/** 2 on two-column doc frames; absent on row-major grid frames. */
	columns?: number;
	/** True when stopwords were printed in dim ink. */
	stopwordDim?: boolean;
	/** Resolution hint forwarded to the provider when re-attaching. */
	detail?: ImageContent["detail"];
}

/** Frame archive persisted under `preserveData[PRESERVE_KEY]`. */
export interface Archive {
	/** Rendered frames ordered oldest to newest, re-derived from {@link text}
	 *  each compaction with foveated quality tiers (HQ/LQ/HQ inside the imaged
	 *  middle). May be empty when the whole archive fits in text. */
	frames: Frame[];
	/** Characters currently readable across all frames plus the text regions. */
	totalChars: number;
	/** Characters dropped so far to respect the archive budget. */
	truncatedChars: number;
	/** Full kept archive source (oldest to newest, normalized, bounded to the
	 *  rendered budget) — the single source re-rendered each compaction. */
	text?: string;
	/** Oldest text region kept verbatim around the imaged middle. */
	textHead?: string;
	/** Newest text region kept verbatim around the imaged middle. */
	textTail?: string;
}

export interface Geometry {
	/** Characters per row (per-column line width when `columns === 2`). */
	cols: number;
	rows: number;
	/** Characters that fit one frame (nominal upper bound on doc shapes,
	 *  where real consumption is wrap-dependent). */
	capacity: number;
}

export interface Options<TMessage = Message> extends SerializeOptions {
	/** Actual raster base64 ceiling, shared with reconstruction. */
	maxFrameDataBytes?: number;
	/** App-level message transformer (same contract as agent-core's `SummaryOptions.convertToLlm`). */
	convertToLlm?: ConvertToLlm<TMessage>;
	/** Model whose provider API and id select the frame shape. */
	model?: ShapeTarget;
	/** Explicit shape override; wins over `model`. */
	shape?: Shape;
	/** Frame edge in pixels. Defaults to the shape's `frameSize`. */
	frameSize?: number;
	/** Upper limit on archive frames; clamped to (and defaulting to) {@link MAX_FRAMES_DEFAULT}. */
	maxFrames?: number;
}

/** Result of rendering one frame. */
export interface RenderedFrame {
	/** Base64-encoded PNG, as returned by the native renderer. */
	data: string;
	cols: number;
	rows: number;
	/** Characters printed (ink toggles excluded; input may be shorter than capacity). */
	chars: number;
}

// ============================================================================
// Compaction data contracts
// ============================================================================

export interface FileOperations {
	read: Set<string>;
	written: Set<string>;
	edited: Set<string>;
}

export interface CompactionDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

export interface CompactionPreparation<TMessage = Message> {
	sourcesToSummarize?: readonly SourceMessage<TMessage>[];
	turnPrefixSources?: readonly SourceMessage<TMessage>[];
	recentSources?: readonly SourceMessage<TMessage>[];
	selectedSources?: readonly (SourceMessage<TMessage> & { spans?: SourceBlockRange[] })[];
	/** UUID of first entry to keep. */
	firstKeptEntryId: string;
	/** Messages that will be archived and discarded. */
	messagesToSummarize: TMessage[];
	/** Messages that will be archived as the split-turn prefix, if any. */
	turnPrefixMessages: TMessage[];
	tokensBefore: number;
	/** Summary from previous compaction, for continuity when no prior snapcompact archive exists. */
	previousSummary?: string;
	/** Preserved opaque compaction payload from the previous compaction, if any. */
	previousPreserveData?: Record<string, unknown>;
	/** File operations extracted by the host agent. */
	fileOps: FileOperations;
}

export interface CompactionResult<T = CompactionDetails> {
	summary: string;
	shortSummary?: string;
	firstKeptEntryId: string;
	tokensBefore: number;
	details?: T;
	preserveData?: Record<string, unknown>;
}

export type ConvertToLlm<TMessage = Message> = (messages: TMessage[]) => Message[];

function defaultConvertToLlm<TMessage>(messages: TMessage[]): Message[] {
	return messages as unknown as Message[];
}

// ============================================================================
// File operation helpers
// ============================================================================

export function createFileOps(): FileOperations {
	return {
		read: new Set(),
		written: new Set(),
		edited: new Set(),
	};
}
const URL_SCHEME_RE = /[a-z][a-z0-9+.-]*:\/\//i;

export function isUrlSchemePath(path: string): boolean {
	return URL_SCHEME_RE.test(path);
}

export function computeFileLists(fileOps: FileOperations): CompactionDetails {
	const modified = new Set([...fileOps.edited, ...fileOps.written].filter(file => !isUrlSchemePath(file)));
	const readFiles = [...fileOps.read].filter(file => !isUrlSchemePath(file) && !modified.has(file)).sort();
	const modifiedFiles = [...modified].sort();
	return { readFiles, modifiedFiles };
}

/**
 * Format file operations as one `<files>` tag: a grouped, prefix-folded
 * directory tree (find-tool shape) with a ` (Read)` / ` (Write)` / ` (RW)`
 * marker per file. `readSet` is the cumulative read set (`fileOps.read`),
 * used to tell modified files that were also read (RW) from blind writes.
 */
const FILE_OPERATION_SUMMARY_LIMIT = 20;

function stripFileOperationTags(summary: string): string {
	// Legacy <read-files>/<modified-files> tags are still stripped so summaries
	// written before the combined <files> tag self-heal on the next compaction.
	return summary
		.replace(/<files>[\s\S]*?<\/files>\s*/g, "")
		.replace(/<read-files>[\s\S]*?<\/read-files>\s*/g, "")
		.replace(/<modified-files>[\s\S]*?<\/modified-files>\s*/g, "")
		.trimEnd();
}

function formatFileList(readFiles: string[], modifiedFiles: string[], readSet?: ReadonlySet<string>): string {
	if (readFiles.length === 0 && modifiedFiles.length === 0) return "";
	const mode = new Map<string, "Read" | "Write" | "RW">();
	for (const file of readFiles) mode.set(file, "Read");
	for (const file of modifiedFiles) mode.set(file, readSet?.has(file) ? "RW" : "Write");
	const all = [...mode.keys()].sort();
	let files = formatGroupedPaths(all.slice(0, FILE_OPERATION_SUMMARY_LIMIT), path => ` (${mode.get(path)})`);
	if (all.length > FILE_OPERATION_SUMMARY_LIMIT) {
		files += `\n[…${all.length - FILE_OPERATION_SUMMARY_LIMIT} files elided…]`;
	}
	return files;
}

function formatFileOperations(readFiles: string[], modifiedFiles: string[], readSet?: ReadonlySet<string>): string {
	const files = formatFileList(readFiles, modifiedFiles, readSet);
	return files.length > 0 ? prompt.render(fileOperationsTemplate, { files }) : "";
}

export function upsertFileOperations(
	summary: string,
	readFiles: string[],
	modifiedFiles: string[],
	readSet?: ReadonlySet<string>,
): string {
	const baseSummary = stripFileOperationTags(summary);
	const fileOperations = formatFileOperations(readFiles, modifiedFiles, readSet);
	if (!fileOperations) return baseSummary;
	if (!baseSummary) return fileOperations;
	return `${baseSummary}\n\n${fileOperations}`;
}

// ============================================================================
// Message serialization
// ============================================================================

/** Default per-tool-result character cap in serialized history. */
export const TOOL_RESULT_MAX_CHARS = 2000;

/** Default per-argument-value character cap inside serialized tool calls
 *  (write/edit bodies otherwise dump whole files into the archive). */
export const TOOL_ARG_MAX_CHARS = 500;

/** Default character cap across one tool call's full serialized argument list. */
export const TOOL_CALL_MAX_CHARS = 2000;

/** Default fraction of a truncation budget spent on the head; the remainder
 *  keeps the tail, where command errors and test failures usually land. */
export const TRUNCATE_HEAD_RATIO = 0.6;

/** Zero-width ink toggles understood by the native renderer (shift-out/in):
 *  text between them prints in dim gray ink without occupying a cell. */
export const DIM_ON = "\u000e";
export const DIM_OFF = "\u000f";

/** Character budgets applied while serializing discarded history for frame
 *  rendering. Pass `Infinity` to disable an individual cap. */
export interface SerializeOptions {
	/** Per-tool-result cap. Defaults to {@link TOOL_RESULT_MAX_CHARS}. */
	toolResultMaxChars?: number;
	/** Per-argument-value cap. Defaults to {@link TOOL_ARG_MAX_CHARS}. */
	toolArgMaxChars?: number;
	/** Whole-argument-list cap per call. Defaults to {@link TOOL_CALL_MAX_CHARS}. */
	toolCallMaxChars?: number;
	/** Head share of each budget, clamped to [0, 1]. Defaults to {@link TRUNCATE_HEAD_RATIO}. */
	truncateHeadRatio?: number;
	/** Print tool-result text in dim gray ink so archived conversation reads
	 *  louder than archived tool noise. Defaults to `true`. */
	dimToolResults?: boolean;
	/** Serialize assistant reasoning as `¶think:` sections. Defaults to `true`.
	 *  Callers archiving for a Claude/Anthropic-dialect model set this `false`:
	 *  the archive frames are replayed as text into every later request, and
	 *  reasoning rendered back to Claude trips its `reasoning_extraction`
	 *  classifier (issue #6093). */
	includeThinking?: boolean;
}

/** Keep the head and tail of `text`, eliding the middle beyond `maxChars`. */
function truncateForSummary(text: string, maxChars: number, headRatio: number): string {
	if (text.length <= maxChars) return text;
	const ratio = Math.min(Math.max(headRatio, 0), 1);
	const headChars = Math.round(maxChars * ratio);
	const tailChars = maxChars - headChars;
	const elided = text.length - maxChars;
	const tail = tailChars > 0 ? text.slice(-tailChars) : "";
	return `${text.slice(0, headChars)} […${elided}ch elided…] ${tail}`;
}

/** One elision marker as emitted by {@link truncateForSummary} (Unicode
 *  ellipses) or as persisted after `normalize()` (ASCII dots). */
const ELIDED_MARKER = String.raw`\[(?:…|\.{3})\d+ch elided(?:…|\.{3})\]`;

/** Unquoted RFC 2045 token used as a media-type parameter name or value.
 *  Quoted-string values (RFC 822) are out of scope. */
const MEDIA_TYPE_TOKEN = String.raw`[\w!#$%&'*+.^|~-]+`;

/** An inline base64 data URL. The payload may be empty or carry one embedded
 *  elision marker so fragments left by pre-guard slices — including a cut
 *  landing exactly on `;base64,` — still match. RFC 2397 allows `*( ";" parameter )`
 *  between type/subtype and the terminal `;base64`; unquoted tokens are matched,
 *  quoted-string values are out of scope. `data:` and `base64` match
 *  case-insensitively (`gi`). Matching starts at `data:`; Markdown wrappers are
 *  recovered by {@link adjacentMarkdownOpenerStart} after each hit. */
const DATA_URL_ATOM = new RegExp(
	String.raw`data:([A-Za-z][\w.+-]*\/[\w.+-]+(?:;${MEDIA_TYPE_TOKEN}=${MEDIA_TYPE_TOKEN})*);base64,` +
		String.raw`([A-Za-z0-9+/=]*(?:\s*${ELIDED_MARKER}\s*[A-Za-z0-9+/=]*)?)` +
		String.raw`(\s*\))?`,
	"gi",
);

const ELIDED_MARKER_RE = new RegExp(String.raw`\s*${ELIDED_MARKER}\s*`);
const MARKDOWN_WHITESPACE_CHAR = /\s/;

/** Canonical base64: 4-char groups with valid terminal padding. */
const CANONICAL_BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$/;

/** A non-canonical payload at least this long is a damaged fragment of a real
 *  data URL (e.g. an archive head cut mid-payload by a structure-blind slice),
 *  not a prose mention like `data:image/png;base64,abc`. */
const DAMAGED_PAYLOAD_MIN_CHARS = 40;

/** Context for {@link elideDataUrls}. `source` text is intact (never sliced),
 *  so a short non-canonical payload is a prose mention and stays untouched.
 *  `archive` text may have been cut by pre-guard structure-blind slices at
 *  any offset — even 0–39 chars past `;base64,` — so every recognized prefix
 *  is suspect and is always elided. */
type DataUrlContext = "source" | "archive";

/** Start of `!?[label](\s*` immediately before `dataIndex`, or `undefined`.
 *  The opener must lie in `[cursor, dataIndex)`. Nested `[` in the label is
 *  kept (the old `[^\]\n]*` class allowed it) by taking the earliest `[` after
 *  a prior `]`, newline, or `cursor`; the scan never walks already-emitted
 *  text, so repeated `](data:...)` stays linear. */
function adjacentMarkdownOpenerStart(text: string, dataIndex: number, cursor: number): number | undefined {
	let i = dataIndex;
	while (i > cursor && MARKDOWN_WHITESPACE_CHAR.test(text.charAt(i - 1))) i--;
	// `](` and any following whitespace must sit in [cursor, dataIndex).
	if (i - 2 < cursor || text.charAt(i - 1) !== "(" || text.charAt(i - 2) !== "]") return undefined;
	let opener = -1;
	for (let j = i - 3; j >= cursor; j--) {
		const c = text.charAt(j);
		if (c === "]" || c === "\n") break;
		if (c === "[") opener = j;
	}
	if (opener < 0) return undefined;
	return opener > cursor && text.charAt(opener - 1) === "!" ? opener - 1 : opener;
}

/** Replace every inline base64 data URL atomically with a deterministic
 *  placeholder. A character cap that slices inside a base64 payload leaves a
 *  recognizable image reference that can never decode; OpenAI-dialect
 *  providers reject such requests as invalid image input, and because the
 *  corrupted text persists in the archive the session re-fails on every later
 *  request. The payload is worthless to a model as text, so the whole atom —
 *  Markdown wrapper included — collapses to its metadata. Payloads already
 *  carrying an elision marker, and non-canonical fragments left by pre-guard
 *  slices, are healed the same way.
 *
 *  The placeholder's `<mime>` is the media type as written: type/subtype plus
 *  any unquoted RFC 2397 `;parameter=value` segments, original case preserved.
 *  Parameters are kept rather than stripped to a bare type/subtype so charset
 *  (and similar) remain visible after elision and the label stays a pure
 *  function of the captured text. */
function elideDataUrls(text: string, context: DataUrlContext = "source", onReplacement?: (start: number, end: number, value: string) => void): string {
	if (!/;base64,/i.test(text)) return text;
	DATA_URL_ATOM.lastIndex = 0;
	let match = DATA_URL_ATOM.exec(text);
	if (match === null) return text;
	const out: string[] = [];
	let cursor = 0;
	while (match !== null) {
		const urlStart = match.index;
		const urlEnd = urlStart + match[0].length;
		const mime = match[1] ?? "";
		const payload = match[2] ?? "";
		const closer = match[3];
		const marker = ELIDED_MARKER_RE.exec(payload);
		const isAtom =
			context === "archive" ||
			marker !== null ||
			CANONICAL_BASE64.test(payload) ||
			payload.length >= DAMAGED_PAYLOAD_MIN_CHARS;
		if (!isAtom) {
			// Advance through short prose too, so a later wrapper cannot swallow
			// a data URL already copied out of its Markdown label.
			out.push(text.slice(cursor, urlEnd));
			cursor = urlEnd;
		} else {
			const b64Chars = marker
				? payload.length - marker[0].length + Number(/\d+/.exec(marker[0])?.[0] ?? 0)
				: payload.length;
			const placeholder = `[data URL omitted: ${mime}, ${b64Chars} base64 chars]`;
			const foundOpener = adjacentMarkdownOpenerStart(text, urlStart, cursor);
			const openerStart = foundOpener !== undefined && foundOpener >= cursor ? foundOpener : undefined;
			const emitStart = openerStart ?? urlStart;
			out.push(text.slice(cursor, emitStart));
			// Swallow the Markdown wrapper only when both delimiters matched;
			// otherwise re-emit whichever half was captured untouched. An opener
			// that starts before the already-emitted cursor would overlap a prior
			// replacement, so that URL is treated as bare.
			if (openerStart !== undefined && closer !== undefined) {
				out.push(placeholder);
				onReplacement?.(emitStart, urlEnd, placeholder);
			} else {
				const opener = openerStart !== undefined ? text.slice(openerStart, urlStart) : "";
				out.push(opener, placeholder, closer ?? "");
				onReplacement?.(emitStart, urlEnd, opener + placeholder + (closer ?? ""));
			}
			cursor = urlEnd;
		}
		match = DATA_URL_ATOM.exec(text);
	}
	out.push(text.slice(cursor));
	return out.join("");
}

const DIM_MARKERS = /[\u000e\u000f]/g;

/** Plain-text history kept verbatim at each chronological edge, in HQ-frame-
 *  capacity units per edge. One page at the start and one at the end preserves
 *  high-fidelity context around the imaged middle while keeping the total text
 *  budget equal to the prior 2-page tail-only scheme. */
const TEXT_EDGE_PAGES = 1;

/** Normalized archive text → plain text: drop zero-width dim toggles and
 *  print newline glyphs as real newlines. */
function toPlainText(text: string): string {
	return stripDimMarkers(text).replaceAll(NEWLINE_GLYPH, "\n");
}

/** Strip stray ink toggles from raw content so it cannot forge dim spans. */
function stripDimMarkers(text: string): string {
	return text.replace(DIM_MARKERS, "");
}


interface SerializationTrace {
	message: Message;
	start: number;
	end: number;
	blockIndex: number;
	rawStart: number;
	rawEnd: number;
	exact: boolean;
}
interface SerializationImage { message: Message; blockIndex: number; offset: number }
interface SerializationCapture { runs: SerializationTrace[]; images: SerializationImage[]; required?: boolean }

export function serializeConversation(messages: Message[], options?: SerializeOptions): string {
	return serializeConversationSource(messages, options);
}

function serializeConversationSource(messages: Message[], options?: SerializeOptions, capture?: SerializationCapture): string {
	const toolResultMaxChars = options?.toolResultMaxChars ?? TOOL_RESULT_MAX_CHARS;
	const toolArgMaxChars = options?.toolArgMaxChars ?? TOOL_ARG_MAX_CHARS;
	const toolCallMaxChars = options?.toolCallMaxChars ?? TOOL_CALL_MAX_CHARS;
	const headRatio = options?.truncateHeadRatio ?? TRUNCATE_HEAD_RATIO;
	const dimToolResults = options?.dimToolResults !== false;
	const includeThinking = options?.includeThinking !== false;
	const parts: string[] = [];
	let lastPrefix: string | null = null;
	let serializedLength = 0;

	const pushPart = (prefix: string, content: string): number => {
		const lastIndex = parts.length - 1;
		let prefixLength: number;
		if (lastIndex >= 0 && lastPrefix === prefix) {
			const sep = parts[lastIndex].endsWith("\n") || content.startsWith("\n") ? "" : "\n";
			parts[lastIndex] += sep + content;
			prefixLength = sep.length;
		} else {
			parts.push(prefix + content);
			lastPrefix = prefix;
			prefixLength = (lastIndex >= 0 ? 2 : 0) + prefix.length;
		}
		const start = serializedLength + prefixLength;
		serializedLength = start + content.length;
		return start;
	};

	const captureRaw = (message: Message, blockIndex: number, raw: string, at: number) => {
		if (!capture) return;
		let start = 0;
		for (const match of raw.matchAll(/[\u000e\u000f]/g)) {
			if (match.index > start) { capture.runs.push({ message, blockIndex, start: at, end: at + match.index - start, rawStart: start, rawEnd: match.index, exact: true }); at += match.index - start; }
			start = match.index + 1;
		}
		if (start < raw.length) capture.runs.push({ message, blockIndex, start: at, end: at + raw.length - start, rawStart: start, rawEnd: raw.length, exact: true });
	};
	const resultByCallId = new Map<string, Extract<Message, { role: "toolResult" }>>();
	const captureResult = (message: Extract<Message, { role: "toolResult" }>, outputStart: number) => {
		if (!capture) return;
		const raw = message.content.filter(block => block.type === "text").map(block => block.text).join("");
		const clean = elideDataUrls(stripDimMarkers(raw));
		const body = truncateForSummary(clean, toolResultMaxChars, headRatio);
		const bodyStart = outputStart + "<out>\n".length + (dimToolResults ? DIM_ON.length : 0);
		const head = clean.length > toolResultMaxChars ? Math.round(toolResultMaxChars * Math.min(Math.max(headRatio, 0), 1)) : clean.length;
		const tail = clean.length > toolResultMaxChars ? toolResultMaxChars - head : 0;
		const intervals = [{ start: 0, end: head, output: bodyStart }, ...(tail > 0 ? [{ start: clean.length - tail, end: clean.length, output: bodyStart + body.length - tail }] : [])];
		let rawOffset = 0;
		for (const [blockIndex, block] of message.content.entries()) {
			if (block.type === "image") {
				const interval = intervals.find(part => part.end >= rawOffset) ?? intervals.at(-1)!;
				capture.images.push({ message, blockIndex, offset: interval.output + Math.max(0, Math.min(rawOffset, interval.end) - interval.start) });
				continue;
			}
			if (block.type !== "text") continue;
			if (clean === raw) for (const interval of intervals) {
				const start = Math.max(rawOffset, interval.start), end = Math.min(rawOffset + block.text.length, interval.end);
				if (end > start) capture.runs.push({ message, blockIndex, start: interval.output + start - interval.start, end: interval.output + end - interval.start, rawStart: start - rawOffset, rawEnd: end - rawOffset, exact: true });
			}
			else if (rawOffset === 0) capture.runs.push({ message, blockIndex, start: bodyStart, end: bodyStart + body.length, rawStart: 0, rawEnd: block.text.length, exact: false });
			rawOffset += block.text.length;
		}
	};


	// Tool results flagged contextually useless (and their paired calls) carry no
	// information worth archiving — skip the whole pair. Surviving results are
	// indexed by tool-call id so each merges into its originating `¶call:` scope.
	const uselessCallIds = new Set<string>();
	const resultTextByCallId = new Map<string, string>();
	for (const msg of messages) {
		if (msg.role !== "toolResult") continue;
		resultByCallId.set(msg.toolCallId, msg);
		if (!capture?.required && msg.useless === true && msg.isError !== true) {
			uselessCallIds.add(msg.toolCallId);
			continue;
		}
		const text = msg.content
			.filter((block): block is { type: "text"; text: string } => block.type === "text")
			.map(block => block.text)
			.join("");
		if (text) resultTextByCallId.set(msg.toolCallId, text);
	}

	// Wrap a raw tool-result body in an `<out>` block, dimming only the body so
	// the frame coloring keeps scope markers and calls loud.
	const renderResultBlock = (rawText: string): string => {
		const body = truncateForSummary(elideDataUrls(stripDimMarkers(rawText)), toolResultMaxChars, headRatio);
		return `<out>\n${dimToolResults ? `${DIM_ON}${body}${DIM_OFF}` : body}\n</out>`;
	};

	const mergedCallIds = new Set<string>();

	for (const msg of messages) {
		if (msg.role === "user") {
			const blocks = typeof msg.content === "string" ? [{ type: "text" as const, text: msg.content }] : msg.content;
			const content = blocks.filter(block => block.type === "text").map(block => stripDimMarkers(block.text)).join("");
			let at = content ? pushPart("¶user:", content) : serializedLength;
			if (capture) for (const [blockIndex, block] of blocks.entries()) {
				if (block.type === "image") { capture.images.push({ message: msg, blockIndex, offset: at }); continue; }
				if (block.type !== "text") continue;
				let rawStart = 0;
				for (const match of block.text.matchAll(/[\u000e\u000f]/g)) {
					if (match.index > rawStart) { const length = match.index - rawStart; capture.runs.push({ message: msg, blockIndex, start: at, end: at + length, rawStart, rawEnd: match.index, exact: true }); at += length; }
					rawStart = match.index + 1;
				}
				if (rawStart < block.text.length) { const length = block.text.length - rawStart; capture.runs.push({ message: msg, blockIndex, start: at, end: at + length, rawStart, rawEnd: block.text.length, exact: true }); at += length; }
			}
		} else if (msg.role === "assistant") {
			// Stream blocks in content order: buffer thinking/text, then flush a
			// separate section for each block type right before each tool call.

			let pendingThinking: { text: string; raw: string; blockIndex: number }[] = [];
			let pendingText: { text: string; raw: string; blockIndex: number }[] = [];
			const flushAssistant = () => {
				for (const [prefix, pending] of [["¶think:", pendingThinking], ["¶ai:", pendingText]] as const) {
					if (!pending.length) continue;
					let at = pushPart(prefix, pending.map(part => part.text).join("\n"));
					for (const part of pending) { captureRaw(msg, part.blockIndex, part.raw, at); at += part.text.length + 1; }
				}
				pendingThinking = []; pendingText = [];
			};

			for (const [blockIndex, block] of msg.content.entries()) {
				if (block.type === "text") {
					const text = stripDimMarkers(block.text);
					if (text.trim()) pendingText.push({ text, raw: block.text, blockIndex });
				} else if (block.type === "thinking") {
					if (!includeThinking) continue;
					const thinking = stripDimMarkers(block.thinking);
					if (thinking.trim()) pendingThinking.push({ text: thinking, raw: block.thinking, blockIndex });
				} else if (block.type === "toolCall") {
					if (uselessCallIds.has(block.id)) continue;
					flushAssistant();
					const args = block.arguments as Record<string, unknown>;
					// Prefer the harness-derived intent, else the raw intent arg; render it as
					// a one-line `//comment` and drop it from the args below.
					const rawIntent =
						typeof block.intent === "string"
							? block.intent
							: typeof args[INTENT_FIELD] === "string"
								? (args[INTENT_FIELD] as string)
								: "";
					const intent = stripDimMarkers(rawIntent).replace(/\s+/g, " ").trim();

					let argumentsComplete = true;
					const argumentText = Object.entries(args).filter(([key]) => key !== INTENT_FIELD).map(([key, value]) => {
						const raw = elideDataUrls(JSON.stringify(value) ?? "undefined");
						if (raw.length > toolArgMaxChars) argumentsComplete = false;
						return key + "=" + truncateForSummary(raw, toolArgMaxChars, headRatio);
					}).join(", ");
					if (argumentText.length > toolCallMaxChars) argumentsComplete = false;
					const argsStr = truncateForSummary(argumentText, toolCallMaxChars, headRatio);
					const lines: string[] = [];
					let firstLine = `${block.name}(${argsStr})`;
					if (intent) {
						firstLine += `//${intent}`;
					}
					lines.push(firstLine);
					const resultText = resultTextByCallId.get(block.id);
					if (resultText !== undefined) {
						mergedCallIds.add(block.id);
						lines.push(renderResultBlock(resultText));
					}
					const at = pushPart("¶call:", lines.join("\n"));
					if (capture) capture.runs.push({ message: msg, blockIndex, start: at, end: at + firstLine.length, rawStart: 0, rawEnd: 0, exact: argumentsComplete });
					const result = resultByCallId.get(block.id);
					if (result && resultText !== undefined) captureResult(result, at + firstLine.length + 1);
				}
			}
			flushAssistant();
		} else if (msg.role === "toolResult") {
			// Paired results already merged into their tool call block above;
			// only orphans (call archived outside this window) render standalone.
			if (uselessCallIds.has(msg.toolCallId) || mergedCallIds.has(msg.toolCallId)) continue;
			const resultText = resultTextByCallId.get(msg.toolCallId);
			if (resultText !== undefined) { const at = pushPart("¶call:", "\n" + renderResultBlock(resultText)); captureResult(msg, at + 1); }
		}
	}

	return parts.join("\n\n");
}

// ============================================================================
// Preserve-data helpers
// ============================================================================

const OPENAI_REMOTE_COMPACTION_PRESERVE_KEY = "openaiRemoteCompaction";

function stripOpenAiRemoteCompactionPreserveData(
	preserveData: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
	if (!preserveData || !(OPENAI_REMOTE_COMPACTION_PRESERVE_KEY in preserveData)) {
		return preserveData;
	}
	const { [OPENAI_REMOTE_COMPACTION_PRESERVE_KEY]: _removed, ...rest } = preserveData;
	return Object.keys(rest).length > 0 ? rest : undefined;
}

// ============================================================================
// Text normalization
// ============================================================================

/** Punctuation and symbol folds applied before the NFKD fallback in
 *  {@link normalize}: quotes, dashes, bullets, arrows, and dot leaders that
 *  have no compatibility decomposition (or one that is itself non-ASCII). */
const CHAR_FOLD: Record<string, string> = {
	// Quotation marks and primes.
	"\u2018": "'",
	"\u2019": "'",
	"\u201a": "'",
	"\u201b": "'",
	"\u201c": '"',
	"\u201d": '"',
	"\u201e": '"',
	"\u2032": "'",
	"\u2033": '"',
	"\u2035": "'",
	"\u2036": '"',
	"\u2039": "<",
	"\u203a": ">",
	// Dashes, hyphens, and the fraction slash NFKD leaves in vulgar fractions.
	"\u2010": "-",
	"\u2011": "-",
	"\u2012": "-",
	"\u2013": "-",
	"\u2014": "-",
	"\u2015": "-",
	"\u2212": "-",
	"\u2044": "/",
	// Dot leaders and ellipses.
	"\u2024": ".",
	"\u2025": "..",
	"\u2026": "...",
	"\u22ef": "...",
	// Bullets.
	"\u2022": "*",
	"\u2023": "*",
	"\u2043": "-",
	"\u2219": "*",
	"\u25cf": "*",
	"\u25a0": "*",
	"\u25aa": "*",
	// Arrows.
	"\u2190": "<-",
	"\u2191": "^",
	"\u2192": "->",
	"\u2193": "v",
	"\u2194": "<->",
	"\u21d0": "<=",
	"\u21d2": "=>",
	"\u21d4": "<=>",
	// Check marks and crosses.
	"\u2713": "v",
	"\u2714": "v",
	"\u2717": "x",
	"\u2718": "x",
};

/** Printed in place of newline runs: the native renderer fills this cell
 *  entirely with pitch-black ink, so line structure survives whitespace
 *  collapsing at a one-cell cost. */
export const NEWLINE_GLYPH = "\u2588";

/** Collapsed in one pass: whitespace plus zero-width format characters (ZWSP,
 *  BOM, directional marks — JS `\s` already counts BOM as whitespace, so they
 *  must fold here, before the per-character pass). */
const COLLAPSIBLE = /[\s\p{Cf}]+/gu;

/** Runs carrying one of these collapse to {@link NEWLINE_GLYPH}. */
const LINE_BREAK = /[\n\r\u2028\u2029]/;

/** Leading/trailing spaces or newline glyphs add no information to a frame. */
const EDGE_RUNS = /^[ \u2588]+|[ \u2588]+$/g;

/** Glyph-less code points skipped outright instead of printing `?`: controls
 *  (bare ESC/BEL/NUL — full ANSI sequences are stripped beforehand),
 *  combining marks the fonts cannot compose, and lone surrogates. */
const UNRENDERABLE = /[\p{Cc}\p{Mn}\p{Me}\p{Cs}]/u;

/** Combining marks NFKD splits off accented letters; dropped so the base
 *  letter prints without the diacritic the bundled fonts cannot compose. */
const COMBINING_MARKS = /\p{M}+/gu;

/** Status-like pictographs that carry meaning in tool output; all other emoji
 *  pictographs drop instead of burning cells as `?`. */
const EMOJI_FOLD: Record<string, string> = {
	"✅": "[OK]",
	"☑": "[OK]",
	"✔": "[OK]",
	"❌": "[FAIL]",
	"❎": "[FAIL]",
	"✖": "[FAIL]",
	"⚠": "[WARN]",
	"🚨": "[ALERT]",
	ℹ: "[INFO]",
	"🐛": "[BUG]",
	"💥": "[CRASH]",
	"🔥": "[HOT]",
	"🔒": "[LOCK]",
	"🔓": "[UNLOCK]",
	"📁": "[DIR]",
	"📂": "[DIR]",
	"📄": "[FILE]",
	"📝": "[NOTE]",
	"🧪": "[TEST]",
	"⏳": "[WAIT]",
	"⌛": "[WAIT]",
	"🚀": "[RUN]",
};

const EMOJI_PICTOGRAPH = /\p{Extended_Pictographic}/u;

export interface NormalizeOptions {
	/** Shape whose font is tried before the embedded Silver fallback. */
	shape?: Pick<Shape, "font">;
	/** Native font name when a full shape is not available. */
	font?: Shape["font"];
}

interface NormalizedText {
	text: string;
	totalGraphics: number;
	fallbackCount: number;
}

/**
 * Aggressive single-code-point ASCII fold via Unicode NFKD: decompose the
 * compatibility form (fullwidth, super/subscripts, ligatures, circled and
 * math-styled alphanumerics, Roman numerals, vulgar fractions, …), strip the
 * combining marks, and keep the ASCII/Latin-1 skeleton — routing any residual
 * punctuation back through {@link CHAR_FOLD}. Returns `undefined` when the code
 * point has no decomposition or still leaves an undrawable glyph, so the
 * caller falls back to `?`.
 */
function isAsciiOrLatin1(cp: number): boolean {
	return (cp >= 0x20 && cp < 0x7f) || (cp >= 0xa0 && cp <= 0xff);
}

function foldToAscii(ch: string): string | undefined {
	const decomposed = ch.normalize("NFKD").replace(COMBINING_MARKS, "");
	if (decomposed === ch) return undefined;
	let out = "";
	for (const part of decomposed) {
		const cp = part.codePointAt(0);
		if (cp !== undefined && isAsciiOrLatin1(cp)) {
			out += part;
			continue;
		}
		const fold = CHAR_FOLD[part];
		if (fold === undefined) return undefined;
		out += fold;
	}
	return out;
}

function renderableUnicodeChars(chars: readonly string[], font: Shape["font"] | undefined): ReadonlySet<string> {
	if (chars.length === 0) return new Set();
	const text = chars.join("");
	const primaryFont = font ?? "5x8";
	const supported = new Set(snapcompactSupportedChars(primaryFont, text));
	if (primaryFont !== "silver") {
		for (const ch of snapcompactSupportedChars("silver", text)) supported.add(ch);
	}
	return supported;
}

function normalizedInputChars(text: string): string[] {
	const stripped = text.includes("\u001b") ? Bun.stripANSI(text) : text;
	const collapsed = stripped
		// A run of pure format chars (BOM is both \s and Cf) vanishes; only a
		// run containing genuine whitespace separates words.
		.replace(COLLAPSIBLE, run => (LINE_BREAK.test(run) ? NEWLINE_GLYPH : /[^\p{Cf}]/u.test(run) ? " " : ""))
		.replace(EDGE_RUNS, "");
	return [...collapsed];
}

function candidateUnicodeChars(chars: readonly string[]): string[] {
	const unique = new Set<string>();
	for (const ch of chars) {
		const cp = ch.codePointAt(0);
		if (cp === undefined || isAsciiOrLatin1(cp) || ch === DIM_ON || ch === DIM_OFF || ch === NEWLINE_GLYPH) {
			continue;
		}
		if (
			CHAR_FOLD[ch] !== undefined ||
			(cp >= 0x2500 && cp <= 0x257f) ||
			EMOJI_FOLD[ch] !== undefined ||
			EMOJI_PICTOGRAPH.test(ch) ||
			foldToAscii(ch) !== undefined ||
			UNRENDERABLE.test(ch)
		) {
			continue;
		}
		unique.add(ch);
	}
	return [...unique];
}

function normalizeWithStats(text: string, options?: NormalizeOptions): NormalizedText {
	const chars = normalizedInputChars(text);
	const font = options?.font ?? options?.shape?.font;
	const supported = renderableUnicodeChars(candidateUnicodeChars(chars), font);
	const out: string[] = [];
	let totalGraphics = 0;
	let fallbackCount = 0;

	for (const ch of chars) {
		const cp = ch.codePointAt(0);
		if (cp === undefined) continue;
		if (isAsciiOrLatin1(cp)) {
			out.push(ch);
			totalGraphics++;
			continue;
		}
		if (ch === DIM_ON || ch === DIM_OFF || ch === NEWLINE_GLYPH) {
			out.push(ch);
			continue;
		}
		const emoji = EMOJI_FOLD[ch];
		if (emoji !== undefined) {
			out.push(emoji);
			totalGraphics++;
			continue;
		}
		const fold = CHAR_FOLD[ch];
		if (fold !== undefined) {
			out.push(fold);
			totalGraphics++;
			continue;
		}
		if (cp >= 0x2500 && cp <= 0x257f) {
			out.push(cp === 0x2502 || cp === 0x2503 ? "|" : cp === 0x2500 || cp === 0x2501 ? "-" : "+");
			totalGraphics++;
			continue;
		}
		if (!EMOJI_PICTOGRAPH.test(ch) && supported.has(ch)) {
			out.push(ch);
			totalGraphics++;
			continue;
		}
		const folded = foldToAscii(ch);
		if (folded !== undefined) {
			out.push(folded);
			totalGraphics++;
		} else if (EMOJI_PICTOGRAPH.test(ch)) {
		} else if (!UNRENDERABLE.test(ch)) {
			out.push("?");
			totalGraphics++;
			fallbackCount++;
		}
	}

	return { text: out.join("").replace(/ +/g, " ").replace(EDGE_RUNS, ""), totalGraphics, fallbackCount };
}

/**
 * Prepare text for printing: strip ANSI escape sequences, collapse horizontal
 * whitespace runs, fold unsupported symbols (including box drawing to ASCII),
 * preserve Unicode glyphs that either the selected font or embedded Silver
 * fallback can render, and drop decorative emoji instead of printing `?`.
 */
export function normalize(text: string, options?: NormalizeOptions): string {
	return normalizeWithStats(text, options).text;
}

/**
 * Scan text with the same font-aware path as {@link normalize}; unsafe means
 * more than 5% of graphic characters would hit the `?` fallback.
 */
export function scanRenderability(
	text: string,
	options?: NormalizeOptions,
): { isSafe: boolean; unrenderableRatio: number } {
	const normalized = normalizeWithStats(text, options);
	const unrenderableRatio = normalized.totalGraphics > 0 ? normalized.fallbackCount / normalized.totalGraphics : 0;
	return { isSafe: unrenderableRatio <= 0.05, unrenderableRatio };
}

// ============================================================================
// Stopword dimming
// ============================================================================

/** High-frequency function words a reader can reconstruct from context; the
 *  dim shapes render them in light gray so content words carry the contrast
 *  (verbatim from `research/bdf.py` `_STOPWORDS`). */
const STOPWORDS: ReadonlySet<string> = new Set(
	(
		"the a an and or of to in on at as is are was were be been by for with that this it its from had has have not but " +
		"he she his her they their them which also who whom when where while will would could should there then than " +
		"into over under about after before between during each such these those some most more other only same so"
	).split(" "),
);

/** Maximal alphabetic runs (ASCII + Latin-1 letters, the fonts' coverage). */
const ALPHA_RUN = /[a-zA-Z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff]+/g;

/** Splitter that keeps the zero-width ink toggles as their own segments. */
const DIM_MARKER_SPLIT = /([\u000e\u000f])/;

/**
 * Wrap each maximal alphabetic run that is a stopword in {@link DIM_ON} /
 * {@link DIM_OFF} so it prints in dim gray ink. Spans that are already dim
 * (e.g. archived tool output) pass through untouched — wrapping there would
 * terminate the enclosing dim span early. Markers are zero-width, so the
 * visible glyph count is unchanged.
 */
export function dimStopwords(text: string): string {
	const parts = text.split(DIM_MARKER_SPLIT);
	let dim = false;
	let out = "";
	for (const part of parts) {
		if (part === DIM_ON) {
			dim = true;
			out += part;
		} else if (part === DIM_OFF) {
			dim = false;
			out += part;
		} else if (dim) {
			out += part;
		} else {
			out += part.replace(ALPHA_RUN, word => (STOPWORDS.has(word.toLowerCase()) ? DIM_ON + word + DIM_OFF : word));
		}
	}
	return out;
}

// ============================================================================
// Doc layout (two word-wrapped newspaper columns)
// ============================================================================

/** Char cells between the two doc columns (research exp14 `GUTTER`). */
const DOC_GUTTER = 3;

/** East Asian Wide / Fullwidth code points that occupy two grid cells when a
 *  narrow bitmap shape draws them through the Silver fallback. Mirrors
 *  `is_wide` in `crates/pi-natives/src/snapcompact.rs`; the two MUST stay in
 *  sync or native layout and this capacity math disagree on cell counts. */
function isWideCodePoint(cp: number): boolean {
	return (
		(cp >= 0x1100 && cp <= 0x115f) ||
		(cp >= 0x2e80 && cp <= 0x2eff) ||
		(cp >= 0x2f00 && cp <= 0x2fdf) ||
		(cp >= 0x3000 && cp <= 0x303e) ||
		(cp >= 0x3041 && cp <= 0x33ff) ||
		(cp >= 0x3400 && cp <= 0x4dbf) ||
		(cp >= 0x4e00 && cp <= 0x9fff) ||
		(cp >= 0xa000 && cp <= 0xa4cf) ||
		(cp >= 0xac00 && cp <= 0xd7a3) ||
		(cp >= 0xf900 && cp <= 0xfaff) ||
		(cp >= 0xfe30 && cp <= 0xfe4f) ||
		(cp >= 0xff00 && cp <= 0xff60) ||
		(cp >= 0xffe0 && cp <= 0xffe6) ||
		(cp >= 0x20000 && cp <= 0x2fffd) ||
		(cp >= 0x30000 && cp <= 0x3fffd)
	);
}

/** Cells one character occupies: 0 for the zero-width dim toggles, 2 for wide
 *  code points in narrow bitmap shapes, 1 otherwise. Mirrors native
 *  `cell_units`. */
function charCells(ch: string, wideCells: boolean): number {
	if (ch === DIM_ON || ch === DIM_OFF) return 0;
	const cp = ch.codePointAt(0);
	return wideCells && cp !== undefined && isWideCodePoint(cp) ? 2 : 1;
}

/** Wide code points span two cells in every shape except the square-celled
 *  Silver shape, which sizes each cell for a full-width glyph already. */
function usesWideCells(shape: Pick<Shape, "font">): boolean {
	return shape.font !== "silver";
}

/** Total grid cells a string occupies (ignoring row wrapping/pads). */
function cellLength(text: string, wideCells: boolean): number {
	let cells = 0;
	for (const ch of text) cells += charCells(ch, wideCells);
	return cells;
}

/** Longest prefix of `text` that fits `width` cells (at least one char). */
function sliceCells(text: string, width: number, wideCells: boolean): string {
	let cells = 0;
	let out = "";
	let placed = false;
	for (const ch of text) {
		const w = charCells(ch, wideCells);
		if (placed && cells + w > width) break;
		out += ch;
		cells += w;
		if (w > 0) placed = true;
	}
	return out;
}

/** Split `text` into pages that each fill at most `capacity` grid cells,
 *  inserting a one-cell pad before a wide glyph that would straddle the right
 *  edge (mirrors native `place_cell`). Pages are contiguous substrings, so each
 *  renders independently starting at cell 0. A single char wider than the whole
 *  budget still rides its page; the native renderer clips it. */
function paginateCells(text: string, capacity: number, cols: number, wideCells: boolean, maxPages = Infinity): string[] {
	const pages: string[] = [];
	let start = 0, offset = 0, cell = 0;
	let hasCell = false;
	for (const char of text) {
		const w = charCells(char, wideCells);
		if (w > 0) {
			let at = cell;
			if (w === 2 && cols >= 2 && at % cols === cols - 1) at += 1;
			if (hasCell && at + w > capacity) {
				pages.push(text.slice(start, offset));
				if (pages.length >= maxPages) return pages;
				start = offset;
				at = 0;
			}
			cell = at + w;
			hasCell = true;
		}
		offset += char.length;
	}
	if (hasCell) pages.push(text.slice(start));
	return pages;
}

/**
 * Greedy word-wrap, no mid-word breaks (hard split only for width+ words) —
 * ported verbatim from `research/exp14_bestgpt.py` `wrap()`. Zero-width dim
 * markers count toward word length here; serialized history places them at
 * word boundaries, so the drift is at most one cell per affected line.
 */
export function wrap(text: string, width: number, wideCells = false): string[] {
	const lines: string[] = [];
	let cur = "";
	let curCells = 0;
	for (const token of text.split(/\s+/)) {
		if (token.length === 0) continue;
		let word = token;
		let wordCells = cellLength(word, wideCells);
		while (wordCells > width) {
			// Pathological; never hit on prose.
			if (cur) {
				lines.push(cur);
				cur = "";
				curCells = 0;
			}
			const head = sliceCells(word, width, wideCells);
			lines.push(head);
			word = word.slice(head.length);
			wordCells = cellLength(word, wideCells);
		}
		if (!cur) {
			cur = word;
			curCells = wordCells;
		} else if (curCells + 1 + wordCells <= width) {
			cur += ` ${word}`;
			curCells += 1 + wordCells;
		} else {
			lines.push(cur);
			cur = word;
			curCells = wordCells;
		}
	}
	if (cur) lines.push(cur);
	return lines;
}

/**
 * Paginate already-normalized text for a doc shape: wrap once at the column
 * width, then slice into pages of `2 * rows` lines, each page `\n`-joined.
 * Every input character lands on exactly one page (whitespace becomes the
 * wrap points).
 */
function docPages(normalized: string, geo: Geometry, wideCells: boolean): string[] {
	const lines = wrap(normalized, geo.cols, wideCells);
	const perPage = 2 * geo.rows;
	const pages: string[] = [];
	for (let offset = 0; offset < lines.length; offset += perPage) {
		pages.push(lines.slice(offset, offset + perPage).join("\n"));
	}
	return pages;
}

// ============================================================================
// Rendering
// ============================================================================

export function geometry(shape: Shape, size: number = shape.frameSize): Geometry {
	const gridCols = Math.floor(size / shape.cellWidth);
	const rows = Math.floor(size / shape.cellHeight / shape.lineRepeat);
	if (shape.columns === 2) {
		const cols = Math.floor((gridCols - DOC_GUTTER) / 2);
		return { cols, rows, capacity: 2 * cols * rows };
	}
	return { cols: gridCols, rows, capacity: gridCols * rows };
}

const NEWLINES = /\n/g;

function nativeRenderOptions(shape: Shape, size: number) {
	return {
		size,
		font: shape.font,
		cellWidth: shape.cellWidth,
		cellHeight: shape.cellHeight,
		stretch: shape.stretch,
		variant: shape.variant,
		lineRepeat: shape.lineRepeat,
		columns: shape.columns,
	};
}

function renderedChars(text: string, shape: Shape, geo: Geometry): number {
	if (shape.columns === 2) {
		let visible = [...text].length - (text.match(DIM_MARKERS)?.length ?? 0);
		visible -= text.match(NEWLINES)?.length ?? 0;
		return Math.min(visible, geo.capacity);
	}
	// Grid: count visible chars that fit within the frame's cell budget, with
	// wide glyphs taking two cells (and a straddle pad) exactly as the renderer.
	const wideCells = usesWideCells(shape);
	let cell = 0;
	let count = 0;
	for (const ch of text) {
		const w = charCells(ch, wideCells);
		if (w === 0) continue;
		let at = cell;
		if (w === 2 && geo.cols >= 2 && at % geo.cols === geo.cols - 1) at += 1;
		if (at + w > geo.capacity) break;
		cell = at + w;
		count++;
	}
	return count;
}

/** Render one snapcompact frame from already-normalized text. Doc shapes
 *  (`columns === 2`) expect one page of `\n`-joined pre-wrapped lines. */
export async function render(text: string, shape: Shape, size: number = shape.frameSize): Promise<RenderedFrame> {
	const geo = geometry(shape, size);
	const { cols, rows } = geo;
	const chars = renderedChars(text, shape, geo);
	const data = await renderSnapcompactPng(text, nativeRenderOptions(shape, size));
	return { data, cols, rows, chars };
}

/** Stateful per-page text finisher: re-opens a dim span the previous page
 *  boundary cut through, then applies stopword dimming when the shape asks
 *  for it (after pagination, so capacity math never sees the markers). */
function pageFinisher(shape: Shape): (page: string) => string {
	let dimOpen = false;
	return page => {
		const text = dimOpen ? DIM_ON + page : page;
		dimOpen = text.lastIndexOf(DIM_ON) > text.lastIndexOf(DIM_OFF);
		return shape.stopwordDim ? dimStopwords(text) : text;
	};
}

/** Options for {@link renderMany} and {@link frames}. */
export interface RenderManyOptions {
	/** Explicit shape; wins over `model`. */
	shape?: Shape;
	/** Model whose provider API and id select the frame shape. */
	model?: ShapeTarget;
	/** Frame edge in px; defaults to the shape's `frameSize`. */
	frameSize?: number;
	/** Hard cap on frames produced; omit for unbounded (caller decides usage). */
	maxFrames?: number;
}

/**
 * Render arbitrary text into snapcompact PNG frames as LLM image blocks
 * (first page first). Empty/whitespace-only input yields no frames.
 */
export async function renderMany(text: string, options?: RenderManyOptions): Promise<ImageContent[]> {
	const shape = options?.shape ?? resolveShapeForText(text, options?.model);
	const frameSize = options?.frameSize ?? shape.frameSize;
	const geo = geometry(shape, frameSize);
	const normalized = normalize(text, { shape });
	const cap = options?.maxFrames;
	// Build the per-frame texts in order first (cheap, synchronous), then fan
	// the native PNG renders out concurrently — render() is async/off-thread,
	// so awaiting each before starting the next leaves throughput on the table.
	const pageTexts: string[] = [];
	const wideCells = usesWideCells(shape);
	if (shape.columns === 2) {
		const finish = pageFinisher(shape);
		for (const page of docPages(normalized, geo, wideCells)) {
			if (cap !== undefined && pageTexts.length >= cap) break;
			pageTexts.push(finish(page));
		}
	} else {
		for (const page of paginateCells(normalized, geo.capacity, geo.cols, wideCells)) {
			if (cap !== undefined && pageTexts.length >= cap) break;
			pageTexts.push(shape.stopwordDim ? dimStopwords(page) : page);
		}
	}
	const rendered = await Promise.all(pageTexts.map(page => render(page, shape, frameSize)));
	return rendered.map(frame => ({
		type: "image",
		data: frame.data,
		mimeType: "image/png",
		...(shape.imageDetail ? { detail: shape.imageDetail } : {}),
	}));
}

/** Frames needed to hold `text` at the given shape/size, without rendering.
 *  For doc shapes this wraps the text once and counts pages of `2 * rows`
 *  lines; for grid shapes it divides by the frame capacity. */
export function frames(text: string, options?: Pick<RenderManyOptions, "shape" | "model" | "frameSize">): number {
	const shape = options?.shape ?? resolveShapeForText(text, options?.model);
	const geo = geometry(shape, options?.frameSize ?? shape.frameSize);
	const normalized = normalize(text, { shape });
	const wideCells = usesWideCells(shape);
	if (shape.columns === 2) return Math.ceil(wrap(normalized, geo.cols, wideCells).length / (2 * geo.rows));
	return paginateCells(normalized, geo.capacity, geo.cols, wideCells).length;
}

// ============================================================================
// Archive helpers
// ============================================================================

/** Validate and extract a persisted frame archive from `preserveData`. */
export function getPreservedArchive(preserveData: Record<string, unknown> | undefined): Archive | undefined {
	const candidate = preserveData?.[PRESERVE_KEY];
	if (!candidate || typeof candidate !== "object") return undefined;
	const archive = candidate as Archive;
	const frames = Array.isArray(archive.frames)
		? archive.frames.filter(
				frame =>
					!!frame &&
					typeof frame.data === "string" &&
					frame.data.length > 0 &&
					typeof frame.mimeType === "string" &&
					typeof frame.cols === "number" &&
					typeof frame.rows === "number" &&
					typeof frame.chars === "number",
			)
		: [];
	const text = typeof archive.text === "string" && archive.text.length > 0 ? archive.text : undefined;
	const textHead = typeof archive.textHead === "string" && archive.textHead.length > 0 ? archive.textHead : undefined;
	const textTail = typeof archive.textTail === "string" && archive.textTail.length > 0 ? archive.textTail : undefined;
	// A text-only archive (everything fit in the plain-text regions) is valid;
	// only an archive carrying neither frames nor text is empty.
	const representation = preserveData?.sourceRepresentation as SourceRepresentation | undefined;
	if (frames.length === 0 && text === undefined && textHead === undefined && textTail === undefined && !representation?.layout.some(part => part.kind === "original-image")) return undefined;
	return {
		frames,
		totalChars: typeof archive.totalChars === "number" ? archive.totalChars : 0,
		truncatedChars: typeof archive.truncatedChars === "number" ? archive.truncatedChars : 0,
		...(text !== undefined ? { text } : {}),
		...(textHead !== undefined ? { textHead } : {}),
		...(textTail !== undefined ? { textTail } : {}),
	};
}

/** Drop the persisted frame archive ({@link PRESERVE_KEY}) from `preserveData`,
 *  returning the remaining state — or `undefined` when nothing else remains, so
 *  an empty `{}` is never persisted. Callers strip the archive once its frames
 *  have been migrated into a new compaction's text, preventing the stale frames
 *  from leaking back into the rebuilt context. */
export function stripPreservedArchive(
	preserveData: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
	if (!preserveData || !(PRESERVE_KEY in preserveData)) return preserveData;
	const { [PRESERVE_KEY]: _removed, ...rest } = preserveData;
	return Object.keys(rest).length > 0 ? rest : undefined;
}

/** Extract persisted archive source text as plain text for LLM summarization. */
export function archiveSourceText(archive: Archive): string | undefined {
	const text =
		archive.text ??
		[archive.textHead, archive.textTail]
			.filter((part): part is string => typeof part === "string" && part.length > 0)
			.join(NEWLINE_GLYPH);
	return text.length > 0 ? elideDataUrls(toPlainText(text), "archive") : undefined;
}

/** Build the text used to choose and preflight a font-aware snapcompact shape. */
export function renderabilityProbeText(
	serialized: string,
	previousPreserveData?: Record<string, unknown>,
	previousSummary?: string,
): string {
	const previousArchive = getPreservedArchive(previousPreserveData);
	const previousText = previousArchive ? (archiveSourceText(previousArchive) ?? "") : "";
	if (previousText.length > 0) return `${previousText}${NEWLINE_GLYPH}${serialized}`;
	if (previousSummary) return `${previousSummary}${NEWLINE_GLYPH}${serialized}`;
	return serialized;
}

/** Options for reconstructing a persisted snapcompact archive into prompt blocks. */
export interface HistoryBlockOptions {
	sourceRepresentation?: SourceRepresentation;
	resolveSourceImage?: (part: Extract<SourceLayoutPart, { kind: "original-image" }>) => ImageContent | undefined;
	/** Actual returned block and position; source parts are emitted by the host. */
	onEmit?: (layoutIndex: number, blockIndex: number, block: TextContent | ImageContent) => void;
	/** Hard cap on image base64 bytes attached to one rebuilt provider request. */
	maxFrameDataBytes?: number;
}

function formatFrameDataBytes(bytes: number): string {
	if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
	if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} KB`;
	return `${bytes} B`;
}

function imagesWithinBudget(
	archive: Archive,
	maxFrameDataBytes: number | undefined,
): { images: ImageContent[]; omittedFrames: number; omittedBytes: number } {
	if (maxFrameDataBytes === undefined) {
		return { images: images(archive), omittedFrames: 0, omittedBytes: 0 };
	}

	let usedBytes = 0;
	let omittedFrames = 0;
	let omittedBytes = 0;
	const keptNewestFirst: Frame[] = [];
	for (let index = archive.frames.length - 1; index >= 0; index--) {
		const frame = archive.frames[index];
		if (!frame) continue;
		const bytes = frame.data.length;
		if (usedBytes + bytes > maxFrameDataBytes) {
			omittedFrames++;
			omittedBytes += bytes;
			continue;
		}
		usedBytes += bytes;
		keptNewestFirst.push(frame);
	}
	keptNewestFirst.reverse();
	return { images: images({ ...archive, frames: keptNewestFirst }), omittedFrames, omittedBytes };
}

function omittedFrameNotice(omittedFrames: number, omittedBytes: number): string {
	return [
		"-------------- snapcompact image middle omitted",
		`${omittedFrames.toLocaleString()} archived image frame${omittedFrames === 1 ? "" : "s"} (${formatFrameDataBytes(omittedBytes)} base64) exceeded the per-request snapcompact payload budget. The compacted summary and visible text edges remain available.`,
		"--------------",
	].join("\n");
}

/** Convert archive frames into LLM image blocks (oldest first). */
export function images(archive: Archive): ImageContent[] {
	return archive.frames.map(frame => ({
		type: "image",
		data: frame.data,
		mimeType: frame.mimeType,
		...(frame.detail ? { detail: frame.detail } : {}),
	}));
}
/** Ordered archive blocks for a compaction summary message, oldest to newest:
 *  the oldest text region, the imaged middle, then the newest text region.
 *  Runtime-only; reconstructed from {@link Archive} on each context rebuild
 *  instead of persisted on the session entry. */
export function historyBlocks(archive: Archive, options: HistoryBlockOptions = {}): (TextContent | ImageContent)[] {
	const blocks: (TextContent | ImageContent)[] = [];

	if (options.sourceRepresentation?.version === 1) {
		let usedBytes = 0;
		let overflow = false;
		for (const [layoutIndex, part] of options.sourceRepresentation.layout.entries()) {
			let block: TextContent | ImageContent | undefined;
			if (part.kind === "source") continue;
			if (part.kind === "text") block = { type: "text", text: toPlainText((archive.text ?? "").slice(part.range.start, part.range.end)) };
			else if (part.kind === "frame") {
				const frame = archive.frames[part.frameIndex];
				if (frame && !overflow && (options.maxFrameDataBytes === undefined || usedBytes + frame.data.length <= options.maxFrameDataBytes)) {
					block = { type: "image", data: frame.data, mimeType: frame.mimeType, ...(frame.detail ? { detail: frame.detail } : {}) }; usedBytes += frame.data.length;
				} else { overflow = true; block = { type: "text", text: toPlainText((archive.text ?? "").slice(part.range.start, part.range.end)) }; }
			} else if (part.kind === "original-image") {
				block = options.resolveSourceImage?.(part);
				if (!block) throw new Error("Original snapcompact source image is unavailable: " + part.entryId + ":" + (part.currentBlockIndex ?? part.blockIndex));
			} else if (part.kind === "gap") {
				if (part.reason === "omitted-messages" && (part.wholeMessages ?? 0) > 0) block = { type: "text", text: "[" + part.wholeMessages + " earlier source messages omitted]" };
				else if (part.reason === "partial-text") block = { type: "text", text: "[truncated]" };
				else if (part.reason === "image-deleted") block = { type: "text", text: "[source image deleted]" };
				else if (part.reason === "unknown-source") block = { type: "text", text: "[earlier source coverage unknown]" };
			}
			const previousPart = options.sourceRepresentation.layout[layoutIndex - 1];
			if (block?.type === "text" && (part.kind === "text" || part.kind === "frame") && previousPart?.kind === "frame" && blocks.at(-1)?.type === "image" && previousPart.range.end === part.range.start) {
				const boundary = part.range.start;
				if (options.sourceRepresentation.coverage.some(run => run.normalized && run.normalized.start < boundary && run.normalized.end > boundary)) block.text = "[continued]\n" + block.text;
			}
			if (block) { options.onEmit?.(layoutIndex, blocks.length, block); blocks.push(block); }
		}
		return blocks;
	}
	const budgeted = imagesWithinBudget(archive, options.maxFrameDataBytes);
	const hasImages = budgeted.images.length > 0;
	const hasOmittedImages = budgeted.omittedFrames > 0;
	if (archive.textHead) {
		const suffix = hasImages
			? "\n-------------- imaged middle below\n"
			: hasOmittedImages
				? `\n${omittedFrameNotice(budgeted.omittedFrames, budgeted.omittedBytes)}\n`
				: "";
		blocks.push({ type: "text", text: elideDataUrls(toPlainText(archive.textHead), "archive") + suffix });
	} else if (hasOmittedImages && !hasImages) {
		blocks.push({ type: "text", text: omittedFrameNotice(budgeted.omittedFrames, budgeted.omittedBytes) });
	}
	// Omitted frames are the OLDEST archived images: the byte budget keeps the
	// newest tail frames, so the gap notice precedes the kept images to keep the
	// reconstructed blocks oldest-to-newest.
	if (hasImages && hasOmittedImages) {
		blocks.push({ type: "text", text: omittedFrameNotice(budgeted.omittedFrames, budgeted.omittedBytes) });
	}
	blocks.push(...budgeted.images);
	if (archive.textTail) {
		const prefix = hasImages
			? "-------------- imaged middle above\n"
			: archive.truncatedChars > 0 || hasOmittedImages
				? "\n-------------- middle history omitted above\n"
				: "";
		const tail = prefix + elideDataUrls(toPlainText(archive.textTail), "archive");
		const lastBlock = blocks[blocks.length - 1];
		if (lastBlock?.type === "text") {
			lastBlock.text += tail;
		} else {
			blocks.push({ type: "text", text: tail });
		}
	}
	return blocks;
}

// ============================================================================
// Compaction entry point
// ============================================================================

/** Denser companion of `high` for the foveated archive middle: same family and
 *  frame size (identical per-frame bill) but a tighter cell. Returns `high`
 *  unchanged for doc layouts, TrueType Unicode shapes, or when no denser
 *  variant exists (foveation off). */
function denseCompanion(high: Shape, api: Api | undefined): Shape {
	if (high.columns === 2 || high.font === "silver") return high;
	const family = billingFamily(api);
	const low = priceShape({ ...SHAPE_VARIANTS[FAMILY_VARIANT_LOW[family]], frameSize: high.frameSize }, family);
	return geometry(low).capacity > geometry(high).capacity ? low : high;
}

/** A rendered page and the exact consumed normalized-source interval. */
interface PlanFrame { text: string; shape: Shape; range: SourceRange }
interface ArchiveLayout {
	frames: PlanFrame[];
	textHead: string;
	textTail: string;
	keptText: string;
	truncatedChars: number;
	/** Original input intervals, in physical order, before packing Archive.text. */
	ranges: SourceRange[];
}

/** Word wrapping carries offsets while consuming tokens, including hard splits.
 * Newline insertion is presentation only; persisted source is never reconstructed
 * from wrapped lines. */
function sourcePages(text: string, shape: Shape, offset = 0, maxPages = Infinity): PlanFrame[] {
	const geo = geometry(shape);
	if (shape.columns !== 2) {
		return paginateCells(text, geo.capacity, geo.cols, usesWideCells(shape), maxPages).map(page => {
			const range = { start: offset, end: offset + page.length };
			offset = range.end;
			return { text: page, shape, range };
		});
	}
	const lines: { text: string; start: number }[] = [];
	let current = "";
	let currentCells = 0;
	let start = 0;
	const wide = usesWideCells(shape);
	const flush = () => { if (current) lines.push({ text: current, start }); current = ""; currentCells = 0; };
	for (const token of text.matchAll(/\S+/g)) {
		if (lines.length > 2 * geo.rows * maxPages) break;
		let word = token[0];
		let at = token.index;
		let cells = cellLength(word, wide);
		while (cells > geo.cols) {
			flush();
			const head = sliceCells(word, geo.cols, wide);
			lines.push({ text: head, start: at });
			at += head.length; word = word.slice(head.length); cells = cellLength(word, wide);
		}
		if (!current) { current = word; currentCells = cells; start = at; }
		else if (currentCells + 1 + cells <= geo.cols) { current += " " + word; currentCells += 1 + cells; }
		else { flush(); current = word; currentCells = cells; start = at; }
	}
	flush();
	const pages: PlanFrame[] = [];
	const perPage = 2 * geo.rows;
	for (let i = 0; i < lines.length && pages.length < maxPages; i += perPage) {
		pages.push({ text: lines.slice(i, i + perPage).map(line => line.text).join("\n"), shape,
			range: { start: offset + (i === 0 ? 0 : lines[i]!.start), end: offset + (lines[i + perPage]?.start ?? text.length) } });
	}
	return pages;
}

/** Vanilla selection chooses coverage exactly once. Insertion never calls this
 * planner on the union: the source-preserving prefix partition below does. */
function planArchive(text: string, high: Shape, low: Shape, maxFrames: number): ArchiveLayout {
	const edge = TEXT_EDGE_PAGES * geometry(high).capacity;
	if (text.length <= 2 * edge) return { frames: [], textHead: text, textTail: "", keptText: text, truncatedChars: 0, ranges: [{ start: 0, end: text.length }] };
	const textHead = text.slice(0, edge);
	const textTail = text.slice(-edge);
	const middleEnd = text.length - edge;
	let planned = sourcePages(text.slice(edge, middleEnd), high, edge);
	if (planned.length > maxFrames) {
		if (high.columns === 2) {
			planned = maxFrames > 0 ? [...planned.slice(0, 1), ...(maxFrames > 1 ? planned.slice(-(maxFrames - 1)) : [])] : [];
		} else {
			const edgeFrames = Math.min(HQ_EDGE_FRAMES, Math.floor(Math.max(0, maxFrames - 1) / 2));
			const head = planned.slice(0, edgeFrames);
			const tail = edgeFrames > 0 ? planned.slice(-edgeFrames) : [];
			const start = head.at(-1)?.range.end ?? edge;
			const end = tail[0]?.range.start ?? middleEnd;
			const dense = sourcePages(text.slice(start, end), low, start);
			const count = maxFrames - 2 * edgeFrames;
			planned = [...head, ...(count > 0 ? dense.slice(-count) : []), ...tail];
		}
	}
	const ranges = [{ start: 0, end: edge }, ...planned.map(page => page.range), { start: middleEnd, end: text.length }];
	const keptText = ranges.map(range => text.slice(range.start, range.end)).join("");
	return { frames: planned, textHead, textTail, keptText, truncatedChars: text.length - keptText.length, ranges };
}

/**
 * Drop `¶think:` sections from serialized archive source text.
 *
 * Archives written before {@link SerializeOptions.includeThinking} existed bake
 * reasoning into their kept source; replaying it to Claude trips the
 * `reasoning_extraction` classifier (issue #6093). Re-compaction re-renders the
 * whole unfolded source, so scrubbing the prior text heals a poisoned session
 * at its next compaction. Conservative by construction: only sections that
 * start with `¶think:` at a section boundary are dropped.
 */
function stripThinkingSections(text: string): string {
	return text
		.split(NEWLINE_GLYPH)
		.map(segment =>
			segment
				.split(/\n\n(?=¶(?:user|think|ai|call):)/)
				.filter(section => !section.startsWith("¶think:"))
				.join("\n\n"),
		)
		.filter(segment => segment.length > 0)
		.join(NEWLINE_GLYPH);
}

interface NormalizationRun { input: SourceRange; output: SourceRange }
interface MappedArchiveInput {
	text: string;
	coverage: SourceCoverageRun[];
	images: { part: Extract<SourceLayoutPart, { kind: "original-image" }>; offset: number; atomicGroup?: SourceMessage<Message>["atomicGroup"] }[];
}

/** Forward normalization map. Expansions/contractions remain indivisible runs;
 * affine runs coalesce, so ordinary ASCII costs one descriptor, not one per char. */
function normalizeTracked(text: string, shape: Shape): { text: string; runs: NormalizationRun[] } {
	const runs: NormalizationRun[] = [];
	const chunks: string[] = [];
	const cache = new Map<string, string>();
	let length = 0;
	const emit = (start: number, end: number, value: string) => {
		if (!value) return;
		const last = runs.at(-1);
		if (last && last.input.end === start && last.output.end === length && last.input.end - last.input.start === last.output.end - last.output.start && end - start === value.length) {
			last.input.end = end; last.output.end += value.length;
		} else runs.push({ input: { start, end }, output: { start: length, end: length + value.length } });
		chunks.push(value); length += value.length;
	};
	const replacements: { start: number; end: number; value: string }[] = [];
	elideDataUrls(text, "source", (start, end, value) => replacements.push({ start, end, value }));
	let replacementIndex = 0;
	// ANSI is consumed in place, not stripped and reverse-matched afterwards.
	const tokens = replacements.length > 0
		? /\u001b(?:\[[0-?]*[ -/]*[@-~]|\][^\u0007]*(?:\u0007|\u001b\\))|[\s\p{Cf}]+|[\s\S]/gu
		: /\u001b(?:\[[0-?]*[ -/]*[@-~]|\][^\u0007]*(?:\u0007|\u001b\\))|[\s\p{Cf}]+|[!-~\u00a1-\u00ac\u00ae-\u00ff]+|[\s\S]/gu;
	for (const match of text.matchAll(tokens)) {
		while (replacementIndex < replacements.length && replacements[replacementIndex]!.end <= match.index) replacementIndex++;
		const replacement = replacements[replacementIndex];
		if (replacement && match.index >= replacement.start && match.index < replacement.end) {
			if (match.index === replacement.start) emit(replacement.start, replacement.end, normalize(replacement.value, { shape }));
			continue;
		}
		const token = match[0];
		let value: string;
		if (token.startsWith("\u001b")) value = "";
		else if (/^[\s\p{Cf}]+$/u.test(token)) value = LINE_BREAK.test(token) ? NEWLINE_GLYPH : /[^\p{Cf}]/u.test(token) ? " " : "";
		else if (token === DIM_ON || token === DIM_OFF || token === NEWLINE_GLYPH) value = token;
		else if (isAsciiOrLatin1(token.charCodeAt(0))) value = token;
		else { value = cache.get(token) ?? normalize(token, { shape }); cache.set(token, value); }
		emit(match.index, match.index + token.length, value);
	}
	const raw = chunks.join("");
	const leading = /^[ \u2588]+/.exec(raw)?.[0].length ?? 0;
	const trailing = /[ \u2588]+$/.exec(raw)?.[0].length ?? 0;
	const end = Math.max(leading, raw.length - trailing);
	const kept: NormalizationRun[] = [];
	for (const run of runs) {
		const start = Math.max(leading, run.output.start), stop = Math.min(end, run.output.end);
		if (stop <= start) continue;
		const affine = run.input.end - run.input.start === run.output.end - run.output.start;
		kept.push({ input: affine ? { start: run.input.start + start - run.output.start, end: run.input.start + stop - run.output.start } : run.input,
			output: { start: start - leading, end: stop - leading } });
	}
	return { text: raw.slice(leading, end), runs: kept };
}

function appendCoverage(runs: SourceCoverageRun[], run: SourceCoverageRun): void {
	const last = runs.at(-1);
	if (last?.normalizedUnit && run.normalizedUnit && last.entryId === run.entryId && last.projection === run.projection && last.status === run.status && last.contribution === run.contribution && last.snapshot.blockIndex === run.snapshot.blockIndex && last.snapshot.start === run.snapshot.start && last.snapshot.end === run.snapshot.end && last.normalized?.end === run.normalized?.start && last.normalizedUnit.end === run.normalizedUnit.start && last.normalizedUnit.length === run.normalizedUnit.length) {
		last.normalized!.end = run.normalized!.end; last.normalizedUnit.end = run.normalizedUnit.end;
		if (last.normalizedUnit.start === 0 && last.normalizedUnit.end === last.normalizedUnit.length) delete last.normalizedUnit;
		return;
	}
	if (last && !last.normalizedUnit && !run.normalizedUnit && last.entryId === run.entryId && last.projection === run.projection && last.status === run.status && last.contribution === run.contribution && last.snapshot.blockIndex === run.snapshot.blockIndex && last.snapshot.end === run.snapshot.start && last.normalized?.end === run.normalized?.start && last.current?.end === run.current?.start &&
		last.normalized && run.normalized && last.snapshot.end - last.snapshot.start === last.normalized.end - last.normalized.start && run.snapshot.end - run.snapshot.start === run.normalized.end - run.normalized.start) {
		last.snapshot.end = run.snapshot.end; last.normalized.end = run.normalized.end;
		if (last.current && run.current) last.current.end = run.current.end;
	} else runs.push(run);
}

function cropCoverage(run: SourceCoverageRun, start: number, end: number, outputStart: number): SourceCoverageRun {
	const normalized = run.normalized!;
	const affine = !run.normalizedUnit && normalized.end - normalized.start === run.snapshot.end - run.snapshot.start;
	const snapshot = affine ? { ...run.snapshot, start: run.snapshot.start + start - normalized.start, end: run.snapshot.start + end - normalized.start } : { ...run.snapshot };
	const current = run.current && affine ? { ...run.current, start: run.current.start + start - normalized.start, end: run.current.start + end - normalized.start } : run.current && { ...run.current };
	const unit = run.normalizedUnit ?? { start: 0, end: normalized.end - normalized.start, length: normalized.end - normalized.start };
	return { ...run, snapshot, current, ...(!affine ? { normalizedUnit: { start: unit.start + start - normalized.start, end: unit.start + end - normalized.start, length: unit.length } } : {}), normalized: { start: outputStart, end: outputStart + end - start } };
}

function captureSources<T>(sources: readonly SourceMessage<T>[], options: Options<T> | undefined, required = false) {
	const messages: Message[] = [];
	const sourceByMessage = new Map<Message, SourceMessage<T>>();
	for (const source of sources) for (const message of (options?.convertToLlm ?? defaultConvertToLlm)([source.message])) {
		messages.push(message); sourceByMessage.set(message, source);
	}

	// The serializer owns actual call/result folding; this is representation
	// metadata for that fold, not a second policy admission/closure pass.
	const results = new Map<string, Message>();
	for (const message of messages) if (message.role === "toolResult") results.set(message.toolCallId, message);
	for (const message of messages) {
		if (message.role !== "assistant") continue;
		const members = [message, ...message.content.flatMap(block => block.type === "toolCall" && results.has(block.id) ? [results.get(block.id)!] : [])];
		if (members.length < 2) continue;
		const first = sourceByMessage.get(message)!;
		const atomicGroup = first.atomicGroup ?? { id: first.entryId, entryIds: [...new Set(members.map(member => sourceByMessage.get(member)!.entryId))] };
		for (const member of members) { const source = sourceByMessage.get(member)!; sourceByMessage.set(member, { ...source, atomicGroup }); }
	}

	const capture: SerializationCapture = { runs: [], images: [], required };
	return { serialized: serializeConversationSource(messages, options, capture), capture, sourceByMessage };
}

function serializeSources<T>(sources: readonly SourceMessage<T>[], options: Options<T> | undefined, shape: Shape, required = false): MappedArchiveInput {
	return mapSerializedSources(captureSources(sources, options, required), sources, shape);
}

function mapSerializedSources<T>(
	{ serialized, capture, sourceByMessage }: ReturnType<typeof captureSources<T>>,
	sources: readonly SourceMessage<T>[],
	shape: Shape,
): MappedArchiveInput {
	const normalized = normalizeTracked(serialized, shape);
	const coverage: SourceCoverageRun[] = [];
	let index = 0;
	for (const trace of capture.runs) {
		const source = sourceByMessage.get(trace.message)!;
		while (index < normalized.runs.length && normalized.runs[index]!.input.end <= trace.start) index++;

		if (trace.rawStart === trace.rawEnd) {
			let start = Infinity, end = 0;
			for (let i = index; i < normalized.runs.length; i++) {
				const mapped = normalized.runs[i]!;
				if (mapped.input.start >= trace.end) break;
				const affine = mapped.input.end - mapped.input.start === mapped.output.end - mapped.output.start;
				start = Math.min(start, affine ? mapped.output.start + Math.max(0, trace.start - mapped.input.start) : mapped.output.start);
				end = Math.max(end, affine ? mapped.output.end - Math.max(0, mapped.input.end - trace.end) : mapped.output.end);
			}
			if (end > start) { const snapshot = { blockIndex: trace.blockIndex, start: 0, end: 0 }; coverage.push({ entryId: source.entryId, ...(source.projection ? { projection: source.projection } : {}), order: source.order, atomicGroup: source.atomicGroup, snapshot,
				...(trace.exact ? { current: { ...snapshot } } : {}), normalized: { start, end }, status: trace.exact ? "exact-current" : "unknown" }); }
			continue;
		}
		for (let i = index; i < normalized.runs.length; i++) {
			const mapped = normalized.runs[i]!;
			if (mapped.input.start >= trace.end) break;
			const start = Math.max(trace.start, mapped.input.start), end = Math.min(trace.end, mapped.input.end);
			const affine = mapped.input.end - mapped.input.start === mapped.output.end - mapped.output.start;
			const output = affine ? { start: mapped.output.start + start - mapped.input.start, end: mapped.output.start + end - mapped.input.start } : mapped.output;
			const snapshot = { blockIndex: trace.blockIndex, start: trace.exact ? trace.rawStart + start - trace.start : trace.rawStart, end: trace.exact ? trace.rawStart + end - trace.start : trace.rawEnd };
			appendCoverage(coverage, { entryId: source.entryId, ...(source.projection ? { projection: source.projection } : {}), order: source.order, ...(source.atomicGroup ? { atomicGroup: source.atomicGroup } : {}), snapshot,
				...(trace.exact ? { current: { ...snapshot } } : {}), normalized: { ...output }, status: trace.exact ? "exact-current" : "unknown" });
		}
	}
	const offsetAt = (offset: number) => {
		let lo = 0, hi = normalized.runs.length;
		while (lo < hi) { const mid = (lo + hi) >>> 1; if (normalized.runs[mid]!.input.end <= offset) lo = mid + 1; else hi = mid; }
		const run = normalized.runs[lo];
		if (!run) return normalized.text.length;
		return run.input.end - run.input.start === run.output.end - run.output.start ? run.output.start + Math.max(0, offset - run.input.start) : run.output.start;
	};
	return pruneSerializedSources({ text: normalized.text, coverage, images: capture.images.map(image => {
		const source = sourceByMessage.get(image.message)!;
		return { part: { kind: "original-image", entryId: source.entryId, ...(source.projection ? { projection: source.projection } : {}), order: source.order, blockIndex: image.blockIndex, currentBlockIndex: image.blockIndex }, offset: offsetAt(image.offset), atomicGroup: source.atomicGroup };
	}) }, sources);
}

/** Apply already-admitted raw intervals to normalized source, retaining the
 * serializer's block positions. This is not raw-message materialization. */
function pruneSerializedSources<T>(input: MappedArchiveInput, sources: readonly SourceMessage<T>[]): MappedArchiveInput {
	if (!sources.some(source => source.spans)) return input;
	const byId = new Map(sources.map(source => [compactionSourceKey(source), source]));
	const ranges: SourceRange[] = [];
	let cursor = 0;
	for (const run of input.coverage) {
		if (!run.normalized) continue;
		const n = run.normalized;
		if (cursor < n.start) ranges.push({ start: cursor, end: n.start });
		const spans = byId.get(compactionSourceKey(run))?.spans;
		if (!spans) ranges.push({ ...n });
		else if (run.current) for (const span of spans) {
			if (span.blockIndex !== run.current.blockIndex) continue;
			const start = Math.max(span.start, run.current.start), end = Math.min(span.end, run.current.end);
			if (end <= start) continue;
			const affine = n.end - n.start === run.current.end - run.current.start;
			ranges.push(affine ? { start: n.start + start - run.current.start, end: n.start + end - run.current.start } : { ...n });
		}
		cursor = n.end;
	}
	if (cursor < input.text.length) ranges.push({ start: cursor, end: input.text.length });
	ranges.sort((a, b) => a.start - b.start);
	const merged: SourceRange[] = [];
	for (const range of ranges) { const last = merged.at(-1); if (last && range.start <= last.end) last.end = Math.max(last.end, range.end); else merged.push({ ...range }); }
	const coverage: SourceCoverageRun[] = [];
	const chunks: string[] = [];
	const mapped: { old: SourceRange; offset: number }[] = [];
	let length = 0, previousEnd = 0;
	for (const range of merged) {
		if (range.start > previousEnd) { chunks.push("[truncated]"); length += 11; }
		mapped.push({ old: range, offset: length });
		for (const run of input.coverage) {
			if (!run.normalized || run.normalized.end <= range.start) continue;
			if (run.normalized.start >= range.end) break;
			const start = Math.max(range.start, run.normalized.start), end = Math.min(range.end, run.normalized.end);
			appendCoverage(coverage, cropCoverage(run, start, end, length + start - range.start));
		}
		chunks.push(input.text.slice(range.start, range.end)); length += range.end - range.start; previousEnd = range.end;
	}
	if (previousEnd < input.text.length) chunks.push("[truncated]");
	const images = input.images.filter(image => { const spans = byId.get(compactionSourceKey(image.part))?.spans; return !spans || spans.some(span => span.blockIndex === image.part.blockIndex); })
		.map(image => { const run = mapped.find(range => range.old.end >= image.offset); return { ...image, offset: run ? run.offset + Math.max(0, image.offset - run.old.start) : length }; });
	return { text: chunks.join(""), coverage, images };
}


interface SourcePiece {
	text: string;
	coverage: SourceCoverageRun[];
	order: number;
	blockIndex: number;
	rawStart: number;
	historicalAtom?: boolean;
	/** Frozen ordinary edge, never recomputed after insertion. */
	edge?: "head" | "tail";
	image?: Extract<SourceLayoutPart, { kind: "original-image" }>;
	notice?: Extract<SourceLayoutPart, { kind: "gap" }>;
}

function coveredRawRanges(runs: readonly SourceCoverageRun[], blockIndex: number): SourceBlockRange[] {
	const covered: SourceBlockRange[] = [];
	const partial = new Map<string, { raw: SourceBlockRange; length: number; ranges: SourceRange[] }>();
	for (const run of runs) {
		if (run.status !== "exact-current" || run.current?.blockIndex !== blockIndex) continue;
		const unit = run.normalizedUnit;
		if (!unit || (unit.start === 0 && unit.end === unit.length)) { covered.push(run.current); continue; }
		const key = run.current.start + ":" + run.current.end + ":" + unit.length;
		const group = partial.get(key);
		if (group) group.ranges.push(unit);
		else partial.set(key, { raw: run.current, length: unit.length, ranges: [unit] });
	}
	for (const group of partial.values()) {
		let end = 0;
		for (const range of group.ranges.sort((a, b) => a.start - b.start)) {
			if (range.start > end) break;
			end = Math.max(end, range.end);
		}
		if (end === group.length) covered.push(group.raw);
	}
	return covered.sort((a, b) => a.start - b.start);
}

function selectedMissing(span: SourceBlockRange, runs: readonly SourceCoverageRun[]): SourceBlockRange[] {
	const covered = coveredRawRanges(runs, span.blockIndex);
	const missing: SourceBlockRange[] = [];
	let at = span.start;
	for (const interval of covered) {
		if (interval.end <= at) continue;
		if (interval.start >= span.end) break;
		if (interval.start > at) missing.push({ ...span, start: at, end: Math.min(interval.start, span.end) });
		at = Math.max(at, interval.end);
	}
	if (at < span.end) missing.push({ ...span, start: at });
	return missing;
}

/** Slice by forward coverage coordinates. Unattributed serializer punctuation
 * stays attached to its neighboring source; no text search recovers identity. */
function sourcePieces(input: MappedArchiveInput, ranges: readonly SourceRange[], headEnd: number, tailStart: number): SourcePiece[] {
	const pieces: SourcePiece[] = [];
	const groupOrder = new Map<string, number>();
	for (const run of input.coverage) if (run.atomicGroup) groupOrder.set(run.atomicGroup.id, Math.min(groupOrder.get(run.atomicGroup.id) ?? Infinity, run.order));
	const historicalGroups = new Set(input.coverage.filter(run => run.atomicGroup && run.status === "historical-not-current").map(run => run.atomicGroup!.id));
	for (const range of ranges) {
		let at = range.start;
		for (const run of input.coverage) {
			const n = run.normalized;
			if (!n || n.end <= range.start) continue;
			if (n.start >= range.end) break;
			const start = Math.max(n.start, range.start), end = Math.min(n.end, range.end);
			if (end <= start) continue;
			const pieceStart = Math.min(at, start);

			const stops = [...input.images.filter(image => image.offset > pieceStart && image.offset < end).map(image => image.offset), end].sort((a, b) => a - b);
			let segmentStart = pieceStart;
			for (const segmentEnd of stops) {
				if (segmentEnd <= segmentStart) continue;
				const rawStart = Math.max(start, segmentStart);
				const cropped = segmentEnd > rawStart ? cropCoverage(run, rawStart, segmentEnd, rawStart - segmentStart) : undefined;
				pieces.push({ text: input.text.slice(segmentStart, segmentEnd), coverage: cropped ? [cropped] : [], order: run.atomicGroup ? groupOrder.get(run.atomicGroup.id)! : run.order,
					blockIndex: run.atomicGroup ? -1 : run.current?.blockIndex ?? run.snapshot.blockIndex, rawStart: run.atomicGroup ? segmentStart : cropped?.current?.start ?? cropped?.snapshot.start ?? run.snapshot.start,
					historicalAtom: !!run.atomicGroup && historicalGroups.has(run.atomicGroup.id),
					...(range.end <= headEnd ? { edge: "head" as const } : range.start >= tailStart ? { edge: "tail" as const } : {}) });
				segmentStart = segmentEnd;
			}
			at = end;
		}
		if (at < range.end) {
			const last = pieces.at(-1);
			if (last && at > range.start) last.text += input.text.slice(at, range.end);
			else {
				const next = input.coverage.find(run => (run.normalized?.start ?? -1) >= range.end);
				const previous = input.coverage.findLast(run => run.normalized && run.normalized.end <= range.start);
				pieces.push({ text: input.text.slice(at, range.end), coverage: [], order: next?.order ?? (previous ? previous.order + 0.5 : Number.NEGATIVE_INFINITY), blockIndex: -1, rawStart: 0,
					...(range.end <= headEnd ? { edge: "head" as const } : range.start >= tailStart ? { edge: "tail" as const } : {}) });
			}
		}
	}
	return pieces;
}

async function renderPlanned(page: PlanFrame, dimOpen: boolean): Promise<{ frame: Frame; dimOpen: boolean }> {
	let text = dimOpen ? DIM_ON + page.text : page.text;
	dimOpen = text.lastIndexOf(DIM_ON) > text.lastIndexOf(DIM_OFF);
	if (page.shape.stopwordDim) text = dimStopwords(text);
	const rendered = await render(text, page.shape);
	return { frame: { ...rendered, mimeType: "image/png", font: page.shape.font, variant: page.shape.variant, lineRepeat: page.shape.lineRepeat,
		...(page.shape.columns === 2 ? { columns: 2 } : {}), ...(page.shape.stopwordDim ? { stopwordDim: true } : {}), ...(page.shape.imageDetail ? { detail: page.shape.imageDetail } : {}) }, dimOpen };
}

/** Rebuild only serializer-generated notices outside mapped source bytes. */
function refreshSelectedUserGaps<T>(pieces: SourcePiece[], selected: readonly SourceMessage<T>[], shape: Shape): void {
	const users = new Map(selected.filter(source => (source.message as Message).role === "user").map(source => [compactionSourceKey(source), source.message as Extract<Message, { role: "user" }>]));
	const ends = new Map<string, number>();
	const last = new Map<string, SourcePiece>();
	for (const piece of pieces) {
		const run = piece.coverage[0];
		if (!run?.current || !run.normalized || !users.has(compactionSourceKey(run)) || run.status !== "exact-current") continue;
		const n = run.normalized;
		let prefix = piece.text.slice(0, n.start).replaceAll("[truncated]", "");
		const suffix = piece.text.slice(n.end).replaceAll("[truncated]", "");
		const key = compactionSourceKey(run) + ":" + run.current.blockIndex;
		const message = users.get(compactionSourceKey(run))!;
		const block = typeof message.content === "string" ? undefined : message.content[run.current.blockIndex];
		const raw = typeof message.content === "string" ? message.content : block?.type === "text" ? block.text : "";
		if (run.current.start > (ends.get(key) ?? 0) && normalize(raw.slice(ends.get(key) ?? 0, run.current.start), { shape })) prefix += "[truncated]";
		const body = piece.text.slice(n.start, n.end);
		piece.text = prefix + body + suffix;
		run.normalized = { start: prefix.length, end: prefix.length + body.length };
		ends.set(key, Math.max(ends.get(key) ?? 0, run.current.end));
		last.set(key, piece);
	}
	for (const [key, piece] of last) {
		const run = piece.coverage[0]!;
		const message = users.get(compactionSourceKey(run))!;
		const block = typeof message.content === "string" ? undefined : message.content[run.current!.blockIndex];
		const raw = typeof message.content === "string" ? message.content : block?.type === "text" ? block.text : "";
		if (normalize(raw.slice(ends.get(key)!), { shape })) {
			const at = run.normalized!.end;
			piece.text = piece.text.slice(0, at) + "[truncated]" + piece.text.slice(at);
		}
	}
}

/** Representation-only monotone writer. Retention and slot shapes are already fixed. */
async function writeArchiveLayout(
	text: string, vanilla: ArchiveLayout, headAnchor: number, tailAnchor: number,
	imageOffsets: readonly { offset: number; part: Extract<SourceLayoutPart, { kind: "original-image" | "gap" }> }[],
	unchanged: boolean, byteBudget: number,
): Promise<{ frames: Frame[]; layout: SourceLayoutPart[]; overflow: boolean }> {
	const layout: SourceLayoutPart[] = [];
	const frames: Frame[] = [];
	let usedBytes = 0, dimOpen = false, overflow = false;
	const emitText = (start: number, end: number) => {
		if (end <= start) return;
		layout.push({ kind: "text", range: { start, end } });
		dimOpen = text.lastIndexOf(DIM_ON, end - 1) > text.lastIndexOf(DIM_OFF, end - 1);
	};
	const emitPages = async (start: number, end: number) => {
		if (end <= start) return;
		if (overflow) { emitText(start, end); return; }
		let at = start;
		while (at < end) {
			if (frames.length >= vanilla.frames.length) { overflow = true; emitText(at, end); break; }
			const shape = vanilla.frames[frames.length]!.shape;
			const page = sourcePages(text.slice(at, end), shape, at, 1)[0];
			if (!page) { emitText(at, end); break; }
			const rendered = await renderPlanned(page, dimOpen);
			if (usedBytes + rendered.frame.data.length > byteBudget) { overflow = true; emitText(at, end); break; }
			layout.push({ kind: "frame", frameIndex: frames.length, range: page.range });
			frames.push(rendered.frame); usedBytes += rendered.frame.data.length; dimOpen = rendered.dimOpen; at = page.range.end;
		}
	};
	if (unchanged) {
		// No insertion: preserve the actual vanilla frame pixels and edge layout.
		let at = 0;
		emitText(0, vanilla.textHead.length); at += vanilla.textHead.length;
		dimOpen = vanilla.textHead.lastIndexOf(DIM_ON) > vanilla.textHead.lastIndexOf(DIM_OFF);
		for (const page of vanilla.frames) {
			const end = at + page.range.end - page.range.start;
			if (!overflow) {
				const rendered = await renderPlanned(page, dimOpen);
				if (usedBytes + rendered.frame.data.length <= byteBudget) { layout.push({ kind: "frame", frameIndex: frames.length, range: { start: at, end } }); frames.push(rendered.frame); usedBytes += rendered.frame.data.length; dimOpen = rendered.dimOpen; }
				else { overflow = true; emitText(at, end); }
			} else emitText(at, end);
			at = end;
		}
		emitText(at, text.length);
		for (const event of imageOffsets) { const index = layout.findIndex(part => (part.kind === "text" || part.kind === "frame") && part.range.start >= event.offset); layout.splice(index < 0 ? layout.length : index, 0, event.part); }
	} else {
		let at = 0;
		const emitInterval = async (end: number) => {
			if (at < headAnchor) { const stop = Math.min(end, headAnchor); emitText(at, stop); at = stop; }
			if (at < end && at < tailAnchor) { const stop = Math.min(end, tailAnchor); await emitPages(at, stop); at = stop; }
			if (at < end) { emitText(at, end); at = end; }
		};
		for (const image of imageOffsets) { await emitInterval(image.offset); layout.push(image.part); }
		await emitInterval(text.length);
	}
	return { frames, layout, overflow };
}

/** Serialize once, retain ordinary ranges once, then place the chronological union. */
export async function compact<T = Message>(preparation: CompactionPreparation<T>, options?: Options<T>): Promise<CompactionResult> {
	if (!preparation.firstKeptEntryId) throw new Error("First kept entry has no ID - session may need migration");
	const previous = getPreservedArchive(preparation.previousPreserveData);
	const prior = preparation.previousPreserveData?.sourceRepresentation as SourceRepresentation | undefined;
	const sourceAware = preparation.sourcesToSummarize !== undefined || preparation.turnPrefixSources !== undefined || prior?.version === 1;
	if (sourceAware && previous && prior?.version !== 1) throw new Error("Legacy snapcompact archive requires original-source rematerialization before source-aware compaction");
	const sources = [...(preparation.sourcesToSummarize ?? []), ...(preparation.turnPrefixSources ?? [])];
	const captured = preparation.sourcesToSummarize !== undefined || preparation.turnPrefixSources !== undefined ? captureSources(sources, options) : undefined;
	const probe = captured?.serialized ?? serializeConversation((options?.convertToLlm ?? defaultConvertToLlm)(preparation.messagesToSummarize.concat(preparation.turnPrefixMessages)), options);
	const base = options?.shape ?? resolveShapeForText(renderabilityProbeText(probe, preparation.previousPreserveData, preparation.previousSummary), options?.model);
	const high = options?.frameSize === undefined ? base : { ...base, frameSize: options.frameSize };
	const low = denseCompanion(high, options?.model?.api);
	const fresh: MappedArchiveInput = captured ? mapSerializedSources(captured, sources, high) : { text: normalize(probe, { shape: high }), coverage: [], images: [] };
	let priorText = previous?.text ?? [previous?.textHead, previous?.textTail].filter(part => !!part).join(NEWLINE_GLYPH);
	if (!prior) {
		priorText = elideDataUrls(priorText, "archive");
		if (options?.includeThinking === false) priorText = stripThinkingSections(priorText);
	}
	const includedPreviousSummary = !priorText && !!preparation.previousSummary;
	if (includedPreviousSummary) priorText = "[Summary of earlier history] " + normalize(preparation.previousSummary!, { shape: high });
	const separator = priorText && fresh.text ? includedPreviousSummary ? " [Recent conversation] " : NEWLINE_GLYPH : "";
	const prefix = priorText.length + separator.length;
	const input: MappedArchiveInput = {
		text: priorText + separator + fresh.text,
		coverage: [...(prior?.coverage ?? []).map(run => ({ ...run, snapshot: { ...run.snapshot }, current: run.current && { ...run.current }, normalized: run.normalized && { ...run.normalized } })),
			...fresh.coverage.map(run => ({ ...run, normalized: run.normalized && { start: run.normalized.start + prefix, end: run.normalized.end + prefix } }))],
		images: fresh.images.map(image => ({ ...image, offset: image.offset + prefix })),
	};
	let priorOffset = 0;
	for (const part of prior?.layout ?? []) {
		if (part.kind === "text" || part.kind === "frame") priorOffset = part.range.end;
		else if (part.kind === "original-image") input.images.push({ part, offset: priorOffset });
	}
	const maxFrames = Math.max(1, Math.min(options?.maxFrames ?? MAX_FRAMES_DEFAULT, MAX_FRAMES_DEFAULT));
	const vanilla = planArchive(input.text, high, low, maxFrames);
	const headEnd = vanilla.textHead.length;
	const tailStart = input.text.length - vanilla.textTail.length;
	for (const run of input.coverage) run.contribution = "ordinary";
	const pieces = sourcePieces(input, vanilla.ranges, headEnd, tailStart);
	const retained = new Set((preparation.recentSources ?? []).map(compactionSourceKey));
	const recentEntryIds = new Set((preparation.recentSources ?? []).map(source => source.entryId));
	let added = false;
	const vanillaCoverage = pieces.flatMap(piece => piece.coverage);
	const images = new Map<string, { part: Extract<SourceLayoutPart, { kind: "original-image" }>; coverage: SourceCoverageRun }>();
	const imagePositions = new Map<string, { order: number; rawStart: number }>();
	for (const image of input.images) { const owner = input.coverage.find(run => run.entryId === image.part.entryId && run.atomicGroup); if (owner?.atomicGroup) imagePositions.set(compactionSourceKey(image.part) + ":" + image.part.blockIndex, { order: Math.min(...input.coverage.filter(run => run.atomicGroup?.id === owner.atomicGroup!.id).map(run => run.order)), rawStart: image.offset }); }
	// Original images already committed remain actual archive input, not former-P shadows.
	for (const image of input.images) {
		if (!prior?.layout.some(part => part.kind === "original-image" && part.entryId === image.part.entryId && part.projection === image.part.projection && part.blockIndex === image.part.blockIndex)) continue;
		const part = image.part;
		const previousRun = input.coverage.find(run => run.entryId === part.entryId && run.projection === part.projection && run.snapshot.blockIndex === part.blockIndex && !run.normalized);
		const atomicGroup = previousRun?.atomicGroup ?? image.atomicGroup;
		images.set(compactionSourceKey(part) + ":" + part.blockIndex, { part, coverage: {
			entryId: part.entryId, ...(part.projection ? { projection: part.projection } : {}), order: part.order, ...(atomicGroup ? { atomicGroup } : {}),
			snapshot: { blockIndex: part.blockIndex, start: 0, end: 1 },
			...(part.currentBlockIndex !== undefined ? { current: { blockIndex: part.currentBlockIndex, start: 0, end: 1 } } : {}),
			status: part.currentBlockIndex !== undefined ? "exact-current" : previousRun?.status ?? "unknown", contribution: "ordinary",
		} });
	}
	const selectedById = new Map((preparation.selectedSources ?? []).map(source => [source.entryId, source]));
	const processed = new Set<string>();
	for (const source of preparation.selectedSources ?? []) {
		if (processed.has(compactionSourceKey(source))) continue;
		const nonUser = (source.message as unknown as Message).role !== "user";
		const group = nonUser && source.atomicGroup ? source.atomicGroup.entryIds.flatMap(id => selectedById.has(id) ? [selectedById.get(id)!] : []) : [source];
		for (const member of group) processed.add(compactionSourceKey(member));
		if (group.every(member => retained.has(compactionSourceKey(member))) || (!nonUser && recentEntryIds.has(source.entryId))) continue;
		const candidate = serializeSources(group, nonUser ? { ...options, toolResultMaxChars: Infinity, toolArgMaxChars: Infinity, toolCallMaxChars: Infinity } : options, high, nonUser);
		for (const run of candidate.coverage) run.contribution = (source.message as unknown as Message).role === "user" ? "selected-user" : "manual-nonuser";

		if (nonUser && candidate.coverage.every(run => {
			if (!run.current) return false;
			const existing = vanillaCoverage.filter(old => old.entryId === run.entryId && old.projection === run.projection);
			return run.current.start === run.current.end
				? coveredRawRanges(existing, run.current.blockIndex).some(raw => raw.start === 0 && raw.end === 0)
				: selectedMissing(run.current, existing).length === 0;
		}) && candidate.images.every(image => images.has(compactionSourceKey(image.part) + ":" + image.part.blockIndex))) continue;
		const existing = vanillaCoverage.filter(run => run.entryId === source.entryId && run.projection === source.projection);
		const spans = source.spans ?? candidate.coverage.filter(run => run.current).map(run => run.current!);
		const wanted = spans.flatMap(span => selectedMissing(span, existing));
		const candidateRanges: SourceRange[] = [];
		for (const run of candidate.coverage) {
			if (!run.current || !run.normalized) continue;
			for (const span of wanted) {
				if (span.blockIndex !== run.current.blockIndex) continue;
				const start = Math.max(span.start, run.current.start), end = Math.min(span.end, run.current.end);
				if (end <= start) continue;
				const affine = !run.normalizedUnit && run.current.end - run.current.start === run.normalized.end - run.normalized.start;

				let missingNormalized = [affine ? { start: run.normalized.start + start - run.current.start, end: run.normalized.start + end - run.current.start } : { ...run.normalized }];
				for (const old of existing) {
					if (old.status !== "exact-current" || !old.current || !old.normalizedUnit || old.current.blockIndex !== run.current.blockIndex || old.current.start !== run.current.start || old.current.end !== run.current.end) continue;
					const shift = run.normalized.start - (run.normalizedUnit?.start ?? 0);
					const coveredStart = shift + old.normalizedUnit.start, coveredEnd = shift + old.normalizedUnit.end;
					missingNormalized = missingNormalized.flatMap(range => {
						if (coveredEnd <= range.start || coveredStart >= range.end) return [range];
						return [...(coveredStart > range.start ? [{ start: range.start, end: coveredStart }] : []), ...(coveredEnd < range.end ? [{ start: coveredEnd, end: range.end }] : [])];
					});
				}
				candidateRanges.push(...missingNormalized);
			}
		}
		// A selected non-user atom uses the standard grouped serializer. It has no
		// invented text-field offsets; replace its already-present serialized pieces
		// with this complete same-source representation rather than doubling them.
		if (nonUser) {
			const memberIds = new Set(group.map(member => member.entryId));
			const freshIds = new Set(sources.map(member => member.entryId));
			for (let i = pieces.length - 1; i >= 0; i--) {
				const runs = pieces[i]!.coverage;
				if (runs.length && runs.every(run => memberIds.has(run.entryId) && (freshIds.has(run.entryId) || (run.status === "exact-current" && run.current && candidate.coverage.some(next => next.entryId === run.entryId && next.current?.blockIndex === run.current!.blockIndex && next.current.start <= run.current!.start && next.current.end >= run.current!.end))))) pieces.splice(i, 1);
			}
			candidateRanges.push({ start: 0, end: candidate.text.length });
		}
		candidateRanges.sort((a, b) => a.start - b.start);
		const merged: SourceRange[] = [];
		for (const range of candidateRanges) { const last = merged.at(-1); if (last && range.start <= last.end) last.end = Math.max(last.end, range.end); else merged.push({ ...range }); }
		if (merged.length) {
			if (!existing.length && merged[0]!.start === candidate.coverage[0]?.normalized?.start) merged[0]!.start = 0;
			pieces.push(...sourcePieces(candidate, merged, -1, Infinity)); added = true;
		}
		for (const image of candidate.images) {
			if (source.spans && !source.spans.some(span => span.blockIndex === image.part.blockIndex)) continue;
			const key = compactionSourceKey(image.part) + ":" + image.part.blockIndex;
			if (nonUser && source.atomicGroup) imagePositions.set(key, { order: Math.min(...group.map(member => member.order)), rawStart: image.offset });
			if (!images.has(key)) {
				const snapshot = { blockIndex: image.part.blockIndex, start: 0, end: 1 };
				images.set(key, { part: image.part, coverage: {
					entryId: image.part.entryId, ...(image.part.projection ? { projection: image.part.projection } : {}), order: image.part.order, ...(image.atomicGroup ? { atomicGroup: image.atomicGroup } : {}),
					snapshot, current: { ...snapshot }, status: "exact-current", contribution: nonUser ? "manual-nonuser" : "selected-user",
				} });
				added = true;
			}
		}
	}
	for (const { part: image, coverage } of images.values()) { const position = imagePositions.get(compactionSourceKey(image) + ":" + image.blockIndex); pieces.push({ text: "", coverage: [coverage], order: position?.order ?? image.order, blockIndex: position ? -1 : image.currentBlockIndex ?? image.blockIndex, rawStart: position?.rawStart ?? 0, image }); }

	const present = new Set(pieces.flatMap(piece => piece.coverage.map(run => run.entryId)));
	const universe = new Map<string, { entryId: string; order: number; atomicGroup?: SourceMessage<T>["atomicGroup"] }>();
	for (const run of input.coverage) universe.set(run.entryId, run);
	for (const source of sources) universe.set(source.entryId, { ...source, atomicGroup: source.atomicGroup ?? universe.get(source.entryId)?.atomicGroup });
	const orderedSources = [...universe.values()].sort((a, b) => a.order - b.order);
	const pendingGaps: typeof orderedSources = [];
	let beforeEntryId: string | undefined;
	const flushGap = (afterEntryId?: string) => {
		if (!pendingGaps.length) return;
		pieces.push({ text: "", coverage: [], order: pendingGaps[0]!.order, blockIndex: -1, rawStart: 0,
			notice: { kind: "gap", reason: "omitted-messages", wholeMessages: pendingGaps.length, beforeEntryId, afterEntryId } });
		pendingGaps.length = 0;
	};
	for (const source of orderedSources) {
		// A retained dependency atom is never split by a whole-message notice.
		const represented = present.has(source.entryId) || source.atomicGroup?.entryIds.some(id => present.has(id));
		if (represented) { flushGap(source.entryId); beforeEntryId = source.entryId; }
		else pendingGaps.push(source);
	}
	flushGap(preparation.recentSources?.[0]?.entryId);
	// Previously omitted IDs have no coverage rows. Keep their durable gap facts;
	// only genuinely restored source IDs inside that gap reduce its count.
	if (prior) {
		const orders = new Map([...input.coverage, ...prior.layout.flatMap(part => part.kind === "source" || part.kind === "original-image" ? [part] : []), ...sources, ...(preparation.recentSources ?? []), ...(preparation.selectedSources ?? [])].map(source => [source.entryId, source.order]));
		const previouslyPresent = new Set(prior.coverage.map(run => run.entryId));
		for (const part of prior.layout) {
			if (part.kind !== "gap" || part.reason !== "omitted-messages" || !part.wholeMessages) continue;
			const before = part.beforeEntryId ? orders.get(part.beforeEntryId) ?? -Infinity : -Infinity;
			const after = part.afterEntryId ? orders.get(part.afterEntryId) ?? Infinity : Infinity;
			let restored = 0;
			for (const id of selectedById.keys()) if (present.has(id) && !previouslyPresent.has(id) && orders.get(id)! > before && orders.get(id)! < after) restored++;
			const wholeMessages = Math.max(0, part.wholeMessages - restored);
			if (wholeMessages) pieces.push({ text: "", coverage: [], order: after, blockIndex: -1, rawStart: -Infinity, notice: { ...part, wholeMessages } });
		}
	}

	// Stable positional union. Equal strings at different source IDs never meet.
	pieces.sort((a, b) => a.order - b.order || Number(a.coverage[0]?.projection === "original") - Number(b.coverage[0]?.projection === "original") || Number(!!b.historicalAtom) - Number(!!a.historicalAtom) || a.blockIndex - b.blockIndex || a.rawStart - b.rawStart || (a.coverage[0]?.normalizedUnit?.start ?? 0) - (b.coverage[0]?.normalizedUnit?.start ?? 0) || Number(!!b.image) - Number(!!a.image));
	if (added) refreshSelectedUserGaps(pieces, preparation.selectedSources ?? [], high);
	const coverage: SourceCoverageRun[] = [];
	const chunks: string[] = [];
	let length = 0;
	let headAnchor = 0, tailAnchor = Infinity;
	const imageOffsets: { offset: number; part: Extract<SourceLayoutPart, { kind: "original-image" | "gap" }> }[] = [];
	for (const piece of pieces) {
		if (piece.image) { imageOffsets.push({ offset: length, part: piece.image }); coverage.push(...piece.coverage); continue; }
		if (piece.notice) { imageOffsets.push({ offset: length, part: piece.notice }); continue; }
		if (piece.edge === "head") headAnchor = length + piece.text.length;
		if (piece.edge === "tail") tailAnchor = Math.min(tailAnchor, length);
		for (const run of piece.coverage) if (run.normalized) appendCoverage(coverage, { ...run, normalized: { start: run.normalized.start + length, end: run.normalized.end + length } });
		chunks.push(piece.text); length += piece.text.length;
	}
	const text = chunks.join("");
	if (!Number.isFinite(tailAnchor)) tailAnchor = text.length;
	const { frames, layout, overflow } = await writeArchiveLayout(text, vanilla, headAnchor, tailAnchor, imageOffsets, !added && images.size === 0, options?.maxFrameDataBytes ?? FRAME_DATA_BYTES_BUDGET);
	const representation: SourceRepresentation = { version: 1, coverage, layout, ...(prior?.aggregate ? { aggregate: prior.aggregate } : preparation.previousSummary && !previous ? { aggregate: { entryIds: [], reason: "summary" as const } } : {}) };
	for (const source of preparation.recentSources ?? []) {
		const selected = selectedById.get(source.entryId);
		let spans = source.spans;
		if (selected && selected.projection === source.projection && spans) {
			if (!selected.spans) spans = undefined;
			else {
				const sorted = [...spans, ...selected.spans].sort((a, b) => a.blockIndex - b.blockIndex || a.start - b.start);
				spans = [];
				for (const span of sorted) {
					const last = spans.at(-1);
					if (last && last.blockIndex === span.blockIndex && span.start <= last.end) last.end = Math.max(last.end, span.end);
					else spans.push({ ...span });
				}
			}
		}
		layout.push({ kind: "source", entryId: source.entryId, ...(source.projection ? { projection: source.projection } : {}), order: source.order, ...(spans ? { spans } : {}) });
		if (selected && selected.projection !== source.projection && !retained.has(compactionSourceKey(selected))) {
			layout.push({ kind: "source", entryId: selected.entryId, ...(selected.projection ? { projection: selected.projection } : {}), order: selected.order, ...(selected.spans ? { spans: selected.spans } : {}), contribution: "selected-user" });
			const message = selected.message as Message;
			const content = "content" in message ? message.content : undefined;
			const selectedSpans = selected.spans ?? (typeof content === "string" ? [{ blockIndex: 0, start: 0, end: content.length }] : Array.isArray(content) ? content.flatMap((block, blockIndex) => block.type === "text" ? [{ blockIndex, start: 0, end: block.text.length }] : block.type === "image" ? [{ blockIndex, start: 0, end: 1 }] : []) : []);
			for (const span of selectedSpans) coverage.push({ entryId: selected.entryId, ...(selected.projection ? { projection: selected.projection } : {}), order: selected.order, snapshot: { ...span }, current: { ...span }, status: "exact-current", contribution: "selected-user" });
		}
	}
	representation.throughEntryId = preparation.recentSources?.at(-1)?.entryId ?? sources.at(-1)?.entryId ?? prior?.throughEntryId;
	const textChars = layout.reduce((sum, part) => sum + (part.kind === "text" ? part.range.end - part.range.start : 0), 0);
	const totalChars = textChars + frames.reduce((sum, frame) => sum + frame.chars, 0);

	let droppedChars = vanilla.truncatedChars;
	if (added) {
		droppedChars = 0;
		const bySource = new Map<string, SourceCoverageRun[]>();
		for (const run of coverage) { const key = compactionSourceKey(run) + ":" + run.snapshot.blockIndex; const existing = bySource.get(key); if (existing) existing.push(run); else bySource.set(key, [run]); }
		for (const run of input.coverage) {
			if (!run.normalized) continue;
			const compatible = (bySource.get(compactionSourceKey(run) + ":" + run.snapshot.blockIndex) ?? []).filter(next => run.status !== "historical-not-current" || next.status === "historical-not-current");
			const represented = compatible.map(next => ({ ...next, current: next.snapshot, status: "exact-current" as const }));
			const missing = selectedMissing(run.snapshot, represented);
			const rawLength = run.snapshot.end - run.snapshot.start;
			const normalizedLength = run.normalized.end - run.normalized.start;
			if (!run.normalizedUnit && rawLength === normalizedLength) droppedChars += missing.reduce((sum, span) => sum + span.end - span.start, 0);
			else {
				const unit = run.normalizedUnit ?? { start: 0, end: normalizedLength };
				const ranges = compatible.filter(next => next.snapshot.start === run.snapshot.start && next.snapshot.end === run.snapshot.end && next.normalized)
					.map(next => next.normalizedUnit ?? { start: 0, end: next.normalized!.end - next.normalized!.start }).sort((a, b) => a.start - b.start);
				let at = unit.start, present = 0;
				for (const range of ranges) {
					const start = Math.max(at, range.start), end = Math.min(unit.end, range.end);
					if (end > start) { present += end - start; at = end; }
				}
				droppedChars += unit.end - unit.start - present;
			}
		}
	}
	// Unmapped callers need no invented source descriptor for a physical suffix.
	const ordinaryTailStart = !sourceAware && overflow ? layout.findLast(part => part.kind === "frame")?.range.end ?? headAnchor : tailAnchor;
	const archive: Archive = { frames, text, totalChars, truncatedChars: (previous?.truncatedChars ?? 0) + droppedChars,
		textHead: text.slice(0, headAnchor), textTail: text.slice(ordinaryTailStart) };
	const { readFiles, modifiedFiles } = computeFileLists(preparation.fileOps);
	const files = formatFileList(readFiles, modifiedFiles, preparation.fileOps.read);
	const cols = [...new Set(frames.map(frame => frame.cols))];
	const summary = !text && !frames.length && !files ? "No prior history." : prompt.render(snapcompactSummaryPrompt, { frameCount: frames.length, multipleFrames: frames.length > 1, docColumns: high.columns === 2, cols: cols.length ? cols.join(" or ") : geometry(high).cols, rows: geometry(high).rows,
		sentenceInk: high.variant === "sent", stopwordDimmed: high.stopwordDim === true, lineRepeated: high.lineRepeat > 1, truncatedChars: archive.truncatedChars,
		includedPreviousSummary: includedPreviousSummary || !!representation.aggregate, files: files || undefined });
	return { summary, shortSummary: `Archived ${totalChars.toLocaleString()} chars of history onto ${frames.length} snapcompact frames (+${textChars.toLocaleString()} chars as text)`,
		firstKeptEntryId: preparation.firstKeptEntryId, tokensBefore: preparation.tokensBefore, details: { readFiles, modifiedFiles },
		preserveData: { ...stripOpenAiRemoteCompactionPreserveData(preparation.previousPreserveData), [PRESERVE_KEY]: archive, ...(sourceAware ? { sourceRepresentation: representation } : {}) } };
}

