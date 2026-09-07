import { RE2JS } from "re2js";
import type { SettingPath, SettingValue } from "../config/settings-schema";

export type PreservationAction = "auto" | "exclude" | "keep";
export type PreservationLimit =
	| { readonly mode: "off" | "all" }
	| { readonly mode: "messages" | "tokens" | "context-percent"; readonly value: number };
export type PreservedUserMessageLimit = "off" | "all" | `messages:${number}` | `tokens:${number}` | `context-percent:${number}`;
export const PRESERVED_USER_MESSAGE_FILTER_KEEP_CAPS = ["keep-last", "keep-first", "uncapped"] as const;
export const PRUNE_LONG_USER_MESSAGE_MODES = ["no", "middle-out", "head-only", "tail-only", "exclude"] as const;
export type PruneLongUserMessageMode = (typeof PRUNE_LONG_USER_MESSAGE_MODES)[number];
export const DEFAULT_MAX_TOKENS_PER_USER_MESSAGE = 2_000;

export const PRESERVED_USER_MESSAGE_CATEGORIES = [
	"longTermRule", "longTermGoal", "lastingSolution", "shortTermTask", "shortTermContext", "venting",
	"restorationGuidance", "preventionGuidance", "contextFreeInstruction", "banter", "question",
] as const;
export type PreservationCategory = (typeof PRESERVED_USER_MESSAGE_CATEGORIES)[number];
export const PRESERVED_USER_MESSAGE_CATEGORY_LABELS: Readonly<Record<PreservationCategory, string>> = {
	longTermRule: "Long-term rule / specification",
	longTermGoal: "Long-term goal / feature",
	lastingSolution: "Lasting solution / guidance",
	shortTermTask: "Short-term task / improvement",
	shortTermContext: "Short-term context / instruction",
	venting: "Venting after a failure",
	restorationGuidance: "Restoration guidance",
	preventionGuidance: "Prevention guidance",
	contextFreeInstruction: "Context-free instruction",
	banter: "Banter / no lasting information",
	question: "Question",
};
export const PRESERVED_USER_MESSAGE_CATEGORY_SETTING_PATHS = {
	longTermRule: "compaction.keepUserMessagesLlmLongTermRule",
	longTermGoal: "compaction.keepUserMessagesLlmLongTermGoal",
	lastingSolution: "compaction.keepUserMessagesLlmLastingSolution",
	shortTermTask: "compaction.keepUserMessagesLlmShortTermTask",
	shortTermContext: "compaction.keepUserMessagesLlmShortTermContext",
	venting: "compaction.keepUserMessagesLlmVenting",
	restorationGuidance: "compaction.keepUserMessagesLlmRestorationGuidance",
	preventionGuidance: "compaction.keepUserMessagesLlmPreventionGuidance",
	contextFreeInstruction: "compaction.keepUserMessagesLlmContextFreeInstruction",
	banter: "compaction.keepUserMessagesLlmBanter",
	question: "compaction.keepUserMessagesLlmQuestion",
} as const;
export const DEFAULT_PRESERVATION_CATEGORY_ACTIONS: Readonly<Record<PreservationCategory, PreservationAction>> = {
	longTermRule: "keep", longTermGoal: "keep", lastingSolution: "keep", shortTermTask: "auto",
	shortTermContext: "auto", venting: "exclude", restorationGuidance: "auto", preventionGuidance: "auto",
	contextFreeInstruction: "auto", banter: "exclude", question: "auto",
};

export interface PreservedUserMessageRegexRule {
	readonly state: PreservationAction;
	readonly caseInsensitive: boolean;
	readonly final?: boolean;
}
export type PreservedUserMessageRegexRules = Readonly<Record<string, PreservedUserMessageRegexRule>>;
export interface PreservationRegexRule {
	readonly condition: string;
	readonly pattern: RE2JS;
	readonly action: PreservationAction;
	readonly ignoreCase: boolean;
	readonly final: boolean;
}
export interface PreservationPolicySettings {
	readonly enabled: boolean;
	readonly first: PreservationLimit;
	readonly recent: PreservationLimit;
	readonly hardRecent: PreservationLimit;
	readonly alwaysCap: (typeof PRESERVED_USER_MESSAGE_FILTER_KEEP_CAPS)[number];
	readonly prune: PruneLongUserMessageMode;
	readonly maxTokens: number;
	readonly heuristics: boolean;
	readonly regexRules: readonly PreservationRegexRule[];
	readonly classifier: boolean;
	readonly categoryActions: Readonly<Record<PreservationCategory, PreservationAction>>;
}

export function isPreservationAction(value: unknown): value is PreservationAction {
	return value === "auto" || value === "exclude" || value === "keep";
}

/** Canonical input validation; legacy messages:0 is resolved only after layer composition. */
export function parsePreservationLimit(value: unknown): PreservationLimit | undefined {
	if (value === "off" || value === "all") return { mode: value };
	if (typeof value !== "string") return undefined;
	const match = /^(messages|tokens|context-percent):([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$/.exec(value);
	if (!match) return undefined;
	const mode = match[1] as "messages" | "tokens" | "context-percent";
	const amount = Number(match[2]);
	if (mode === "context-percent") {
		if (!Number.isFinite(amount) || amount < 0 || amount > 100) return undefined;
	} else if (!Number.isSafeInteger(amount) || amount < (mode === "messages" ? 1 : 0)) return undefined;
	return { mode, value: amount };
}

export function serializePreservationLimit(limit: PreservationLimit): PreservedUserMessageLimit {
	const value = "value" in limit ? `${limit.mode}:${limit.value}` : limit.mode;
	if (!parsePreservationLimit(value)) throw new Error("Invalid preservation limit");
	return value as PreservedUserMessageLimit;
}

export function validatePreservedUserMessageRegexCondition(condition: string, caseInsensitive = true): string {
	if (!condition.trim()) throw new Error("Condition cannot be empty");
	RE2JS.compile(condition, caseInsensitive ? RE2JS.CASE_INSENSITIVE : 0);
	return condition;
}

function regexRule(value: unknown): PreservedUserMessageRegexRule | undefined {
	if (isPreservationAction(value)) return { state: value, caseInsensitive: true, final: false };
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	const raw = value as Partial<PreservedUserMessageRegexRule>;
	if (!isPreservationAction(raw.state)) return undefined;
	if (raw.caseInsensitive !== undefined && typeof raw.caseInsensitive !== "boolean") return undefined;
	if (raw.final !== undefined && typeof raw.final !== "boolean") return undefined;
	return { state: raw.state, caseInsensitive: raw.caseInsensitive ?? true, final: raw.final ?? false };
}

export function compilePreservedUserMessageRegexRules(value: unknown): readonly PreservationRegexRule[] {
	if (!value || typeof value !== "object" || Array.isArray(value)) return [];
	const rules: PreservationRegexRule[] = [];
	for (const [condition, raw] of Object.entries(value)) {
		const rule = regexRule(raw);
		if (!rule || !condition.trim()) continue;
		try {
			rules.push({ condition, pattern: RE2JS.compile(condition, rule.caseInsensitive ? RE2JS.CASE_INSENSITIVE : 0),
				action: rule.state, ignoreCase: rule.caseInsensitive, final: rule.final ?? false });
		} catch {
			// Invalid persisted RE2 conditions have no policy effect; UI validation blocks new saves.
		}
	}
	return rules;
}

export function normalizePreservedUserMessageRegexRules(value: unknown): PreservedUserMessageRegexRules {
	return Object.fromEntries(compilePreservedUserMessageRegexRules(value).map(rule => [rule.condition,
		{ state: rule.action, caseInsensitive: rule.ignoreCase, final: rule.final }]));
}

export function readPreservationPolicySettings(settings: { get<P extends SettingPath>(path: P): SettingValue<P> }): PreservationPolicySettings {
	const categoryActions = { ...DEFAULT_PRESERVATION_CATEGORY_ACTIONS };
	for (const category of PRESERVED_USER_MESSAGE_CATEGORIES) {
		const action = settings.get(PRESERVED_USER_MESSAGE_CATEGORY_SETTING_PATHS[category]);
		if (isPreservationAction(action)) categoryActions[category] = action;
	}
	const maxTokens = settings.get("compaction.maxTokensPerUserMessage");
	return {
		enabled: settings.get("compaction.keepUserMessages"),
		first: parsePreservationLimit(settings.get("compaction.keepFirstLimit")) ?? { mode: "tokens", value: 0 },
		recent: parsePreservationLimit(settings.get("compaction.keepLastLimit")) ?? { mode: "tokens", value: 0 },
		hardRecent: parsePreservationLimit(settings.get("compaction.keepRecentUserMessagesLimit")) ?? { mode: "tokens", value: 0 },
		alwaysCap: settings.get("compaction.keepUserMessagesFilterKeepCap"),
		prune: settings.get("compaction.pruneLongUserMessages"),
		maxTokens: Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : DEFAULT_MAX_TOKENS_PER_USER_MESSAGE,
		heuristics: settings.get("compaction.keepUserMessagesHeuristic"),
		regexRules: settings.get("compaction.keepUserMessagesRegex") ? compilePreservedUserMessageRegexRules(settings.get("compaction.keepUserMessagesRegexRules")) : [],
		classifier: settings.get("compaction.keepUserMessagesClassifierFilter"),
		categoryActions,
	};
}

export const MESSAGE_OVERRIDE_CUSTOM_TYPE = "com.omp.compaction.message-override.v1";
export const USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE = "com.omp.compaction.user-message-classification.v1";
export const INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE = `${USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE}.invalidated`;
export const MAX_PRESERVED_USER_MESSAGE_CATEGORY_MASK = (1 << PRESERVED_USER_MESSAGE_CATEGORIES.length) - 1;
export interface CompactionMessageOverrideData { messageIds: string[]; state: PreservationAction }
export interface PreservedUserMessageClassification { id: string; mask: number }
export interface PreservedUserMessageClassificationData { v: 1; c: Array<string | number> }
export type ClassificationDecodeResult =
	| { status: "valid"; classifications: PreservedUserMessageClassification[] }
	| { status: "unsupported" | "malformed"; raw: unknown };

function validClassification(id: unknown, mask: unknown): boolean {
	return typeof id === "string" && id.length > 0 && typeof mask === "number" && Number.isInteger(mask)
		&& mask >= 0 && mask <= MAX_PRESERVED_USER_MESSAGE_CATEGORY_MASK;
}

export function packPreservedUserMessageClassifications(classifications: readonly PreservedUserMessageClassification[]): PreservedUserMessageClassificationData {
	const c: Array<string | number> = [];
	for (const { id, mask } of classifications) {
		if (!validClassification(id, mask)) throw new Error("Invalid user-message classification");
		c.push(id, mask);
	}
	return { v: 1, c };
}

/** All-or-nothing validation prevents a malformed record from inventing partial success. */
export function decodePreservedUserMessageClassifications(value: unknown): ClassificationDecodeResult {
	if (!value || typeof value !== "object" || Array.isArray(value)) return { status: "malformed", raw: value };
	const data = value as Partial<PreservedUserMessageClassificationData>;
	if (data.v !== 1) return { status: typeof data.v === "number" ? "unsupported" : "malformed", raw: value };
	if (!Array.isArray(data.c) || data.c.length % 2 !== 0) return { status: "malformed", raw: value };
	const classifications: PreservedUserMessageClassification[] = [];
	for (let index = 0; index < data.c.length; index += 2) {
		const id = data.c[index];
		const mask = data.c[index + 1];
		if (!validClassification(id, mask)) return { status: "malformed", raw: value };
		classifications.push({ id: id as string, mask: mask as number });
	}
	return { status: "valid", classifications };
}

export function unpackPreservedUserMessageClassifications(value: unknown): PreservedUserMessageClassification[] {
	const result = decodePreservedUserMessageClassifications(value);
	return result.status === "valid" ? result.classifications : [];
}

/** Accept the established array payload and singular historical override without changing source identity. */
export function decodeCompactionMessageOverride(value: unknown): CompactionMessageOverrideData | undefined {
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	const data = value as { messageIds?: unknown; messageId?: unknown; state?: unknown };
	if (!isPreservationAction(data.state)) return undefined;
	const ids = data.messageIds ?? (typeof data.messageId === "string" ? [data.messageId] : undefined);
	if (!Array.isArray(ids) || ids.length === 0 || ids.some(id => typeof id !== "string" || id.length === 0)) return undefined;
	return { messageIds: [...new Set(ids as string[])], state: data.state };
}

/** Caller rewrites the same journal entry, preserving its id/parent/order; binary legacy data is not manual state. */
export function migrateLegacyCompactionPin(value: unknown): CompactionMessageOverrideData | undefined {
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	const data = value as { messageId?: unknown; pinned?: unknown };
	if (typeof data.messageId !== "string" || !data.messageId || (data.pinned !== undefined && typeof data.pinned !== "boolean")) return undefined;
	return { messageIds: [data.messageId], state: data.pinned === false ? "auto" : "keep" };
}

export interface CompactionOverridePrompt {
	text: string;
	compactionOverride: "keep" | "exclude";
}

/** Parse only the outer semantic prefix. Empty recognized bodies let ingress report usage without a source write. */
export function parseCompactionOverridePrompt(text: string): CompactionOverridePrompt | undefined {
	const match = /^\/(keep|once)(?::|\s+|$)([\s\S]*)$/.exec(text);
	if (!match) return undefined;
	return { text: match[2].trim(), compactionOverride: match[1] === "keep" ? "keep" : "exclude" };
}

/** Restore exactly one semantic command; a nested-looking body is literal user content. */
export function restoreCompactionOverridePrompt(text: string, state: "keep" | "exclude" | undefined): string {
	if (!state) return text;
	return `/${state === "keep" ? "keep" : "once"} ${text}`;
}
