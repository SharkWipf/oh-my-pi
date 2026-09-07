import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { Tokenizer } from "@oh-my-pi/pi-agent-core/tokenizer";
import type { UserMessage } from "@oh-my-pi/pi-ai";
import { PreservedMessageIndex, type PolicyKind, type PolicyLimit, type PolicyRange } from "./preserved-message-index";
import {
	PRESERVED_USER_MESSAGE_CATEGORIES,
	MESSAGE_OVERRIDE_CUSTOM_TYPE,
	USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
	INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
	decodeCompactionMessageOverride,
	unpackPreservedUserMessageClassifications,
	decodePreservedUserMessageClassifications,
	type PreservationLimit,
	type PreservationAction,
	type PreservationPolicySettings,
} from "./preserved-message-settings";
import type { SessionEntry, SessionMessageEntry } from "./session-entries";

/** Current authored source coverage, not a physical frame or a persisted archive descriptor. */
export interface PreservationSourceSpan {
	sourceId: string;
	/** String user content has block index zero. */
	blockIndex: number;
	/** UTF-16 half-open text interval; absent for an indivisible non-text block. */
	text?: { start: number; end: number };
}

/** Materialized on demand, never retained as a second source-message pool. */
export interface PreservationCandidate {
	sourceId: string;
	message: AgentMessage;
	spans: readonly PreservationSourceSpan[];
	rawTokens: number;
	quotaTokens: number;
	truncated: boolean;
	/** Immutable non-text/marker content may exceed the configured pruning target. */
	limitation?: { kind: "immutable-content-exceeds-limit"; limit: number; tokens: number };
}

export interface PreservationStages {
	heuristic: PreservationAction;
	regex: PreservationAction;
	classifier: PreservationAction;
	finalRegex: PreservationAction;
	manual: PreservationAction;
	/** Stored policy still resolves normally while hard-recent temporarily bypasses it. */
	resolved: PreservationAction;
}

export interface PreservationAtom {
	/** Canonical first source member; admission is whole-closure. */
	id: string;
	memberIds: readonly string[];
	entries: readonly SessionMessageEntry[];
	quotaTokens: number;
}

export function isPreservationUser(entry: SessionEntry): entry is SessionMessageEntry & { message: UserMessage } {
	return (
		entry.type === "message" &&
		entry.message.role === "user" &&
		entry.message.synthetic !== true &&
		entry.message.attribution !== "agent"
	);
}

const PURE_ACKNOWLEDGMENT =
	/^(?:ok(ay)?|k|kk|yep|yeah|yup|yea|sounds? (?:good|great|fine)|looks? (?:good|great|fine)|lgtm|lgbtm|tbgm|thx|thanks|thank (?:you|u)|ty|tysm|perfect|great|nice|cool|awesome|excellent|amazing|brilliant|good|fine|alright|all (?:good|set)|sure(?: thing)?|of course|absolutely|definitely|indeed|exactly|right|correct|got it|gotcha|understood|noted|roger|copy (?:that|cat)?|affirmative|ack(?:s)?|on it|will do|no problem|np|no worries)\s*[!.]?\s*$/i;

function userText(message: UserMessage): string {
	if (typeof message.content === "string") return message.content;
	return message.content
		.filter(block => block.type === "text")
		.map(block => block.text)
		.join(" ");
}

/** Compatibility reduction within a stage: Keep > Never > neutral Auto. */
function strongest(left: PreservationAction, right: PreservationAction): PreservationAction {
	if (left === "keep" || right === "keep") return "keep";
	return left === "exclude" || right === "exclude" ? "exclude" : "auto";
}

export function evaluatePreservationPolicy(
	message: UserMessage,
	policy: PreservationPolicySettings,
	manual: PreservationAction = "auto",
	categoryMask?: number,
): PreservationStages {
	const text = userText(message);
	const hasNonText = typeof message.content !== "string" && message.content.some(block => block.type !== "text");
	const heuristic =
		policy.heuristics && !hasNonText && (!text.trim() || PURE_ACKNOWLEDGMENT.test(text.trim())) ? "exclude" : "auto";
	let regex: PreservationAction = "auto";
	let finalRegex: PreservationAction = "auto";
	for (const rule of policy.regexRules) {
		if (!rule.pattern.matcher(text).find()) continue;
		if (rule.final) finalRegex = strongest(finalRegex, rule.action);
		else regex = strongest(regex, rule.action);
	}
	let classifier: PreservationAction = "auto";
	if (policy.classifier && categoryMask !== undefined) {
		for (let bit = 0; bit < PRESERVED_USER_MESSAGE_CATEGORIES.length; bit++) {
			if (categoryMask & (1 << bit))
				classifier = strongest(classifier, policy.categoryActions[PRESERVED_USER_MESSAGE_CATEGORIES[bit]!]);
		}
	}
	let resolved: PreservationAction = heuristic;
	for (const stage of [regex, classifier, finalRegex, manual]) if (stage !== "auto") resolved = stage;
	return { heuristic, regex, classifier, finalRegex, manual, resolved };
}

function fullSpans(id: string, message: AgentMessage): PreservationSourceSpan[] {
	if (!("content" in message)) return [];
	if (typeof message.content === "string")
		return [{ sourceId: id, blockIndex: 0, text: { start: 0, end: message.content.length } }];
	if (!Array.isArray(message.content)) return [];
	return message.content.map((block, blockIndex) => ({
		sourceId: id,
		blockIndex,
		...(block.type === "text" ? { text: { start: 0, end: block.text.length } } : {}),
	}));
}

function safePrefix(text: string, end: number): number {
	if (end > 0 && end < text.length && /[\uD800-\uDBFF]/.test(text[end - 1]!) && /[\uDC00-\uDFFF]/.test(text[end]!))
		return end - 1;
	return end;
}

function safeSuffix(text: string, start: number): number {
	if (
		start > 0 &&
		start < text.length &&
		/[\uD800-\uDBFF]/.test(text[start - 1]!) &&
		/[\uDC00-\uDFFF]/.test(text[start]!)
	)
		return start + 1;
	return start;
}

/** Prune text only, retaining original non-text blocks at their authored positions. */
function pruneAt(
	entry: SessionMessageEntry & { message: UserMessage },
	head: number,
	tail: number,
): Pick<PreservationCandidate, "message" | "spans"> {
	const source = entry.message;
	const blocks =
		typeof source.content === "string" ? [{ type: "text" as const, text: source.content }] : source.content;
	let length = 0;
	for (const block of blocks) if (block.type === "text") length += block.text.length;
	const content: typeof blocks = [];
	const spans: PreservationSourceSpan[] = [];
	let offset = 0;
	let marked = false;
	for (let blockIndex = 0; blockIndex < blocks.length; blockIndex++) {
		const block = blocks[blockIndex]!;
		if (block.type !== "text") {
			content.push(block);
			spans.push({ sourceId: entry.id, blockIndex });
			continue;
		}
		const end = safePrefix(block.text, Math.max(0, Math.min(block.text.length, head - offset)));
		const start = safeSuffix(block.text, Math.max(end, Math.min(block.text.length, length - tail - offset)));
		if (end === block.text.length || start === 0) {
			content.push(block);
			spans.push({ sourceId: entry.id, blockIndex, text: { start: 0, end: block.text.length } });
		} else {
			let text = block.text.slice(0, end);
			if (end > 0) spans.push({ sourceId: entry.id, blockIndex, text: { start: 0, end } });
			if (!marked && start > end) {
				text += "[truncated]";
				marked = true;
			}
			text += block.text.slice(start);
			if (start < block.text.length)
				spans.push({ sourceId: entry.id, blockIndex, text: { start, end: block.text.length } });
			if (text) content.push({ ...block, text });
		}
		offset += block.text.length;
	}
	return {
		message: {
			...source,
			content:
				typeof source.content === "string"
					? content
							.filter(block => block.type === "text")
							.map(block => block.text)
							.join("")
					: content,
			providerPayload: undefined,
		},
		spans,
	};
}

/** q uses the source estimator, before ordinary allocation or physical representation exists. */
export function preservationCandidate(
	entry: SessionMessageEntry,
	policy: Pick<PreservationPolicySettings, "prune" | "maxTokens">,
	tokenizer: Tokenizer,
	raw = false,
): PreservationCandidate | undefined {
	const rawTokens = tokenizer.countMessage(entry.message);
	const base = { sourceId: entry.id, rawTokens, quotaTokens: rawTokens, truncated: false };
	if (raw || entry.message.role !== "user" || policy.prune === "no" || rawTokens <= policy.maxTokens) {
		return { ...base, message: entry.message, spans: fullSpans(entry.id, entry.message) };
	}
	if (policy.prune === "exclude") return undefined;
	const source = entry as SessionMessageEntry & { message: UserMessage };
	let length = 0;
	if (typeof source.message.content === "string") length = source.message.content.length;
	else for (const block of source.message.content) if (block.type === "text") length += block.text.length;
	const at = (retained: number) =>
		pruneAt(
			source,
			policy.prune === "tail-only" ? 0 : policy.prune === "head-only" ? retained : Math.ceil(retained / 2),
			policy.prune === "head-only" ? 0 : policy.prune === "tail-only" ? retained : Math.floor(retained / 2),
		);
	let chosen = at(0);
	let tokens = tokenizer.countMessage(chosen.message);
	if (tokens > policy.maxTokens)
		return {
			...base,
			...chosen,
			quotaTokens: tokens,
			truncated: length > 0,
			limitation: { kind: "immutable-content-exceeds-limit", limit: policy.maxTokens, tokens },
		};
	// Text tokenization need not be strictly monotone. Every accepted candidate is measured;
	// maximal character retention is not promised, and no over-limit trial is published.
	let low = 0;
	let high = length;
	while (low < high) {
		const middle = Math.ceil((low + high) / 2);
		const candidate = at(middle);
		const price = tokenizer.countMessage(candidate.message);
		if (price <= policy.maxTokens) {
			low = middle;
			chosen = candidate;
			tokens = price;
		} else high = middle - 1;
	}
	return { ...base, ...chosen, quotaTokens: tokens, truncated: low < length };
}

/** Reserve complete N once; callers retain their allocator's actual zero-budget cut behavior. */
export function prechargeNonUsers(
	atoms: Iterable<PreservationAtom>,
	ordinaryBudget: number,
	priceEntry: (entry: SessionMessageEntry) => number,
): { residualBudget: number; tokens: number; sourceIds: ReadonlySet<string> } {
	const sourceIds = new Set<string>();
	let tokens = 0;
	for (const atom of atoms)
		for (const entry of atom.entries) {
			if (sourceIds.has(entry.id)) continue;
			sourceIds.add(entry.id);
			tokens += priceEntry(entry);
		}
	return { residualBudget: Math.max(0, ordinaryBudget - tokens), tokens, sourceIds };
}

export interface PreservationBuildOptions {
	/** Existing session/branch/reset ownership; benign same-branch appends remain current. */
	isCurrent(): boolean;
	yieldControl?: () => Promise<void>;
	/** Elapsed-time scheduling quantum, not a history-size or product limit. */
	sliceMs?: number;
}

export interface PreservationRow {
	id: string;
	memberIds: readonly string[];
	entry: SessionMessageEntry;
	manual: PreservationAction;
	categoryStatus: "absent" | "valid" | "unsupported" | "invalidated";
	categoryMask?: number;
}

export interface PreservationManualGroup {
	id: string;
	memberIds: readonly string[];
	members: readonly { sourceId: string; state: PreservationAction; revisionId: string | null }[];
}

export interface PreservationMembership extends Iterable<string> {
	readonly size: number;
	has(id: string): boolean;
	values(): IterableIterator<string>;
}

export interface PreservationReasons {
	first: boolean;
	recent: boolean;
	hardRecent: boolean;
	always: boolean;
	capDenied: boolean;
}

export interface PreservationSelection {
	/** User selections are additive; never zero these sources in an ordinary cut walk. */
	P: PreservationMembership;
	/** Complete non-user sources, precharged inside the method's ordinary budget. */
	N: PreservationMembership;
	H: PreservationMembership;
	/** Missing effective model maximum leaves percentage-derived selection provisional. */
	unavailableLimits: readonly ("first" | "recent" | "hardRecent" | "always")[];
	blockers: Partial<Record<"first" | "recent" | "hardRecent" | "always", string>>;
	quota: Record<"P" | "N" | "H" | "first" | "recent" | "always", { count: number; tokens: number }>;
	reasons(id: string): PreservationReasons;
	positions(id: string): { first?: number; recent?: number };
	candidate(id: string): PreservationCandidate | undefined;
	candidates(): IterableIterator<PreservationCandidate>;
	nonUserAtoms(): IterableIterator<PreservationAtom>;
}

export interface PreservationSelectionOptions {
	maximumContext?: number;
	enabled?: boolean;
	first?: PreservationLimit;
	recent?: PreservationLimit;
	hardRecent?: PreservationLimit;
	alwaysCap?: PreservationPolicySettings["alwaysCap"];
}

interface PendingExchange {
	members: string[];
	remaining: number;
}

/** One branch-local policy index over journal references, with lazy source materialization. */
export class PreservedMessageQuery {
	readonly #index = new PreservedMessageIndex();
	readonly #positions = new Map<string, number>();
	readonly #users: number[] = [];
	readonly #rows: number[] = [];
	readonly #overrides = new Map<string, Exclude<PreservationAction, "auto">>();
	readonly #classifications = new Map<string, number>();
	readonly #classificationProblems = new Map<string, "unsupported" | "malformed" | "invalidated">();
	readonly #overrideRevisions = new Map<string, { id: string; position: number }>();
	readonly #nonAutoGroups = new Set<number>();
	readonly #appended: SessionEntry[] = [];
	readonly #baseLength: number;
	readonly #groups = new Map<string, readonly string[]>();
	readonly #pending = new Map<string, PendingExchange[]>();
	readonly #manualAtoms = new Map<number, readonly string[]>();
	#entries: readonly SessionEntry[];
	#offset: number;
	#policy: PreservationPolicySettings;
	readonly #tokenizer: Tokenizer;
	readonly #ownership: PreservationBuildOptions;
	#published = false;

	private constructor(
		entries: readonly SessionEntry[],
		offset: number,
		policy: PreservationPolicySettings,
		tokenizer: Tokenizer,
		options: PreservationBuildOptions,
	) {
		this.#entries = entries;
		this.#baseLength = entries.length;
		this.#offset = offset;
		this.#policy = policy;
		this.#tokenizer = tokenizer;
		this.#ownership = options;
	}

	/** Cold/global work publishes only a complete, still-current owner. */
	static async build(
		entries: readonly SessionEntry[],
		policy: PreservationPolicySettings,
		tokenizer: Tokenizer,
		options: PreservationBuildOptions,
	): Promise<PreservedMessageQuery | undefined> {
		let offset = entries.length;
		let deadline = performance.now() + (options.sliceMs ?? 4);
		const cooperate = async () => {
			await (options.yieldControl?.() ?? new Promise<void>(resolve => setTimeout(resolve, 0)));
			deadline = performance.now() + (options.sliceMs ?? 4);
		};
		while (offset > 0 && entries[offset - 1]!.type !== "reset_boundary") {
			offset--;
			if (performance.now() >= deadline) {
				await cooperate();
				if (!options.isCurrent()) return undefined;
			}
		}
		const query = new PreservedMessageQuery(entries, offset, policy, tokenizer, options);
		for (let position = 0; position < entries.length - offset; position++) {
			query.#appendEntry(position);
			if (performance.now() >= deadline) {
				await cooperate();
				if (!options.isCurrent()) return undefined;
			}
		}
		// Later journal controls can override any earlier source. Price only after folding them.
		for (const position of query.#rows) {
			query.#refreshPosition(position);
			if (performance.now() >= deadline) {
				await cooperate();
				if (!options.isCurrent()) return undefined;
			}
		}
		if (!options.isCurrent()) return undefined;
		query.#published = true;
		return query;
	}

	#assertCurrent(): void {
		if (!this.#published || !this.#ownership.isCurrent())
			throw new Error("Preservation source branch is no longer current");
	}

	#entry(position: number): SessionMessageEntry | undefined {
		const index = position + this.#offset;
		const entry = index < this.#baseLength ? this.#entries[index] : this.#appended[index - this.#baseLength];
		return entry?.type === "message" ? entry : undefined;
	}

	#appendEntry(position: number): void {
		const index = position + this.#offset;
		const entry = (index < this.#baseLength ? this.#entries[index] : this.#appended[index - this.#baseLength])!;
		this.#index.append({
			user: false,
			raw: 0,
			candidate: 0,
			eligible: false,
			always: false,
			manual: false,
			nonUserCount: 0,
		});
		if (entry.type === "custom") {
			if (entry.customType === MESSAGE_OVERRIDE_CUSTOM_TYPE) {
				const data = decodeCompactionMessageOverride(entry.data);
				if (data)
					for (const id of data.messageIds) {
						this.#overrideRevisions.set(id, { id: entry.id, position });
						if (data.state === "auto") this.#overrides.delete(id);
						else this.#overrides.set(id, data.state);
					}
			} else if (entry.customType === USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE) {
				const decoded = decodePreservedUserMessageClassifications(entry.data);
				if (decoded.status === "valid")
					for (const { id, mask } of decoded.classifications) {
						this.#classifications.set(id, mask);
						this.#classificationProblems.delete(id);
					}
				else {
					const packed = entry.data as { c?: unknown } | undefined;
					if (Array.isArray(packed?.c))
						for (let i = 0; i < packed.c.length; i += 2) {
							const id = packed.c[i];
							if (typeof id === "string" && this.#positions.has(id))
								this.#classificationProblems.set(id, decoded.status);
						}
				}
			} else if (entry.customType === INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE) {
				const packed = entry.data as { c?: unknown } | undefined;
				if (Array.isArray(packed?.c))
					for (let i = 0; i < packed.c.length; i += 2) {
						const id = packed.c[i];
						if (typeof id === "string" && this.#positions.has(id))
							this.#classificationProblems.set(id, "invalidated");
					}
			}
			return;
		}
		if (entry.type !== "message") return;
		this.#positions.set(entry.id, position);
		const initial = (entry as SessionMessageEntry & { compactionOverride?: PreservationAction }).compactionOverride;
		if (initial === "keep" || initial === "exclude") {
			this.#overrides.set(entry.id, initial);
			this.#overrideRevisions.set(entry.id, { id: entry.id, position });
		}
		if (isPreservationUser(entry)) this.#users.push(position);
		if (entry.message.role === "assistant") {
			const calls = entry.message.content.filter(block => block.type === "toolCall");
			if (calls.length) {
				const exchange: PendingExchange = { members: [entry.id], remaining: calls.length };
				for (const call of calls) {
					const pending = this.#pending.get(call.id);
					if (pending) pending.push(exchange);
					else this.#pending.set(call.id, [exchange]);
				}
				return;
			}
		} else if (entry.message.role === "toolResult") {
			const pending = this.#pending.get(entry.message.toolCallId);
			const exchange = pending?.shift();
			if (!pending?.length) this.#pending.delete(entry.message.toolCallId);
			if (exchange) {
				exchange.members.push(entry.id);
				if (--exchange.remaining === 0) {
					for (const id of exchange.members) this.#groups.set(id, exchange.members);
					const anchor = this.#positions.get(exchange.members[0]!)!;
					// A delayed result can complete an earlier row; maintain source order.
					let low = 0,
						high = this.#rows.length;
					while (low < high) {
						const middle = (low + high) >>> 1;
						if (this.#rows[middle]! < anchor) low = middle + 1;
						else high = middle;
					}
					this.#rows.splice(low, 0, anchor);
				}
			}
			return;
		}
		this.#rows.push(position);
	}

	#manual(memberIds: readonly string[]): PreservationAction {
		let keep = false;
		for (const id of memberIds) {
			const state = this.#overrides.get(id);
			if (state === "exclude") return "exclude";
			if (state === "keep") keep = true;
		}
		return keep ? "keep" : "auto";
	}

	#refreshPosition(position: number): void {
		const entry = this.#entry(position);
		if (!entry) return;
		const members = this.#groups.get(entry.id) ?? [entry.id];
		const anchor = this.#positions.get(members[0]!)!;
		if (anchor !== position) {
			this.#refreshPosition(anchor);
			return;
		}
		if (
			entry.message.role === "toolResult" ||
			(entry.message.role === "assistant" &&
				entry.message.content.some(block => block.type === "toolCall") &&
				!this.#groups.has(entry.id))
		)
			return;
		if (members.some(id => this.#overrides.has(id))) this.#nonAutoGroups.add(position);
		else this.#nonAutoGroups.delete(position);
		const manual = this.#manual(members);
		if (isPreservationUser(entry)) {
			const stages = evaluatePreservationPolicy(
				entry.message,
				this.#policy,
				manual,
				this.#classifications.get(entry.id),
			);
			const candidate =
				stages.resolved === "exclude"
					? undefined
					: preservationCandidate(entry, this.#policy, this.#tokenizer, manual === "keep");
			this.#index.update(position, {
				user: true,
				raw: this.#tokenizer.countMessage(entry.message),
				candidate: candidate?.quotaTokens ?? NaN,
				eligible: stages.resolved !== "exclude",
				always: stages.resolved === "keep",
				manual: manual === "keep",
				nonUserCount: 0,
			});
		} else {
			let raw = 0;
			if (manual === "keep") {
				this.#manualAtoms.set(position, members);
				for (const id of members)
					raw += this.#tokenizer.countMessage(this.#entry(this.#positions.get(id)!)!.message);
			} else this.#manualAtoms.delete(position);
			this.#index.update(position, {
				user: false,
				raw,
				candidate: 0,
				eligible: false,
				always: manual === "keep",
				manual: false,
				nonUserCount: manual === "keep" ? members.length : 0,
			});
		}
	}

	rowCount(role: "user" | "all" = "user"): number {
		this.#assertCurrent();
		return (role === "user" ? this.#users : this.#rows).length;
	}

	rowAt(index: number, role: "user" | "all" = "user"): PreservationRow | undefined {
		this.#assertCurrent();
		const position = (role === "user" ? this.#users : this.#rows)[index];
		if (position === undefined) return undefined;
		const entry = this.#entry(position)!;
		const memberIds = this.#groups.get(entry.id) ?? [entry.id];
		return {
			id: entry.id,
			memberIds,
			entry,
			manual: this.#manual(memberIds),
			categoryMask: isPreservationUser(entry) ? this.#classifications.get(entry.id) : undefined,
			categoryStatus: this.classificationStatus(entry.id),
		};
	}

	rowIndexOf(id: string, role: "user" | "all" = "user"): number {
		this.#assertCurrent();
		const anchor = this.#groups.get(id)?.[0] ?? id;
		const position = this.#positions.get(anchor);
		if (position === undefined) return -1;
		const rows = role === "user" ? this.#users : this.#rows;
		let low = 0,
			high = rows.length;
		while (low < high) {
			const middle = (low + high) >>> 1;
			if (rows[middle]! < position) low = middle + 1;
			else high = middle;
		}
		return rows[low] === position ? low : -1;
	}

	inspect(id: string): PreservationStages | undefined {
		this.#assertCurrent();
		const position = this.#positions.get(id);
		const entry = position === undefined ? undefined : this.#entry(position);
		if (!entry) return undefined;
		const manual = this.#manual(this.#groups.get(id) ?? [id]);
		return isPreservationUser(entry)
			? evaluatePreservationPolicy(entry.message, this.#policy, manual, this.#classifications.get(id))
			: { heuristic: "auto", regex: "auto", classifier: "auto", finalRegex: "auto", manual, resolved: manual };
	}

	/** Apply a committed complete-source transition; no persistence or optimistic UI lies here. */
	applyManualOverride(ids: readonly string[], state: PreservationAction): readonly string[] {
		this.#assertCurrent();
		const affected = new Set<string>();
		for (const id of ids)
			if (this.#positions.has(id)) for (const member of this.#groups.get(id) ?? [id]) affected.add(member);
		for (const id of affected) {
			if (state === "auto") this.#overrides.delete(id);
			else this.#overrides.set(id, state);
		}
		const anchors = new Set<number>();
		for (const id of affected) anchors.add(this.#positions.get(this.#groups.get(id)?.[0] ?? id)!);
		for (const position of anchors) this.#refreshPosition(position);
		return [...affected];
	}

	applyClassifications(facts: readonly { id: string; mask: number }[]): readonly string[] {
		this.#assertCurrent();
		const affected: string[] = [];
		for (const { id, mask } of facts) {
			const position = this.#positions.get(id);
			const entry = position === undefined ? undefined : this.#entry(position);
			if (!entry || !isPreservationUser(entry) || !Number.isInteger(mask) || mask < 0 || mask > 2047) continue;
			this.#classifications.set(id, mask);
			this.#refreshPosition(position!);
			this.#classificationProblems.delete(id);
			affected.push(id);
		}
		return affected;
	}

	/** Known source rewrites retain IDs; tokenizer invalidation and classification dependency clearing belong to the journal writer. */
	refreshSources(ids: readonly string[], entries?: readonly SessionEntry[]): void {
		this.#assertCurrent();
		if (entries) {
			this.#entries = entries;
			for (const id of ids) {
				const position = this.#positions.get(id);
				if (position === undefined) continue;
				const index = position + this.#offset;
				if (index >= this.#baseLength) this.#appended[index - this.#baseLength] = entries[index]!;
			}
		}
		const positions = new Set<number>();
		for (const id of ids) {
			const position = this.#positions.get(this.#groups.get(id)?.[0] ?? id);
			if (position !== undefined) positions.add(position);
		}
		for (const position of positions) this.#refreshPosition(position);
	}

	/** Hot same-branch extension; source rewrites or divergent suffixes require rebuild. */
	append(entries: readonly SessionEntry[]): boolean {
		this.#assertCurrent();
		const oldLength = this.#baseLength + this.#appended.length;
		const previous = this.#appended.at(-1) ?? this.#entries[this.#baseLength - 1];
		if (entries.length < oldLength || (oldLength > 0 && entries[oldLength - 1] !== previous)) return false;
		return this.appendEntries(entries.slice(oldLength));
	}

	/** Bounded source-owner delta: no fresh getBranch allocation on ordinary sends. */
	appendEntries(entries: readonly SessionEntry[]): boolean {
		this.#assertCurrent();
		let parentId = (this.#appended.at(-1) ?? this.#entries[this.#baseLength - 1])?.id ?? null;
		for (const entry of entries) {
			if (entry.type === "reset_boundary" || entry.parentId !== parentId) return false;
			parentId = entry.id;
		}
		const affected = new Set<number>();
		for (const entry of entries) {
			const position = this.#index.length;
			this.#appended.push(entry);
			this.#appendEntry(position);
			if (entry.type === "message") affected.add(this.#positions.get(this.#groups.get(entry.id)?.[0] ?? entry.id)!);
			else if (entry.type === "custom" && entry.customType === MESSAGE_OVERRIDE_CUSTOM_TYPE) {
				for (const id of decodeCompactionMessageOverride(entry.data)?.messageIds ?? []) {
					const target = this.#positions.get(this.#groups.get(id)?.[0] ?? id);
					if (target !== undefined) affected.add(target);
				}
			} else if (entry.type === "custom" && entry.customType === USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE) {
				for (const { id } of unpackPreservedUserMessageClassifications(entry.data)) {
					const target = this.#positions.get(id);
					if (target !== undefined) affected.add(target);
				}
			}
		}
		for (const position of affected) this.#refreshPosition(position);
		return true;
	}

	get resetId(): string | null {
		return this.#offset > 0 ? this.#entries[this.#offset - 1]!.id : null;
	}

	nonAutoCount(): number {
		this.#assertCurrent();
		return this.#nonAutoGroups.size;
	}

	positionOf(id: string): number | undefined {
		this.#assertCurrent();
		return this.#positions.get(id);
	}

	getManualGroup(id: string): PreservationManualGroup | undefined {
		this.#assertCurrent();
		const row = this.rowAt(this.rowIndexOf(id, "all"), "all");
		if (!row) return undefined;
		return {
			id: row.id,
			memberIds: row.memberIds,
			members: row.memberIds.map(sourceId => ({
				sourceId,
				state: this.#overrides.get(sourceId) ?? "auto",
				revisionId: this.#overrideRevisions.get(sourceId)?.id ?? null,
			})),
		};
	}

	getManualGroups(options: { nonAutoOnly?: boolean } = {}): IterableIterator<PreservationManualGroup> {
		this.#assertCurrent();
		const sourceEnd = this.#index.length;
		let remaining = this.#nonAutoGroups.size;
		const sparse = this.#nonAutoGroups.values();
		const owner = this;
		return (function* () {
			let after = -1;
			for (;;) {
				owner.#assertCurrent();
				let position: number;
				if (options.nonAutoOnly) {
					if (remaining-- <= 0) return;
					const next = sparse.next();
					if (next.done) return;
					position = next.value;
				} else {
					let low = 0,
						high = owner.#rows.length;
					while (low < high) {
						const middle = (low + high) >>> 1;
						if (owner.#rows[middle]! <= after) low = middle + 1;
						else high = middle;
					}
					position = owner.#rows[low] ?? sourceEnd;
					if (position >= sourceEnd) return;
					after = position;
				}
				if (position >= sourceEnd) continue;
				const group = owner.getManualGroup(owner.#entry(position)!.id);
				if (
					!group ||
					group.memberIds.some(
						id =>
							owner.#positions.get(id)! >= sourceEnd ||
							(owner.#overrideRevisions.get(id)?.position ?? -1) >= sourceEnd,
					)
				)
					continue;
				if (options.nonAutoOnly && !group.members.some(member => member.state !== "auto")) continue;
				yield group;
			}
		})();
	}

	classificationStatus(id: string): PreservationRow["categoryStatus"] {
		this.#assertCurrent();
		if (this.#classifications.has(id)) return "valid";
		const problem = this.#classificationProblems.get(id);
		return problem === "invalidated" ? "invalidated" : problem ? "unsupported" : "absent";
	}

	invalidateClassifications(ids: readonly string[]): void {
		this.#assertCurrent();
		for (const id of ids) {
			const position = this.#positions.get(id);
			if (position === undefined) continue;
			this.#classifications.delete(id);
			this.#classificationProblems.set(id, "invalidated");
			this.#refreshPosition(position);
		}
	}

	/** Inspect configured source pricing even when no window/cap currently admits the source. */
	inspectCandidate(id: string, raw = false): PreservationCandidate | undefined {
		this.#assertCurrent();
		const position = this.#positions.get(id);
		const entry = position === undefined ? undefined : this.#entry(position);
		return entry
			? preservationCandidate(entry, this.#policy, this.#tokenizer, raw || this.#overrides.get(id) === "keep")
			: undefined;
	}

	select(options: PreservationSelectionOptions): PreservationSelection {
		this.#assertCurrent();
		const enabled = options.enabled ?? this.#policy.enabled;
		const unavailableLimits: ("first" | "recent" | "hardRecent" | "always")[] = [];
		const resolve = (limit: PreservationLimit, group: "first" | "recent" | "hardRecent" | "always"): PolicyLimit => {
			if (limit.mode !== "context-percent") return limit as PolicyLimit;
			if (
				options.maximumContext === undefined ||
				!Number.isFinite(options.maximumContext) ||
				options.maximumContext <= 0
			) {
				unavailableLimits.push(group);
				return { mode: "off" };
			}
			return { mode: "tokens", value: (options.maximumContext * limit.value) / 100 };
		};
		const firstLimit = options.first ?? this.#policy.first;
		const recentLimit = options.recent ?? this.#policy.recent;
		const cap = options.alwaysCap ?? this.#policy.alwaysCap;
		const H = this.#index.query(
			enabled ? resolve(options.hardRecent ?? this.#policy.hardRecent, "hardRecent") : { mode: "off" },
			"recent",
			"raw",
			this.#index.length,
			enabled,
		);
		const h = H.count ? H.start : this.#index.length;
		const linked = cap === "keep-first" ? firstLimit : recentLimit;
		const alwaysLimit =
			cap === "uncapped" || linked.mode === "off" || linked.mode === "all"
				? { mode: "all" as const }
				: resolve(linked, "always");
		const always = this.#index.query(alwaysLimit, cap === "keep-first" ? "first" : "recent", "always", h, enabled);
		const first = this.#index.query(
			enabled ? resolve(firstLimit, "first") : { mode: "off" },
			"first",
			"eligible",
			h,
			enabled,
		);
		const recent = this.#index.query(
			enabled ? resolve(recentLimit, "recent") : { mode: "off" },
			"recent",
			"eligible",
			h,
			enabled,
		);
		const ranges: { range: PolicyRange; kind: PolicyKind }[] = [
			{ range: H, kind: "raw" },
			{ range: first, kind: "eligible" },
			{ range: recent, kind: "eligible" },
			{ range: always, kind: "always" },
		];
		const totalP = this.#index.measureUnion(ranges, h, enabled);
		const allAlways = { start: 0, end: this.#index.length, count: 0, tokens: 0 };
		const included = (position: number, range: PolicyRange, kind: PolicyKind) =>
			this.#index.includes(position, range, kind, h, enabled);
		const selectedUser = (position: number) => {
			const entry = this.#entry(position);
			return (
				!!entry && isPreservationUser(entry) && ranges.some(({ range, kind }) => included(position, range, kind))
			);
		};
		const userAlways = this.#index.measureUnion([{ range: always, kind: "always" }], h, enabled);
		const totalN = { count: always.count - userAlways.count, tokens: always.tokens - userAlways.tokens };
		const owner = this;
		const membership = (
			size: number,
			has: (id: string) => boolean,
			values: () => IterableIterator<string>,
		): PreservationMembership => ({ size, has, values, [Symbol.iterator]: values });
		const P = membership(
			totalP.count,
			id => {
				owner.#assertCurrent();
				const position = owner.#positions.get(id);
				return position !== undefined && selectedUser(position);
			},
			function* () {
				owner.#assertCurrent();
				// Merge indexed lane iterators; demand-driven output never scans unselected history.
				const iterators = ranges.map(({ range, kind }) => owner.#index.iterate(range, kind, h, enabled));
				const heads = iterators.map(iterator => iterator.next());
				for (;;) {
					let next = Infinity;
					for (const head of heads) if (!head.done && head.value < next) next = head.value;
					if (next === Infinity) return;
					if (selectedUser(next)) yield owner.#entry(next)!.id;
					for (let index = 0; index < heads.length; index++)
						if (!heads[index]!.done && heads[index]!.value === next) heads[index] = iterators[index]!.next();
				}
			},
		);
		const hard = membership(
			H.count,
			id => {
				owner.#assertCurrent();
				const position = owner.#positions.get(id);
				return position !== undefined && included(position, H, "raw");
			},
			function* () {
				owner.#assertCurrent();
				for (const position of owner.#index.iterate(H, "raw", h, enabled)) yield owner.#entry(position)!.id;
			},
		);
		const N = membership(
			totalN.count,
			id => {
				owner.#assertCurrent();
				const position = owner.#positions.get(owner.#groups.get(id)?.[0] ?? id);
				return position !== undefined && owner.#manualAtoms.has(position) && included(position, always, "always");
			},
			function* () {
				owner.#assertCurrent();
				const positions: number[] = [];
				for (const anchor of owner.#index.iterate(always, "always", h, enabled))
					for (const id of owner.#manualAtoms.get(anchor) ?? []) positions.push(owner.#positions.get(id)!);
				positions.sort((a, b) => a - b);
				for (const position of positions) yield owner.#entry(position)!.id;
			},
		);
		const candidate = (id: string): PreservationCandidate | undefined => {
			owner.#assertCurrent();
			const position = owner.#positions.get(id);
			if (position === undefined || (!P.has(id) && !N.has(id))) return undefined;
			return preservationCandidate(
				owner.#entry(position)!,
				owner.#policy,
				owner.#tokenizer,
				N.has(id) || hard.has(id) || owner.#overrides.get(id) === "keep",
			);
		};
		return {
			P,
			N,
			H: hard,
			quota: { P: totalP, N: totalN, H, first, recent, always },
			unavailableLimits,
			blockers: {
				first: first.blocker === undefined ? undefined : owner.#entry(first.blocker)?.id,
				recent: recent.blocker === undefined ? undefined : owner.#entry(recent.blocker)?.id,
				hardRecent: H.blocker === undefined ? undefined : owner.#entry(H.blocker)?.id,
				always: always.blocker === undefined ? undefined : owner.#entry(always.blocker)?.id,
			},
			positions(id) {
				owner.#assertCurrent();
				const position = owner.#positions.get(id);
				if (position === undefined) return {};
				const rank: { first?: number; recent?: number } = {};
				if (included(position, first, "eligible"))
					rank.first = owner.#index.measureUnion(
						[{ range: { ...first, end: position + 1 }, kind: "eligible" }],
						h,
						enabled,
					).count;
				if (included(position, recent, "eligible"))
					rank.recent = owner.#index.measureUnion(
						[{ range: { ...recent, start: position }, kind: "eligible" }],
						h,
						enabled,
					).count;
				return rank;
			},
			reasons(id) {
				owner.#assertCurrent();
				const position = owner.#positions.get(owner.#groups.get(id)?.[0] ?? id);
				if (position === undefined)
					return { first: false, recent: false, hardRecent: false, always: false, capDenied: false };
				const admitted = included(position, always, "always");
				return {
					first: included(position, first, "eligible"),
					recent: included(position, recent, "eligible"),
					hardRecent: included(position, H, "raw"),
					always: admitted,
					capDenied: !unavailableLimits.includes("always") && !admitted && included(position, allAlways, "always"),
				};
			},
			candidate,
			*candidates() {
				const users = P.values();
				const others = N.values();
				let u = users.next(),
					n = others.next();
				while (!u.done || !n.done) {
					if (!u.done && (n.done || owner.#positions.get(u.value)! < owner.#positions.get(n.value)!)) {
						yield candidate(u.value)!;
						u = users.next();
					} else if (!n.done) {
						yield candidate(n.value)!;
						n = others.next();
					}
				}
			},
			*nonUserAtoms() {
				owner.#assertCurrent();
				for (const position of owner.#index.iterate(always, "always", h, enabled)) {
					const memberIds = owner.#manualAtoms.get(position);
					if (!memberIds) continue;
					const entries = memberIds.map(id => owner.#entry(owner.#positions.get(id)!)!);
					yield {
						id: owner.#entry(position)!.id,
						memberIds,
						entries,
						quotaTokens: entries.reduce((sum, entry) => sum + owner.#tokenizer.countMessage(entry.message), 0),
					};
				}
			},
		};
	}
}

/** Successful active-branch facts for backfill, without pricing sources or constructing UI rows. */
export async function readPreservedUserMessageClassificationMasks(
	entries: readonly SessionEntry[],
	options: PreservationBuildOptions,
): Promise<ReadonlyMap<string, number> | undefined> {
	let deadline = performance.now() + (options.sliceMs ?? 4);
	let offset = entries.length;
	while (offset > 0 && entries[offset - 1]!.type !== "reset_boundary") {
		offset--;
		if (performance.now() >= deadline) {
			await (options.yieldControl?.() ?? new Promise<void>(resolve => setTimeout(resolve, 0)));
			if (!options.isCurrent()) return undefined;
			deadline = performance.now() + (options.sliceMs ?? 4);
		}
	}
	const users = new Set<string>();
	const masks = new Map<string, number>();
	for (let index = offset; index < entries.length; index++) {
		const entry = entries[index]!;
		if (isPreservationUser(entry)) users.add(entry.id);
		else if (entry.type === "custom" && entry.customType === USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE) {
			for (const { id, mask } of unpackPreservedUserMessageClassifications(entry.data))
				if (users.has(id)) masks.set(id, mask);
		}
		if (performance.now() >= deadline) {
			await (options.yieldControl?.() ?? new Promise<void>(resolve => setTimeout(resolve, 0)));
			if (!options.isCurrent()) return undefined;
			deadline = performance.now() + (options.sliceMs ?? 4);
		}
	}
	return options.isCurrent() ? masks : undefined;
}
