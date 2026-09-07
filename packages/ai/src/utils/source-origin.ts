import { type FluentType, type } from "@oh-my-pi/omptype";
import type { SourceRewrite } from "../compaction-source";
import type { Message } from "../types";

export interface NativeSourcePart {
	entryId: string;
	order: number;
	blockIndex: number | string;
	coverage: "full" | "partial" | "derived";
	representation: "native" | "json-quoted-block" | "original-image" | "transformed-text";
	sourceSpan?: { start: number; end: number };
	transportSpan?: { start: number; end: number };
	/** Original block extent at capture. */
	sourceLength?: number;
	/** Current correspondence after controlled rewrites; sourceSpan stays immutable. */
	currentSourceSpan?: { start: number; end: number };
	currentBlockIndex?: number | string;
	currentSourceLength?: number;
	/** Omission denotes an untouched capture, not an unresolved rewrite. */
	status?: "exact-current" | "historical-not-current" | "unknown";
	/** Actual emitted item.content position, not the original source block index. */
	transportBlockIndex?: number;
}

export type NativeItemOrigin =
	| { kind: "source"; parts: NativeSourcePart[] }
	| { kind: "synthetic"; reason: string; anchorEntryId?: string }
	| { kind: "aggregate"; compactionEntryId: string; coveredSources?: NativeSourcePart[] }
	| { kind: "unknown"; reason: string };

const nativePositionSchema = type("number.integer").narrow(value => value >= 0);
const nativeRangeSchema = type({ start: nativePositionSchema, end: nativePositionSchema }).narrow(range => range.end >= range.start);
const nativeSourcePartSchema: FluentType<NativeSourcePart> = type({
	entryId: "string",
	order: type("number").narrow(Number.isFinite),
	blockIndex: nativePositionSchema.or(type("string")),
	coverage: "'full' | 'partial' | 'derived'",
	representation: "'native' | 'json-quoted-block' | 'original-image' | 'transformed-text'",
	"sourceSpan?": nativeRangeSchema,
	"transportSpan?": nativeRangeSchema,
	"sourceLength?": nativePositionSchema,
	"currentSourceSpan?": nativeRangeSchema,
	"currentBlockIndex?": nativePositionSchema.or(type("string")),
	"currentSourceLength?": nativePositionSchema,
	"status?": "'exact-current' | 'historical-not-current' | 'unknown'",
	"transportBlockIndex?": nativePositionSchema,
});
const nativeItemOriginSchema: FluentType<NativeItemOrigin> = type({ kind: "'source'", parts: nativeSourcePartSchema.array() })
	.or(type({ kind: "'synthetic'", reason: "string", "anchorEntryId?": "string" }))
	.or(type({ kind: "'aggregate'", compactionEntryId: "string", "coveredSources?": nativeSourcePartSchema.array() }))
	.or(type({ kind: "'unknown'", reason: "string" }));
const nativeItemOriginsSchema = nativeItemOriginSchema.array();
const nativeSourcePartsSchema = nativeSourcePartSchema.array();

/** Validate persisted aggregate input coverage without fabricating an item wrapper. */
export function validateNativeSourceParts(value: unknown): NativeSourcePart[] | undefined {
	return nativeSourcePartsSchema.allows(value) ? value : undefined;
}

/** Validate persisted local metadata once at its untyped boundary; never infer missing origins. */
export function validateNativeItemOrigins(value: unknown): NativeItemOrigin[] | undefined {
	return nativeItemOriginsSchema.allows(value) ? value : undefined;
}

const origins = new WeakMap<object, NativeItemOrigin>();
const unknownOrigin: NativeItemOrigin = { kind: "unknown", reason: "legacy-map-absent" };
let sourceBindingGeneration = 0;

/** O(1) cache guard; ordinary emission transfers do not change source bindings. */
export function getSourceOriginBindingGeneration(): number {
	return sourceBindingGeneration;
}

export function getSourceOrigin(value: object): NativeItemOrigin | undefined {
	return origins.get(value);
}

export function setSourceOrigin<T extends object>(value: T, origin: NativeItemOrigin): T {
	origins.set(value, origin);
	return value;
}

/** Transfer only at the operation that constructs a known equivalent object. */
export function transferSourceOrigin<T extends object>(from: object, to: T): T {
	const origin = origins.get(from);
	if (origin) origins.set(to, origin);
	return to;
}

/** A known whole-block transform preserves logical coverage, never original character offsets. */
export function transferTransformedSourceOrigin<T extends object>(
	from: object,
	to: T,
	coverage: NativeSourcePart["coverage"] = "full",
): T {
	const origin = origins.get(from);
	if (!origin) return to;
	if (origin.kind !== "source") return setSourceOrigin(to, origin);
	return setSourceOrigin(to, {
		kind: "source",
		parts: origin.parts.map(({ sourceSpan: _sourceSpan, transportSpan: _transportSpan, currentSourceSpan: _currentSourceSpan, ...part }) => ({
			...part,
			coverage: part.coverage === "full" ? coverage : part.coverage,
			representation: "transformed-text",
		})),
	});
}

/** Combine only actual emitted components; unknown input never gains source credit. */
export function combineSourceOrigins(values: readonly unknown[]): NativeItemOrigin {
	const parts: NativeSourcePart[] = [];
	let sole: NativeItemOrigin | undefined;
	for (const value of values) {
		const origin = value && typeof value === "object" ? origins.get(value) : undefined;
		sole ??= origin;
		if (origin?.kind === "source") parts.push(...origin.parts);
	}
	return parts.length ? { kind: "source", parts } : sole ?? unknownOrigin;
}

/** Assemble an emitted content array with its actual transport positions. */
export function combineContentSourceOrigins(content: readonly unknown[]): NativeItemOrigin {
	const origin = combineSourceOrigins(content);
	if (origin.kind !== "source") return origin;
	const parts: NativeSourcePart[] = [];
	for (let index = 0; index < content.length; index++) {
		const block = content[index];
		const blockOrigin = block && typeof block === "object" ? origins.get(block) : undefined;
		if (blockOrigin?.kind !== "source") continue;
		for (const part of blockOrigin.parts) parts.push({ ...part, transportBlockIndex: index });
	}
	return { kind: "source", parts };
}

/** Known message rewrite: derive coverage from the blocks actually emitted, not the old array. */
export function transferMessageSourceOrigin<T extends Message>(from: Message, to: T): T {
	if (typeof to.content === "string") return transferSourceOrigin(from, to);
	return setSourceOrigin(to, combineContentSourceOrigins(to.content));
}

export function bindMessageSource(message: Message, entryId: string, order: number): void {
	sourceBindingGeneration++;
	const parts: NativeSourcePart[] = [];
	const bind = (block: object, blockIndex: number | string, text?: string, image = false): void => {
		const part: NativeSourcePart = {
			entryId,
			order,
			blockIndex,
			coverage: "full",
			representation: image ? "original-image" : "native",
			...(text === undefined ? {} : {
				sourceLength: text.length,
				sourceSpan: { start: 0, end: text.length },
				transportSpan: { start: 0, end: text.length },
			}),
		};
		parts.push(part);
		setSourceOrigin(block, { kind: "source", parts: [part] });
	};
	if (typeof message.content === "string") bind(message, 0, message.content);
	else {
		for (let index = 0; index < message.content.length; index++) {
			const block = message.content[index]!;
			bind(block, index, block.type === "text" ? block.text : block.type === "thinking" ? block.thinking : undefined, block.type === "image");
		}
	}
	if (message.role === "toolResult" && message.providerMetadata) {
		bind(message.providerMetadata, "metadata");
		bind(message.providerMetadata.screenshot, "metadata.screenshot", undefined, true);
		for (let index = 0; index < message.providerMetadata.acknowledgedSafetyChecks.length; index++) {
			bind(message.providerMetadata.acknowledgedSafetyChecks[index]!, `metadata.acknowledgedSafetyChecks.${index}`);
		}
	}
	setSourceOrigin(message, { kind: "source", parts });
	const payload = message.providerPayload;
	if (message.role !== "assistant" || payload?.type !== "openaiResponsesHistory" || payload.dt !== true) return;
	// A delta is this assistant response, unlike a full replay snapshot. Native
	// components get their own source names, never guessed normalized-block matches.
	importItemOrigins(payload.items, validateNativeItemOrigins(payload.origins));
	for (let itemIndex = 0; itemIndex < payload.items.length; itemIndex++) {
		const item = payload.items[itemIndex]!;
		const existing = getSourceOrigin(item);
		if (existing && (existing.kind !== "unknown" || existing.reason !== "legacy-map-absent")) continue;
		const nativePartsStart = parts.length;
		const component = `providerPayload.${itemIndex}`;
		const text = typeof item.content === "string" ? item.content : typeof item.arguments === "string" ? item.arguments : undefined;
		bind(item, component, text);
		if (Array.isArray(item.content)) {
			for (let blockIndex = 0; blockIndex < item.content.length; blockIndex++) {
				const block: unknown = item.content[blockIndex];
				if (!block || typeof block !== "object") continue;
				const blockText = "text" in block && typeof block.text === "string" ? block.text : undefined;
				bind(block, `${component}.content.${blockIndex}`, blockText, "type" in block && block.type === "input_image");
				parts[parts.length - 1]!.transportBlockIndex = blockIndex;
			}
		}
		setSourceOrigin(item, { kind: "source", parts: parts.splice(nativePartsStart) });
	}
	payload.origins = exportItemOrigins(payload.items);
}

/** Serialize the sidecar separately; never put provenance fields inside provider items. */
export function exportItemOrigins(items: readonly object[]): NativeItemOrigin[] {
	return items.map(item => origins.get(item) ?? unknownOrigin);
}

/** Import only the persisted map. Missing legacy maps cannot be reconstructed from content. */
export function importItemOrigins(items: readonly object[], itemOrigins?: readonly NativeItemOrigin[]): void {
	if (!itemOrigins) return;
	for (let index = 0; index < items.length; index++) {
		const item = items[index]!;
		const origin = itemOrigins[index] ?? unknownOrigin;
		setSourceOrigin(item, origin);
		const payload = "content" in item ? item.content : "output" in item ? item.output : undefined;
		const content = Array.isArray(payload) ? payload : payload && typeof payload === "object" ? [payload] : undefined;
		if (origin.kind !== "source" || !Array.isArray(content)) continue;
		const blocks = new Map<number, NativeSourcePart[]>();
		for (const part of origin.parts) {
			const position = part.transportBlockIndex;
			if (position === undefined || !Number.isInteger(position) || position < 0) continue;
			const block = content[position];
			if (!block || typeof block !== "object") continue;
			const parts = blocks.get(position);
			if (parts) parts.push(part);
			else blocks.set(position, [part]);
		}
		for (const [position, parts] of blocks) setSourceOrigin(content[position], { kind: "source", parts });
	}
}

/** JSON-shaped native data clone with identity transfer during each actual recursive clone. */
export function cloneWithSourceOrigins<T>(value: T): T {
	if (!value || typeof value !== "object") return value;
	if (Array.isArray(value)) return transferSourceOrigin(value, value.map(item => cloneWithSourceOrigins(item))) as T;
	const clone: Record<string, unknown> = {};
	for (const [key, child] of Object.entries(value)) clone[key] = cloneWithSourceOrigins(child);
	return transferSourceOrigin(value, clone) as T;
}

/** Arbitrary hooks can mutate in place: invalidate the actual returned tree, not an index-aligned copy. */
export function invalidateSourceOrigins(value: unknown, reason = "externally-mutated"): void {
	const seen = new WeakSet<object>();
	sourceBindingGeneration++;
	const visit = (node: unknown): void => {
		if (!node || typeof node !== "object" || seen.has(node)) return;
		seen.add(node);
		setSourceOrigin(node, { kind: "unknown", reason });
		if ("type" in node && node.type === "openaiResponsesHistory" && "items" in node && Array.isArray(node.items) && "origins" in node) {
			node.origins = node.items.map(() => ({ kind: "unknown", reason }));
		}
		if ("replacementHistory" in node && Array.isArray(node.replacementHistory) && "replacementOrigins" in node) {
			node.replacementOrigins = node.replacementHistory.map(() => ({ kind: "unknown", reason }));
		}
		for (const child of Object.values(node)) visit(child);
	};
	visit(value);
}

function occurrenceKey(item: object): string | object {
	const origin = origins.get(item);
	if (origin?.kind === "aggregate") return JSON.stringify(["aggregate", origin.compactionEntryId]);
	if (origin?.kind !== "source" || origin.parts.length === 0) return item;
	return JSON.stringify(origin.parts.map(part => [
		part.entryId, part.order, part.blockIndex, part.coverage, part.representation,
		part.sourceSpan?.start, part.sourceSpan?.end, part.transportSpan?.start, part.transportSpan?.end,
		part.transportBlockIndex, part.status ?? "exact-current", part.currentBlockIndex ?? part.blockIndex,
		(part.currentSourceSpan ?? part.sourceSpan)?.start, (part.currentSourceSpan ?? part.sourceSpan)?.end,
	]));
}

function sourceOrder(item: object): number | undefined {
	const origin = origins.get(item);
	if (origin?.kind !== "source" || !origin.parts.length) return undefined;
	let order = Infinity;
	for (const part of origin.parts) order = Math.min(order, part.order);
	return order;
}

/** Authoritative full snapshot plus uncovered prefix occurrences; equality is identity, never wire IDs/text. */
export function mergeSourceHistory<T extends object>(prefix: readonly T[], history: readonly T[]): T[] {
	if (!prefix.length) return [...history];
	if (!history.length) return [...prefix];
	const remaining = new Map<string | object, number>();
	for (const item of history) {
		const key = occurrenceKey(item);
		remaining.set(key, (remaining.get(key) ?? 0) + 1);
	}
	const missing: T[] = [];
	for (const item of prefix) {
		const key = occurrenceKey(item);
		const count = remaining.get(key) ?? 0;
		if (count) remaining.set(key, count - 1);
		else missing.push(item);
	}
	if (!missing.length) return [...history];
	const result: T[] = [];
	let prefixIndex = 0;
	for (const item of history) {
		const order = sourceOrder(item);
		while (prefixIndex < missing.length) {
			const candidate = missing[prefixIndex]!;
			const candidateOrder = sourceOrder(candidate);
			if (candidateOrder !== undefined && (order === undefined || candidateOrder > order)) break;
			result.push(candidate);
			prefixIndex++;
		}
		result.push(item);
	}
	while (prefixIndex < missing.length) result.push(missing[prefixIndex++]!);
	return result;
}

/** Remap current correspondence only; never rewrite captured native bytes or source coordinates. */
export function remapNativeItemOrigins(itemOrigins: readonly NativeItemOrigin[], rewrites: readonly SourceRewrite[]): NativeItemOrigin[] {
	const byEntry = new Map(rewrites.map(rewrite => [rewrite.entryId, rewrite]));
	const remapPart = (part: NativeSourcePart): NativeSourcePart[] => {
		const rewrite = byEntry.get(part.entryId);
		if (!rewrite || part.status === "historical-not-current" || part.status === "unknown") return [part];
		const historical = (status: "historical-not-current" | "unknown"): NativeSourcePart[] => {
			const { currentSourceSpan: _current, currentBlockIndex: _block, currentSourceLength: _length, ...captured } = part;
			return [{ ...captured, status }];
		};
		if (!rewrite.blocks) return historical("unknown");
		const currentBlock = part.currentBlockIndex ?? part.blockIndex;
		const block = rewrite.blocks.find(candidate => candidate.oldBlockIndex === currentBlock);
		if (!block) return typeof currentBlock === "number" ? [part] : historical("unknown");
		if (block.newBlockIndex === null) return historical("historical-not-current");
		const edits = block.textEdits;
		const priorLength = part.currentSourceLength ?? part.sourceLength;
		if (!edits?.length) return [{ ...part, currentBlockIndex: block.newBlockIndex, currentSourceLength: priorLength, status: "exact-current" }];
		const currentSourceLength = priorLength === undefined ? undefined : priorLength + edits.reduce((sum, edit) => sum + edit.replacementLength - (edit.end - edit.start), 0);
		const current = part.currentSourceSpan ?? part.sourceSpan;
		const capture = part.sourceSpan;
		if (!current || !capture || current.end - current.start !== capture.end - capture.start) return historical("unknown");
		const result: NativeSourcePart[] = [];
		let cursor = current.start;
		let delta = 0;
		const emit = (start: number, end: number, retained: boolean, shift: number): void => {
			if (end <= start) return;
			const sourceSpan = { start: capture.start + start - current.start, end: capture.start + end - current.start };
			const transport = part.transportSpan;
			const transportSpan = transport && transport.end - transport.start === current.end - current.start
				? { start: transport.start + start - current.start, end: transport.start + end - current.start }
				: undefined;
			result.push({
				...part,
				coverage: "partial",
				sourceSpan,
				transportSpan,
				currentBlockIndex: retained ? block.newBlockIndex! : undefined,
				currentSourceSpan: retained ? { start: start + shift, end: end + shift } : undefined,
				currentSourceLength: retained ? currentSourceLength : undefined,
				status: retained ? "exact-current" : "historical-not-current",
			});
		};
		for (const edit of edits) {
			if (edit.end <= current.start && edit.start < current.start) {
				delta += edit.replacementLength - (edit.end - edit.start);
				continue;
			}
			if (edit.start >= current.end) break;
			const start = Math.max(cursor, edit.start);
			emit(cursor, Math.min(start, current.end), true, delta);
			const end = Math.min(current.end, edit.end);
			emit(start, end, false, 0);
			cursor = Math.max(start, end);
			delta += edit.replacementLength - (edit.end - edit.start);
		}
		emit(cursor, current.end, true, delta);
		return result.length ? result : historical("historical-not-current");
	};
	return itemOrigins.map(origin => {
		const parts = origin.kind === "source" ? origin.parts : origin.kind === "aggregate" ? origin.coveredSources : undefined;
		if (!parts || !parts.some(part => byEntry.has(part.entryId))) return origin;
		const mapped = parts.flatMap(remapPart);
		return origin.kind === "source" ? { ...origin, parts: mapped } : { ...origin, coveredSources: mapped };
	});
}

