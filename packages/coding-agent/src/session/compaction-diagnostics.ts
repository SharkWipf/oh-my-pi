import { type AgentMessage, IMAGE_TOKEN_ESTIMATE, type Tokenizer } from "@oh-my-pi/pi-agent-core";
import type {
	CompactionDiagnostics,
	ContextInventoryRow,
	DiagnosticTokenQuantity,
} from "@oh-my-pi/pi-agent-core/compaction";
import type { Context, Message, Model } from "@oh-my-pi/pi-ai";
import type { SourceLayoutPart, SourceRepresentation } from "@oh-my-pi/pi-ai/compaction-source";
import { FRAME_TOKEN_ESTIMATE } from "@oh-my-pi/snapcompact";
import { getArchiveFrameAccounting, getInlinePhysical, getSourceOrigin, type NativeItemOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";
import { estimateToolSchemaTokens } from "../modes/utils/context-usage";
import { convertToLlm } from "./messages";

export interface CompactionDiagnosticsInput {
	messages: readonly AgentMessage[];
	/** Actual OMP-prehook Context after all owned transforms; never recreated by this builder. */
	preparedContext?: Context;
	tokenizer: Tokenizer;
	fixedCosts: {
		systemPromptTokens: number;
		toolsTokens: number;
		systemContextTokens: number;
		skillsTokens: number;
	};
	/** Actual emitted prompt segments, tool descriptors, context segments, and rendered skill entries. */
	fixedCounts?: { systemPrompt: number; tools: number; context: number; skills: number };
	model: Pick<Model, "provider" | "id"> & { contextWindow?: Model["contextWindow"] };
	method: string;
	settings: Record<string, unknown>;
	before: DiagnosticTokenQuantity;
	target: CompactionDiagnostics["target"];
	snapshot?: CompactionDiagnostics["snapshot"];
	sourceRepresentation?: SourceRepresentation;
	/** Owning compaction entry; required to resolve an archive-frame locator into this descriptor. */
	sourceRepresentationEntryId?: string;
	/** Entry IDs aligned with messages, supplied by the context reconstruction owner. */
	messageSourceIds?: readonly (string | undefined)[];
	/** Exact emitter mapping. blockIndex names summary.blocks, not its lead-in text. */
	sourceLocations?: readonly { messageIndex: number; blockIndex?: number; layoutIndex: number }[];
	selectionReasons?: ReadonlyMap<string, readonly string[]>;
	selectedUserSourceIds?: ReadonlySet<string>;
	addedUserSourceIds?: ReadonlySet<string>;
	manualNonUserSourceIds?: ReadonlySet<string>;
	/** Raw message IDs after the representation owner's committed replay frontier. */
	postCompactionSourceIds?: ReadonlySet<string>;
	/** Separate shape-family estimate for disclosure; the inventory retains the generic archive price. */
	frameTokenEstimate?: number;
}

function quantity(tokens: number | null, description: string, basis: DiagnosticTokenQuantity["basis"] = "local-estimate"): DiagnosticTokenQuantity {
	return { tokens, basis: tokens === null ? "unknown" : basis, description };
}

function locationKey(messageIndex: number, blockIndex?: number): string {
	return blockIndex === undefined ? `message:${messageIndex}` : `message:${messageIndex}/block:${blockIndex}`;
}

/** Describes stored source membership without rebuilding or allocating shared-frame costs. */
export function describeCompactionSourceRepresentation(representation: SourceRepresentation | undefined, id: string): string[] {
	if (!representation) return ["Source representation unavailable; historical coverage is unknown."];
	const descriptions: string[] = [];
	const sourceRuns = representation.coverage.filter(run => run.entryId === id);
	for (const run of sourceRuns) {
		const snapshot = `captured block ${run.snapshot.blockIndex} [${run.snapshot.start}, ${run.snapshot.end})`;
		const current = run.current ? `current block ${run.current.blockIndex} [${run.current.start}, ${run.current.end})` : "current correspondence unavailable";
		descriptions.push(`${run.status}: ${snapshot}; ${current}; contribution ${run.contribution ?? "unknown"}.`);
	}
	for (let index = 0; index < representation.layout.length; index++) {
		const part = representation.layout[index]!;
		if (part.kind === "source" && part.entryId === id) {
			const spans = part.spans?.map(span => `block ${span.blockIndex} [${span.start}, ${span.end})`).join(", ");
			descriptions.push(`Occurrence ${index}: source text/blocks, ${spans ?? "complete original atomic member"}.`);
		} else if (part.kind === "original-image" && part.entryId === id) {
			descriptions.push(`Occurrence ${index}: original image captured at block ${part.blockIndex}; ${part.currentBlockIndex === undefined ? "current reference unresolved" : `current block ${part.currentBlockIndex}`}.`);
		} else if (part.kind === "text" || part.kind === "frame") {
			const covered = sourceRuns.some(run => run.normalized && run.normalized.start < part.range.end && run.normalized.end > part.range.start);
			if (covered) descriptions.push(`Occurrence ${index}: ${part.kind === "frame" ? `frame ${part.frameIndex}` : "archive text"}, archive range [${part.range.start}, ${part.range.end}); may share physical content with other sources, no per-source token allocation.`);
		}
	}
	if (representation.aggregate?.entryIds.includes(id)) descriptions.push(`${representation.aggregate.reason} aggregate reference only; not exact retained source position or coverage.`);
	return descriptions.length > 0 ? descriptions : ["No coverage for this source is recorded in this representation; ordinary/later context membership is separate."];
}

function sourceCoverage(part: SourceLayoutPart | undefined, representation: SourceRepresentation | undefined): Pick<ContextInventoryRow, "coverage" | "sourceIds" | "contributions" | "note"> {
	if (!part || !representation) return { coverage: "unknown" };
	if (part.kind === "gap") return { coverage: "aggregate", note: part.reason };
	if (part.kind === "source" && part.contribution) return { coverage: "source", sourceIds: [part.entryId], contributions: [part.contribution] };
	const ids = new Set<string>();
	const contributions = new Set<NonNullable<ContextInventoryRow["contributions"]>[number]>();
	let historical = part.kind === "original-image" && part.currentBlockIndex === undefined;
	let unknown = false;
	let missingContribution = false;
	if (part.kind === "source" || part.kind === "original-image") ids.add(part.entryId);
	for (const run of representation.coverage) {
		const intersects = part.kind === "text" || part.kind === "frame"
			? run.normalized && run.normalized.start < part.range.end && run.normalized.end > part.range.start
			: run.entryId === part.entryId && (part.kind === "original-image"
				? run.snapshot.blockIndex === part.blockIndex
				: !part.spans || part.spans.some(span => run.current && span.blockIndex === run.current.blockIndex && span.start < run.current.end && span.end > run.current.start));
		if (!intersects) continue;
		ids.add(run.entryId);
		historical ||= run.status === "historical-not-current";
		unknown ||= run.status === "unknown";
		if (run.contribution) contributions.add(run.contribution);
		else missingContribution = true;
	}
	return {
		coverage: ids.size === 0 || unknown ? "unknown" : historical ? "historical" : "source",
		...(ids.size > 0 ? { sourceIds: [...ids] } : {}),
		contributions: missingContribution ? [] : [...contributions],
	};
}

function preparedOriginCoverage(origin: NativeItemOrigin | undefined, input: CompactionDiagnosticsInput, rawPart?: Extract<SourceLayoutPart, { kind: "source" }>): Pick<ContextInventoryRow, "coverage" | "sourceIds" | "contributions"> {
	if (!origin || origin.kind === "unknown") return { coverage: "unknown" };
	if (origin.kind !== "source") return { coverage: "aggregate" };
	const ids = new Set<string>();
	const contributions = new Set<NonNullable<ContextInventoryRow["contributions"]>[number]>();
	let historical = false;
	let unknown = false;
	let missingContribution = false;
	for (const part of origin.parts) {
		ids.add(part.entryId);
		historical ||= part.status === "historical-not-current";
		unknown ||= part.status === "unknown";
		if (part.coverage === "derived" || part.status === "unknown") { missingContribution = true; continue; }
		if (input.postCompactionSourceIds?.has(part.entryId) || (input.method === "uncompacted" && !input.sourceRepresentation)) { contributions.add("ordinary"); continue; }
		const range = part.sourceSpan ?? (part.sourceLength !== undefined ? { start: 0, end: part.sourceLength } : part.representation === "original-image" ? { start: 0, end: 1 } : undefined);
		const currentRange = part.currentSourceSpan ?? (part.status === undefined || part.status === "exact-current" ? range : undefined);
		const currentBlock = part.currentBlockIndex ?? part.blockIndex;
		let wholePart = false;
		if (!rawPart && input.sourceRepresentation) for (const layout of input.sourceRepresentation.layout) {
			if (layout.kind !== "source" || layout.entryId !== part.entryId || !layout.contribution || part.status === "historical-not-current") continue;
			if (layout.spans && (!currentRange || !layout.spans.some(span => span.blockIndex === currentBlock && span.start <= currentRange.start && span.end >= currentRange.end))) continue;
			contributions.add(layout.contribution);
			wholePart = true;
		}
		if (wholePart) continue;
		if (!range) { missingContribution = true; continue; }
		const covered: { start: number; end: number }[] = [];
		for (const run of input.sourceRepresentation?.coverage ?? []) {
			if (run.entryId !== part.entryId || run.snapshot.blockIndex !== part.blockIndex || run.snapshot.start >= range.end || run.snapshot.end <= range.start) continue;
			covered.push(run.snapshot);
			if (run.contribution) contributions.add(run.contribution);
			else missingContribution = true;
		}
		covered.sort((left, right) => left.start - right.start);
		let through = range.start;
		for (const span of covered) {
			if (span.start > through) break;
			through = Math.max(through, span.end);
		}
		missingContribution ||= through < range.end || covered.length === 0;
	}
	return { coverage: unknown || ids.size === 0 ? "unknown" : historical ? "historical" : "source", sourceIds: [...ids], contributions: missingContribution ? [] : [...contributions] };
}

function appendNativeRows(rows: ContextInventoryRow[], items: Array<Record<string, unknown>>, tokenizer: Tokenizer, location: string): number {
	let known = 0;
	for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
		const item = items[itemIndex]!;
		const firstRow = rows.length;
		let blockIndex = 0;
		const emit = (kind: ContextInventoryRow["kind"], label: string, tokens: number | null, size: ContextInventoryRow["payloadSize"], blocks = 1) => {
			rows.push({
				location: `${location}/item:${itemIndex}/block:${blockIndex++}`,
				kind, label, coverage: "aggregate", selectionReasons: [],
				quantity: quantity(tokens, tokens === null ? "Opaque native item cost is unknown" : kind === "original-image" ? "Native image baseline estimate, not a provider invoice" : "Visible native content under the local tokenizer estimator", kind === "original-image" ? "image-estimate" : "local-estimate"),
				counts: { messages: rows.length === firstRow && (item.type === "message" || item.type === undefined && typeof item.role === "string") ? 1 : 0, items: rows.length === firstRow ? 1 : 0, blocks, frames: 0, images: kind === "original-image" ? 1 : 0 },
				...(size ? { payloadSize: size } : {}), controls: ["native compaction", "model"],
				note: "Native returned content has no exact source-position attribution.",
			});
			if (tokens !== null) known += tokens;
		};
		const text = (value: string, label: string, kind: "text" | "tool" = "text") => emit(kind, label, tokenizer.countTokens(value), { value: Buffer.byteLength(value, "utf8"), unit: "utf8-bytes" });
		const opaque = (value: unknown, label: string, blocks = 1) => emit("native", label, null, { value: Buffer.byteLength(JSON.stringify(value) ?? "", "utf8"), unit: "utf8-bytes" }, blocks);
		const content = (value: unknown) => {
			if (typeof value === "string") { text(value, "Native message text"); return; }
			if (!Array.isArray(value)) { opaque(value, "Unresolved native content"); return; }
			for (const block of value) {
				if (!block || typeof block !== "object") { opaque(block, "Unresolved native block"); continue; }
				if ((block.type === "input_text" || block.type === "output_text" || block.type === "summary_text") && typeof block.text === "string") {
					text(block.text, `Native ${String(item.role ?? item.type)} text`);
				} else if (block.type === "input_image") {
					const url = typeof block.image_url === "string" ? block.image_url : undefined;
					const prefix = url?.startsWith("data:") ? url.indexOf(";base64,") : -1;
					const size: ContextInventoryRow["payloadSize"] = url && prefix !== undefined && prefix >= 0
						? { value: url.length - prefix - 8, unit: "base64-characters" }
						: { value: Buffer.byteLength(JSON.stringify(block), "utf8"), unit: "utf8-bytes" };
					emit("original-image", "Native input image", IMAGE_TOKEN_ESTIMATE, size);
				} else opaque(block, "Unresolved native block");
			}
		};
		if ((item.type === "message" || item.type === undefined) && typeof item.role === "string") {
			content(item.content);
		} else if (item.type === "function_call" && typeof item.name === "string" && typeof item.arguments === "string") {
			emit("tool", "Native function call", tokenizer.countTokens([item.name, item.arguments]), { value: Buffer.byteLength(item.name, "utf8") + Buffer.byteLength(item.arguments, "utf8"), unit: "utf8-bytes" });
		} else if (item.type === "function_call_output") {
			if (typeof item.output === "string") text(item.output, "Native function result", "tool");
			else content(item.output);
		} else if (item.type === "reasoning") {
			if (Array.isArray(item.summary)) content(item.summary);
			if (item.encrypted_content !== undefined) opaque(item.encrypted_content, "Encrypted native reasoning", 0);
			else if (rows.length === firstRow) opaque(item, "Opaque native reasoning reference", 0);
		} else opaque(item, item.type === "compaction" ? "Opaque native compaction" : "Unresolved native item", 0);
		// An empty native message still occupies one exact item/message boundary.
		if (rows.length === firstRow) emit("text", "Empty native message", 0, { value: 0, unit: "utf8-bytes" }, 0);
	}
	return known;
}

function coalesceAdjacentRows(rows: ContextInventoryRow[]): void {
	let length = 0;
	const equal = (a: readonly string[] | undefined, b: readonly string[] | undefined) => a === b || (!!a && !!b && a.length === b.length && a.every(value => b.includes(value)));
	for (const row of rows) {
		const previous = rows[length - 1];
		const mergeable = row.kind === "text" || row.kind === "tool" || row.kind === "retained" || row.kind === "post-compaction";
		if (mergeable && previous && previous.kind === row.kind && previous.label === row.label && previous.coverage === row.coverage && previous.note === row.note && previous.quantity.tokens !== null && row.quantity.tokens !== null && previous.quantity.basis === row.quantity.basis && previous.payloadSize?.unit === row.payloadSize?.unit && equal(previous.selectionReasons, row.selectionReasons) && equal(previous.contributions, row.contributions) && equal(previous.controls, row.controls)) {
			const delimiter = previous.location.indexOf(" .. ");
			previous.location = (delimiter < 0 ? previous.location : previous.location.slice(0, delimiter)) + " .. " + row.location;
			previous.quantity.tokens += row.quantity.tokens;
			previous.counts.messages += row.counts.messages;
			previous.counts.blocks += row.counts.blocks;
			if (row.counts.items) previous.counts.items = (previous.counts.items ?? 0) + row.counts.items;
			if (previous.payloadSize && row.payloadSize) previous.payloadSize.value += row.payloadSize.value;
			if (row.sourceIds) previous.sourceIds = [...new Set([...(previous.sourceIds ?? []), ...row.sourceIds])];
		} else rows[length++] = row;
	}
	rows.length = length;
}

/** Inventories one actual reconstruction. No rendering, provider calls, hooks, or live settings reads. */
export function buildCompactionDiagnostics(input: CompactionDiagnosticsInput): CompactionDiagnostics {
	const rows: ContextInventoryRow[] = [];
	const framePrice = FRAME_TOKEN_ESTIMATE;
	const prepared = input.preparedContext;
	const prompt = prepared?.systemPrompt ?? [];
	const fixedCosts = prepared ? {
		systemPromptTokens: input.tokenizer.countTokens(prompt[0] ?? ""),
		toolsTokens: estimateToolSchemaTokens(prepared.tools ?? [], input.tokenizer),
		systemContextTokens: input.tokenizer.countTokens(prompt.slice(1)), skillsTokens: 0,
	} : input.fixedCosts;
	const fixedCounts = prepared ? { systemPrompt: Math.min(1, prompt.length), tools: prepared.tools?.length ?? 0, context: Math.max(0, prompt.length - 1), skills: 0 } : input.fixedCounts;
	const locations = new Map<string, SourceLayoutPart>();
	const identityLocations = new WeakMap<object, SourceLayoutPart>();
	for (const location of input.sourceLocations ?? []) {
		const part = input.sourceRepresentation?.layout[location.layoutIndex];
		if (!part) continue;
		locations.set(locationKey(location.messageIndex, location.blockIndex), part);
		if (!prepared) continue;
		const source = input.messages[location.messageIndex];
		if (!source) continue;
		const blocks = source.role === "compactionSummary" ? source.blocks : "content" in source && Array.isArray(source.content) ? source.content : undefined;
		const value = location.blockIndex === undefined ? source : blocks?.[location.blockIndex];
		if (value) identityLocations.set(value, part);
	}
	const pushFixed = (kind: "system-prompt" | "tools" | "context" | "skills", tokens: number, label: string, items: number | undefined) => {
		rows.push({
			location: `fixed:${kind}`, kind, label, coverage: "fixed", selectionReasons: [],
			quantity: quantity(tokens, "Existing non-message tokenizer estimate"),
			counts: { messages: 0, blocks: 0, frames: 0, images: 0, ...(items === undefined ? {} : { items }) }, controls: ["model", kind],
			...(items === undefined ? { note: "Structural item count was not captured." } : {}),
		});
	};
	pushFixed("system-prompt", fixedCosts.systemPromptTokens, "System prompt", fixedCounts?.systemPrompt);
	pushFixed("tools", fixedCosts.toolsTokens, "Tool descriptors", fixedCounts?.tools);
	pushFixed("context", fixedCosts.systemContextTokens, "System context", fixedCounts?.context);
	if (!prepared) pushFixed("skills", fixedCosts.skillsTokens, "Skills", fixedCounts?.skills);
	let convertedEstimate = 0;
	let imageAdjustment = 0;
	let excludedOpaqueEstimate = 0;
	let nativePlaceholderEstimate = 0;
	let nativeKnownEstimate = 0;
	let hasUnknown = false;

	const inventoryMessages: readonly AgentMessage[] = prepared?.messages ?? input.messages;
	for (let messageIndex = 0; messageIndex < inventoryMessages.length; messageIndex++) {
		const original = inventoryMessages[messageIndex]!;
		const sourceId = prepared ? undefined : input.messageSourceIds?.[messageIndex];
		const next = inventoryMessages[messageIndex + 1];
		// A prepared Context is already transformed; never convert or run hooks again.
		const converted = prepared ? [prepared.messages[messageIndex]!] : convertToLlm(original.role === "assistant" && next ? [original, next] : [original]);
		const fragmentCount = !prepared && original.role === "assistant" && next ? converted.length - convertToLlm([next]).length : converted.length;
		const fragments = fragmentCount === converted.length ? converted : converted.slice(0, fragmentCount);
		for (let fragmentIndex = 0; fragmentIndex < fragments.length; fragmentIndex++) {
			const message = fragments[fragmentIndex]!;
			convertedEstimate += input.tokenizer.countMessage(message);
			const messageLocation = `${locationKey(messageIndex)}/fragment:${fragmentIndex}`;
			const rowStart = rows.length;
			const emittedBlock = (index: number | undefined): object | undefined => index !== undefined && Array.isArray(message.content) ? message.content[index] : undefined;
			const locatedPart = (index: number | undefined): SourceLayoutPart | undefined => {
				if (prepared) {
					const block = emittedBlock(index);
					const archiveFrame = block ? getArchiveFrameAccounting(block) : undefined;
					if (archiveFrame && archiveFrame.compactionEntryId === input.sourceRepresentationEntryId) {
						const archivePart = input.sourceRepresentation?.layout[archiveFrame.layoutIndex];
						if (archivePart?.kind === "frame") return archivePart;
					}
					return (block ? identityLocations.get(block) : undefined) ?? identityLocations.get(message);
				}
				const archiveIndex = original.role === "compactionSummary" && index !== undefined ? index - 1 : index;
				return locations.get(locationKey(messageIndex, archiveIndex)) ?? locations.get(locationKey(messageIndex));
			};
			const add = (
				kind: ContextInventoryRow["kind"], label: string, tokens: number | null,
				blockIndex: number | undefined, bytes: number | undefined, image = false,
				note?: string,
			) => {
				const part = locatedPart(blockIndex);
				const block = emittedBlock(blockIndex);
				const physical = prepared && block ? getInlinePhysical(block) : undefined;
				const origin = getSourceOrigin(block ?? message);
				const coverage = physical ? { coverage: physical.owner === "tool" ? "aggregate" as const : "fixed" as const, contributions: ["ordinary" as const] } : kind === "native" || kind === "summary" ? { coverage: "aggregate" as const } : part?.kind === "source" && !part.contribution && origin?.kind === "source" ? preparedOriginCoverage(origin, input, part) : part ? sourceCoverage(part, input.sourceRepresentation) : prepared ? preparedOriginCoverage(origin, input) : sourceId
					? { coverage: "source" as const, sourceIds: [sourceId] }
					: { coverage: "unknown" as const };
				const reasons = new Set<string>();
				for (const id of coverage.sourceIds ?? []) for (const reason of input.selectionReasons?.get(id) ?? []) reasons.add(reason);
				const actualKind = physical?.kind === "note" ? physical.owner === "system" ? "system-prompt" : physical.owner === "context" ? "context" : "tool" : part?.kind === "gap" ? "gap" : kind === "text" && sourceId && !part ? input.postCompactionSourceIds?.has(sourceId) ? "post-compaction" : "retained" : kind;
				rows.push({
					location: `${messageLocation}/${blockIndex === undefined ? "aggregate" : `block:${blockIndex}`}${note === "Opaque replay content" ? "/opaque" : ""}`,
					kind: actualKind, label, ...coverage, selectionReasons: [...reasons],
					quantity: quantity(tokens, tokens === null ? "Provider replay cost is not locally measurable" : physical?.kind === "frame" ? "Live inline renderer image estimate, not a provider invoice" : image ? "Generic local image/frame estimate, not a provider invoice" : "Emitted content under the local tokenizer estimator", image ? "image-estimate" : "local-estimate"),
					counts: { messages: rows.length === rowStart ? 1 : 0, blocks: 1, frames: kind === "frame" ? 1 : 0, images: image ? 1 : 0 },
					...(bytes === undefined ? {} : { payloadSize: { value: bytes, unit: image ? "base64-characters" as const : "utf8-bytes" as const } }),
					controls: physical ? [physical.owner === "tool" ? "snapcompact.toolResults" : "snapcompact.systemPrompt", "model"] : image ? kind === "frame" ? ["snapcompact.shape", "archive frame window", "archive payload budget", "model"] : ["images", "model"] : ["compaction", "model"],
					...(note ? { note } : {}),
				});
				hasUnknown ||= tokens === null;
			};
			const payload = "providerPayload" in message ? message.providerPayload : undefined;
			if (payload?.type === "openaiResponsesHistory" && (payload.provider === undefined || payload.provider === input.model.provider)) {
				nativePlaceholderEstimate += input.tokenizer.countMessage(message);
				const nativeStart = rows.length;
				nativeKnownEstimate += appendNativeRows(rows, payload.items, input.tokenizer, messageLocation);
				for (let index = nativeStart; index < rows.length; index++) hasUnknown ||= rows[index]!.quantity.tokens === null;
				continue;
			}
			const content = typeof message.content === "string" ? [{ type: "text" as const, text: message.content }] : message.content;
			for (let blockIndex = 0; blockIndex < content.length; blockIndex++) {
				const block = content[blockIndex]!;
				const single = { ...message, content: [block] } as Message;
				const fullTokens = input.tokenizer.countMessage(single);
				const visibleTokens = input.tokenizer.countMessage(single, { excludeEncryptedReasoning: true });
				if (block.type === "image") {
					const part = locatedPart(blockIndex);
					const physical = prepared ? getInlinePhysical(block) : undefined;
					const archiveFrame = getArchiveFrameAccounting(block);
					const frame = physical?.kind === "frame" || archiveFrame !== undefined || part?.kind === "frame" || (original.role === "compactionSummary" && part?.kind !== "original-image");
					const tokens = physical?.kind === "frame" ? physical.estimatedTokens : frame ? framePrice : IMAGE_TOKEN_ESTIMATE;
					imageAdjustment += tokens - fullTokens;
					add(frame ? "frame" : "original-image", physical?.kind === "frame" ? `Inline ${physical.owner} frame` : frame ? "Rasterized archive frame" : "Original or unclassified image", tokens, blockIndex, block.data.length, true, archiveFrame ? "Archive identity is recorded; historical renderer-specific price is not. Charge is the generic local archive estimate." : undefined);
					continue;
				}
				const opaque = block.type === "redactedThinking" || block.type === "anthropicServerTool" || (block.type === "thinking" && !!block.thinkingSignature);
				if (block.type !== "redactedThinking" && block.type !== "anthropicServerTool") {
					const text = block.type === "text" ? block.text : block.type === "thinking" ? block.thinking : block.type === "toolCall" ? block.name + JSON.stringify(block.arguments) : "";
					const kind = original.role === "compactionSummary" && blockIndex === 0 || original.role === "branchSummary" ? "summary" : block.type === "toolCall" || message.role === "toolResult" ? "tool" : "text";
					add(kind, kind === "summary" ? "Compaction summary and wrapper" : `${message.role} ${block.type}`, visibleTokens, blockIndex, Buffer.byteLength(text, "utf8"));
				}
				if (opaque) {
					excludedOpaqueEstimate += fullTokens - visibleTokens;
					const raw = block.type === "redactedThinking" ? block.data : block.type === "thinking" ? block.thinkingSignature! : JSON.stringify(block.block);
					add("native", `${message.role} opaque replay`, null, blockIndex, Buffer.byteLength(raw, "utf8"), false, "Opaque replay content");
					if (block.type === "thinking") rows[rows.length - 1]!.counts.blocks = 0;
				}
			}
		}
	}
	coalesceAdjacentRows(rows);
	let total = 0;
	let ordinary = 0;
	let addedUser = 0;
	let shared = 0;
	let unallocated = false;
	const selectedSources = new Set<string>();
	const manualSources = new Set<string>();
	for (const run of input.sourceRepresentation?.coverage ?? []) {
		if (run.contribution === "selected-user") selectedSources.add(run.entryId);
		if (run.contribution === "manual-nonuser") manualSources.add(run.entryId);
	}
	for (const row of rows) {
		const tokens = row.quantity.tokens;
		if (tokens === null) continue;
		total += tokens;
		if (row.contributions !== undefined) {
			if (row.contributions.length === 0) { shared += tokens; unallocated = true; }
			else if (row.contributions.includes("selected-user")) {
				if (row.contributions.length === 1) addedUser += tokens;
				else shared += tokens;
			} else ordinary += tokens;
			continue;
		}
		const ids = row.sourceIds;
		if (!ids?.length) {
			if (row.coverage === "fixed" || row.kind === "summary") ordinary += tokens;
			else { shared += tokens; unallocated = true; }
			continue;
		}
		let additions = 0;
		for (const id of ids) if (input.addedUserSourceIds?.has(id)) additions++;
		if (additions === ids.length) addedUser += tokens;
		else if (additions > 0) shared += tokens;
		else ordinary += tokens;
	}
	const fixedTotal = fixedCosts.systemPromptTokens + fixedCosts.toolsTokens + fixedCosts.systemContextTokens + fixedCosts.skillsTokens;
	const reconciled = fixedTotal + convertedEstimate + imageAdjustment - excludedOpaqueEstimate - nativePlaceholderEstimate + nativeKnownEstimate;
	const notes = [
		prepared ? "Actual supplied OMP-prehook Context: transformed prompt, descriptors and message blocks only. Inline frame prices come from live renderer facts; no hooks or rendering rerun. Skills remain included in transformed prompt text without separate attribution." : "Local emitted-content estimate after message conversion; includes summary wrappers, excludes unmeasurable opaque replay. Not a provider invoice or a post-hook observation.",
		`Tokenizer encoding: ${input.tokenizer.encoding ?? "unresolved; tokenizer fallback applies"}. Image baseline: ${IMAGE_TOKEN_ESTIMATE}; archive frame estimate: ${framePrice}.`,
		`Reconciliation: fixed ${fixedTotal} + converted-message estimate ${convertedEstimate} + image replacement delta ${imageAdjustment} - opaque estimate ${excludedOpaqueEstimate} - native fallback estimate ${nativePlaceholderEstimate} + native visible estimate ${nativeKnownEstimate} = known local subtotal ${reconciled}.`,
		"Selection quotas overlap; ordinary, added-user, and shared/unallocated rows partition physical known charges, not quota membership.",
	];
	if (input.frameTokenEstimate !== undefined) notes.push(`Separate shape-family estimate: ${input.frameTokenEstimate} tokens/frame; not substituted for the ${FRAME_TOKEN_ESTIMATE} generic archive charge in this local domain.`);
	if (!input.selectedUserSourceIds) notes.push("Selected-user source count contains known contributed additions only; overlapping ordinary policy membership may be unavailable.");
	if (unallocated) notes.push("Shared/unallocated includes emitted content without exact source attribution; no exact per-selection allocation is claimed.");
	if (reconciled !== total) notes.push(`Row subtotal ${total} differs from aggregate estimator ${reconciled}; tokenizer fragmentation is not an exact provider serialization count.`);
	return {
		snapshot: input.snapshot ?? (prepared ? "OMP-prehook" : "recorded-at-compaction"), model: `${input.model.provider}/${input.model.id}`, method: input.method,
		contextWindow: typeof input.model.contextWindow === "number" && Number.isFinite(input.model.contextWindow) && input.model.contextWindow > 0 ? input.model.contextWindow : null,
		settings: structuredClone(input.settings), target: { ...input.target }, before: { ...input.before },
		total: quantity(total, hasUnknown ? "Known local subtotal; opaque/native replay cost remains unknown" : "Sum of disjoint emitted-content local estimates"), rows,
		distribution: {
			ordinary: quantity(ordinary, "Ordinary known physical charges, including fixed context"),
			addedUser: quantity(addedUser, "Known physical charges with exclusively added-user source coverage"),
			shared: quantity(shared, "Known mixed-source or unallocated physical charges; no counterfactual group price"),
			manualNonUserSources: input.manualNonUserSourceIds?.size ?? manualSources.size,
			selectedUserSources: input.selectedUserSourceIds?.size ?? selectedSources.size,
			addedUserSources: input.addedUserSourceIds?.size ?? selectedSources.size,
		},
		...(hasUnknown ? { warning: "Opaque/native replay has unknown local token cost; the known subtotal is not the complete request size." } : {}), notes,
	};
}