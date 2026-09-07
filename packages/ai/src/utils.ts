import { $env, isRecord, parseImageMetadata } from "@oh-my-pi/pi-utils";
import type { ResponseInput, ResponseInputItem } from "./providers/openai-responses-wire";
import { redactSensitiveCredentials } from "./providers/transform-messages";
import type { AssistantMessage, CacheRetention, ImageContent, OpenAIResponsesHistoryPayload, ProviderPayload } from "./types";
import { exportItemOrigins, getSourceOrigin, importItemOrigins, type NativeItemOrigin, transferSourceOrigin } from "./utils/source-origin";

type OpenAIResponsesReplayItem = ResponseInput[number];
const NON_WHITESPACE_RE = /\S/;

export { isRecord } from "@oh-my-pi/pi-utils";
/**
 * Read a header value ignoring key casing. HTTP header names are
 * case-insensitive, but `Record<string, string>` header bags are not, so a
 * config-authored `User-Agent` and a caller-authored `user-agent` are the same
 * header to every provider that lowercases before merging.
 */
export function getHeaderCaseInsensitive(
	headers: Record<string, string> | undefined,
	headerName: string,
): string | undefined {
	if (!headers) return undefined;
	const normalizedName = headerName.toLowerCase();
	for (const [key, value] of Object.entries(headers)) {
		if (key.toLowerCase() === normalizedName) return value;
	}
	return undefined;
}

export function normalizeSystemPrompts(systemPrompt: readonly string[] | string | undefined | null): string[] {
	if (systemPrompt === undefined || systemPrompt === null) return [];
	const prompts = Array.isArray(systemPrompt) ? systemPrompt : typeof systemPrompt === "string" ? [systemPrompt] : [];
	return prompts
		.map(prompt => redactSensitiveCredentials(prompt.toWellFormed()))
		.filter(prompt => prompt.trim().length > 0);
}

export function normalizeToolCallId(id: string): string {
	const sanitized = id.replace(/[^a-zA-Z0-9_-]/g, "_");
	return sanitized.length > 64 ? sanitized.slice(0, 64) : sanitized;
}

type ResponsesToolItemIdPrefix = "fc" | "ctc";

/** Preserve opaque call IDs for Responses replay while normalizing or generating the separate item ID. */
export function normalizeResponsesToolCallId(
	id: string,
	itemPrefix: ResponsesToolItemIdPrefix = "fc",
): { callId: string; itemId: string } {
	const [callId, itemId] = id.split("|");
	if (callId && itemId) {
		return { callId, itemId: normalizeResponsesItemId(itemId, itemPrefix) };
	}
	const hash = Bun.hash(id).toString(36);
	return { callId: id, itemId: `${itemPrefix}_${hash}` };
}

function getExplicitIdPrefix(id: string): string | undefined {
	return id.match(/^([a-zA-Z][a-zA-Z0-9]*)_/)?.[1];
}

function normalizeResponsesItemId(itemId: string, fallbackPrefix: ResponsesToolItemIdPrefix): string {
	const prefix = getExplicitIdPrefix(itemId);
	const isAllowedPrefix = prefix
		? fallbackPrefix === "ctc"
			? prefix === "ctc"
			: prefix === "fc" || prefix === "fcr"
		: false;
	if (!prefix || !isAllowedPrefix) {
		return `${fallbackPrefix}_${Bun.hash(itemId).toString(36)}`;
	}
	return truncateResponseItemId(itemId, prefix);
}

/**
 * Truncate an OpenAI Responses API item ID to 64 characters.
 * IDs exceeding the limit are replaced with a hash-based ID using the given prefix.
 */
export function truncateResponseItemId(id: string, prefix: string): string {
	if (id.length <= 64) return id;
	return `${prefix}_${Bun.hash(id).toString(36)}`;
}

interface OpenAIResponsesReplaySanitizeOptions {
	supportsImageDetailOriginal?: boolean;
	supportsComputerUse?: boolean;
}
/**
 * Removes response-only lifecycle status from item types that reject it when replayed as input.
 *
 * Returns the original array when no item needs sanitization.
 */
export function stripOpenAIResponsesOutputOnlyStatusesForReplay<TItem extends { type?: unknown; status?: unknown }>(
	items: TItem[],
): TItem[] {
	let sanitized: TItem[] | undefined;
	for (let index = 0; index < items.length; index++) {
		const item = items[index]!;
		const rejectsOutputStatus =
			item.type === "message" ||
			item.type === "function_call" ||
			item.type === "custom_tool_call" ||
			item.type === "compaction" ||
			item.type === "compaction_summary";
		if (!rejectsOutputStatus || !Object.hasOwn(item, "status")) {
			sanitized?.push(item);
			continue;
		}
		if (!sanitized) sanitized = items.slice(0, index);
		const withoutStatus = { ...item };
		delete withoutStatus.status;
		sanitized.push(transferSourceOrigin(item, withoutStatus));
	}
	return sanitized ?? items;
}

/**
 * Clamp `detail: "original"` only where Responses input_image parts live —
 * top-level items and `message.content[]`. Avoids a deep tree walk/clone of
 * every history node on providers that reject native-resolution images.
 */
function clampReplayItemImageDetail(
	item: Record<string, unknown>,
	supportsImageDetailOriginal: boolean,
): Record<string, unknown> {
	if (supportsImageDetailOriginal) return item;

	if (item.type === "input_image" && item.detail === "original") {
		return transferSourceOrigin(item, { ...item, detail: "auto" });
	}

	if (item.type !== "message" || !Array.isArray(item.content)) return item;

	let changed = false;
	const content = item.content.map(part => {
		if (!part || typeof part !== "object" || Array.isArray(part)) return part;
		const record = part as Record<string, unknown>;
		if (record.type !== "input_image" || record.detail !== "original") return part;
		changed = true;
		return transferSourceOrigin(record, { ...record, detail: "auto" });
	});
	return changed ? transferSourceOrigin(item, { ...item, content }) : item;
}

function isOpenAIResponsesClientInputBoundary(item: Record<string, unknown>): boolean {
	if (item.type === "message") return item.role !== "assistant";
	if (item.type === undefined && typeof item.role === "string") return item.role !== "assistant";

	switch (item.type) {
		case "input_text":
		case "input_image":
		case "input_file":
		case "input_audio":
		case "function_call_output":
		case "custom_tool_call_output":
		case "computer_call_output":
		case "local_shell_call_output":
		case "shell_call_output":
		case "apply_patch_call_output":
		case "mcp_approval_response":
		case "compaction":
		case "compaction_summary":
		case "compaction_trigger":
		case "item_reference":
			return true;
		case "additional_tools":
			return item.role !== "assistant";
		case "tool_search_output":
			return item.execution !== "server";
		default:
			return false;
	}
}

function collectOpenAIResponsesComputerLinkedReasoningItems(
	items: Array<Record<string, unknown>>,
	requireLaterOutput: boolean,
): Set<Record<string, unknown>> {
	let computerCallsWithLaterOutputs: Set<Record<string, unknown>> | undefined;
	if (requireLaterOutput) {
		computerCallsWithLaterOutputs = new Set();
		const laterComputerOutputCallIds = new Set<string>();
		for (let index = items.length - 1; index >= 0; index--) {
			const item = items[index]!;
			if (item.type === "computer_call_output" && typeof item.call_id === "string") {
				laterComputerOutputCallIds.add(item.call_id);
			} else if (
				item.type === "computer_call" &&
				typeof item.id === "string" &&
				typeof item.call_id === "string" &&
				laterComputerOutputCallIds.has(item.call_id)
			) {
				computerCallsWithLaterOutputs.add(item);
			}
		}
	}

	const computerLinkedReasoningItems = new Set<Record<string, unknown>>();
	const responseReasoningItems: Array<Record<string, unknown>> = [];
	for (const item of items) {
		if (isOpenAIResponsesClientInputBoundary(item)) {
			responseReasoningItems.length = 0;
		} else if (item.type === "reasoning") {
			responseReasoningItems.push(item);
		} else if (
			item.type === "computer_call" &&
			typeof item.id === "string" &&
			(!computerCallsWithLaterOutputs || computerCallsWithLaterOutputs.has(item))
		) {
			for (const reasoningItem of responseReasoningItems) computerLinkedReasoningItems.add(reasoningItem);
		}
	}
	return computerLinkedReasoningItems;
}

const provisionalOpenAIResponsesComputerReasoningItems = new WeakSet<object>();

export function sanitizeOpenAIResponsesHistoryItemsForReplay(
	items: Array<Record<string, unknown>>,
	options: OpenAIResponsesReplaySanitizeOptions = {},
): ResponseInput {
	const supportsImageDetailOriginal = options.supportsImageDetailOriginal !== false;
	const computerLinkedReasoningItems =
		options.supportsComputerUse === false
			? undefined
			: collectOpenAIResponsesComputerLinkedReasoningItems(items, false);
	const sanitized = items.flatMap(item => {
		const preserveForComputer = computerLinkedReasoningItems?.has(item) === true;
		const sanitizedItem = sanitizeOpenAIResponsesHistoryItemForReplay(
			item,
			supportsImageDetailOriginal,
			preserveForComputer,
		);
		if (preserveForComputer && sanitizedItem?.type === "reasoning") {
			provisionalOpenAIResponsesComputerReasoningItems.add(sanitizedItem);
		}
		return sanitizedItem ? [sanitizedItem] : [];
	});
	return stripOpenAIResponsesOutputOnlyStatusesForReplay(sanitized);
}

function collectOpenAIResponsesReasoningItemsWithSurvivingOutputIds(
	items: Array<Record<string, unknown>>,
): Set<Record<string, unknown>> {
	const retainedReasoningItems = new Set<Record<string, unknown>>();
	let responseReasoningItems: Array<Record<string, unknown>> = [];
	let hasSurvivingOutputId = false;
	const finishResponse = (): void => {
		if (hasSurvivingOutputId) {
			for (const reasoningItem of responseReasoningItems) retainedReasoningItems.add(reasoningItem);
		}
		responseReasoningItems = [];
		hasSurvivingOutputId = false;
	};

	for (const item of items) {
		if (isOpenAIResponsesClientInputBoundary(item)) {
			finishResponse();
		} else if (item.type === "reasoning") {
			responseReasoningItems.push(item);
		} else if (item.type !== "computer_call" && typeof item.id === "string") {
			hasSurvivingOutputId = true;
		}
	}
	finishResponse();
	return retainedReasoningItems;
}

/** Strip reasoning IDs whose only linked native output is a computer call that will be demoted. */
export function stripOpenAIResponsesComputerLinkedReasoningIdsForReplay(items: ResponseInput): ResponseInput {
	const records = items as unknown as Array<Record<string, unknown>>;
	const linkedReasoningItems = collectOpenAIResponsesComputerLinkedReasoningItems(records, false);
	const retainedReasoningItems = collectOpenAIResponsesReasoningItemsWithSurvivingOutputIds(records);
	let sanitized: ResponseInput | undefined;

	for (let index = 0; index < items.length; index++) {
		const item = items[index]!;
		const record = records[index]!;
		if (
			item.type !== "reasoning" ||
			typeof record.id !== "string" ||
			!linkedReasoningItems.has(record) ||
			retainedReasoningItems.has(record)
		) {
			sanitized?.push(item);
			continue;
		}
		if (!sanitized) sanitized = items.slice(0, index);
		const { id: _id, ...withoutId } = record;
		sanitized.push(transferSourceOrigin(item, withoutId) as unknown as ResponseInput[number]);
	}
	return sanitized ?? items;
}

/**
 * Finalize provisional native-computer reasoning IDs after the complete
 * Responses input has been rebuilt, model-adapted, and orphan-repaired.
 */
export function stripUnpairedOpenAIResponsesComputerReasoningIdsForReplay(items: ResponseInput): ResponseInput {
	const records = items as unknown as Array<Record<string, unknown>>;
	const linkedReasoningItems = collectOpenAIResponsesComputerLinkedReasoningItems(records, true);
	let sanitized: ResponseInput | undefined;

	for (let index = 0; index < items.length; index++) {
		const item = items[index]!;
		const record = records[index]!;
		if (
			item.type !== "reasoning" ||
			!provisionalOpenAIResponsesComputerReasoningItems.has(item) ||
			typeof record.id !== "string" ||
			linkedReasoningItems.has(record)
		) {
			sanitized?.push(item);
			continue;
		}
		if (!sanitized) sanitized = items.slice(0, index);
		const { id: _id, ...withoutId } = record;
		sanitized.push(transferSourceOrigin(item, withoutId) as unknown as ResponseInput[number]);
	}
	return sanitized ?? items;
}

/**
 * Sanitize assistant-native Responses history for replay.
 *
 * Returns `undefined` for hidden-empty turns that only contain reasoning and an
 * empty assistant message, allowing callers to rebuild visible transcript
 * history instead of replaying stale native state.
 */
export function sanitizeOpenAIResponsesAssistantHistoryItemsForReplay(
	items: Array<Record<string, unknown>>,
	options: OpenAIResponsesReplaySanitizeOptions = {},
): ResponseInput | undefined {
	const sanitized = sanitizeOpenAIResponsesHistoryItemsForReplay(items, options);
	let hasReplayableAssistantOutput = false;

	for (const item of sanitized) {
		if (item.type === "reasoning") continue;
		if (item.type !== "message" || item.role !== "assistant") {
			hasReplayableAssistantOutput = true;
			break;
		}
		if (typeof item.content === "string") {
			if (NON_WHITESPACE_RE.test(item.content)) {
				hasReplayableAssistantOutput = true;
				break;
			}
			continue;
		}
		for (const part of item.content) {
			if (part.type === "output_text" && NON_WHITESPACE_RE.test(part.text)) {
				hasReplayableAssistantOutput = true;
				break;
			}
			if (part.type === "refusal" && NON_WHITESPACE_RE.test(part.refusal)) {
				hasReplayableAssistantOutput = true;
				break;
			}
		}
		if (hasReplayableAssistantOutput) break;
	}

	return hasReplayableAssistantOutput ? sanitized : undefined;
}

/**
 * Drop hidden-only fallback assistant replay after a native Responses snapshot is rejected.
 */
export function sanitizeOpenAIResponsesAssistantFallbackItemsForReplay(items: ResponseInput): ResponseInput {
	const sanitized: ResponseInput = [];

	for (const item of items) {
		if (item.type === "reasoning") continue;
		if (item.type !== "message" || item.role !== "assistant") {
			sanitized.push(item);
			continue;
		}

		let hasVisibleText = false;
		if (typeof item.content === "string") {
			hasVisibleText = NON_WHITESPACE_RE.test(item.content);
		} else {
			for (const part of item.content) {
				if (part.type === "output_text" && NON_WHITESPACE_RE.test(part.text)) {
					hasVisibleText = true;
					break;
				}
				if (part.type === "refusal" && NON_WHITESPACE_RE.test(part.refusal)) {
					hasVisibleText = true;
					break;
				}
			}
		}

		if (hasVisibleText) sanitized.push(item);
	}

	return sanitized;
}

function sanitizeOpenAIResponsesHistoryItemForReplay(
	item: Record<string, unknown>,
	supportsImageDetailOriginal: boolean,
	preserveReasoningItemIds: boolean,
): OpenAIResponsesReplayItem | undefined {
	if (item.type === "function_call") {
		if (typeof item.arguments !== "string" || item.arguments.trim().length === 0) return undefined;
		try {
			JSON.parse(item.arguments);
		} catch {
			return undefined;
		}
	}
	if (item.type === "item_reference") return undefined;
	if (item.type === "image_generation_call") return sanitizeOpenAIResponsesImageGenerationCallForReplay(item);
	if (item.type === "reasoning") {
		return sanitizeOpenAIResponsesReasoningItemForReplay(item, preserveReasoningItemIds);
	}
	const { id: _id, ...sanitizedItem } = item;
	transferSourceOrigin(item, sanitizedItem);
	if (item.type === "computer_call" && typeof item.id === "string") sanitizedItem.id = item.id;

	return clampReplayItemImageDetail(
		sanitizedItem,
		supportsImageDetailOriginal,
	) as unknown as OpenAIResponsesReplayItem;
}

function sanitizeOpenAIResponsesReasoningItemForReplay(
	item: Record<string, unknown>,
	preserveItemId: boolean,
): OpenAIResponsesReplayItem {
	const sanitizedItem: Record<string, unknown> = { type: "reasoning" };
	if (preserveItemId && typeof item.id === "string") sanitizedItem.id = item.id;
	if (Array.isArray(item.summary)) sanitizedItem.summary = item.summary;
	if (Array.isArray(item.content)) sanitizedItem.content = item.content;
	if (typeof item.encrypted_content === "string" || item.encrypted_content === null) {
		sanitizedItem.encrypted_content = item.encrypted_content;
	}
	return transferSourceOrigin(item, sanitizedItem) as unknown as OpenAIResponsesReplayItem;
}

function sanitizeOpenAIResponsesImageGenerationCallForReplay(
	item: Record<string, unknown>,
): ResponseInputItem.ImageGenerationCall | undefined {
	if (typeof item.id !== "string" || typeof item.result !== "string" || item.result.length === 0) {
		return undefined;
	}
	return transferSourceOrigin<ResponseInputItem.ImageGenerationCall>(item, {
		id: truncateResponseItemId(item.id, "ig"),
		type: "image_generation_call",
		status: "completed",
		result: item.result,
	});
}

export type OpenAIResponsesLogicalImage =
	| { type: "base64"; data: string }
	| { type: "reference"; input: boolean; image_url?: string; file_id?: string; detail: unknown };

export interface OpenAIResponsesLogicalVisitor {
	text?(logical: Record<string, unknown>, origin: NativeItemOrigin): void;
	image?(image: OpenAIResponsesLogicalImage, origin: NativeItemOrigin): void;
	/** Original visible field fragments, before role/history rendering. */
	sourceText?(text: string): void;
}

/** MIME detection belongs to materialization, never source-image counting. */
export function materializeOpenAIResponsesImage(image: OpenAIResponsesLogicalImage): Record<string, unknown> {
	if (image.type === "reference") {
		const { type: _type, input: _input, ...reference } = image;
		return { type: "input_image", ...reference };
	}
	const content = openAIResponsesImageContent(image.data);
	return { type: "input_image", detail: "auto", image_url: `data:${content.mimeType};base64,${content.data}` };
}

export function openAIResponsesImageContent(data: string): ImageContent {
	return { type: "image", data, mimeType: parseImageMetadata(Buffer.from(data, "base64"))?.mimeType ?? "image/png" };
}
/** Shared normalized source projection; signatures and transport controls stay opaque. */
export function getAssistantLogicalBlock(block: AssistantMessage["content"][number]): Record<string, unknown> | undefined {
	switch (block.type) {
		case "text": return { type: "text", text: block.text };
		case "thinking": return { type: "thinking", thinking: block.thinking };
		case "toolCall": return {
			type: "toolCall", id: block.id, name: block.name, arguments: block.arguments,
			rawBlock: block.rawBlock, intent: block.intent, customWireName: block.customWireName,
			...(block.providerMetadata?.type === "computer" ? { actions: block.providerMetadata.actions, pendingSafetyChecks: block.providerMetadata.pendingSafetyChecks } : {}),
		};
		case "anthropicServerTool": return { type: block.type, block: block.block };
		case "image": case "redactedThinking": case "fallback": return undefined;
	}
}

/** Visit only typed logical Responses fields; opaque provider state is never traversed. */
export function visitOpenAIResponsesLogicalContent(
	value: object, visitor: OpenAIResponsesLogicalVisitor,
	inheritedOrigin?: NativeItemOrigin,
): void {
	const item = value as Record<string, unknown>;
	if (item.type === "compaction" || item.type === "compaction_summary") return;
	const origin = getSourceOrigin(item) ?? inheritedOrigin ?? { kind: "unknown", reason: "unmapped-native-history" };
	if (item.type === "image_generation_call" && typeof item.result === "string" && item.result.length > 0) {
		const imageOrigin: NativeItemOrigin = origin.kind === "source" ? { kind: "source", parts: origin.parts.slice(0, 1).map(part => ({
			...part, blockIndex: `${part.blockIndex}.result`, representation: "original-image", sourceSpan: undefined, transportSpan: undefined, transportBlockIndex: undefined,
		})) } : origin;
		visitor.text?.({ type: item.type }, origin);
		visitor.image?.({ type: "base64", data: item.result }, imageOrigin);
		return;
	}
	if (item.type === "input_image" || item.type === "computer_screenshot" || (item.type === "image" && typeof item.url === "string")) {
		const image: OpenAIResponsesLogicalImage = { type: "reference", input: item.type !== "image", detail: item.detail ?? "auto",
			...(typeof item.image_url === "string" ? { image_url: item.image_url } : item.type === "image" ? { image_url: String(item.url) } : {}),
			...(typeof item.file_id === "string" ? { file_id: item.file_id } : {}) };
		if (image.image_url || image.file_id) visitor.image?.(image, origin.kind === "source" ? { kind: "source", parts: origin.parts.map(part => ({
			...part, representation: "original-image", sourceSpan: undefined, transportSpan: undefined, transportBlockIndex: undefined,
		})) } : origin);
		return;
	}
	// Only typed logical fields cross into visible history. Provider signatures,
	// encrypted state, replay identifiers and unknown extensions stay opaque.
	let logical: Record<string, unknown>;
	let nativeContent: unknown;
	let output: unknown;
	let outputs: unknown;
	switch (item.type) {
		case "text": case "thinking": case "toolCall":
			logical = getAssistantLogicalBlock(item as unknown as AssistantMessage["content"][number])!; break;
		case "message":
			logical = { type: item.type, role: item.role };
			nativeContent = item.content;
			break;
		case "input_text": case "output_text": case "reasoning_text": case "summary_text":
			logical = { type: item.type, text: item.text, annotations: item.annotations };
			break;
		case "refusal": logical = { type: item.type, refusal: item.refusal }; break;
		case "reasoning": logical = { type: item.type, summary: item.summary }; nativeContent = item.content; break;
		case "function_call": logical = { type: item.type, call_id: item.call_id, name: item.name, arguments: item.arguments }; break;
		case "custom_tool_call": logical = { type: item.type, call_id: item.call_id, name: item.name, input: item.input }; break;
		case "function_call_output": case "custom_tool_call_output":
			logical = { type: item.type, call_id: item.call_id }; output = item.output; break;
		case "computer_call":
			logical = { type: item.type, call_id: item.call_id, action: item.action, actions: item.actions, pending_safety_checks: item.pending_safety_checks }; break;
		case "computer_call_output":
			logical = { type: item.type, call_id: item.call_id, acknowledged_safety_checks: item.acknowledged_safety_checks }; output = item.output; break;
		case "web_search_call": logical = { type: item.type, action: item.action }; break;
		case "file_search_call": logical = { type: item.type, queries: item.queries, results: item.results }; break;
		case "tool_search_call": logical = { type: item.type, call_id: item.call_id, execution: item.execution, arguments: item.arguments }; break;
		case "tool_search_output": logical = { type: item.type, call_id: item.call_id, execution: item.execution, tools: item.tools }; break;
		case "mcp_call": logical = { type: item.type, name: item.name, server_label: item.server_label, arguments: item.arguments, output: item.output, error: item.error }; break;
		case "mcp_list_tools": logical = { type: item.type, server_label: item.server_label, tools: item.tools, error: item.error }; break;
		case "code_interpreter_call": logical = { type: item.type, code: item.code }; outputs = item.outputs; break;
		case "logs": logical = { type: item.type, logs: item.logs }; break;
		default: return;
	}
	const outputIsContent = Array.isArray(output) || (isRecord(output) && output.type === "computer_screenshot");
	if (nativeContent !== undefined && !Array.isArray(nativeContent)) logical.content = nativeContent;
	if (output !== undefined && !outputIsContent) logical.output = output;
	const metadataOrigin = origin.kind === "source" && Array.isArray(nativeContent) ? { kind: "source" as const, parts: origin.parts.filter(part => part.transportBlockIndex === undefined) } : origin;
	if (visitor.sourceText) for (const [field, value] of Object.entries(logical)) {
		if (field === "type" || field === "role" || field === "id" || field === "call_id" || value === undefined || value === null) continue;
		if (field === "summary" && Array.isArray(value)) {
			for (const part of value) if (isRecord(part) && part.type === "summary_text" && typeof part.text === "string") visitor.sourceText(part.text);
		} else if (typeof value === "string") visitor.sourceText(value);
		else if (!Array.isArray(value) || value.length > 0) visitor.sourceText(JSON.stringify(value));
	}
	visitor.text?.(logical, metadataOrigin);
	for (const [field, value] of [["content", nativeContent], ["output", output], ["outputs", outputs]] as const) {
		const children = Array.isArray(value) ? value : field === "output" && outputIsContent ? [value] : [];
		for (let index = 0; index < children.length; index++) {
			const child: unknown = children[index];
			if (!isRecord(child)) continue;
			const childOrigin: NativeItemOrigin = origin.kind === "source" ? { kind: "source", parts: origin.parts.slice(0, 1).map(part => ({
				...part, blockIndex: `${part.blockIndex}.${field}${Array.isArray(value) ? `.${index}` : ""}`,
				sourceSpan: undefined, transportSpan: undefined, transportBlockIndex: undefined,
			})) } : origin;
			visitOpenAIResponsesLogicalContent(child, visitor, childOrigin);
		}
	}
}

/** Current delta components only. Ingress mirrors remain owned by current content, even after deletion. */
export function visitOpenAIResponsesSourceContent(message: AssistantMessage, visitor: OpenAIResponsesLogicalVisitor): void {
	const payload = message.providerPayload;
	if (payload?.type !== "openaiResponsesHistory" || payload.dt !== true) return;
	importItemOrigins(payload.items, payload.origins);
	for (let index = 0; index < payload.items.length; index++) {
		const item = payload.items[index]!;
		const origin = getSourceOrigin(item);
		if (origin?.kind === "source" && origin.parts.some(part => part.status === "historical-not-current")) continue;
		const mirrored = origin?.kind === "source"
			? origin.parts.some(part => typeof (part.currentBlockIndex ?? part.blockIndex) === "number")
			: payload.contentBlocks?.some(mapping => mapping.itemIndex === index) === true;
		if (!mirrored) visitOpenAIResponsesLogicalContent(item, visitor);
	}
}

export function createOpenAIResponsesHistoryPayload(
	provider: string,
	items: Array<Record<string, unknown>>,
	incremental = true,
	contentBlocks?: OpenAIResponsesHistoryPayload["contentBlocks"],
): OpenAIResponsesHistoryPayload {
	return {
		type: "openaiResponsesHistory",
		provider,
		...(incremental ? { dt: true } : {}),
		items,
		origins: exportItemOrigins(items),
		...(contentBlocks?.length ? { contentBlocks } : {}),
	};
}

export function getOpenAIResponsesHistoryPayload(
	providerPayload: ProviderPayload | undefined,
	currentProvider: string,
	fallbackProvider?: string,
): OpenAIResponsesHistoryPayload | undefined {
	if (providerPayload?.type !== "openaiResponsesHistory" || !Array.isArray(providerPayload.items)) {
		return undefined;
	}
	const payloadProvider = providerPayload.provider ?? fallbackProvider ?? currentProvider;
	if (payloadProvider !== currentProvider) return undefined;
	importItemOrigins(providerPayload.items, providerPayload.origins);
	return { ...providerPayload, provider: payloadProvider };
}

export function getOpenAIResponsesHistoryItems(
	providerPayload: ProviderPayload | undefined,
	currentProvider: string,
	fallbackProvider?: string,
): Array<Record<string, unknown>> | undefined {
	return getOpenAIResponsesHistoryPayload(providerPayload, currentProvider, fallbackProvider)?.items;
}

/**
 * Resolve cache retention preference: explicit request option first, then the
 * `PI_CACHE_RETENTION` env override (`long` | `short` | `none`), then the
 * provider-supplied fallback.
 */
export function resolveCacheRetention(
	cacheRetention?: CacheRetention,
	fallback: CacheRetention = "short",
): CacheRetention {
	if (cacheRetention) return cacheRetention;
	const env = $env.PI_CACHE_RETENTION;
	if (env === "long" || env === "short" || env === "none") return env;
	return fallback;
}
