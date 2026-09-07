import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, ComputerToolCallMetadata, ComputerToolResultMetadata, Context, ImageContent, Message, TextContent } from "@oh-my-pi/pi-ai";
import { combineContentSourceOrigins, exportItemOrigins, getSourceOrigin, importItemOrigins, setSourceOrigin, transferTransformedSourceOrigin, validateNativeItemOrigins } from "@oh-my-pi/pi-ai/utils/source-origin";
import { isRecord } from "@oh-my-pi/pi-utils";
import type { SessionContext } from "../session/session-context";
import type { JsonValue, SecretObfuscator } from "./obfuscator";
import { collectJsonRegexSecretValues, mapJsonStrings } from "./placeholder-scan";

// ═══════════════════════════════════════════════════════════════════════════
// Display restore (inbound, persisted/provider → local display)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Restore secret placeholders for local display. Only message kinds the model
 * itself authored from obfuscated context carry placeholders — assistant
 * content and the LLM-written branch/compaction summaries. User, developer, and
 * tool-result messages are persisted with their literal text, so operator-authored
 * placeholder-shaped text must survive untouched; those roles are never walked.
 */
export function deobfuscateSessionContext(
	sessionContext: SessionContext,
	obfuscator: SecretObfuscator | undefined,
): SessionContext {
	if (!obfuscator?.hasSecrets()) return sessionContext;
	const messages = deobfuscateAgentMessages(obfuscator, sessionContext.messages);
	return messages === sessionContext.messages ? sessionContext : { ...sessionContext, messages };
}

export function deobfuscateAgentMessages(obfuscator: SecretObfuscator, messages: AgentMessage[]): AgentMessage[] {
	const deob = (text: string): string => obfuscator.deobfuscate(text);
	let changed = false;
	const result = messages.map((message): AgentMessage => {
		switch (message.role) {
			case "assistant": {
				const content = deobfuscateAssistantContent(obfuscator, message.content);
				if (content === message.content) return message;
				changed = true;
				return setSourceOrigin({ ...message, content }, combineContentSourceOrigins(content));
			}
			case "branchSummary": {
				const summary = deob(message.summary);
				if (summary === message.summary) return message;
				changed = true;
				return transferTransformedSourceOrigin(message, { ...message, summary });
			}
			case "compactionSummary": {
				const summary = deob(message.summary);
				const shortSummary = message.shortSummary === undefined ? undefined : deob(message.shortSummary);
				const blocks = message.blocks === undefined ? undefined : deobfuscateTextBlocks(obfuscator, message.blocks);
				if (summary === message.summary && shortSummary === message.shortSummary && blocks === message.blocks) {
					return message;
				}
				changed = true;
				return transferTransformedSourceOrigin(message, { ...message, summary, shortSummary, blocks });
			}
			default:
				return message;
		}
	});
	return changed ? result : messages;
}

/**
 * Restore placeholders in assistant content: visible text and tool-call
 * arguments/intent/rawBlock. Thinking and signatures are opaque
 * provider-replay/hidden-reasoning data and pass through byte-identical.
 */
export function deobfuscateAssistantContent(
	obfuscator: SecretObfuscator,
	content: AssistantMessage["content"],
): AssistantMessage["content"] {
	if (!obfuscator.hasSecrets()) return content;
	const deob = (text: string): string => obfuscator.deobfuscate(text);
	let changed = false;
	const result = content.map((block): AssistantMessage["content"][number] => {
		if (block.type === "text") {
			const text = deob(block.text);
			if (text === block.text) return block;
			changed = true;
			return transferTransformedSourceOrigin(block, { ...block, text });
		}

		if (block.type === "toolCall") {
			const args = deobfuscateToolArguments(obfuscator, block.arguments);
			const intent = block.intent === undefined ? undefined : deob(block.intent);
			const rawBlock = block.rawBlock === undefined ? undefined : deob(block.rawBlock);
			if (args === block.arguments && intent === block.intent && rawBlock === block.rawBlock) return block;
			changed = true;
			return transferTransformedSourceOrigin(block, { ...block, arguments: args, intent, rawBlock });
		}
		return block;
	});
	return changed ? result : content;
}

/**
 * Restore placeholders inside a tool call's arguments. Arguments are arbitrary
 * model-authored JSON, so tool-call arguments are the ONLY place a recursive
 * JSON walk runs.
 */
export function deobfuscateToolArguments(
	obfuscator: SecretObfuscator,
	args: Record<string, unknown>,
): Record<string, unknown> {
	if (!obfuscator.hasSecrets()) return args;
	return mapJsonStrings(args as JsonValue, s => obfuscator.deobfuscate(s)) as Record<string, unknown>;
}

/** Redact secrets inside a tool call's arguments (same JSON-walk exception as {@link deobfuscateToolArguments}). */
export function obfuscateToolArguments(
	obfuscator: SecretObfuscator,
	args: Record<string, unknown>,
	sharedRegexSecretValues?: ReadonlySet<string>,
): Record<string, unknown> {
	if (!obfuscator.hasSecrets()) return args;
	const regexSecretValues = sharedRegexSecretValues ?? collectJsonRegexSecretValues(obfuscator, args as JsonValue);
	return mapJsonStrings(args as JsonValue, s => obfuscator.obfuscate(s, regexSecretValues)) as Record<string, unknown>;
}

// ═══════════════════════════════════════════════════════════════════════════
// Outbound obfuscation (local → provider)
// ═══════════════════════════════════════════════════════════════════════════

type UserFacingMessage = Extract<Message, { role: "user" | "developer" | "toolResult" }>;

/** Obfuscate `text` blocks of a content array; image and other blocks pass through. */
function obfuscateTextBlocks(
	obfuscator: SecretObfuscator,
	content: (TextContent | ImageContent)[],
	sharedRegexSecretValues?: ReadonlySet<string>,
): (TextContent | ImageContent)[] {
	let changed = false;
	const result = content.map((block): TextContent | ImageContent => {
		if (block.type !== "text") return block;
		const text = obfuscator.obfuscate(block.text, sharedRegexSecretValues);
		if (text === block.text) return block;
		changed = true;
		return transferTransformedSourceOrigin(block, { ...block, text });
	});
	return changed ? result : content;
}

/** Restore placeholders in `text` blocks of a content array; image and other blocks pass through. */
function deobfuscateTextBlocks(
	obfuscator: SecretObfuscator,
	content: (TextContent | ImageContent)[],
): (TextContent | ImageContent)[] {
	let changed = false;
	const result = content.map((block): TextContent | ImageContent => {
		if (block.type !== "text") return block;
		const text = obfuscator.deobfuscate(block.text);
		if (text === block.text) return block;
		changed = true;
		return transferTransformedSourceOrigin(block, { ...block, text });
	});
	return changed ? result : content;
}

/**
 * Re-obfuscate assistant content before it returns to a provider after session
 * restoration, removing friendly prefixes made unsafe by this batch. A changed
 * thinking block loses its byte-bound replay signature.
 */
function obfuscateAssistantContentForReplay(
	obfuscator: SecretObfuscator,
	content: AssistantMessage["content"],
	sharedRegexSecretValues: ReadonlySet<string>,
): AssistantMessage["content"] {
	const obfuscate = (text: string): string =>
		obfuscator.stripUnsafeFriendlyPlaceholderPrefixes(
			obfuscator.obfuscate(text, sharedRegexSecretValues),
			sharedRegexSecretValues,
		);
	let changed = false;
	const result = content.map((block): AssistantMessage["content"][number] => {
		if (block.type === "text") {
			const text = obfuscate(block.text);
			if (text === block.text) return block;
			changed = true;
			return transferTransformedSourceOrigin(block, { ...block, text });
		}
		if (block.type === "thinking") {
			const thinking = obfuscate(block.thinking);
			if (thinking === block.thinking) return block;
			changed = true;
			return transferTransformedSourceOrigin(block, { ...block, thinking, thinkingSignature: undefined });
		}
		if (block.type === "toolCall") {
			const args = mapJsonStrings(block.arguments as JsonValue, obfuscate) as Record<string, unknown>;
			const intent = block.intent === undefined ? undefined : obfuscate(block.intent);
			const rawBlock = block.rawBlock === undefined ? undefined : obfuscate(block.rawBlock);
			if (args === block.arguments && intent === block.intent && rawBlock === block.rawBlock) return block;
			changed = true;
			return transferTransformedSourceOrigin(block, { ...block, arguments: args, intent, rawBlock });
		}
		return block;
	});
	return changed ? result : content;
}

/** Copy only arrays whose explicitly selected visible text changes. */
function mapChanged<T>(items: T[], map: (item: T) => T): T[] {
	let result: T[] | undefined;
	for (let index = 0; index < items.length; index++) {
		const item = items[index]!;
		const next = map(item);
		if (next !== item && !result) result = items.slice(0, index);
		result?.push(next);
	}
	return result ?? items;
}

function mapComputerMetadata<T extends ComputerToolCallMetadata | ComputerToolResultMetadata>(metadata: T, map: (text: string) => string): T {
	if ("actions" in metadata) {
		const actions = mapChanged(metadata.actions, action => {
			if (action.type !== "type") return action;
			const text = map(action.text);
			return text === action.text ? action : transferTransformedSourceOrigin(action, { ...action, text });
		});
		const pendingSafetyChecks = mapChanged(metadata.pendingSafetyChecks, check => {
			if (typeof check.message !== "string") return check;
			const message = map(check.message);
			return message === check.message ? check : transferTransformedSourceOrigin(check, { ...check, message });
		});
		return actions === metadata.actions && pendingSafetyChecks === metadata.pendingSafetyChecks
			? metadata : { ...metadata, actions, pendingSafetyChecks };
	}
	const acknowledgedSafetyChecks = mapChanged(metadata.acknowledgedSafetyChecks, check => {
		if (typeof check.message !== "string") return check;
		const message = map(check.message);
		return message === check.message ? check : transferTransformedSourceOrigin(check, { ...check, message });
	});
	return acknowledgedSafetyChecks === metadata.acknowledgedSafetyChecks ? metadata : { ...metadata, acknowledgedSafetyChecks };
}

function mapAssistantMetadata(block: AssistantMessage["content"][number], map: (text: string) => string): AssistantMessage["content"][number] {
	if (block.type === "toolCall" && block.providerMetadata?.type === "computer") {
		const providerMetadata = mapComputerMetadata(block.providerMetadata, map);
		return providerMetadata === block.providerMetadata ? block : transferTransformedSourceOrigin(block, { ...block, providerMetadata });
	}
	if (block.type !== "anthropicServerTool") return block;
	const server = block.block;
	if (server.type === "server_tool_use") {
		// Server-tool input is model-authored arguments, not arbitrary result metadata.
		const input = mapJsonStrings(server.input as JsonValue, map) as typeof server.input;
		return input === server.input ? block : transferTransformedSourceOrigin(block, { ...block, block: { ...server, input } });
	}
	if (server.type !== "web_search_tool_result" || !Array.isArray(server.content)) return block;
	const content = mapChanged(server.content, result => {
		if (!isRecord(result) || result.type !== "web_search_result") return result;
		const title = typeof result.title === "string" ? map(result.title) : result.title;
		const url = typeof result.url === "string" ? map(result.url) : result.url;
		return title === result.title && url === result.url ? result : { ...result, title, url };
	});
	return content === server.content ? block : transferTransformedSourceOrigin(block, { ...block, block: { ...server, content } });
}

function mapNativeSearchItem(item: Record<string, unknown>, map: (text: string) => string): Record<string, unknown> {
	if (item.type !== "web_search_call" || !isRecord(item.action)) return item;
	const action = item.action;
	if (action.type === "search") {
		const query = typeof action.query === "string" ? map(action.query) : action.query;
		const queries = Array.isArray(action.queries) ? mapChanged(action.queries, query => typeof query === "string" ? map(query) : query) : action.queries;
		const sources = Array.isArray(action.sources) ? mapChanged(action.sources, source => {
			if (!isRecord(source) || source.type !== "url" || typeof source.url !== "string") return source;
			const url = map(source.url);
			return url === source.url ? source : { ...source, url };
		}) : action.sources;
		return query === action.query && queries === action.queries && sources === action.sources
			? item : { ...item, action: { ...action, ...(query !== action.query ? { query } : {}), ...(queries !== action.queries ? { queries } : {}), ...(sources !== action.sources ? { sources } : {}) } };
	}
	if (action.type === "open_page" || action.type === "find_in_page") {
		const url = typeof action.url === "string" ? map(action.url) : action.url;
		const pattern = action.type === "find_in_page" && typeof action.pattern === "string" ? map(action.pattern) : action.pattern;
		return url === action.url && pattern === action.pattern ? item
			: { ...item, action: { ...action, ...(url !== action.url ? { url } : {}), ...(pattern !== action.pattern ? { pattern } : {}) } };
	}
	return item;
}

/** Only provider-visible typed fields; identifiers, opaque replay state and image bytes stay untouched. */
function mapVisibleMetadata(message: Message, map: (text: string) => string): Message {
	if (message.role === "assistant") {
		const content = mapChanged(message.content, block => mapAssistantMetadata(block, map));
		let providerPayload = message.providerPayload;
		if (providerPayload?.type === "openaiResponsesHistory") {
			const items = mapChanged(providerPayload.items, item => mapNativeSearchItem(item, map));
			if (items !== providerPayload.items) {
				importItemOrigins(providerPayload.items, validateNativeItemOrigins(providerPayload.origins));
				for (let index = 0; index < items.length; index++) {
					if (items[index] !== providerPayload.items[index]) transferTransformedSourceOrigin(providerPayload.items[index]!, items[index]!);
				}
				providerPayload = { ...providerPayload, items, origins: exportItemOrigins(items) };
			}
		}
		return content === message.content && providerPayload === message.providerPayload ? message
			: transferTransformedSourceOrigin(message, { ...message, content, providerPayload });
	}
	if (message.role !== "toolResult") return message;
	const providerMetadata = message.providerMetadata?.type === "computer" ? mapComputerMetadata(message.providerMetadata, map) : message.providerMetadata;
	let details = message.details;
	// The historical tool-result projection emits this visible explanation; other unknown details are not traversed.
	if (isRecord(details) && typeof details.explanation === "string") {
		const explanation = map(details.explanation);
		if (explanation !== details.explanation) details = { ...details, explanation };
	}
	if (providerMetadata === message.providerMetadata && details === message.details) return message;
	const result = transferTransformedSourceOrigin(message, { ...message, providerMetadata, details });
	const original = getSourceOrigin(message);
	const transformed = getSourceOrigin(result);
	if (original?.kind === "source" && transformed?.kind === "source" && original.parts.some(part => part.representation === "original-image")) {
		setSourceOrigin(result, { kind: "source", parts: transformed.parts.map((part, index) => original.parts[index]!.representation === "original-image" ? original.parts[index]! : part) });
	}
	return result;
}

function collectMessageRegexSecretValues(obfuscator: SecretObfuscator, messages: Message[]): Set<string> {
	const values = new Set<string>();
	const addText = (text: string | undefined): void => {
		if (text === undefined) return;
		for (const value of obfuscator.collectRegexSecretValuesForObfuscation(text)) {
			values.add(value);
		}
	};
	for (const message of messages) {
		mapVisibleMetadata(message, text => { addText(text); return text; });
		if (message.role === "assistant") {
			for (const block of message.content) {
				if (block.type === "text") addText(block.text);
				else if (block.type === "thinking") addText(block.thinking);
				else if (block.type === "toolCall") {
					for (const value of collectJsonRegexSecretValues(obfuscator, block.arguments as JsonValue)) {
						values.add(value);
					}
					addText(block.intent);
					addText(block.rawBlock);
				}
			}
			continue;
		}
		if (
			message.role !== "user" &&
			message.role !== "toolResult" &&
			!(message.role === "developer" && message.attribution === "user")
		) {
			continue;
		}
		const target = message as UserFacingMessage;
		if (typeof target.content === "string") {
			addText(target.content);
			continue;
		}
		for (const block of target.content) {
			if (block.type === "text") addText(block.text);
		}
	}
	return values;
}

/**
 * Redact secrets from outbound messages. User messages, tool results, and
 * user-authored developer messages (e.g. `@file` mentions) are obfuscated.
 * Assistant replay content is re-obfuscated too, because session restoration
 * expands keyed placeholders locally before the next provider request. Inline
 * image bytes are never walked.
 */
export function obfuscateMessages(obfuscator: SecretObfuscator, messages: Message[]): Message[] {
	if (!obfuscator.hasSecrets()) return messages;
	const sharedRegexSecretValues = collectMessageRegexSecretValues(obfuscator, messages);
	const obfuscateMetadata = (text: string): string => obfuscator.stripUnsafeFriendlyPlaceholderPrefixes(obfuscator.obfuscate(text, sharedRegexSecretValues), sharedRegexSecretValues);
	let changed = false;
	const result = messages.map((message): Message => {
		const metadata = mapVisibleMetadata(message, obfuscateMetadata);
		if (metadata !== message) changed = true;
		message = metadata;
		if (
			message.role !== "user" &&
			message.role !== "toolResult" &&
			!(message.role === "developer" && message.attribution === "user")
		) {
			if (message.role !== "assistant") return message;
			const content = obfuscateAssistantContentForReplay(obfuscator, message.content, sharedRegexSecretValues);
			if (content === message.content) return message;
			changed = true;
			return setSourceOrigin({ ...message, content }, combineContentSourceOrigins(content));
		}
		const target = message as UserFacingMessage;
		if (typeof target.content === "string") {
			const content = obfuscator.obfuscate(target.content, sharedRegexSecretValues);
			if (content === target.content) return message;
			changed = true;
			return transferTransformedSourceOrigin(message, { ...target, content } as Message);
		}
		const content = obfuscateTextBlocks(obfuscator, target.content, sharedRegexSecretValues);
		if (content === target.content) return message;
		changed = true;
		return setSourceOrigin({ ...target, content } as Message, combineContentSourceOrigins(content));
	});
	return changed ? result : messages;
}

/**
 * Redact outbound provider context. Only conversation messages are rewritten;
 * the static system prompt and tool schemas pass through unchanged.
 */
export function obfuscateProviderContext(obfuscator: SecretObfuscator | undefined, context: Context): Context {
	if (!obfuscator?.hasSecrets()) return context;
	const messages = obfuscateMessages(obfuscator, context.messages);
	return messages === context.messages ? context : { ...context, messages };
}
