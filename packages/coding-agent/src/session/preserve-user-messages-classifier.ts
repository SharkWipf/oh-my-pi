import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import { Tokenizer } from "@oh-my-pi/pi-agent-core/tokenizer";
import type { AssistantMessage, Context, ImageContent, Message, Model, TextContent } from "@oh-my-pi/pi-ai";
import { providerImageBudget } from "@oh-my-pi/snapcompact";
import classifierPrompt from "../prompts/system/preserve-user-messages-classifier.md" with { type: "text" };
import type { SessionEntry } from "./session-entries";

type UserContent = (TextContent | ImageContent)[];
type AssistantContent = (TextContent | { type: "toolCall"; name: string; arguments: Record<string, unknown> })[];

export interface PreservedUserMessageClassifierInput {
	currentMessage: UserContent;
	previousUserMessage: UserContent | null;
	previousAssistantMessages: AssistantContent[];
	/** Source identity, not provider input; retained alongside the immutable content snapshot. */
	sourceIds: { current: string; previousUser: string | null; previousAssistants: string[] };
}

type MessageProjection = { role: "user"; content: UserContent } | { role: "assistant"; content: AssistantContent };

/** Only original real-user content and visible assistant text/tool calls are dependencies. */
export function projectPreservedUserMessageClassifierMessage(message: AgentMessage): MessageProjection | null {
	return projectClassifierMessage(message, true);
}

/** Validation borrows arguments only for its synchronous comparison; snapshots own independent arguments. */
function projectClassifierMessage(message: AgentMessage, cloneArguments: boolean): MessageProjection | null {
	if (message.role === "user" && message.synthetic !== true && message.attribution !== "agent") {
		const content = typeof message.content === "string" ? [{ type: "text" as const, text: message.content }] : message.content;
		return {
			role: "user",
			content: content.map(part => {
				if (part.type === "text") return { type: "text", text: part.text };
				return {
					type: "image", data: part.data, mimeType: part.mimeType,
					...(part.detail !== undefined ? { detail: part.detail } : {}),
				};
			}),
		};
	}
	if (message.role !== "assistant") return null;
	const content: AssistantContent = [];
	for (const part of message.content) {
		if (part.type === "text" && part.text.length > 0) content.push({ type: "text", text: part.text });
		else if (part.type === "toolCall") {
			content.push({ type: "toolCall", name: part.name, arguments: cloneArguments ? structuredClone(part.arguments) : part.arguments });
		}
	}
	return content.length > 0 ? { role: "assistant", content } : null;
}

function inputSnapshot(
	current: { id: string; content: UserContent },
	previousUser: { id: string; content: UserContent } | null,
	assistants: { id: string; content: AssistantContent }[],
): PreservedUserMessageClassifierInput {
	return {
		currentMessage: current.content,
		previousUserMessage: previousUser?.content ?? null,
		previousAssistantMessages: assistants.map(item => item.content),
		sourceIds: { current: current.id, previousUser: previousUser?.id ?? null, previousAssistants: assistants.map(item => item.id) },
	};
}

class ClassifierInputCursor {
	#previousUser: { id: string; content: UserContent } | null = null;
	#assistants: { id: string; content: AssistantContent }[] = [];

	push(entry: SessionEntry): { entryId: string; input: PreservedUserMessageClassifierInput } | undefined {
		if (entry.type === "reset_boundary") {
			this.#previousUser = null;
			this.#assistants = [];
			return undefined;
		}
		if (entry.type !== "message") return undefined;
		const projection = projectPreservedUserMessageClassifierMessage(entry.message);
		if (!projection) return undefined;
		if (projection.role === "user") {
			const current = { id: entry.id, content: projection.content };
			const input = inputSnapshot(current, this.#previousUser, this.#assistants);
			this.#previousUser = current;
			return { entryId: entry.id, input };
		}
		if (this.#assistants.length === 2) this.#assistants.shift();
		this.#assistants.push({ id: entry.id, content: projection.content });
		return undefined;
	}
}

/** Streams a single ancestor-ordered branch; clear boundaries reset all relative context. */
export function* iteratePreservedUserMessageClassifierInputs(
	entries: Iterable<SessionEntry>,
): Generator<{ entryId: string; input: PreservedUserMessageClassifierInput }> {
	const cursor = new ClassifierInputCursor();
	for (const entry of entries) {
		const target = cursor.push(entry);
		if (target) yield target;
	}
}

/** Cooperates even across long stretches with no eligible user; cancellation is checked after every await. */
export async function* iteratePreservedUserMessageClassifierInputsCooperatively(
	entries: Iterable<SessionEntry>,
	isCurrent: () => boolean,
): AsyncGenerator<{ entryId: string; input: PreservedUserMessageClassifierInput }> {
	const cursor = new ClassifierInputCursor();
	let deadline = performance.now() + 4;
	for (const entry of entries) {
		if (!isCurrent()) return;
		const target = cursor.push(entry);
		if (target) yield target;
		if (performance.now() >= deadline) {
			await new Promise<void>(resolve => setImmediate(resolve));
			if (!isCurrent()) return;
			deadline = performance.now() + 4;
		}
	}
}

export function buildPreservedUserMessageClassifierInput(
	entries: readonly SessionEntry[],
	targetId: string,
): PreservedUserMessageClassifierInput | undefined {
	for (const item of iteratePreservedUserMessageClassifierInputs(entries)) {
		if (item.entryId === targetId) return item.input;
	}
	return undefined;
}

/** Cold synchronous fallback: absent neighbors can require the full prefix. Use cooperative iteration for large scans. */
export function buildPreservedUserMessageClassifierInputFromLookup(
	targetId: string,
	getEntry: (id: string) => SessionEntry | undefined,
): PreservedUserMessageClassifierInput | undefined {
	const target = getEntry(targetId);
	if (target?.type !== "message") return undefined;
	const current = projectPreservedUserMessageClassifierMessage(target.message);
	if (current?.role !== "user") return undefined;
	let previousUser: { id: string; content: UserContent } | null = null;
	const assistants: { id: string; content: AssistantContent }[] = [];
	let parentId = target.parentId;
	while (parentId !== null && (previousUser === null || assistants.length < 2)) {
		const entry = getEntry(parentId);
		if (!entry || entry.type === "reset_boundary") break;
		parentId = entry.parentId;
		if (entry.type !== "message") continue;
		// Project only eligible neighbors still needed by this window.
		if (entry.message.role === "user" && previousUser !== null) continue;
		if (entry.message.role === "assistant" && assistants.length === 2) continue;
		const projection = projectPreservedUserMessageClassifierMessage(entry.message);
		if (projection?.role === "user") previousUser = { id: entry.id, content: projection.content };
		else if (projection?.role === "assistant") assistants.unshift({ id: entry.id, content: projection.content });
	}
	return inputSnapshot({ id: target.id, content: current.content }, previousUser, assistants);
}

export function preservedUserMessageClassifierInputsEqual(
	left: PreservedUserMessageClassifierInput | undefined,
	right: PreservedUserMessageClassifierInput | undefined,
): boolean {
	return Bun.deepEquals(left, right);
}

function matchesClassifierSource(
	sourceId: string,
	role: MessageProjection["role"],
	content: UserContent | AssistantContent,
	getEntry: (id: string) => SessionEntry | undefined,
): boolean {
	const entry = getEntry(sourceId);
	if (entry?.type !== "message" || entry.id !== sourceId) return false;
	const projection = projectClassifierMessage(entry.message, false);
	return projection?.role === role && Bun.deepEquals(content, projection.content);
}

/**
 * At most four direct source lookups. Caller MUST independently establish active ancestry
 * and unchanged neighbor selection through the source-mutation invalidation owner.
 * This compares named source bodies only; it cannot discover newly eligible intervening messages.
 */
export function matchesPreservedUserMessageClassifierSources(
	input: PreservedUserMessageClassifierInput,
	getEntry: (id: string) => SessionEntry | undefined,
): boolean {
	if (!matchesClassifierSource(input.sourceIds.current, "user", input.currentMessage, getEntry)) return false;
	if (input.sourceIds.previousUser !== null &&
		!matchesClassifierSource(input.sourceIds.previousUser, "user", input.previousUserMessage!, getEntry)) return false;
	for (let i = 0; i < input.sourceIds.previousAssistants.length; i++) {
		if (!matchesClassifierSource(input.sourceIds.previousAssistants[i]!, "assistant", input.previousAssistantMessages[i]!, getEntry)) return false;
	}
	return true;
}

export interface PreservedUserMessageClassifierRequestOptions {
	maxTokens?: number;
	/** Existing side-request conversion/secret obfuscation; must preserve original images. */
	convertToLlm?: (messages: AgentMessage[]) => Message[];
}

function outputAllowance(model: Model, requested?: number): number {
	// Existing online side-classifier allowance accommodates providers retaining reasoning.
	return Math.min(requested ?? 4096, model.maxTokens ?? 4096);
}

/** Complete CURRENT is mandatory; only auxiliary messages may be omitted to fit. */
export function buildPreservedUserMessageClassifierRequest(
	input: PreservedUserMessageClassifierInput,
	model: Model,
	options: PreservedUserMessageClassifierRequestOptions = {},
): Context {
	const currentImages = input.currentMessage.filter((part): part is ImageContent => part.type === "image");
	const modelName = `${model.provider}/${model.id}`;
	if (currentImages.length > 0 && !model.input.includes("image")) {
		throw new Error(`User-message classifier: ${modelName} does not support the current message's original images. Configure a vision model.`);
	}
	const imageLimit = providerImageBudget(model.provider);
	if (currentImages.length > imageLimit) {
		throw new Error(`User-message classifier: current message has ${currentImages.length} original images; ${modelName} allows ${imageLimit}. Configure a supported model.`);
	}
	if (!model.contextWindow || model.contextWindow <= 0) {
		throw new Error(`User-message classifier: ${modelName} has no known input allowance. Configure a supported model.`);
	}
	const allowance = model.contextWindow - outputAllowance(model, options.maxTokens);
	const tokenizer = new Tokenizer(model);
	const build = (previous: UserContent | null, assistants: AssistantContent[], omitted: boolean): Context => {
		const auxiliaryImages: ImageContent[] = [];
		const auxiliary = {
			previousUserMessage: previous?.map(part => {
				if (part.type === "text") return part;
				auxiliaryImages.push(part);
				return { type: "image", index: auxiliaryImages.length };
			}) ?? null,
			previousAssistantMessages: assistants,
			omitted,
		};
		const content: UserContent = [
			{ type: "text", text: `AUXILIARY CONTEXT (JSON; previous-user images follow):\n${JSON.stringify(auxiliary)}` },
			...auxiliaryImages,
			{ type: "text", text: "CURRENT MESSAGE (all following blocks):" },
			...input.currentMessage,
		];
		let messages: Message[] = [{ role: "user", content, timestamp: 0 }];
		if (options.convertToLlm) {
			const expectedImages = [...auxiliaryImages, ...currentImages];
			messages = options.convertToLlm([{ role: "user", content: content.map(part => ({ ...part })), timestamp: 0 }]);
			const actualImages: ImageContent[] = [];
			for (const message of messages) {
				if (typeof message.content === "string") continue;
				for (const part of message.content) if (part.type === "image") actualImages.push(part);
			}
			if (actualImages.length !== expectedImages.length || actualImages.some((part, i) => part.data !== expectedImages[i]!.data || part.mimeType !== expectedImages[i]!.mimeType)) {
				throw new Error("User-message classifier: side-request conversion changed original images; classification was not sent.");
			}
		}
		return { systemPrompt: [classifierPrompt], messages };
	};
	const fits = (context: Context): boolean => {
		const text = [...(context.systemPrompt ?? [])];
		let imageTokens = 0;
		let images = 0;
		for (const message of context.messages) {
			if (typeof message.content === "string") { text.push(message.content); continue; }
			for (const part of message.content) {
				if (part.type === "text") text.push(part.text);
				else if (part.type === "image") {
					images++;
					imageTokens += tokenizer.countMessage({ role: "user", content: [part], timestamp: 0 });
				}
			}
		}
		return images <= imageLimit && tokenizer.checkTokenBudget(text, allowance - imageTokens).fits;
	};
	const mandatory = build(null, [], true);
	if (!fits(mandatory)) {
		throw new Error(`User-message classifier: complete current message exceeds ${modelName}'s input allowance. Configure a larger supported model; current content was not truncated.`);
	}
	let previous = input.previousUserMessage;
	let omitted = false;
	if (previous?.some(part => part.type === "image") && !model.input.includes("image")) {
		// Retain the actual chronological anchor rather than substituting an older user.
		previous = previous.filter(part => part.type === "text");
		omitted = true;
	}
	const assistants = input.previousAssistantMessages.slice();
	for (;;) {
		const request = build(previous, assistants, omitted);
		if (fits(request)) return request;
		omitted = true;
		if (assistants.length > 0) assistants.shift();
		else return mandatory;
	}
}

export interface ClassifyPreservedUserMessageOptions extends PreservedUserMessageClassifierRequestOptions {
	model: Model;
	/** Caller owns role/credentials, existing side transport, provider identity, cancellation and usage. */
	complete: (context: Context, maxTokens: number) => Promise<AssistantMessage>;
}

export async function classifyPreservedUserMessage(
	input: PreservedUserMessageClassifierInput,
	options: ClassifyPreservedUserMessageOptions,
): Promise<number> {
	const request = buildPreservedUserMessageClassifierRequest(input, options.model, options);
	const response = await options.complete(request, outputAllowance(options.model, options.maxTokens));
	if (response.stopReason !== "stop") {
		throw new Error(`User-message classifier failed (${response.stopReason}): ${response.errorMessage ?? "incomplete classification"}`);
	}
	const text = response.content.filter(part => part.type === "text").map(part => part.text).join("");
	const mask = parsePreservedUserMessageCategoryMask(text);
	if (mask === undefined) throw new Error("User-message classifier returned invalid labels; expected <labels> followed by exactly eleven binary digits and </labels>.");
	return mask;
}

export function parsePreservedUserMessageCategoryMask(text: string): number | undefined {
	const match = /^<labels>([01]{11})<\/labels>$/.exec(text.trim());
	if (!match) return undefined;
	let mask = 0;
	for (let i = 0; i < 11; i++) if (match[1]![i] === "1") mask |= 1 << i;
	return mask;
}
