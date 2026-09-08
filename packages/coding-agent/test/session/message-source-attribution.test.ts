import { describe, expect, it } from "bun:test";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, Message, TextContent, ToolResultMessage, UserMessage } from "@oh-my-pi/pi-ai";
import {
	bindMessageSource,
	getSourceOrigin,
	invalidateSourceOrigins,
	type NativeSourcePart,
	setSourceOrigin,
} from "@oh-my-pi/pi-ai/utils/source-origin";
import {
	convertToLlm,
	type CustomMessage,
	INTERRUPTED_THINKING_MESSAGE_TYPE,
	replaceLlmImagesWithText,
	SKILL_PROMPT_MESSAGE_TYPE,
	wrapSteeringForModel,
} from "../../src/session/messages";

function parts(value: object): NativeSourcePart[] {
	const origin = getSourceOrigin(value);
	if (origin?.kind !== "source") throw new Error(`Expected source, got ${JSON.stringify(origin)}`);
	return origin.parts;
}

function custom(content: CustomMessage["content"], customType = "note"): CustomMessage {
	return { role: "custom", content, customType, display: false, attribution: "user", timestamp: 1 };
}

function blocks(message: Message): Exclude<Message["content"], string> {
	if (!Array.isArray(message.content)) throw new Error("Expected array content");
	return message.content;
}

describe("source-aware session conversion", () => {
	it("keeps duplicate occurrences distinct through cached conversion and append growth", () => {
		const first = custom("duplicate");
		const second = custom("duplicate", SKILL_PROMPT_MESSAGE_TYPE);
		bindMessageSource(first, "first", 0);
		bindMessageSource(second, "second", 1);
		const messages: AgentMessage[] = [first, second];
		const converted = convertToLlm(messages);
		expect(convertToLlm(messages)).toBe(converted);
		const copiedInput = convertToLlm(messages.slice());
		expect(copiedInput[0]).toBe(converted[0]);
		expect(copiedInput[1]).toBe(converted[1]);
		expect(converted.map(message => parts(message).map(part => part.entryId))).toEqual([["first"], ["second"]]);
		expect(converted.map(message => parts(blocks(message)[0]!).map(part => part.entryId))).toEqual([
			["first"], ["second"],
		]);
		expect(JSON.parse(JSON.stringify(converted))).toEqual([
			{ role: "developer", content: [{ type: "text", text: "duplicate" }], attribution: "user", timestamp: 1 },
			{ role: "user", content: [{ type: "text", text: "duplicate" }], attribution: "user", timestamp: 1 },
		]);
		messages.push(first);
		expect(convertToLlm(messages).map(message => parts(message).map(part => part.entryId))).toEqual([
			["first"], ["second"], ["first"],
		]);
		expect(converted).toHaveLength(2);
	});

	it("refreshes populated caches on binding, rebinding, and explicit hook invalidation", () => {
		const message = custom("duplicate");
		const messages: AgentMessage[] = [message];
		convertToLlm(messages);
		bindMessageSource(message, "first", 0);
		expect(parts(convertToLlm(messages)[0]!).map(part => part.entryId)).toEqual(["first"]);
		bindMessageSource(message, "second", 1);
		const rebound = convertToLlm(messages);
		expect(parts(rebound[0]!).map(part => part.entryId)).toEqual(["second"]);
		expect(convertToLlm(messages)).toBe(rebound);
		message.content = "hook replacement";
		invalidateSourceOrigins(messages);
		const invalidated = convertToLlm(messages)[0]!;
		expect(blocks(invalidated)).toEqual([{ type: "text", text: "hook replacement" }]);
		expect(getSourceOrigin(invalidated)).toMatchObject({ kind: "unknown" });
	});

	it("does not infer source identity for an unbound hook copy with identical text", () => {
		const original = custom("duplicate");
		bindMessageSource(original, "original", 0);
		convertToLlm([original]);
		const converted = convertToLlm([{ ...original }])[0]!;
		expect(getSourceOrigin(converted)).toMatchObject({ kind: "unknown" });
		expect(getSourceOrigin(blocks(converted)[0]!)).toBeUndefined();
	});

	it("splits custom text and images by source block without crediting the role label", () => {
		const message = custom([
			{ type: "text", text: "same" },
			{ type: "image", data: "AA==", mimeType: "image/png" },
			{ type: "text", text: "same" },
		]);
		bindMessageSource(message, "mixed", 3);
		const converted = convertToLlm([message]);
		expect(converted.map(message => parts(message).map(part => [part.entryId, part.blockIndex, part.transportBlockIndex]))).toEqual([
			[["mixed", 0, 0], ["mixed", 2, 1]],
			[["mixed", 1, 1]],
		]);
		expect(getSourceOrigin(blocks(converted[1]!)[0]!)).toMatchObject({ kind: "synthetic" });
		expect(parts(blocks(converted[1]!)[1]!)[0]?.representation).toBe("original-image");
	});

	it("carries steering text without inventing source offsets for the envelope", () => {
		const message: UserMessage = {
			role: "user", steering: true, timestamp: 1,
			content: [
				{ type: "text", text: "same" },
				{ type: "image", data: "AA==", mimeType: "image/png" },
				{ type: "text", text: "same" },
			],
		};
		bindMessageSource(message, "steer", 4);
		const converted = convertToLlm(wrapSteeringForModel([message]));
		const textOrigin = parts(blocks(converted[0]!)[0]!);
		expect(textOrigin.map(part => part.blockIndex)).toEqual([0, 2]);
		for (const part of textOrigin) {
			expect(part.coverage).toBe("full");
			expect(part.representation).toBe("transformed-text");
			expect(part.sourceSpan).toBeUndefined();
		}
		expect(parts(blocks(converted[0]!)[1]!)[0]).toMatchObject({ blockIndex: 1, representation: "original-image" });
	});

	it("removes image credit without mutating the original cached request", () => {
		const message: UserMessage = {
			role: "user", timestamp: 1,
			content: [
				{ type: "image", data: "AA==", mimeType: "image/png" },
				{ type: "image", data: "AQ==", mimeType: "image/png" },
				{ type: "text", text: "same" },
			],
		};
		bindMessageSource(message, "images", 5);
		const messages: AgentMessage[] = [message];
		const converted = convertToLlm(messages);
		const scrubbed = replaceLlmImagesWithText(converted, "[image omitted]");
		expect(blocks(scrubbed[0]!)).toEqual([{ type: "text", text: "[image omitted]" }, { type: "text", text: "same" }]);
		expect(getSourceOrigin(blocks(scrubbed[0]!)[0]!)).toMatchObject({ kind: "synthetic" });
		expect(parts(scrubbed[0]!).map(part => part.blockIndex)).toEqual([2]);
		expect(parts(converted[0]!).map(part => part.blockIndex)).toEqual([0, 1, 2]);
		expect(convertToLlm(messages)).toBe(converted);
		expect(replaceLlmImagesWithText(scrubbed, "[image omitted]")).toBe(scrubbed);
	});

	it("drops stripped thinking credit and marks merged pruned text as partial", () => {
		const assistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "answer" }, { type: "thinking", thinking: "unfinished" }],
			api: "anthropic-messages", provider: "anthropic", model: "fixture", stopReason: "aborted", timestamp: 1,
			usage: {
				input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
		};
		bindMessageSource(assistant, "assistant", 6);
		const converted = convertToLlm([assistant, custom("continuity", INTERRUPTED_THINKING_MESSAGE_TYPE)]);
		expect(parts(converted[0]!).map(part => part.blockIndex)).toEqual([0]);
		const tool: ToolResultMessage = {
			role: "toolResult", toolCallId: "call", toolName: "read", isError: false, timestamp: 2, prunedAt: 3,
			content: [
				{ type: "text", text: "a" },
				{ type: "image", data: "AA==", mimeType: "image/png" },
				{ type: "text", text: "b" },
			],
		};
		bindMessageSource(tool, "tool", 7);
		const result = convertToLlm([tool])[0]!;
		expect((blocks(result)[0] as TextContent).text).toBe("ab");
		expect(parts(blocks(result)[0]!).map(part => [part.blockIndex, part.coverage, part.sourceSpan])).toEqual([
			[0, "partial", undefined], [2, "partial", undefined],
		]);
		expect(parts(blocks(result)[1]!)[0]).toMatchObject({ blockIndex: 1, representation: "original-image" });
	});

	it("keeps opaque summaries aggregate and archive lead-ins synthetic", () => {
		const summary: AgentMessage = { role: "branchSummary", summary: "summary", fromId: "old", timestamp: 1 };
		setSourceOrigin(summary, { kind: "aggregate", compactionEntryId: "compaction" });
		expect(getSourceOrigin(convertToLlm([summary])[0]!)).toEqual({ kind: "aggregate", compactionEntryId: "compaction" });
		const block: TextContent = { type: "text", text: "retained" };
		bindMessageSource({ role: "user", content: [block], timestamp: 1 }, "retained", 8);
		const archive = convertToLlm([
			{ role: "compactionSummary", summary: "archive", blocks: [block], tokensBefore: 10, timestamp: 1 },
		])[0]!;
		expect(getSourceOrigin(blocks(archive)[0]!)).toMatchObject({ kind: "synthetic" });
		expect(parts(archive).map(part => part.entryId)).toEqual(["retained"]);
	});
});
