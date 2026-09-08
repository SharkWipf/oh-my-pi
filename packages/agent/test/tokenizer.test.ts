import { afterEach, describe, expect, test, vi } from "bun:test";
import type { AssistantMessage, ImageContent, TextContent, ToolResultMessage } from "@oh-my-pi/pi-ai";
import { createOpenAIResponsesHistoryPayload } from "@oh-my-pi/pi-ai/utils";
import { bindMessageSource, remapNativeItemOrigins } from "@oh-my-pi/pi-ai/utils/source-origin";
import * as natives from "@oh-my-pi/pi-natives";
import { Tokenizer, tokenizerEncodingForModel } from "../src/tokenizer";
import type { AgentMessage } from "../src/types";

afterEach(() => {
	vi.restoreAllMocks();
});

// Contract: the catalog resolves model identity once as Model.tokenizer; the
// agent maps that catalog property to the matching native counter. A wrong
// row silently skews every context-budget and compaction decision.
describe("tokenizerEncodingForModel", () => {
	test("maps every catalog tokenizer family to its native counter", () => {
		expect(tokenizerEncodingForModel({ tokenizer: "claude-v3" })).toBe(natives.Encoding.ClaudeV3);
		expect(tokenizerEncodingForModel({ tokenizer: "claude-v47" })).toBe(natives.Encoding.ClaudeV47);
		expect(tokenizerEncodingForModel({ tokenizer: "claude-v5" })).toBe(natives.Encoding.ClaudeV5);
		expect(tokenizerEncodingForModel({ tokenizer: "claude-v5-sonnet" })).toBe(natives.Encoding.ClaudeV5Sonnet);
		expect(tokenizerEncodingForModel({ tokenizer: "qwen3" })).toBe(natives.Encoding.Qwen3);
		expect(tokenizerEncodingForModel({ tokenizer: "deepseek-v3" })).toBe(natives.Encoding.DeepSeekV3);
		expect(tokenizerEncodingForModel({ tokenizer: "kimi-k2" })).toBe(natives.Encoding.KimiK2);
		expect(tokenizerEncodingForModel({ tokenizer: "glm5" })).toBe(natives.Encoding.Glm5);
	});

	test("leaves unknown catalog models on the estimate policy", () => {
		expect(tokenizerEncodingForModel({})).toBeNull();
		expect(tokenizerEncodingForModel(undefined)).toBeNull();
	});
});

describe("Tokenizer", () => {
	test("counts each original image once across user, developer, tool and hook content", () => {
		const tokenizer = new Tokenizer();
		const content: (TextContent | ImageContent)[] = [
			{ type: "text", text: "Inspect these images" },
			{ type: "image", data: "cG5n", mimeType: "image/png", detail: "low" },
			{ type: "image", data: "cG5n", mimeType: "image/png", detail: "high" },
			{ type: "image", data: "cG5n", mimeType: "image/png", detail: "auto" },
		];
		const messages: AgentMessage[] = [
			{ role: "user", content, timestamp: 0 },
			{ role: "developer", content, timestamp: 0 },
			{ role: "toolResult", toolCallId: "read_1", toolName: "read", content, isError: false, timestamp: 0 },
			{ role: "hookMessage", customType: "images", content, display: false, timestamp: 0 },
		];
		const expected = tokenizer.countTokens("Inspect these images") + 3 * 1200;
		for (const message of messages) {
			expect(tokenizer.countMessage(message)).toBe(expected);
			expect(tokenizer.countMessage(message, { excludeEncryptedReasoning: true })).toBe(expected);
		}
		expect(tokenizer.countMessages(messages)).toBe(4 * expected);
	});

	test("charges a metadata-only computer screenshot without charging content mirrors again", () => {
		const tokenizer = new Tokenizer();
		const screenshot: ToolResultMessage = {
			role: "toolResult",
			toolCallId: "call_screen",
			toolName: "computer",
			content: [],
			isError: false,
			timestamp: 0,
			providerMetadata: {
				type: "computer",
				screenshot: { type: "computer_screenshot", file_id: "file-screen" },
				acknowledgedSafetyChecks: [],
			},
		};
		expect(tokenizer.countMessage(screenshot)).toBe(1200);
		expect(tokenizer.countMessage(screenshot, { excludeEncryptedReasoning: true })).toBe(1200);
		const image: ImageContent = { type: "image", data: "cG5n", mimeType: "image/png" };
		// The computer-result serializer emits its one screenshot instead of content images.
		expect(tokenizer.countMessage({ ...screenshot, content: [image, { ...image }] })).toBe(1200);
		expect(
			tokenizer.countMessage({ ...screenshot, providerMetadata: undefined, content: [image, { ...image }] }),
		).toBe(2400);
	});

	test("prices current native logical text, computer metadata and images without counting opaque snapshots", () => {
		const tokenizer = new Tokenizer();
		const actions = [{ type: "type" as const, text: "visible action" }];
		const checks = [{ id: "check", code: "reason", message: "visible safety" }];
		const search = { type: "search", queries: ["visible query"] };
		const items = [
			{
				type: "code_interpreter_call",
				code: "visible code",
				outputs: [
					{ type: "logs", logs: "visible log" },
					{ type: "image", url: "data:image/png;base64,cG5n" },
				],
				encrypted_content: "opaque".repeat(1000),
			},
			{ type: "web_search_call", action: search, signature: "opaque".repeat(1000) },
			{ type: "unknown_extension", text: "opaque".repeat(1000) },
		];
		const message: AssistantMessage = {
			role: "assistant",
			provider: "openai",
			api: "openai-responses",
			model: "fixture",
			timestamp: 0,
			stopReason: "stop",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			content: [
				{
					type: "toolCall",
					id: "call",
					name: "computer",
					arguments: {},
					providerMetadata: { type: "computer", providerItemId: "item", actions, pendingSafetyChecks: checks },
				},
			],
			providerPayload: createOpenAIResponsesHistoryPayload("openai", items),
		};
		const metadata = ["computer", "{}", JSON.stringify(actions), JSON.stringify(checks)];
		const expected =
			tokenizer.countTokens(["visible code", "visible log", JSON.stringify(search), ...metadata]) + 1200;
		expect(tokenizer.countMessage(message)).toBe(expected);
		expect(tokenizer.countMessage(message, { excludeEncryptedReasoning: true })).toBe(expected);
		expect(
			tokenizer.countMessage({
				...message,
				providerPayload: createOpenAIResponsesHistoryPayload("openai", items, false),
			}),
		).toBe(tokenizer.countTokens(metadata));
	});

	test("charges normalized generated images once and never revives deleted native mirrors after JSON reload", () => {
		const tokenizer = new Tokenizer();
		const image: ImageContent = { type: "image", data: "cG5n", mimeType: "image/png" };
		const text: TextContent = { type: "text", text: "equal source" };
		const message: AssistantMessage = {
			role: "assistant",
			provider: "openai",
			api: "openai-responses",
			model: "fixture",
			timestamp: 0,
			stopReason: "stop",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			content: [image, { ...image }, text],
			providerPayload: createOpenAIResponsesHistoryPayload(
				"openai",
				[
					{ type: "image_generation_call", result: image.data },
					{ type: "image_generation_call", result: image.data },
					{ type: "message", role: "assistant", content: [{ type: "output_text", text: text.text }] },
				],
				true,
				[
					{ itemIndex: 0, contentIndex: 0 },
					{ itemIndex: 1, contentIndex: 1 },
					{ itemIndex: 2, contentIndex: 2 },
				],
			),
		};
		expect(tokenizer.countMessage(message)).toBe(2400 + tokenizer.countTokens(text.text));
		bindMessageSource(message, "source", 0);
		const payload = message.providerPayload;
		if (payload?.type !== "openaiResponsesHistory" || !payload.origins)
			throw new Error("Expected captured source origins");
		const rewritten: AssistantMessage = JSON.parse(
			JSON.stringify({
				...message,
				content: [image, text],
				providerPayload: {
					...payload,
					origins: remapNativeItemOrigins(payload.origins, [
						{
							entryId: "source",
							blocks: [
								{ oldBlockIndex: 0, newBlockIndex: null },
								{ oldBlockIndex: 1, newBlockIndex: 0 },
								{ oldBlockIndex: 2, newBlockIndex: 1 },
							],
						},
					]),
				},
			}),
		);
		expect(tokenizer.countMessage(rewritten)).toBe(1200 + tokenizer.countTokens(text.text));
		expect(tokenizer.countMessage(rewritten, { excludeEncryptedReasoning: true })).toBe(
			1200 + tokenizer.countTokens(text.text),
		);
	});

	test("defaults to null encoding and byte estimation", () => {
		const tokenizer = new Tokenizer();
		expect(tokenizer.encoding).toBeNull();
		expect(tokenizer.countTokens("hello world")).toBe(3);
	});

	test("encoding is fixed at construction from the catalog model", () => {
		expect(new Tokenizer({ tokenizer: "claude-v47" }).encoding).toBe(natives.Encoding.ClaudeV47);
		expect(new Tokenizer({ tokenizer: "claude-v5" }).encoding).toBe(natives.Encoding.ClaudeV5);
		expect(new Tokenizer({}).encoding).toBeNull();
		expect(new Tokenizer(undefined).encoding).toBeNull();
	});

	test("separate instances do not interfere with each other", () => {
		const t1 = new Tokenizer({ tokenizer: "claude-v47" });
		const t2 = new Tokenizer({ tokenizer: "qwen3" });
		const t3 = new Tokenizer({});

		expect(t1.encoding).toBe(natives.Encoding.ClaudeV47);
		expect(t2.encoding).toBe(natives.Encoding.Qwen3);
		expect(t3.encoding).toBeNull();

		const t4 = new Tokenizer({ tokenizer: "claude-v3" });
		expect(t4.encoding).toBe(natives.Encoding.ClaudeV3);
		expect(t1.encoding).toBe(natives.Encoding.ClaudeV47);
		expect(t2.encoding).toBe(natives.Encoding.Qwen3);
		expect(t3.encoding).toBeNull();
	});
});

describe("countTokens with modes", () => {
	test("approximate mode uses fast estimation", () => {
		const tokenizer = new Tokenizer();
		expect(tokenizer.countTokens("hello world", "approximate")).toBe(3);
	});

	test("upperbound mode uses byte length", () => {
		const tokenizer = new Tokenizer();
		expect(tokenizer.countTokens("hello world", "upperbound")).toBe(11);
	});

	test("strict mode uses native counting regardless of encoding", () => {
		const noEncoding = new Tokenizer();
		expect(noEncoding.countTokens("hello world", "strict")).toBe(2);
		const claudeEncoding = new Tokenizer({ tokenizer: "claude-v47" });
		expect(claudeEncoding.countTokens("hello world", "strict")).toBeGreaterThan(0);
	});

	test("mode is per-call; encoding stays independently model-scoped in strict mode", () => {
		// approximate/upperbound skip the encoding entirely under NODE_ENV=test
		// (fast estimate for a snappy suite); strict is testEnv-independent, so
		// it is the mode that proves per-instance encoding isolation here.
		const claude = new Tokenizer({ tokenizer: "claude-v47" });
		const generic = new Tokenizer({});
		expect(claude.countTokens("hello world", "strict")).not.toBe(generic.countTokens("hello world", "strict"));
	});

	test("falls back conservatively when native encoding is unknown", () => {
		vi.spyOn(natives, "countTokens").mockImplementation(() => {
			throw new Error('value "DeepSeekV3" does not match any variant of enum Encoding');
		});
		const tokenizer = new Tokenizer({ tokenizer: "deepseek-v3" });
		expect(tokenizer.countTokens("hello world", "strict")).toBe(11);
		expect(tokenizer.countTokens("hello world", "upperbound")).toBe(11);
		expect(tokenizer.checkTokenBudget("x".repeat(40), 20)).toEqual({
			fits: false,
			tokens: 40,
			exact: false,
		});
	});

	test("does not swallow unrelated native tokenizer errors", () => {
		vi.spyOn(natives, "countTokens").mockImplementation(() => {
			throw new Error("native tokenizer exploded");
		});
		expect(() => new Tokenizer({ tokenizer: "deepseek-v3" }).countTokens("hello world", "strict")).toThrow(
			"native tokenizer exploded",
		);
	});
});
