import { describe, expect, it } from "bun:test";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { convertCodexResponsesMessages } from "../src/providers/openai-codex-responses";
import { streamAzureOpenAIResponses } from "../src/providers/azure-openai-responses";
import { streamOpenAIResponses } from "../src/providers/openai-responses";
import { buildResponsesInput, convertResponsesAssistantMessage, escapeReplayedControlTokens, repairOrphanResponsesToolCalls, repairOrphanResponsesToolOutputs } from "../src/providers/openai-shared";
import type { ResponseInput } from "../src/providers/openai-responses-wire";
import type { AssistantMessage, Context, Message, Model, UserMessage } from "../src/types";
import { createOpenAIResponsesHistoryPayload } from "../src/utils";
import { bindMessageSource, cloneWithSourceOrigins, exportItemOrigins, getSourceOrigin, importItemOrigins, type NativeItemOrigin, setSourceOrigin } from "../src/utils/source-origin";

const openai = getBundledModel<"openai-responses">("openai", "gpt-5.4");
const azure = getBundledModel<"azure-openai-responses">("azure", "gpt-5.4");
const zeroUsage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };

function assistant(model: Model, content: AssistantMessage["content"]): AssistantMessage {
	return { role: "assistant", content, model: model.id, api: model.api, provider: model.provider, usage: zeroUsage, stopReason: "toolUse", timestamp: 1 };
}

function sourceParts(item: object) {
	const origin = getSourceOrigin(item);
	if (origin?.kind !== "source") throw new Error(`Expected source, got ${JSON.stringify(origin)}`);
	return origin.parts;
}

function serialize(model: Model<"openai-responses" | "azure-openai-responses">, messages: Message[], replay = false): ResponseInput {
	return buildResponsesInput({ model, context: { messages }, strictResponsesPairing: true, supportsImageDetailOriginal: model.compat.supportsImageDetailOriginal, nativeHistory: { replay, filterReasoning: false } });
}

function fixture(model: Model): Message[] {
	const messages: Message[] = [
		{ role: "user", content: [{ type: "image", data: "AA==", mimeType: "image/png" }, { type: "text", text: "same" }], timestamp: 0 },
		assistant(model, [
			{ type: "thinking", thinking: "reason", thinkingSignature: JSON.stringify({ type: "reasoning", id: "rs_fixture", summary: [{ type: "summary_text", text: "reason" }] }) },
			{ type: "toolCall", id: "call_a", name: "read", arguments: { path: "a" } },
			{ type: "text", text: "same" },
			{ type: "toolCall", id: "call_b", name: "read", arguments: { path: "b" } },
		]),
		{ role: "toolResult", toolCallId: "call_a", toolName: "read", content: [{ type: "image", data: "AQ==", mimeType: "image/png" }, { type: "text", text: "result" }], isError: false, timestamp: 2 },
		{ role: "toolResult", toolCallId: "call_b", toolName: "read", content: [{ type: "text", text: "one" }, { type: "text", text: "two" }], isError: false, timestamp: 3 },
		{ role: "user", content: "same", timestamp: 4 },
	];
	messages.forEach((message, index) => bindMessageSource(message, `entry-${index}`, index));
	return messages;
}

describe("OpenAI source origins at actual serializers", () => {
	for (const model of [openai, azure]) {
		it(`${model.provider} retains block identity through one-to-many emission, text hoist and tool images without changing wire bytes`, () => {
			const messages = fixture(model);
			const items = serialize(model, messages, true);
			expect(JSON.stringify(items)).toBe(JSON.stringify(serialize(model, JSON.parse(JSON.stringify(messages)), true)));
			expect(items.map(item => sourceParts(item).map(part => [part.entryId, part.blockIndex]))).toEqual([
				[["entry-0", 1], ["entry-0", 0]],
				[["entry-1", 0]],
				[["entry-1", 2]],
				[["entry-1", 1]],
				[["entry-1", 3]],
				[["entry-2", 0], ["entry-2", 1]],
				[["entry-3", 0], ["entry-3", 1]],
				[["entry-4", 0]],
			]);
			const output = items.find(item => item.type === "function_call_output" && item.call_id === "call_a");
			if (!output || output.type !== "function_call_output" || !Array.isArray(output.output)) throw new Error("Missing image result");
			expect(sourceParts(output.output[0])[0]).toMatchObject({ entryId: "entry-2", blockIndex: 0, representation: "original-image" });
			const textOutput = items.find(item => item.type === "function_call_output" && item.call_id === "call_b");
			expect(sourceParts(textOutput!)).toMatchObject([
				{ sourceSpan: { start: 0, end: 3 }, transportSpan: { start: 0, end: 3 } },
				{ sourceSpan: { start: 0, end: 3 }, transportSpan: { start: 4, end: 7 } },
			]);
		});
	}

	it("imports prior full-snapshot maps and delta maps instead of attributing the snapshot to its carrier", () => {
		const users: UserMessage[] = [0, 1, 2].map(index => ({ role: "user", content: "identical", timestamp: index }));
		users.forEach((user, index) => bindMessageSource(user, `historical-${index}`, index));
		const original = serialize(openai, users);
		const full = assistant(openai, [{ type: "text", text: "carrier fallback" }]);
		full.providerPayload = JSON.parse(JSON.stringify(createOpenAIResponsesHistoryPayload("openai", original as unknown as Record<string, unknown>[], false)));
		bindMessageSource(full, "carrier", 9);
		const replacement = serialize(openai, [...users, full], true);
		expect(replacement).toEqual(original);
		expect(replacement.map(item => sourceParts(item)[0].entryId)).toEqual(["historical-0", "historical-1", "historical-2"]);
		const delta = assistant(openai, [{ type: "text", text: "delta" }]);
		bindMessageSource(delta, "delta-entry", 10);
		delta.providerPayload = JSON.parse(JSON.stringify(createOpenAIResponsesHistoryPayload("openai", convertResponsesAssistantMessage(delta, openai, 0, new Set()) as unknown as Record<string, unknown>[], true)));
		const appended = serialize(openai, [full, delta], true);
		expect(appended.map(item => sourceParts(item)[0].entryId)).toEqual(["historical-0", "historical-1", "historical-2", "delta-entry"]);
		const legacy = assistant(openai, [{ type: "text", text: "fallback" }]);
		legacy.providerPayload = { type: "openaiResponsesHistory", provider: "openai", dt: false, items: JSON.parse(JSON.stringify(original)) };
		bindMessageSource(legacy, "legacy-carrier", 11);
		expect(exportItemOrigins(serialize(openai, [legacy], true)).every(origin => origin.kind === "unknown")).toBe(true);
	});

	it("transfers exact clone positions and treats escaped text, omitted images and synthetic repairs honestly", () => {
		const user: UserMessage = { role: "user", content: [{ type: "text", text: "<|channel|>analysis" }, { type: "image", data: "AA==", mimeType: "image/png" }], timestamp: 0 };
		bindMessageSource(user, "escaped", 0);
		const item = escapeReplayedControlTokens(serialize(openai, [user]))[0];
		expect(sourceParts(item)[0]).toMatchObject({ entryId: "escaped", blockIndex: 0, representation: "transformed-text", coverage: "full" });
		expect(sourceParts(item)[0].sourceSpan).toBeUndefined();
		expect(sourceParts(item)[1]).toMatchObject({ blockIndex: 1, representation: "original-image" });
		const clone = JSON.parse(JSON.stringify([item]));
		importItemOrigins(clone, exportItemOrigins([item]));
		expect(sourceParts(clone[0].content[1])[0].blockIndex).toBe(1);
		expect(exportItemOrigins(escapeReplayedControlTokens(clone))).toEqual(exportItemOrigins([item]));
		const noImages = serialize({ ...openai, input: ["text"] }, [user])[0];
		expect(sourceParts(noImages).map(part => part.blockIndex)).toEqual([0]);
		const call = assistant(openai, [{ type: "toolCall", id: "call_missing", name: "read", arguments: {} }]);
		bindMessageSource(call, "interrupted", 1);
		const repaired = repairOrphanResponsesToolCalls(convertResponsesAssistantMessage(call, openai, 0, new Set()));
		expect(getSourceOrigin(repaired[1])).toMatchObject({ kind: "synthetic", reason: "interrupted-tool-output" });
		const result = setSourceOrigin({ type: "function_call_output", call_id: "orphan", output: "x".repeat(17000) } as ResponseInput[number], { kind: "source", parts: [{ entryId: "orphan-source", order: 2, blockIndex: 0, coverage: "full", representation: "native", sourceSpan: { start: 0, end: 17000 } }] });
		expect(sourceParts(repairOrphanResponsesToolOutputs([result])[0])[0]).toMatchObject({ entryId: "orphan-source", coverage: "derived", representation: "transformed-text" });
	});

	for (const baseModel of [openai, azure, getBundledModel<"openai-codex-responses">("openai-codex", "gpt-5.4")]) {
		it(baseModel.provider + " escapes ordinary computer acknowledgments without changing screenshot identity or the non-Harmony request", () => {
			const marker = "<|channel|>analysis";
			const escaped = String.raw`<\|channel\|>analysis`;
			const model = { ...baseModel, supportsComputerUse: true };
			const call = assistant(model, [{ type: "toolCall", id: "call_screen|item_screen", name: "computer", arguments: {}, providerMetadata: { type: "computer", providerItemId: "item_screen", actions: [{ type: "screenshot" }], pendingSafetyChecks: [] } }]);
			const screenshot = { type: "computer_screenshot" as const, image_url: "data:image/png;base64,AA==" };
			const check = { id: "ack_" + marker, code: "code_" + marker, message: marker };
			const result: Message = { role: "toolResult", toolCallId: "call_screen|item_screen", toolName: "computer", content: [], isError: false, timestamp: 2, providerMetadata: { type: "computer", screenshot, acknowledgedSafetyChecks: [check] } };
			bindMessageSource(call, "screen-call", 0);
			bindMessageSource(result, "screen-result", 1);
			const before = JSON.stringify([call, result]);
			const convert = (activeModel: typeof model) => activeModel.api === "openai-codex-responses"
				? convertCodexResponsesMessages(activeModel, { messages: [call, result] })
				: buildResponsesInput({ model: activeModel, context: { messages: [call, result] }, strictResponsesPairing: true, supportsImageDetailOriginal: true });
			const ordinary = convert(model);
			const items = convert({ ...model, identity: { ...model.identity, class: "gpt-oss" } });
			const output = items.find(item => item.type === "computer_call_output");
			if (!output || output.type !== "computer_call_output") throw new Error("Missing computer output");
			expect(output).toMatchObject({ call_id: "call_screen", output: screenshot, acknowledged_safety_checks: [{ ...check, message: escaped }] });
			expect(ordinary.find(item => item.type === "computer_call_output")).toMatchObject({ output: screenshot, acknowledged_safety_checks: [check] });
			expect(sourceParts(output).find(part => part.blockIndex === "metadata.screenshot")).toMatchObject({ entryId: "screen-result", coverage: "full", representation: "original-image" });
			expect(sourceParts(output.output)[0]).toMatchObject({ entryId: "screen-result", blockIndex: "metadata.screenshot", representation: "original-image" });
			expect(sourceParts(output).find(part => part.blockIndex === "metadata.acknowledgedSafetyChecks.0")).toMatchObject({ entryId: "screen-result", representation: "transformed-text" });
			const replay = JSON.parse(JSON.stringify(items));
			importItemOrigins(replay, exportItemOrigins(items));
			const replayedOutput = replay.find((item: { type: string }) => item.type === "computer_call_output");
			expect(sourceParts(replayedOutput.output)[0]).toMatchObject({ entryId: "screen-result", blockIndex: "metadata.screenshot", representation: "original-image" });
			expect(JSON.stringify([call, result])).toBe(before);
		});
	}

	for (const model of [openai, azure]) {
		for (const replacement of [false, true]) {
			it(`${model.provider} invalidates ${replacement ? "replacement" : "in-place"} hooks at the controlled HTTP boundary without reinjection`, async () => {
				let observed: Record<string, unknown> | undefined;
				let prehook: NativeItemOrigin[] | undefined;
				let hookPayload: { input: ResponseInput } | undefined;
				const source: UserMessage = { role: "user", content: "protected", timestamp: 0 };
				bindMessageSource(source, "protected-source", 0);
				const options = {
					apiKey: "offline-fixture", statefulResponses: false, azureBaseUrl: "https://offline.invalid/openai/v1", azureApiVersion: "v1",
					onPayload: (payload: unknown) => {
						const request = payload as { input: ResponseInput };
						prehook = exportItemOrigins(request.input);
						hookPayload = replacement ? cloneWithSourceOrigins(request) : request;
						hookPayload.input = hookPayload.input.filter(item => "role" in item && item.role === "user");
						const item = hookPayload.input[0] as { content: Array<{ text: string }> };
						item.content[0].text = "hook owns this";
						return replacement ? hookPayload : undefined;
					},
					fetch: async (_url: string | URL | Request, init?: RequestInit) => {
						observed = JSON.parse(String(init?.body));
						expect(exportItemOrigins(hookPayload!.input).every(origin => origin.kind === "unknown")).toBe(true);
						return new Response(JSON.stringify({ error: { message: "offline boundary stop" } }), { status: 400, headers: { "content-type": "application/json" } });
					},
				};
				const context: Context = { messages: [source] };
				const result = await (model.api === "azure-openai-responses" ? streamAzureOpenAIResponses(model, context, options) : streamOpenAIResponses(model, context, options)).result();
				expect(result.stopReason).toBe("error");
				expect(prehook).toContainEqual(expect.objectContaining({ kind: "source", parts: expect.arrayContaining([expect.objectContaining({ entryId: "protected-source" })]) }));
				expect(JSON.stringify(observed)).toContain("hook owns this");
				expect(JSON.stringify(observed)).not.toContain("protected");
			});
		}
	}
});
