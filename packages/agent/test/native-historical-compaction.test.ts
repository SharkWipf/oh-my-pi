import { expect, test } from "bun:test";
import { type CompactionPreparation, compact, DEFAULT_COMPACTION_SETTINGS } from "../src/compaction/compaction";
import { IMAGE_TOKEN_ESTIMATE, Tokenizer } from "../src/tokenizer";
import { createFileOps } from "../src/compaction/utils";
import { buildCompleteNativeAtomItems, buildOpenAiNativeHistory, composeOpenAiHistoricalInput, requestOpenAiRemoteCompaction, trimRemoteCompactionInputToContextWindow } from "../src/compaction/openai";
import { type ConvertToLlm, defaultConvertToLlm } from "../src/compaction/messages";
import { getCompactionV2PreserveData } from "../src/compaction/compaction-v2-streaming";
import type { AssistantMessage, Message, ToolResultMessage, UserMessage } from "@oh-my-pi/pi-ai/types";
import { bindMessageSource, exportItemOrigins, getSourceOrigin, transferTransformedSourceOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { streamOpenAIResponses } from "@oh-my-pi/pi-ai/providers/openai-responses";

const image = { type: "image" as const, mimeType: "image/png", data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jM1sAAAAASUVORK5CYII=" };
const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
function fixture() {
	const old: UserMessage = { role: "user", content: "equal user", timestamp: 1 };
	const assistant: AssistantMessage = {
		role: "assistant", api: "openai-responses", provider: "openai", model: "gpt-5", usage, stopReason: "toolUse", timestamp: 2,
		content: [
			{ type: "text", text: "before tools", textSignature: "opaque-text-signature" },
			{ type: "thinking", thinking: "full plaintext thinking", thinkingSignature: JSON.stringify({ type: "reasoning", encrypted_content: "opaque-thinking-signature" }) },
			{ type: "toolCall", id: "call_a", name: "read", arguments: { path: "full/path", nested: { content: "unabridged" } }, rawBlock: "<read full/path>", intent: "inspect", thoughtSignature: "opaque-thought-signature" },
			{ type: "toolCall", id: "call_b", name: "edit", arguments: { input: "*** Begin Patch\nfull patch\n*** End Patch" }, customWireName: "apply_patch", rawBlock: "<edit full patch>" },
		],
	};
	const first: ToolResultMessage = { role: "toolResult", toolCallId: "call_a", toolName: "read", content: [{ type: "text", text: "result before image" }, { ...image }, { type: "text", text: "result after image" }], isError: true, details: { explanation: "readable failure metadata" }, timestamp: 3 };
	const second: ToolResultMessage = { role: "toolResult", toolCallId: "call_b", toolName: "edit", content: [{ type: "text", text: "complete patch result" }], isError: false, timestamp: 4 };
	const recent: UserMessage = { role: "user", content: "equal user", timestamp: 5 };
	const messages: Message[] = [old, assistant, first, second, recent];
	const sources = messages.map((message, order) => ({ entryId: `local-source-${order}`, order, message }));
	for (const source of sources) bindMessageSource(source.message, source.entryId, source.order);
	return { messages, sources, old, recent, nonUsers: [assistant, first, second] };
}
function model(baseUrl = "http://127.0.0.1:1/v1") {
	return buildModel({ id: "gpt-5", name: "offline native fixture", api: "openai-responses", provider: "openai", baseUrl, remoteCompaction: { v2StreamingEnabled: true }, reasoning: true, input: ["text", "image"], contextWindow: 400000, maxTokens: 128000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } });
}

function historicalPayloads(input: Array<Record<string, unknown>>) {
	return input.flatMap(item => Array.isArray(item.content) ? item.content : []).flatMap(block => {
		if (block.type !== "input_text") return [];
		try { const value = JSON.parse(block.text); return value.type === "historical_context" ? [value] : []; } catch { return []; }
	});
}

test("V1 replaces a complete multi-tool atom with chronological logical history and real interleaved images", async () => {
	const data = fixture();
	const native = buildOpenAiNativeHistory(data.messages.slice(1), model());
	const candidates = buildOpenAiNativeHistory([data.old, data.recent], model());
	const input = composeOpenAiHistoricalInput(native, candidates, data.nonUsers);
	expect(input.map(item => getSourceOrigin(item))).toEqual(expect.arrayContaining([
		expect.objectContaining({ kind: "source", parts: expect.arrayContaining([expect.objectContaining({ entryId: "local-source-1", blockIndex: 1, representation: "json-quoted-block", coverage: "full" })]) }),
	]));
	expect(input.map(item => getSourceOrigin(item)).flatMap(origin => origin?.kind === "source" ? [origin.parts[0]!.entryId] : [])).toEqual(data.sources.map(source => source.entryId));
	expect(input.every(item => item.role === "user")).toBe(true);
	const logical = historicalPayloads(input);
	expect(logical.find(value => value.block?.type === "thinking")?.block.thinking).toBe("full plaintext thinking");
	expect(logical.filter(value => value.block?.type === "toolCall").map(value => value.block)).toEqual([
		{ type: "toolCall", id: "call_a", name: "read", arguments: { path: "full/path", nested: { content: "unabridged" } }, rawBlock: "<read full/path>", intent: "inspect" },
		{ type: "toolCall", id: "call_b", name: "edit", arguments: { input: "*** Begin Patch\nfull patch\n*** End Patch" }, rawBlock: "<edit full patch>", customWireName: "apply_patch" },
	]);
	expect(logical.find(value => value.block?.toolCallId === "call_a")?.block).toMatchObject({ isError: true, details: { explanation: "readable failure metadata" } });
	const toolContent = input[2]!.content as Array<Record<string, unknown>>;
	const imageIndex = toolContent.findIndex(block => block.type === "input_image");
	expect(JSON.parse(String(toolContent[imageIndex - 1]!.text)).block.text).toBe("result before image");
	expect(toolContent[imageIndex]!.image_url).toBe(`data:${image.mimeType};base64,${image.data}`);
	expect(JSON.parse(String(toolContent[imageIndex + 1]!.text)).block.text).toBe("result after image");
	expect(JSON.stringify(input)).not.toMatch(/opaque-(text|thinking|thought)-signature|local-source-/);

	let sent: unknown;
	const canonical = [...input.map(item => ({ ...item, provider_extension: { untouched: true } })), { type: "compaction", encrypted_content: "opaque-result", extra: "canonical" }];
	const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) { sent = await request.json(); return Response.json({ output: canonical }); } });
	try {
		const result = await requestOpenAiRemoteCompaction(model(`${server.url}v1`), "offline-key", input, "compact fixture");
		expect(sent).toMatchObject({ input });
		expect(result.replacementHistory).toEqual(canonical);
		expect(result.replacementOrigins?.every(origin => origin.kind === "unknown")).toBe(true);
		expect(new Set(result.allUserSources?.map(part => part.entryId))).toEqual(new Set(data.sources.map(source => source.entryId)));
	} finally { server.stop(true); }
});

test("V2 failure rebuilds V1 historical N from frozen sources, not the attempted native payload", async () => {
	const data = fixture();
	const requirementMarker = "frozen-memory-requirement-v2-v1-selected-p-n";
	const requests: Array<{ path: string; input: Array<Record<string, unknown>>; wire: string }> = [];
	let canonical: Array<Record<string, unknown>> = [];
	const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
		const wire = await request.text();
		const body = JSON.parse(wire) as { input: Array<Record<string, unknown>> };
		const path = new URL(request.url).pathname;
		requests.push({ path, input: body.input, wire });
		if (!path.endsWith("/compact")) {
			data.nonUsers[0]!.content = [{ type: "text", text: "mutation after operation froze" }];
			return Response.json({ error: { message: "unsupported streaming compaction" } }, { status: 400 });
		}
		canonical = [...body.input.filter(item => item.role === "user"), { type: "compaction", encrypted_content: "canonical-fallback" }];
		return Response.json({ output: canonical });
	} });
	try {
		const preparation = {
			firstKeptEntryId: "local-source-4", messagesToSummarize: data.messages.slice(1, 4), turnPrefixMessages: [], recentMessages: [data.recent],
			isSplitTurn: false, tokensBefore: 100000, fileOps: createFileOps(), settings: { ...DEFAULT_COMPACTION_SETTINGS, remoteEnabled: true, remoteStreamingV2Enabled: true },
			sourcesToSummarize: data.sources.slice(1, 4), recentSources: data.sources.slice(4), selectedSources: data.sources.slice(0, 4),
		} as CompactionPreparation;
		const result = await compact(preparation, model(`${server.url}v1`), "offline-key", undefined, undefined, { remoteSystemPrompt: [requirementMarker] });
		expect(requests.map(request => request.path)).toEqual(["/v1/responses", "/v1/responses/compact"]);
		for (const request of requests) expect(request.wire.split(requirementMarker)).toHaveLength(2);
		expect(requests[0]!.input.filter(item => item.type === "function_call" || item.type === "custom_tool_call")).toHaveLength(2);
		expect(requests[1]!.input.every(item => item.role === "user")).toBe(true);
		expect(historicalPayloads(requests[1]!.input).find(value => value.block?.type === "thinking")?.block.thinking).toBe("full plaintext thinking");
		expect(JSON.stringify(requests[1]!.input)).not.toContain("mutation after operation froze");
		expect(result.preserveData?.openaiRemoteCompaction).toMatchObject({ replacementHistory: canonical });
		expect(result.retentionTarget).toBeUndefined();
	} finally { server.stop(true); }
});

test("complete native atoms supplement plaintext and raw metadata without duplicating executable tool calls", () => {
	const data = fixture();
	const native = buildOpenAiNativeHistory(data.nonUsers, model());
	const complete = buildCompleteNativeAtomItems(native, data.nonUsers);
	expect(complete.filter(item => item.type === "function_call" || item.type === "custom_tool_call")).toHaveLength(2);
	expect(historicalPayloads(complete).find(value => value.block?.thinking)?.block.thinking).toBe("full plaintext thinking");
	expect(historicalPayloads(complete).find(value => value.block?.rawBlock === "<read full/path>")?.block.intent).toBe("inspect");
	expect(exportItemOrigins(complete).every(origin => origin.kind !== "unknown")).toBe(true);
});


test("request trimming never replaces admitted N and marks ordinary rewritten output incomplete", () => {
	const data = fixture();
	const assistant = data.nonUsers[0] as AssistantMessage;
	assistant.content = assistant.content.filter(block => block.type === "toolCall" && block.id === "call_b");
	const result = data.nonUsers[2] as ToolResultMessage;
	result.content = [{ type: "text", text: "full tool output ".repeat(10000) }];
	bindMessageSource(result, "local-source-3", 3);
	const input = buildOpenAiNativeHistory([data.nonUsers[0]!, result], model());
	const ordinary = trimRemoteCompactionInputToContextWindow(input, new Tokenizer(model()), 2000, "compact");
	expect(ordinary.rewrittenOutputs).toBe(1);
	expect(getSourceOrigin(ordinary.input.at(-1)!)).toMatchObject({ kind: "source", parts: [expect.objectContaining({ entryId: "local-source-3", coverage: "derived", representation: "transformed-text" })] });
	const protectedIds = new Set(data.sources.slice(1, 4).map(source => source.entryId));
	const protectedInput = trimRemoteCompactionInputToContextWindow(input, new Tokenizer(model()), 2000, "compact", undefined, protectedIds);
	expect(protectedInput.input.at(-1)?.output).toEqual(input.at(-1)?.output);
	expect(protectedInput.rewrittenOutputs).toBe(0);
	expect(getSourceOrigin(protectedInput.input.at(-1)!)).toMatchObject({ kind: "source", parts: [expect.objectContaining({ coverage: "full" })] });
});


test("compact V2 sends complete N and partial P without changing ordinary retained membership", async () => {
	const data = fixture();
	data.old.content = "prefix keep suffix";
	const requirementMarker = "frozen-memory-requirement-v2-selected-partial-p-n";
	const requests: Array<Array<Record<string, unknown>>> = [];
	const wires: string[] = [];
	const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
		const wire = await request.text();
		wires.push(wire);
		const body = JSON.parse(wire) as { input: Array<Record<string, unknown>> };
		requests.push(body.input);
		const events = [
			{ type: "response.output_item.done", item: { type: "compaction", encrypted_content: "same native summary" } },
			{ type: "response.completed", response: { usage: { input_tokens: 30000, output_tokens: 1 } } },
		];
		return new Response(events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } });
	} });
	try {
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "local-source-4", messagesToSummarize: data.messages.slice(1, 4), turnPrefixMessages: [], recentMessages: [data.recent],
			isSplitTurn: false, tokensBefore: 100000, fileOps: createFileOps(), settings: { ...DEFAULT_COMPACTION_SETTINGS, remoteEnabled: true, remoteStreamingV2Enabled: true, v2RetainedMessageBudget: 1 },
			sourcesToSummarize: data.sources.slice(1, 4), recentSources: data.sources.slice(4), selectedSources: data.sources.slice(1, 4),
		};
		const convertToLlm: ConvertToLlm = messages => defaultConvertToLlm(messages).map(message =>
			message.role === "user" && typeof message.content === "string" && message.content.includes("keep")
				? transferTransformedSourceOrigin(message, { ...message, content: message.content.replace("keep", "KEEP") }) : message);
		const baseline = await compact(preparation, model(`${server.url}v1`), "offline-key", undefined, undefined, { convertToLlm, remoteSystemPrompt: [requirementMarker] });
		const selected = await compact({ ...preparation, selectedSources: [{ ...data.sources[0]!, spans: [{ blockIndex: 0, start: 7, end: 11 }] }, ...data.sources.slice(1, 4)] }, model(`${server.url}v1`), "offline-key", undefined, undefined, { convertToLlm, remoteSystemPrompt: [requirementMarker] });
		expect(requests).toHaveLength(2);
		for (const wire of wires) expect(wire.split(requirementMarker)).toHaveLength(2);
		const sent = requests[1]!;
		expect(sent.filter(item => item.type === "function_call" || item.type === "custom_tool_call")).toHaveLength(2);
		expect(sent.filter(item => item.type === "function_call_output" || item.type === "custom_tool_call_output")).toHaveLength(2);
		expect(historicalPayloads(sent).find(value => value.block?.thinking)?.block.thinking).toBe("full plaintext thinking");
		expect(historicalPayloads(sent).find(value => value.block?.rawBlock === "<read full/path>")?.block.intent).toBe("inspect");
		expect(JSON.stringify(sent)).toContain(`data:${image.mimeType};base64,${image.data}`);
		expect(JSON.stringify(sent)).toContain("[truncated]KEEP[truncated]");
		expect(JSON.stringify(sent)).not.toContain("prefix keep suffix");
		const baselineHistory = getCompactionV2PreserveData(baseline.preserveData)!.replacementHistory;
		const selectedHistory = getCompactionV2PreserveData(selected.preserveData)!.replacementHistory;
		const isOldSource = (item: Record<string, unknown>) => {
			const origin = getSourceOrigin(item);
			return origin?.kind === "source" && origin.parts.some(part => part.entryId === "local-source-0");
		};
		expect(selectedHistory.filter(item => !isOldSource(item))).toEqual(baselineHistory);
		expect(selectedHistory.filter(isOldSource)).toHaveLength(1);
		expect(JSON.stringify(selectedHistory)).toContain("[truncated]KEEP[truncated]");
		expect(JSON.stringify(selectedHistory)).toContain(`data:${image.mimeType};base64,${image.data}`);
	} finally { server.stop(true); }
});

test("historical selected computer results retain metadata-only screenshots and safety context", () => {
	const assistant: AssistantMessage = { role: "assistant", api: "openai-responses", provider: "openai", model: "gpt-5", usage, stopReason: "toolUse", timestamp: 1, content: [
		{ type: "toolCall", id: "call_screen|cu_screen", name: "computer", arguments: {}, providerMetadata: { type: "computer", providerItemId: "cu_screen", actions: [{ type: "screenshot" }], pendingSafetyChecks: [{ id: "check", message: "pending safety" }] } },
	] };
	const result: ToolResultMessage = { role: "toolResult", toolCallId: "call_screen|cu_screen", toolName: "computer", content: [], isError: false, timestamp: 2, providerMetadata: { type: "computer", screenshot: { type: "computer_screenshot", image_url: `data:${image.mimeType};base64,${image.data}` }, acknowledgedSafetyChecks: [{ id: "check", message: "acknowledged safety" }] } };
	bindMessageSource(assistant, "computer-call", 0);
	bindMessageSource(result, "computer-result", 1);
	const converted = defaultConvertToLlm([assistant, result]);
	const native = buildOpenAiNativeHistory(converted, { ...model(), supportsComputerUse: true });
	const complete = buildCompleteNativeAtomItems(native, converted);
	expect(complete.filter(item => item.type === "computer_call")).toHaveLength(1);
	expect(complete.filter(item => item.type === "computer_call_output")).toHaveLength(1);
	const historical = composeOpenAiHistoricalInput(native, [], converted);
	expect(historical.every(item => item.type === "message" && item.role === "user")).toBe(true);
	const images = historical.flatMap(item => Array.isArray(item.content) ? item.content : []).filter(block => block.type === "input_image");
	expect(images).toHaveLength(1);
	expect(images[0].image_url).toBe(result.providerMetadata?.screenshot.image_url);
	expect(getSourceOrigin(images[0])).toMatchObject({ kind: "source", parts: [{ entryId: "computer-result", blockIndex: "metadata.screenshot", coverage: "full", representation: "original-image" }] });
	expect(historicalPayloads(historical).find(value => value.block?.type === "toolCall")?.block.pendingSafetyChecks).toEqual([{ id: "check", message: "pending safety" }]);
	expect(historicalPayloads(historical).find(value => value.block?.type === "toolResult")?.block.acknowledgedSafetyChecks).toEqual([{ id: "check", message: "acknowledged safety" }]);
});

for (const streaming of [false, true]) {
	test(`compact V${streaming ? 2 : 1} preserves selected response-native server metadata without replaying an executable duplicate`, async () => {
		const requests: Array<Array<Record<string, unknown>>> = [];
		const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
			const body = await request.json() as { input: Array<Record<string, unknown>> };
			requests.push(body.input);
			if (!streaming) return Response.json({ output: [{ type: "compaction", encrypted_content: "native-summary" }] });
			const events = [{ type: "response.output_item.done", item: { type: "compaction", encrypted_content: "native-summary" } }, { type: "response.completed", response: { usage: { input_tokens: 30000, output_tokens: 1 } } }];
			return new Response(events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } });
		} });
		try {
			const assistant: AssistantMessage = { role: "assistant", api: "openai-responses", provider: "openai", model: "gpt-5", usage, stopReason: "stop", timestamp: 1, content: [{ type: "text", text: "visible synthesis" }], providerPayload: { type: "openaiResponsesHistory", provider: "openai", dt: true, items: [{ type: "web_search_call", id: "ws_selected", status: "completed", action: { type: "search", queries: ["selected native query"], sources: [{ type: "url", url: "https://example.invalid/selected-source" }] } }] } };
			const recent: UserMessage = { role: "user", content: "current question", timestamp: 2 };
			bindMessageSource(assistant, "selected-server", 0);
			bindMessageSource(recent, "recent-user", 1);
			const selected = { entryId: "selected-server", order: 0, message: assistant };
			const preparation: CompactionPreparation = { firstKeptEntryId: "recent-user", messagesToSummarize: [assistant], turnPrefixMessages: [], recentMessages: [recent], isSplitTurn: false, tokensBefore: 100000, fileOps: createFileOps(), settings: { ...DEFAULT_COMPACTION_SETTINGS, remoteEnabled: true, remoteStreamingV2Enabled: streaming, v2RetainedMessageBudget: 1 }, sourcesToSummarize: [selected], recentSources: [{ entryId: "recent-user", order: 1, message: recent }], selectedSources: [selected] };
			await compact(preparation, model(`${server.url}v1`), "offline-key");
			expect(requests).toHaveLength(1);
			const input = requests[0]!;
			const executable = input.filter(item => item.type === "web_search_call");
			expect(executable).toHaveLength(streaming ? 1 : 0);
			const logical = historicalPayloads(input);
			expect(logical.some(value => value.block?.text === "visible synthesis")).toBe(true);
			const search = streaming ? executable[0] : logical.find(value => value.block?.type === "web_search_call")?.block;
			expect(search?.action).toEqual({ type: "search", queries: ["selected native query"], sources: [{ type: "url", url: "https://example.invalid/selected-source" }] });
		} finally { server.stop(true); }
	});
}

test("selected historical native deltas retain interleaved image components but never admit full snapshots", () => {
	const assistant: AssistantMessage = { role: "assistant", api: "openai-responses", provider: "openai", model: "gpt-5", usage, stopReason: "stop", timestamp: 1, content: [{ type: "text", text: "visible" }], providerPayload: { type: "openaiResponsesHistory", provider: "openai", dt: true, items: [
		{ type: "function_call", id: "fc_native", call_id: "call_native", name: "read", arguments: "{}" },
		{ type: "function_call_output", call_id: "call_native", output: [{ type: "input_text", text: "native before" }, { type: "input_image", image_url: `data:${image.mimeType};base64,${image.data}`, detail: "original" }, { type: "input_text", text: "native after" }] },
	] } };
	bindMessageSource(assistant, "native-delta", 0);
	const input = composeOpenAiHistoricalInput([], [], [assistant]);
	const content = input.flatMap(item => Array.isArray(item.content) ? item.content : []);
	const imageIndex = content.findIndex(block => block.type === "input_image");
	expect(JSON.parse(content[imageIndex - 1].text).block.text).toBe("native before");
	expect(content[imageIndex]).toMatchObject({ image_url: `data:${image.mimeType};base64,${image.data}`, detail: "original" });
	expect(JSON.parse(content[imageIndex + 1].text).block.text).toBe("native after");
	expect(getSourceOrigin(content[imageIndex])).toMatchObject({ kind: "source", parts: [{ entryId: "native-delta", blockIndex: "providerPayload.1.output.1", representation: "original-image", coverage: "full" }] });
	expect(input.some(item => item.type === "function_call" || item.type === "function_call_output")).toBe(false);
	if (assistant.providerPayload?.type !== "openaiResponsesHistory") throw new Error("Expected native payload");
	assistant.providerPayload.dt = false;
	const snapshot = composeOpenAiHistoricalInput([], [], [assistant]);
	expect(historicalPayloads(snapshot).map(value => value.block).filter(Boolean)).toEqual([{ type: "text", text: "visible" }]);
});

test("selected generated and code-interpreter images remain original image INPUT alongside native atoms", async () => {
	const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
	const jpeg = await new Bun.Image(Buffer.from(png, "base64")).jpeg().toBase64();
	const assistant: AssistantMessage = { role: "assistant", api: "openai-responses", provider: "openai", model: "gpt-5", usage, stopReason: "stop", timestamp: 1, content: [], providerPayload: { type: "openaiResponsesHistory", provider: "openai", dt: true, items: [
		{ type: "image_generation_call", id: "ig_selected", status: "completed", result: jpeg },
		{ type: "code_interpreter_call", id: "ci_selected", status: "completed", container_id: "container_fixture", code: "render()", outputs: [{ type: "logs", logs: "before plot" }, { type: "image", url: `data:image/png;base64,${png}` }, { type: "logs", logs: "after plot" }] },
		{ type: "reasoning", id: "rs_selected", summary: [{ type: "summary_text", text: "readable reasoning" }], encrypted_content: "opaque-native-reasoning" },
	] } };
	bindMessageSource(assistant, "generated-images", 0);
	const sourceTokenizer = new Tokenizer(model());
	expect(sourceTokenizer.countMessage(assistant, { excludeEncryptedReasoning: true })).toBe(2 * IMAGE_TOKEN_ESTIMATE + sourceTokenizer.countTokens(["render()", "before plot", "after plot", "readable reasoning"]));
	const native = buildOpenAiNativeHistory([assistant], model());
	const complete = buildCompleteNativeAtomItems(native, [assistant]);
	expect(complete.filter(item => item.type === "image_generation_call")).toHaveLength(1);
	expect(complete.filter(item => item.type === "code_interpreter_call")).toHaveLength(1);
	const historical = composeOpenAiHistoricalInput(native, [], [assistant]);
	expect(historical.every(item => item.type === "message" && item.role === "user")).toBe(true);
	for (const input of [complete, historical]) {
		const content = input.flatMap(item => Array.isArray(item.content) ? item.content : []);
		const images = content.filter(block => block.type === "input_image");
		expect(images.map(block => block.image_url)).toEqual([`data:image/jpeg;base64,${jpeg}`, `data:image/png;base64,${png}`]);
		expect(getSourceOrigin(images[0])).toMatchObject({ kind: "source", parts: [{ entryId: "generated-images", blockIndex: "providerPayload.0.result", representation: "original-image", coverage: "full" }] });
		expect(getSourceOrigin(images[1])).toMatchObject({ kind: "source", parts: [{ entryId: "generated-images", blockIndex: "providerPayload.1.outputs.1", representation: "original-image", coverage: "full" }] });
		const plotIndex = content.indexOf(images[1]);
		expect(JSON.parse(content[plotIndex - 1].text).block.logs).toBe("before plot");
		expect(JSON.parse(content[plotIndex + 1].text).block.logs).toBe("after plot");
		expect(content.filter(block => block.type === "input_text").every(block => !block.text.includes(jpeg) && !block.text.includes(png) && !block.text.includes("opaque-native-reasoning"))).toBe(true);
	}
	expect(historicalPayloads(historical).find(value => value.block?.type === "reasoning")?.block.summary).toEqual([{ type: "summary_text", text: "readable reasoning" }]);
});

test("actual generated ingress and persisted replay quote each logical image once without byte deduplication", async () => {
	const generated = [0, 1].map(index => ({ type: "image_generation_call", id: `ig_${index}`, status: "completed", result: image.data }));
	const interpreter = { type: "code_interpreter_call", id: "ci_ingress", status: "completed", container_id: "offline", code: "plot()", outputs: [{ type: "logs", logs: "before" }, { type: "image", url: `data:${image.mimeType};base64,${image.data}` }, { type: "logs", logs: "after" }] };
	const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch() {
		const events = [...generated, interpreter].map((item, output_index) => ({ type: "response.output_item.done", item, output_index }));
		return new Response([...events, { type: "response.completed", response: { status: "completed", usage: { input_tokens: 1, output_tokens: 1 } } }].map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } });
	} });
	try {
		const active = model(`${server.url}v1`);
		const ingress = await streamOpenAIResponses(active, { messages: [{ role: "user", content: "draw", timestamp: 0 }] }, { apiKey: "offline-key", statefulResponses: false }).result();
		expect(ingress.stopReason).toBe("stop");
		expect(ingress.content.filter(block => block.type === "image")).toHaveLength(2);
		const replay: AssistantMessage = JSON.parse(JSON.stringify(ingress));
		bindMessageSource(replay, "generated-ingress", 0);
		const tokenizer = new Tokenizer(active);
		expect(tokenizer.countMessage(replay, { excludeEncryptedReasoning: true })).toBe(3 * IMAGE_TOKEN_ESTIMATE + tokenizer.countTokens(["plot()", "before", "after"]));
		const native = buildOpenAiNativeHistory([replay], active);
		for (const input of [buildCompleteNativeAtomItems(native, [replay]), composeOpenAiHistoricalInput(native, [], [replay])]) {
			const images = input.flatMap(item => Array.isArray(item.content) ? item.content : []).filter(block => block.type === "input_image");
			expect(images.map(block => block.image_url)).toEqual(Array(3).fill(`data:${image.mimeType};base64,${image.data}`));
		}
		expect(native.filter(item => item.type === "image_generation_call")).toHaveLength(2);
	} finally { server.stop(true); }
});
