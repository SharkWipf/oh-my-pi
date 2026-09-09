import { describe, expect, test } from "bun:test";
import { buildTransformedCodexRequestBody } from "@oh-my-pi/pi-ai/providers/openai-codex-responses";
import type { AssistantMessage, Message, ToolResultMessage } from "@oh-my-pi/pi-ai/types";
import { bindMessageSource, cloneWithSourceOrigins, exportItemOrigins, getSourceOrigin, importItemOrigins, setSourceOrigin, type NativeSourcePart } from "@oh-my-pi/pi-ai/utils/source-origin";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { buildCompactionV2ReplacementHistory, buildCompactionV2RequestFromBody, getCompactionV2PreserveData, requestCompactionV2Streaming, storeCompactionV2PreserveData } from "../src/compaction/compaction-v2-streaming";
import { compact, DEFAULT_COMPACTION_SETTINGS, type CompactionPreparation } from "../src/compaction/compaction";
import { createFileOps } from "../src/compaction/utils";
import { Tokenizer } from "../src/tokenizer";

const compaction = () => ({ type: "compaction", encrypted_content: "fixture-opaque" });
function user(entryId: string, order: number, text: string): Record<string, unknown> {
	const message: Message = { role: "user", content: text, timestamp: 0 };
	bindMessageSource(message, entryId, order);
	return setSourceOrigin({ role: "user", content: text }, getSourceOrigin(message)!);
}
function candidate(original: Record<string, unknown>, text: string, start: number, end: number, marker = "[truncated]"): Record<string, unknown> {
	const origin = getSourceOrigin(original)!;
	if (origin.kind !== "source") throw Error("fixture missing source");
	return setSourceOrigin({ role: "user", content: marker + text }, { kind: "source", parts: origin.parts.map(part => ({ ...part, coverage: "partial", sourceSpan: { start, end }, transportSpan: { start: marker.length, end: marker.length + text.length } })) });
}
function sourceIds(items: Record<string, unknown>[]): string[] {
	return items.flatMap(item => { const origin = getSourceOrigin(item); return origin?.kind === "source" ? [origin.parts[0].entryId] : []; });
}

describe("native V2 protected source union", () => {
	test("native retained delivery and selected original do not share a source slot", () => {
		const delivered = user("same-entry", 0, "DELIVERED EXPANSION");
		const original = user("same-entry", 0, "/task original");
		const origin = getSourceOrigin(original)!;
		if (origin.kind !== "source") throw Error("missing fixture source");
		for (const part of origin.parts) part.projection = "original";
		const result = buildCompactionV2ReplacementHistory([delivered], compaction(), 100, { userCandidates: [original], nonUserItems: [], nonUserSourceIds: [], nonUserTokens: 0 });
		expect(result.replacementHistory.map(item => item.content)).toEqual(["DELIVERED EXPANSION", "/task original", undefined]);
		expect(result.replacementHistory.at(-1)?.type).toBe("compaction");
	});
	test("P does not refund overlapping ordinary sources or change its cut; equal text IDs remain distinct", () => {
		const items = [user("old", 0, "same"), user("middle", 1, "same"), user("new", 2, "same")];
		const vanilla = buildCompactionV2ReplacementHistory(items, compaction(), 1).replacementHistory;
		const selected = buildCompactionV2ReplacementHistory(items, compaction(), 1, { userCandidates: [items[0], items[2]], nonUserItems: [], nonUserSourceIds: [], nonUserTokens: 0 }).replacementHistory;
		expect(sourceIds(vanilla)).toEqual(["new"]);
		expect(sourceIds(selected)).toEqual(["old", "new"]);
		expect(selected[1]).toBe(items[2]);
		expect(selected.some(item => item === items[1])).toBe(false);
	});

	test("actual budget-one string prefix and independent tail candidate share one source slot", () => {
		const text = `ABCD${"x".repeat(192)}EFGH`;
		const original = user("partial", 0, text);
		const tail = candidate(original, text.slice(132), 132, 200);
		const result = buildCompactionV2ReplacementHistory([original], compaction(), 1, { userCandidates: [tail], nonUserItems: [], nonUserSourceIds: [], nonUserTokens: 0 });
		expect(result.replacementHistory[0].content).toBe(`ABCD[truncated]${text.slice(132)}`);
		expect(sourceIds(result.replacementHistory)).toEqual(["partial"]);
		const origin = getSourceOrigin(result.replacementHistory[0]);
		expect(origin?.kind === "source" && origin.parts.map(part => part.sourceSpan)).toEqual([{ start: 0, end: 4 }, { start: 132, end: 200 }]);
		expect(tail.content).toBe(`[truncated]${text.slice(132)}`);
	});

	test("complete union removes derived markers without deleting ordinary source text", () => {
		const original = user("covered", 0, "ABCDEFGHIJKLMNOP");
		const tail = candidate(original, "EFGHIJKLMNOP", 4, 16);
		const result = buildCompactionV2ReplacementHistory([original], compaction(), 1, { userCandidates: [tail], nonUserItems: [], nonUserSourceIds: [], nonUserTokens: 0 });
		expect(result.replacementHistory[0].content).toBe(original.content);
		expect(getSourceOrigin(result.replacementHistory[0])?.kind).toBe("source");
	});

	test("complete multi-tool N is precharged before allocation, with zero-residual native minimum retained", () => {
		const source = (entryId: string, order: number, blockIndex: number): NativeSourcePart => ({ entryId, order, blockIndex, coverage: "full", representation: "native" });
		const atom = [
			setSourceOrigin({ type: "function_call", call_id: "c1", name: "read", arguments: '{"path":"one"}' }, { kind: "source", parts: [source("assistant", 0, 0)] }),
			setSourceOrigin({ type: "function_call", call_id: "c2", name: "read", arguments: '{"path":"two"}' }, { kind: "source", parts: [source("assistant", 0, 1)] }),
			setSourceOrigin({ type: "function_call_output", call_id: "c1", output: "full first output" }, { kind: "source", parts: [source("tool1", 1, 0)] }),
			setSourceOrigin({ type: "function_call_output", call_id: "c2", output: "full second output" }, { kind: "source", parts: [source("tool2", 2, 0)] }),
		];
		const older = user("older", 3, "old!");
		const newest = user("newest", 4, "new!more");
		const result = buildCompactionV2ReplacementHistory([...atom, older, newest], compaction(), 2, { userCandidates: [], nonUserSourceIds: ["assistant", "tool1", "tool2"], nonUserItems: atom, nonUserTokens: 40 });
		expect(result.retentionTarget).toEqual({ configuredTokens: 2, manualNonUserTokens: 40, residualTokens: 1 });
		expect(result.replacementHistory.slice(0, 4)).toEqual(atom);
		expect(result.replacementHistory[4].content).toBe("new!");
		expect(sourceIds(result.replacementHistory)).toEqual(["assistant", "assistant", "tool1", "tool2", "newest"]);
		expect(result.replacementHistory.filter(item => item.type === "function_call_output").map(item => item.output)).toEqual(["full first output", "full second output"]);
		const residual = buildCompactionV2ReplacementHistory([...atom, older, newest], compaction(), 42, {
			userCandidates: [], nonUserSourceIds: ["assistant", "tool1", "tool2"], nonUserItems: atom, nonUserTokens: 40,
		});
		expect(residual.retentionTarget).toEqual({ configuredTokens: 42, manualNonUserTokens: 40, residualTokens: 2 });
		expect(residual.replacementHistory.slice(0, 4)).toEqual(atom);
		expect(residual.replacementHistory[4].content).toBe("new!more");
		expect(sourceIds(residual.replacementHistory)).toEqual(["assistant", "assistant", "tool1", "tool2", "newest"]);
	});

	test("original image survives partial ordinary retention once and reload retains exact text/image origins", async () => {
		const spec = getBundledModel<"openai-codex-responses">("openai-codex", "gpt-5.4");
		if (!spec) throw Error("missing bundled Codex model");
		const model = buildModel<"openai-codex-responses">({ ...spec, api: "openai-codex-responses" });
		const message: Message = { role: "user", content: [{ type: "text", text: "ABCDEFGHIJKLMNOP" }, { type: "image", mimeType: "image/png", data: "aW1hZ2U=" }], timestamp: 0 };
		bindMessageSource(message, "image-source", 0);
		const body = await buildTransformedCodexRequestBody(model, { messages: [message] }, undefined);
		const input = body.input as Record<string, unknown>[];
		const result = buildCompactionV2ReplacementHistory(input, compaction(), 1, { userCandidates: input, nonUserItems: [], nonUserSourceIds: [], nonUserTokens: 0 });
		expect(result.retainedImageCount).toBe(1);
		expect(result.replacementHistory[0].content).toEqual(input[0].content);
		const data = JSON.parse(JSON.stringify(storeCompactionV2PreserveData({ ...result, compactionItem: result.replacementHistory.at(-1)!, usedTokens: 20 }, model)));
		const reloaded = getCompactionV2PreserveData(data)!;
		expect(sourceIds(reloaded.replacementHistory)).toEqual(["image-source"]);
		expect(exportItemOrigins(reloaded.replacementHistory)).toEqual(exportItemOrigins(result.replacementHistory));
	});

	test("provider-boundary stream keeps local selections off wire and returns frozen ordinary-plus-P union", async () => {
		const spec = getBundledModel("openai", "gpt-5.4");
		if (!spec) throw Error("missing bundled OpenAI model");
		const model = buildModel({ ...spec, remoteCompaction: { enabled: true, v2StreamingEnabled: true } });
		const older = user("old-wire", 0, "older");
		const newest = user("new-wire", 1, "new!");
		const input = [older, newest];
		let captured: Record<string, unknown> | undefined;
		const request = buildCompactionV2RequestFromBody(model, { model: model.id, input, stream: true }, { retainedMessageBudget: 1, preservation: { userCandidates: [older], nonUserItems: [], nonUserSourceIds: [], nonUserTokens: 0 } });
		const response = await requestCompactionV2Streaming(model, "fixture-key", request, undefined, { fetch: async (_url, init) => {
			captured = JSON.parse(String(init?.body));
			return new Response(`data: ${JSON.stringify({ type: "response.output_item.done", item: compaction() })}\n\ndata: ${JSON.stringify({ type: "response.completed", response: { usage: { input_tokens: 9, output_tokens: 1, total_tokens: 10 } } })}\n\n`, { headers: { "content-type": "text/event-stream" } });
		} });
		expect(captured?.input).toEqual([...input, { type: "compaction_trigger" }]);
		expect(JSON.stringify(captured)).not.toContain("entryId");
		expect(sourceIds(response.replacementHistory)).toEqual(["old-wire", "new-wire"]);
		const persistedItems = JSON.parse(JSON.stringify(response.replacementHistory));
		importItemOrigins(persistedItems, response.replacementOrigins);
		expect(sourceIds(persistedItems)).toEqual(["old-wire", "new-wire"]);
	});
	test("actual compact freezes complete original N before conversion inflation and keeps the ordinary cut under P overlap", async () => {
		const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
		let captured: Record<string, unknown> | undefined;
		const endpoint = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
			captured = await request.json() as Record<string, unknown>;
			return new Response(
				`data: ${JSON.stringify({ type: "response.output_item.done", item: compaction() })}

data: ${JSON.stringify({ type: "response.completed", response: { usage: { input_tokens: 10000, output_tokens: 1 } } })}

`,
				{ headers: { "content-type": "text/event-stream" } },
			);
		} });
		try {
			const model = buildModel({ id: "gpt-5", name: "localhost source quote", api: "openai-responses", provider: "openai", baseUrl: `${endpoint.url}v1`, remoteCompaction: { enabled: true, v2StreamingEnabled: true }, supportsComputerUse: true, reasoning: true, input: ["text", "image"], contextWindow: 400000, maxTokens: 128000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } });
			const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
			for (const generated of [false, true]) {
				const assistant: AssistantMessage = { role: "assistant", api: "openai-responses", provider: "openai", model: model.id, usage, stopReason: generated ? "stop" : "toolUse", timestamp: 1, content: generated ? [] : [{ type: "toolCall", id: "call_screen|cu_screen", name: "computer", arguments: {}, providerMetadata: { type: "computer", providerItemId: "cu_screen", actions: [{ type: "screenshot" }], pendingSafetyChecks: [] } }], ...(generated ? { providerPayload: { type: "openaiResponsesHistory" as const, provider: "openai", dt: true, items: [{ type: "image_generation_call" as const, id: "ig_only", status: "completed", result: png }, { type: "code_interpreter_call" as const, id: "ci_logical", status: "completed", container_id: "local", code: "print('logical-code')", outputs: [{ type: "logs" as const, logs: "logical execution output" }] }] } } : {}) };
				const screenshot: ToolResultMessage = { role: "toolResult", toolCallId: "call_screen|cu_screen", toolName: "computer", content: [], isError: false, timestamp: 2, providerMetadata: { type: "computer", screenshot: { type: "computer_screenshot", image_url: `data:image/png;base64,${png}` }, acknowledgedSafetyChecks: [] } };
				const originals: Message[] = generated ? [assistant] : [assistant, screenshot];
				const sources = originals.map((message, order) => ({ entryId: `N-${order}`, order, message }));
				const tokenizer = new Tokenizer(model);
				const quote = originals.reduce((tokens, message) => tokens + tokenizer.countMessage(message), 0);
				expect(quote).toBeGreaterThan(1200);
				const ordinary = ["excluded", "m".repeat(8000), "n".repeat(4000)].map((content, index) => ({ entryId: ["old", "middle", "newest"][index], order: index + 2, message: { role: "user" as const, content, timestamp: index + 3 } }));
				let baseline: Array<Record<string, unknown>> = [];
				for (const inflated of [false, true]) for (const overlap of [false, true]) {
					const preparation: CompactionPreparation = { firstKeptEntryId: "newest", messagesToSummarize: [...originals, ...ordinary.slice(0, 2).map(source => source.message)], turnPrefixMessages: [], recentMessages: [ordinary[2].message], isSplitTurn: false, tokensBefore: 10000, fileOps: createFileOps(), settings: { ...DEFAULT_COMPACTION_SETTINGS, v2RetainedMessageBudget: 3000 }, sourcesToSummarize: [...sources, ...ordinary.slice(0, 2)], recentSources: [ordinary[2]], selectedSources: [...sources, ...sources, ...(overlap ? [ordinary[2]] : [])] };
					const result = await compact(preparation, model, "localhost-key", undefined, undefined, { remoteSystemPrompt: ["local source quote fixture"], convertToLlm(messages) {
						if (inflated) for (const message of messages) if (message.role === "assistant" || message.role === "toolResult") message.content.push({ type: "text", text: "conversion inflation ".repeat(1000) });
						return cloneWithSourceOrigins(messages) as Message[];
					} });
					if (inflated) for (const message of originals) if (message.role === "assistant" || message.role === "toolResult") message.content.pop();
					expect(result.retentionTarget).toEqual({ configuredTokens: 3000, manualNonUserTokens: quote, residualTokens: Math.max(1, 3000 - quote) });
					const history = getCompactionV2PreserveData(result.preserveData)!.replacementHistory;
					const users = history.filter(item => sourceIds([item]).some(id => ["old", "middle", "newest"].includes(id)));
					expect(sourceIds(users)).toEqual(["middle", "newest"]);
					const middle = users[0].content as Array<{ type: string; text: string }>;
					const middleChars = middle.reduce((length, block) => length + (block.text?.length ?? 0), 0);
					expect(middleChars).toBeLessThanOrEqual((3000 - quote - 1000) * 4);
					expect(middleChars).toBeGreaterThan((3000 - quote - 1030) * 4);
					if (!inflated && !overlap) baseline = users;
					else expect(users).toEqual(baseline);
					expect(history.filter(item => item.type === (generated ? "image_generation_call" : "computer_call"))).toHaveLength(1);
					if (generated) expect(history.filter(item => item.type === "code_interpreter_call")).toMatchObject([{ code: "print('logical-code')", outputs: [{ type: "logs", logs: "logical execution output" }] }]);
					else expect(history.filter(item => item.type === "computer_call_output")).toHaveLength(1);
					const imageItems = history.flatMap(item => Array.isArray(item.content) ? item.content : []).filter(block => block.type === "input_image");
					if (generated) expect(imageItems.map(block => block.image_url)).toEqual([`data:image/png;base64,${png}`]);
					else expect(history.find(item => item.type === "computer_call_output")?.output).toEqual({ type: "computer_screenshot", image_url: `data:image/png;base64,${png}` });
					expect(JSON.stringify(captured)).not.toContain("nonUserTokens");
					expect(JSON.stringify(captured)).not.toContain("entryId");
				}
			}
		} finally { endpoint.stop(true); }
	});
});
