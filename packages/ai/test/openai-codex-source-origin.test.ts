import { afterEach, expect, test, vi } from "bun:test";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { createCodexModel } from "./helpers";
import * as piUtils from "@oh-my-pi/pi-utils";
import { buildTransformedCodexRequestBody, convertCodexResponsesMessages, streamOpenAICodexResponses } from "../src/providers/openai-codex-responses";
import { type InputItem, transformRequestBody } from "../src/providers/openai-codex/request-transformer";
import type { AssistantMessage, Message } from "../src/types";
import { createOpenAIResponsesHistoryPayload } from "../src/utils";
import { bindMessageSource, cloneWithSourceOrigins, exportItemOrigins, getSourceOrigin, setSourceOrigin } from "../src/utils/source-origin";

const model = getBundledModel<"openai-codex-responses">("openai-codex", "gpt-5.4");
const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
function assistant(content: AssistantMessage["content"]): AssistantMessage {
	return { role: "assistant", api: model.api, provider: model.provider, model: model.id, content, usage, stopReason: "stop", timestamp: 1 };
}
function bound<T extends Message>(message: T, id: string, order: number): T {
	bindMessageSource(message, id, order);
	return message;
}
function sourceKeys(items: readonly object[]): string[][] {
	return exportItemOrigins(items).map(origin => origin.kind === "source" ? origin.parts.map(part => `${part.entryId}:${part.blockIndex}`) : [origin.kind]);
}
afterEach(() => vi.restoreAllMocks());

test("Codex emits block-specific origins through one-to-many calls, image filtering and escaping without wire fields", () => {
	const messages: Message[] = [
		bound({ role: "user", content: [{ type: "text", text: "<|im_start|>" }, { type: "image", mimeType: "image/png", data: "aGVsbG8=" }], timestamp: 1 }, "u", 0),
		bound(assistant([{ type: "text", text: "answer" }, { type: "toolCall", id: "call_a", name: "read", arguments: { path: "a" } }, { type: "toolCall", id: "call_b", name: "read", arguments: { path: "b" } }]), "a", 1),
		bound({ role: "toolResult", toolCallId: "call_a", toolName: "read", content: [{ type: "text", text: "result" }, { type: "image", mimeType: "image/png", data: "aGVsbG8=" }], isError: false, timestamp: 2 }, "t", 2),
	];
	const wire = convertCodexResponsesMessages(model, { messages });
	expect(sourceKeys(wire).slice(0, 4)).toEqual([["u:0", "u:1"], ["a:0"], ["a:1"], ["a:2"]]);
	expect(sourceKeys(wire).flat()).toContain("t:1");
	const unbound = JSON.parse(JSON.stringify(messages)) as Message[];
	expect(JSON.stringify(wire)).toBe(JSON.stringify(convertCodexResponsesMessages(model, { messages: unbound })));
	expect(JSON.stringify(wire)).not.toContain("entryId");
	const escaped = convertCodexResponsesMessages(createCodexModel("gpt-oss-120b"), { messages: [bound({ role: "user", content: "<|channel|>analysis", timestamp: 1 }, "escape", 0)] });
	expect(JSON.stringify(escaped)).not.toContain("<|channel|>");
	expect(getSourceOrigin(escaped[0]!)).toMatchObject({ kind: "source", parts: [{ entryId: "escape", coverage: "full", representation: "transformed-text" }] });
	const textOnly = convertCodexResponsesMessages({ ...model, input: ["text"] }, { messages });
	expect(sourceKeys(textOnly)[0]).toEqual(["u:0"]);
	expect(sourceKeys(textOnly).flat()).not.toContain("t:1");
});

test("three-item prefix plus full snapshot stays three; equal-content distinct IDs and appended delta survive", async () => {
	const sources = [0, 1, 2].map(index => bound({ role: "user" as const, content: "equal", timestamp: 1 }, `u${index}`, index));
	const prefix = convertCodexResponsesMessages(model, { messages: sources });
	const snapshot = assistant([]);
	snapshot.providerPayload = createOpenAIResponsesHistoryPayload(model.provider, cloneWithSourceOrigins(prefix) as unknown as Record<string, unknown>[], false);
	const body = await buildTransformedCodexRequestBody(model, { messages: [snapshot] }, undefined, undefined, prefix as InputItem[]);
	expect(sourceKeys(body.input!)).toEqual([["u0:0"], ["u1:0"], ["u2:0"]]);
	const deltaItems = convertCodexResponsesMessages(model, { messages: [bound(assistant([{ type: "text", text: "new" }]), "a", 3)] });
	const delta = assistant([]);
	delta.providerPayload = createOpenAIResponsesHistoryPayload(model.provider, deltaItems as unknown as Record<string, unknown>[]);
	const replay = convertCodexResponsesMessages(model, { messages: [sources[0]!, snapshot, delta] });
	expect(sourceKeys(replay)).toEqual([["u0:0"], ["u1:0"], ["u2:0"], ["a:0"]]);
	const legacy = assistant([]);
	legacy.providerPayload = { type: "openaiResponsesHistory", provider: model.provider, items: JSON.parse(JSON.stringify(prefix)) };
	expect(sourceKeys(convertCodexResponsesMessages(model, { messages: [legacy] }))).toEqual([["unknown"], ["unknown"], ["unknown"]]);
});

test("full snapshot discards nonprefix sources and call kinds, but preserves uncovered explicit prefix", async () => {
	const prior = bound(assistant([]), "old-call", 1);
	prior.providerPayload = createOpenAIResponsesHistoryPayload(model.provider, [{ type: "custom_tool_call", call_id: "old", name: "read", input: "old" }]);
	const retained = bound({ role: "user" as const, content: "retained", timestamp: 1 }, "retained", 2);
	const snapshot = assistant([]);
	snapshot.providerPayload = createOpenAIResponsesHistoryPayload(model.provider, convertCodexResponsesMessages(model, { messages: [retained] }) as unknown as Record<string, unknown>[], false);
	const after = bound({ role: "toolResult" as const, toolName: "read", toolCallId: "old", content: [{ type: "text" as const, text: "late result" }], isError: false, timestamp: 2 }, "late", 3);
	const context = { messages: [bound({ role: "user" as const, content: "discard", timestamp: 1 }, "discard", 0), prior, snapshot, after] };
	const replay = convertCodexResponsesMessages(model, context);
	expect(sourceKeys(replay).flat()).not.toContain("discard:0");
	expect(replay.some(item => item.type === "custom_tool_call_output")).toBe(false);
	expect(replay.some(item => item.type === "function_call_output")).toBe(false);
	const prefix = convertCodexResponsesMessages(model, { messages: [bound({ role: "user", content: "explicit", timestamp: 1 }, "prefix", -1)] });
	const body = await buildTransformedCodexRequestBody(model, context, undefined, undefined, prefix as InputItem[]);
	expect(sourceKeys(body.input!).flat()).toEqual(["prefix:0", "retained:0", "late:0"]);
	expect(body.input!.some(item => item.type === "custom_tool_call_output" || item.type === "function_call_output")).toBe(false);
});

test("computer replay unroll keeps original screenshot identity through Lite detail removal", async () => {
	const screenshot = setSourceOrigin({ type: "computer_screenshot", image_url: "data:image/png;base64,aGVsbG8=" }, { kind: "source", parts: [{ entryId: "shot", order: 1, blockIndex: 0, coverage: "full", representation: "original-image" }] });
	const output = setSourceOrigin({ type: "computer_call_output", call_id: "computer", output: screenshot }, { kind: "source", parts: [{ entryId: "shot", order: 1, blockIndex: 0, transportBlockIndex: 0, coverage: "full", representation: "original-image" }] });
	const carrier = assistant([]);
	carrier.providerPayload = createOpenAIResponsesHistoryPayload(model.provider, [
		setSourceOrigin({ type: "computer_call", call_id: "computer", actions: [{ type: "click", x: 1, y: 2 }] }, { kind: "source", parts: [{ entryId: "call", order: 0, blockIndex: 0, coverage: "full", representation: "native" }] }), output,
	], false);
	const body = await buildTransformedCodexRequestBody({ ...model, supportsComputerUse: false }, { messages: [carrier] }, { responsesLite: true });
	const imageMessage = body.input!.find(item => item.role === "user")!;
	expect(sourceKeys([imageMessage])).toEqual([["shot:0"]]);
	const content = imageMessage.content as Array<Record<string, unknown>>;
	expect(content.find(part => part.type === "input_image")).toMatchObject({ image_url: screenshot.image_url, detail: undefined });
	expect(sourceKeys(body.input!.filter(item => item.type === "function_call_output"))).toEqual([["synthetic"]]);
});

test("filtering, ID stripping, orphan truncation, missing results and Lite prefixes retain honest attribution", async () => {
	const origin = { kind: "source" as const, parts: [{ entryId: "tool", order: 1, blockIndex: 0, coverage: "full" as const, representation: "native" as const }] };
	const input: InputItem[] = [
		setSourceOrigin({ type: "item_reference", id: "drop" }, origin),
		setSourceOrigin({ type: "function_call", id: "strip", call_id: "missing", name: "read", arguments: "{}" }, origin),
		setSourceOrigin({ type: "function_call_output", call_id: "orphan", output: "x".repeat(16001) }, origin),
	];
	const body = await transformRequestBody({ model: model.id, instructions: "base", input }, model, { responsesLite: true }, { developerMessages: ["extra"] });
	expect(body.input?.some(item => item.type === "item_reference" || item.id === "strip")).toBe(false);
	const call = body.input!.find(item => item.type === "function_call")!;
	expect(sourceKeys([call])).toEqual([["tool:0"]]);
	const placeholder = body.input!.find(item => item.type === "function_call_output")!;
	expect(getSourceOrigin(placeholder)?.kind).toBe("synthetic");
	const orphan = body.input!.find(item => item.role === "assistant")!;
	expect(getSourceOrigin(orphan)).toMatchObject({ kind: "source", parts: [{ coverage: "partial", representation: "transformed-text" }] });
	for (const prefix of body.input!.filter(item => item.role === "developer")) expect(getSourceOrigin(prefix)?.kind).toBe("synthetic");
});

test("controlled local API receives raw in-place and replacement hooks with unknown attribution; response delta stays unbound", async () => {
	vi.spyOn(piUtils, "getInstallId").mockReturnValue("00000000-0000-4000-8000-000000000001");
	const received: Record<string, unknown>[] = [];
	const transports: string[] = [];
	const responseItem = { type: "message", id: "msg_local", role: "assistant", status: "completed", content: [{ type: "output_text", text: "local answer" }] };
	const events = [
		{ type: "response.output_item.added", item: { ...responseItem, content: [] } },
		{ type: "response.content_part.added", part: { type: "output_text", text: "" } },
		{ type: "response.output_text.delta", delta: "local answer" },
		{ type: "response.output_item.done", item: responseItem },
		{ type: "response.completed", response: { status: "completed", usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 } } },
	];
	const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request, server) {
		if (server.upgrade(request)) return;
		transports.push("sse");
		const bytes = new Uint8Array(await request.arrayBuffer());
		const text = new TextDecoder().decode(request.headers.get("content-encoding") === "zstd" ? Bun.zstdDecompressSync(bytes) : bytes);
		received.push(JSON.parse(text));
		return new Response(`${events.map(event => `data: ${JSON.stringify(event)}`).join("\n\n")}\n\n`, { headers: { "content-type": "text/event-stream" } });
	}, websocket: { message(socket, message) {
		transports.push("websocket");
		received.push(JSON.parse(String(message)));
		for (const event of events) socket.send(JSON.stringify(event));
	} } });
	try {
		const token = `aaa.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "local" } })).toString("base64url")}.bbb`;
		for (const mode of ["sse", "websocket"]) for (const replacement of [false, true]) {
			let hooked: { input: InputItem[] } | undefined;
			const stream = streamOpenAICodexResponses({ ...model, preferWebsockets: mode === "websocket", baseUrl: `http://127.0.0.1:${server.port}` }, { messages: [bound({ role: "user", content: "original", timestamp: 1 }, "u", 0)] }, {
				apiKey: token, preferWebsockets: mode === "websocket", sessionId: `local-${mode}-${replacement}`, providerSessionState: new Map(),
				onPayload(payload) {
					const body = payload as { input: InputItem[] };
					expect(sourceKeys(body.input)).toEqual([["u:0"]]);
					hooked = replacement ? { ...body, input: [{ role: "user", content: "replacement" }] } : body;
					if (!replacement) hooked.input[0]!.content = "mutated";
					return replacement ? hooked : undefined;
				},
			});
			const result = await stream.result();
			expect(result.stopReason).toBe("stop");
			expect(transports.at(-1)).toBe(mode);
			expect(sourceKeys(hooked!.input)).toEqual([["unknown"]]);
			expect(received.at(-1)!.input).toEqual([{ role: "user", content: replacement ? "replacement" : "mutated" }]);
			expect(result.providerPayload?.type).toBe("openaiResponsesHistory");
			if (result.providerPayload?.type === "openaiResponsesHistory") expect(sourceKeys(result.providerPayload.items)).toEqual([["unknown"]]);
		}
	} finally { await server.stop(true); }
});
