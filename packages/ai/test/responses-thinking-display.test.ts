import { describe, expect, test } from "bun:test";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { streamAzureOpenAIResponses } from "../src/providers/azure-openai-responses";
import { streamOpenAICodexResponses } from "../src/providers/openai-codex-responses";
import { streamOpenAIResponses } from "../src/providers/openai-responses";
import type { AssistantMessageEvent, FetchImpl, Model } from "../src/types";

const reasoning = (text = "") => ({
	type: "reasoning",
	id: "rs_display",
	summary: text ? [{ type: "summary_text", text }] : [],
	encrypted_content: "encrypted-replay",
});
const added = () => ({ type: "response.output_item.added", output_index: 0, item: reasoning() });
const delta = (text: string, raw = false) => ({
	type: raw ? "response.reasoning_text.delta" : "response.reasoning_summary_text.delta",
	item_id: "rs_display",
	output_index: 0,
	summary_index: 0,
	delta: text,
});
const done = (text: string) => ({ type: "response.output_item.done", output_index: 0, item: reasoning(text) });
const terminal = (incomplete = false, output?: unknown[]) => ({
	type: incomplete ? "response.incomplete" : "response.completed",
	response: {
		id: "resp_display",
		status: incomplete ? "incomplete" : "completed",
		...(incomplete ? { incomplete_details: { reason: "max_output_tokens" } } : {}),
		output,
		usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
	},
});

for (const api of ["openai-responses", "azure-openai-responses", "openai-codex-responses"] as const) {
	const model = buildModel({
		api,
		provider: api === "openai-codex-responses" ? "openai-codex" : api === "azure-openai-responses" ? "azure" : "openai",
		id: "gpt-5.3-codex-spark",
		name: "Display test",
		baseUrl: "https://example.invalid/v1",
		reasoning: true,
		input: ["text"],
		contextWindow: 8192,
		maxTokens: 2048,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		preferWebsockets: false,
	});
	async function run(events: unknown[], abort = false) {
		const controller = new AbortController();
		const fetch: FetchImpl = async () => {
			const sse = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join("");
			let sent = false;
			const body = abort ? new ReadableStream<Uint8Array>({
				pull(stream) {
					if (!sent) {
						sent = true;
						stream.enqueue(new TextEncoder().encode(sse));
					} else {
						controller.abort(new Error("synthetic display abort"));
						stream.error(controller.signal.reason);
					}
				},
			}) : sse;
			return new Response(body, { headers: { "content-type": "text/event-stream" } });
		};
		const options = {
			fetch, signal: controller.signal,
			apiKey: `a.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "test" } })).toString("base64url")}.b`,
			azureBaseUrl: model.baseUrl, azureApiVersion: "v1",
		};
		const context = { messages: [{ role: "user" as const, content: "Synthetic display fixture", timestamp: 0 }] };
		const stream = api === "openai-codex-responses"
			? streamOpenAICodexResponses(model as Model<"openai-codex-responses">, context, options)
			: api === "azure-openai-responses"
				? streamAzureOpenAIResponses(model as Model<"azure-openai-responses">, context, options)
				: streamOpenAIResponses(model as Model<"openai-responses">, context, options);
		const emitted: AssistantMessageEvent[] = [];
		for await (const event of stream) emitted.push(structuredClone(event));
		const result = await stream.result();
		const deltas = emitted.flatMap(event => event.type === "thinking_delta" ? [event.delta] : []);
		return { emitted, result, deltas, thinking: result.content.find(block => block.type === "thinking") };
	}
	describe(`${api} append-only thinking display`, () => {
		test("late summary supersedes buffered raw text and reaches delta consumers before end", async () => {
			const { emitted, result, deltas, thinking } = await run([
				added(), delta("opaque raw", true), done("Readable summary"), terminal(),
			]);
			expect(result.stopReason).toBe("stop");
			expect(deltas).toEqual(["Readable summary"]);
			expect(thinking?.thinking).toBe("Readable summary");
			expect(JSON.parse(thinking!.thinkingSignature!)).toEqual(reasoning("Readable summary"));
			expect(emitted.filter(event => event.type.startsWith("thinking_")).map(event => event.type)).toEqual([
				"thinking_start", "thinking_delta", "thinking_end",
			]);
		});
		test("final extension emits only the unseen suffix", async () => {
			const { deltas, thinking } = await run([added(), delta("Plan"), done("Plan carefully"), terminal()]);
			expect(deltas).toEqual(["Plan", " carefully"]);
			expect(thinking?.thinking).toBe("Plan carefully");
		});
		test("divergent authoritative text does not rewrite already emitted display", async () => {
			const { deltas, thinking } = await run([added(), delta("Original"), done("Corrected"), terminal()]);
			expect(deltas).toEqual(["Original"]);
			expect(thinking?.thinking).toBe("Original");
			expect(JSON.parse(thinking!.thinkingSignature!)).toEqual(reasoning("Corrected"));
		});
		test("raw-only completion remains readable without duplicate deltas", async () => {
			const { deltas, thinking } = await run([added(), delta("Raw ", true), delta("thinking", true), done(""), terminal()]);
			expect(deltas).toEqual(["Raw thinking"]);
			expect(thinking?.thinking).toBe("Raw thinking");
		});
		test("terminal snapshot completes a missing item.done even on token cutoff", async () => {
			const { deltas, result, thinking } = await run([added(), delta("Plan"), terminal(true, [reasoning("Plan further")])]);
			expect(result.stopReason).toBe("length");
			expect(deltas).toEqual(["Plan", " further"]);
			expect(thinking?.thinking).toBe("Plan further");
			expect(JSON.parse(thinking!.thinkingSignature!)).toEqual(reasoning("Plan further"));
		});
		test("missing item.done and terminal snapshot retains buffered raw fallback", async () => {
			const { deltas, thinking } = await run([added(), delta("Only raw", true), terminal()]);
			expect(deltas).toEqual(["Only raw"]);
			expect(thinking?.thinking).toBe("Only raw");
		});
		test("summary text done without item.done emits a late summary once", async () => {
			const { deltas, thinking } = await run([
				added(), delta("opaque raw", true),
				{
					type: "response.reasoning_summary_text.done",
					item_id: "rs_display",
					output_index: 0,
					summary_index: 0,
					text: "Late done summary",
				},
				terminal(),
			]);
			expect(deltas).toEqual(["Late done summary"]);
			expect(thinking?.thinking).toBe("Late done summary");
		});
		test("truncated transport does not publish unfinalized raw reasoning", async () => {
			const { deltas, emitted, result } = await run([added(), delta("unfinished raw", true)]);
			expect(result.stopReason).toBe("error");
			expect(deltas).toEqual([]);
			expect(emitted.some(event => event.type === "thinking_end")).toBe(false);
		});
		test("abort retains failure semantics without publishing buffered raw thinking", async () => {
			const { deltas, emitted, result } = await run([added(), delta("unfinished raw", true)], true);
			expect(result.stopReason).toBe("aborted");
			expect(deltas).toEqual([]);
			expect(emitted.some(event => event.type === "thinking_end")).toBe(false);
		});
	});
}
