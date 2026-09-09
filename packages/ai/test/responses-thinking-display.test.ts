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
const envelope = (text = "Readable reasoning") => JSON.stringify({
	codex_json: true, codex_event_type: "item.completed", codex_item_type: "reasoning", text,
});
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
	async function run(events: unknown[], abort = false, cutoff = false, wireSuffix = "") {
		const controller = new AbortController();
		const fetch: FetchImpl = async () => {
			const sse = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join("") + wireSuffix;
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
			onPayload: (payload: unknown) => {
				if (cutoff && payload && typeof payload === "object") {
					Object.assign(payload, { stream_options: { reasoning_summary_delivery: "sequential_cutoff" } });
				}
			},
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
		test("decodes buffered fallback envelopes when item.done and terminal snapshots are absent", async () => {
			for (const raw of [true, false]) {
				const { deltas, thinking } = await run([added(), delta(envelope().slice(0, 1), raw), delta(envelope().slice(1), raw), terminal()]);
				expect(deltas).toEqual(["Readable reasoning"]);
				expect(thinking?.thinking).toBe("Readable reasoning");
			}
		});
		test("preserves clean-summary precedence over raw envelopes and decodes terminal summary snapshots", async () => {
			const preferred = await run([added(), delta(envelope("Raw detail"), true), done("Clean summary"), terminal()]);
			expect(preferred.deltas).toEqual(["Clean summary"]);
			const snapshot = await run([added(), terminal(true, [reasoning(envelope())])]);
			expect(snapshot.result.stopReason).toBe("length");
			expect(snapshot.deltas).toEqual(["Readable reasoning"]);
		});
		if (api === "openai-codex-responses") {
			test("decodes atomic cutoff summaries and separate buffered envelope parts", async () => {
				const atomic = await run([added(),
					{ type: "response.reasoning_summary_text.done", item_id: "rs_display", summary_index: 0, text: envelope() },
					done(envelope("Readable reasoning extended")), terminal()], false, true);
				expect(atomic.deltas).toEqual(["Readable reasoning", " extended"]);
				const fallback = await run([added(), delta(envelope("First")),
					{ type: "response.reasoning_summary_part.done", item_id: "rs_display", summary_index: 0 },
					delta(envelope("Second")), done(""), terminal()], false, true);
				expect(fallback.deltas.join("")).toBe("First\n\nSecond");
			});
		}
		test("keeps envelope-shaped assistant message text literal", async () => {
			const item = { type: "message", id: "msg_literal", role: "assistant", status: "completed", content: [{ type: "output_text", text: envelope(), annotations: [] }] };
			const { result, emitted } = await run([
				{ type: "response.output_item.added", output_index: 0, item: { ...item, content: [] } },
				{ type: "response.output_text.delta", output_index: 0, item_id: item.id, content_index: 0, delta: envelope() },
				{ type: "response.output_item.done", output_index: 0, item }, terminal(),
			]);
			expect(result.content.find(block => block.type === "text")?.text).toBe(envelope());
			expect(emitted.flatMap(event => event.type === "text_delta" ? [event.delta] : []).join("")).toBe(envelope());
		});
		test("decodes raw-only envelopes without rewriting the replay item", async () => {
			const item = { ...reasoning(), content: [{ type: "reasoning_text", text: envelope() }] };
			const { result, deltas, thinking } = await run([added(), delta(envelope().slice(0, 20), true), delta(envelope().slice(20), true),
				{ type: "response.output_item.done", output_index: 0, item }, terminal()]);
			expect(result.stopReason).toBe("stop");
			expect(deltas).toEqual(["Readable reasoning"]);
			expect(JSON.parse(thinking!.thinkingSignature!)).toEqual(item);
		});
		test("decodes final summary wrappers as append-only continuations", async () => {
			const { deltas, thinking } = await run([added(), delta("Readable"), done(envelope()), terminal()]);
			expect(deltas).toEqual(["Readable", " reasoning"]);
			expect(JSON.parse(thinking!.thinkingSignature!)).toEqual(reasoning(envelope()));
		});
		test("buffers split summary wrappers through text.done without leaking prefixes", async () => {
			const { deltas, thinking } = await run([added(), ...Array.from(envelope()).map(text => delta(text)),
				{ type: "response.reasoning_summary_text.done", item_id: "rs_display", output_index: 0, summary_index: 0, text: envelope() },
				done(envelope()), terminal()]);
			expect(deltas).toEqual(["Readable reasoning"]);
			expect(thinking?.thinking).toBe("Readable reasoning");
		});
		test("retains literal and malformed JSON instead of guessing envelopes", async () => {
			for (const text of [
				JSON.stringify({ text: "literal", type: "reasoning" }),
				envelope().replace("true", "false"),
				envelope().replace("item.completed", "item.started"),
				envelope().replace('reasoning"', 'message"'),
				envelope().slice(0, -1),
				"Example: " + envelope(),
				"\x60\x60\x60json\n" + envelope() + "\n\x60\x60\x60",
				JSON.stringify({ ...JSON.parse(envelope()), literal: true }),
			]) {
				const { deltas, thinking } = await run([added(), delta(text.slice(0, 1)), delta(text.slice(1)), done(text), terminal()]);
				expect(deltas.join("")).toBe(text);
				expect(thinking?.thinking).toBe(text);
			}
		});
		test("does not expose uncompleted envelopes on malformed SSE or abort", async () => {
			for (const abort of [false, true]) {
				const { result, deltas } = await run(
					[added(), delta(envelope().slice(0, -1))], abort, false, abort ? "" : "data: {invalid JSON}\n\n",
				);
				expect(result.stopReason).toBe(abort ? "aborted" : "error");
				expect(deltas).toEqual([]);
			}
		}, 20_000);
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
			expect(emitted.filter(event => event.type === "thinking_end").map(event => event.content).join("")).toBe("");
		}, 20_000);
		test("abort retains failure semantics without publishing buffered raw thinking", async () => {
			const { deltas, emitted, result } = await run([added(), delta("unfinished raw", true)], true);
			expect(result.stopReason).toBe("aborted");
			expect(deltas).toEqual([]);
			expect(emitted.some(event => event.type === "thinking_end")).toBe(false);
		});
	});
}
