import { describe, expect, it } from "bun:test";
import type { Context, Message } from "@oh-my-pi/pi-ai";
import { bindMessageSource, cloneWithSourceOrigins, getSourceOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { decorateContextImages, inlineContextImages } from "../src/blob-broker/context-images";
import { obfuscateMessages } from "../src/secrets/message-transform";
import { SecretObfuscator } from "../src/secrets/obfuscator";
import { DateCwdReminderInjector } from "../src/session/date-cwd-reminder";
import { clampProviderContextImages, dropUnreadableContextImages } from "../src/session/provider-image-budget";
import { convertImageToPng, normalizeModelContextMessages } from "../src/utils/image-loading";

const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const model = buildModel({ id: "origin-proof", name: "Origin proof", provider: "openai", api: "openai-responses", baseUrl: "http://127.0.0.1:1", reasoning: false, input: ["text", "image"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 1024 });
function user(content: Message["content"]): Message { return { role: "user", content, timestamp: 0 } as Message; }
function parts(value: object) { const origin = getSourceOrigin(value); expect(origin?.kind).toBe("source"); if (origin?.kind !== "source") throw new Error("Missing source origin"); return origin.parts; }

describe("native SDK transform origins", () => {
	it("keeps equal-text IDs distinct through cloning and secret replacement without invented original offsets", () => {
		const secret = "UNIQUE_SECRET_12345";
		const messages = [user(secret), user([{ type: "text", text: secret }, { type: "image", data: PNG, mimeType: "image/png" }])];
		messages.forEach((message, index) => bindMessageSource(message, `u${index}`, index));
		const output = obfuscateMessages(new SecretObfuscator([{ type: "plain", content: secret }]), cloneWithSourceOrigins(messages));
		expect(JSON.stringify(output)).not.toContain(secret);
		expect(parts(output[0]!)[0]!.entryId).toBe("u0");
		expect(parts(output[1]!)[0]!.entryId).toBe("u1");
		for (const message of output) {
			const text = parts(message).find(part => part.blockIndex === 0)!;
			expect(text.coverage).toBe("full");
			expect(text.representation).toBe("transformed-text");
			expect(text.sourceSpan).toBeUndefined();
			expect(text.transportSpan).toBeUndefined();
		}
		expect(parts(output[1]!).find(part => part.blockIndex === 1)?.representation).toBe("original-image");
		expect(JSON.stringify(output)).not.toContain("entryId");
	});

	it("redacts typed provider-visible metadata without changing opaque replay state", () => {
		const secret = "FABRICATED_SECRET_92837";
		const text = `quoted \"text\" \\ Ω <|channel|> ${secret}`;
		const opaque = { encrypted_content: secret, signature: secret, image_url: `data:image/png;base64,${PNG}`, extra: { text: secret } };
		const messages: Message[] = [{
			role: "assistant", provider: "openai", model: model.id, api: "openai-responses", timestamp: 1, stopReason: "toolUse",
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			content: [
				{ type: "toolCall", id: secret, name: "computer", arguments: { text }, providerMetadata: { type: "computer", providerItemId: secret, actions: [{ type: "type", text }, { type: "keypress", keys: [secret] }], pendingSafetyChecks: [{ id: secret, code: secret, message: text }] } },
				{ type: "anthropicServerTool", block: { type: "server_tool_use", id: secret, name: "web_search", input: { query: text } } },
				{ type: "anthropicServerTool", block: { type: "web_search_tool_result", tool_use_id: secret, content: [{ type: "web_search_result", title: text, url: `https://example.invalid/${secret}`, encrypted_content: secret }] } },
			],
			providerPayload: { type: "openaiResponsesHistory", dt: true, provider: "openai", items: [{ type: "web_search_call", id: secret, status: "completed", action: { type: "search", queries: [text], sources: [{ type: "url", url: `https://example.invalid/${secret}` }] }, opaque }, { type: "reasoning", id: secret, encrypted_content: secret }] },
		}, {
			role: "toolResult", toolCallId: secret, toolName: "computer", isError: false, timestamp: 2, content: [], details: { explanation: text, opaque },
			providerMetadata: { type: "computer", screenshot: { type: "computer_screenshot", file_id: secret }, acknowledgedSafetyChecks: [{ id: secret, code: secret, message: text }] },
		}];
		messages.forEach((message, index) => bindMessageSource(message, `metadata-${index}`, index));
		const original = JSON.stringify(messages);
		const obfuscator = new SecretObfuscator([{ type: "plain", content: secret }]);
		const output = obfuscateMessages(obfuscator, messages);
		const assistant = output[0]!;
		const result = output[1]!;
		if (assistant.role !== "assistant" || result.role !== "toolResult") throw new Error("Expected original roles");
		const call = assistant.content[0]!;
		const server = assistant.content[1]!;
		const search = assistant.content[2]!;
		if (call.type !== "toolCall" || server.type !== "anthropicServerTool" || server.block.type !== "server_tool_use" || search.type !== "anthropicServerTool" || !Array.isArray(search.block.content)) throw new Error("Expected typed metadata");
		const payload = assistant.providerPayload;
		if (payload?.type !== "openaiResponsesHistory") throw new Error("Expected native replay");
		const action = payload.items[0]!.action as { queries: string[]; sources: { url: string }[] };
		const details = result.details as { explanation: string; opaque: typeof opaque };
		const visible = [call.arguments.text, call.providerMetadata!.actions[0]!.type === "type" ? call.providerMetadata!.actions[0]!.text : "", call.providerMetadata!.pendingSafetyChecks[0]!.message, server.block.input!.query, search.block.content[0].title, result.providerMetadata!.acknowledgedSafetyChecks[0]!.message, details.explanation, action.queries[0]];
		for (const value of visible) {
			expect(value).not.toContain(secret);
			expect(obfuscator.deobfuscate(value as string)).toBe(text);
		}
		expect(action.sources[0]!.url).not.toContain(secret);
		expect(search.block.content[0].url).not.toContain(secret);
		expect(call.id).toBe(secret);
		expect(call.providerMetadata!.providerItemId).toBe(secret);
		expect(call.providerMetadata!.actions[1]).toEqual({ type: "keypress", keys: [secret] });
		expect(call.providerMetadata!.pendingSafetyChecks[0]!.id).toBe(secret);
		expect(call.providerMetadata!.pendingSafetyChecks[0]!.code).toBe(secret);
		expect(search.block.content[0].encrypted_content).toBe(secret);
		expect(result.providerMetadata!.screenshot).toEqual({ type: "computer_screenshot", file_id: secret });
		expect(parts(result).find(part => part.blockIndex === "metadata.screenshot")?.representation).toBe("original-image");
		expect(parts(result.providerMetadata!.acknowledgedSafetyChecks[0]!)[0]?.representation).toBe("transformed-text");
		expect(details.opaque).toBe(opaque);
		expect(payload.items[0]!.opaque).toBe(opaque);
		expect(payload.items[1]).toEqual({ type: "reasoning", id: secret, encrypted_content: secret });
		expect(payload.origins?.[0]?.kind).toBe("source");
		expect(parts(payload.items[0]!)[0]!.representation).toBe("transformed-text");
		expect(JSON.stringify(messages)).toBe(original);
		expect(obfuscateMessages(obfuscator, output)).toBe(output);
	});

	it("keeps per-source image identity through real encoding, cached normalization and URL round trips", async () => {
		const webp = await new Bun.Image(Buffer.from(PNG, "base64")).webp().toBase64();
		const messages = [user([{ type: "image", data: webp, mimeType: "image/webp" }]), user([{ type: "image", data: webp, mimeType: "image/webp" }])];
		messages.forEach((message, index) => bindMessageSource(message, `image${index}`, index));
		const normalized = await normalizeModelContextMessages(messages, { ...model, imageInputDecoder: "stb" });
		const context = decorateContextImages({ messages: normalized }, () => "https://example.invalid/image");
		const inline = await inlineContextImages(context, async () => null);
		for (let index = 0; index < inline.messages.length; index++) {
			const content = inline.messages[index]!.content;
			if (typeof content === "string" || content[0]?.type !== "image") throw new Error("Expected normalized image");
			expect(content[0].mimeType).not.toBe("image/webp");
			expect(content[0].url).toBeUndefined();
			const png = await convertImageToPng(content[0]);
			expect(parts(png)[0]!.entryId).toBe(`image${index}`);
			expect(parts(png)[0]!.representation).toBe("original-image");
		}
	});

	it("removes only omitted image coverage in generic and native content", async () => {
		const message = user([{ type: "text", text: "kept" }, { type: "image", data: "broken", mimeType: "image/png" }]);
		bindMessageSource(message, "broken-image", 0);
		const guarded = await dropUnreadableContextImages({ messages: [message] }, model);
		expect(parts(guarded.messages[0]!).map(part => part.blockIndex)).toEqual([0]);
		const content = guarded.messages[0]!.content;
		if (typeof content === "string") throw new Error("Expected blocks");
		expect(getSourceOrigin(content[1]!)?.kind).toBe("synthetic");
		const native = { type: "message", role: "user", content: [{ type: "input_text", text: "kept" }, { type: "input_image", image_url: "data:image/png;base64,broken" }] };
		const nativeOrigin = { kind: "source" as const, parts: parts(message).map((part, index) => ({ ...part, transportBlockIndex: index })) };
		const nativeMessage = { role: "user" as const, content: "kept", timestamp: 0, providerPayload: { type: "openaiResponsesHistory" as const, items: [native], origins: [nativeOrigin] } };
		const nativeGuarded = await dropUnreadableContextImages({ messages: [nativeMessage] }, model);
		const payload = (nativeGuarded.messages[0] as typeof nativeMessage).providerPayload;
		expect(payload.origins[0]!.kind).toBe("source");
		expect(payload.origins[0]!.parts.map(part => part.blockIndex)).toEqual([0]);
		const many = user(Array.from({ length: 101 }, () => ({ type: "image" as const, data: PNG, mimeType: "image/png" })));
		bindMessageSource(many, "budget", 1);
		const clamped = clampProviderContextImages({ messages: [many] }, { ...model, provider: "anthropic" });
		const surviving = parts(clamped.messages[0]!);
		expect(surviving[0]!.blockIndex).toBeGreaterThan(0);
		expect(surviving.at(-1)!.blockIndex).toBe(100);
		expect(surviving[0]!.transportBlockIndex).toBe(0);
	});

	it("counts reminder bytes as controls, not newly authored source text", () => {
		const message = user("original");
		bindMessageSource(message, "reminder", 0);
		const context: Context = { systemPrompt: ["system"], messages: [message] };
		const injector = new DateCwdReminderInjector();
		const output = injector.transform(context, "2026-09-07", "/controlled");
		const part = parts(output.messages[0]!)[0]!;
		expect(part.sourceSpan).toEqual({ start: 0, end: 8 });
		expect(part.transportSpan!.start).toBeGreaterThan(0);
		expect((output.messages[0]!.content as string).slice(part.transportSpan!.start, part.transportSpan!.end)).toBe("original");
		const changed = injector.transform(context, "2026-09-08", "/controlled");
		expect(getSourceOrigin(changed.messages.at(-1)!)?.kind).toBe("synthetic");
	});
});
