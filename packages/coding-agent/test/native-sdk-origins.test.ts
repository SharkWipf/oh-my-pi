import { describe, expect, it } from "bun:test";
import type { Context, Message } from "@oh-my-pi/pi-ai";
import { bindMessageSource, cloneWithSourceOrigins, getSourceOrigin, setSourceOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";
import { clearCustomApis, registerCustomApi } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { TempDir } from "@oh-my-pi/pi-utils";
import { ModelRegistry } from "../src/config/model-registry";
import { Settings } from "../src/config/settings";
import { createAgentSession } from "../src/sdk";
import { AuthStorage } from "../src/session/auth-storage";
import { SessionManager } from "../src/session/session-manager";
import { createAssistantMessage } from "./helpers/agent-session-setup";
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

it("honors actual SDK context and provider hooks once, without retaining unsupported attribution", async () => {
	using directory = TempDir.createSync("native-sdk-origins-");
	const auth = await AuthStorage.create(directory.join("auth.db"));
	const api = "native-origin-hook-proof";
	const requestModel = buildModel({ ...model, api, provider: "native-origin-hook-proof" });
	auth.setRuntimeApiKey(requestModel.provider, "local-only");
	let contextCalls = 0;
	let payloadCalls = 0;
	const observed: Array<{ context: Context; payload: unknown }> = [];
	registerCustomApi(api, (_model, context, options) => {
		const stream = new AssistantMessageEventStream();
		void (async () => {
			const payload = { input: [{ role: "user", content: "original" }] };
			setSourceOrigin(payload.input[0]!, { kind: "source", parts: [{ entryId: "wire", order: 0, blockIndex: 0, coverage: "full", representation: "native" }] });
			const result = await options?.onPayload?.(payload, requestModel);
			observed.push({ context, payload: result ?? payload });
			const message = createAssistantMessage("ok");
			stream.push({ type: "done", reason: "stop", message });
		})();
		return stream;
	});
	const { session } = await createAgentSession({
		cwd: directory.path(), agentDir: directory.path(), sessionManager: SessionManager.inMemory(directory.path()),
		authStorage: auth, modelRegistry: new ModelRegistry(auth, directory.join("models.yml")),
		settings: Settings.isolated({ "compaction.enabled": false }), model: requestModel, disableExtensionDiscovery: true,
		extensions: [pi => {
			pi.on("context", event => {
				contextCalls++;
				const target = event.messages.find(message => message.role === "user")!;
				bindMessageSource(target as Message, "hook-input", 0);
				if (target.role === "user") target.content = "context-authority";
			});
			pi.on("before_provider_request", event => {
				payloadCalls++;
				const payload = event.payload as { input: Array<{ content: string }> };
				if (payloadCalls === 1) payload.input[0]!.content = "in-place-authority";
				else return { input: [{ content: "replacement-authority" }] };
			});
		}],
		skills: [], contextFiles: [], promptTemplates: [], slashCommands: [], enableMCP: false, enableLsp: false, skipPythonPreflight: true, taskDepth: 1, agentId: "SubAgent",
	});
	try {
		await session.sendUserMessage("first");
		await session.sendUserMessage("second");
		expect([contextCalls, payloadCalls]).toEqual([2, 2]);
		for (const [index, result] of observed.entries()) {
			const user = result.context.messages.find(message => message.role === "user")!;
			expect(JSON.stringify(user.content)).toContain("context-authority");
			expect(getSourceOrigin(user)?.kind).toBe("unknown");
			const item = (result.payload as { input: Array<{ content: string }> }).input[0]!;
			expect(item.content).toBe(index === 0 ? "in-place-authority" : "replacement-authority");
			expect(getSourceOrigin(item)?.kind).toBe("unknown");
		}
	} finally { await session.dispose(); auth.close(); clearCustomApis(); }
});
