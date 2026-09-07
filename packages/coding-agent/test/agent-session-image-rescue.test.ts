import assert from "node:assert/strict";
import { test } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "../src/config/model-registry";
import { Settings } from "../src/config/settings";
import { AgentSession } from "../src/session/agent-session";
import { convertToLlm } from "../src/session/messages";
import { SessionManager } from "../src/session/session-manager";
import { MemorySessionStorage } from "../src/session/session-storage";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";

test("automatic image rescue preserves selected originals but explicit image removal overrides preservation", async () => {
	const bundled = getBundledModel("openai", "gpt-5.5");
	assert(bundled);
	const model = { ...bundled, tokenizer: undefined, contextWindow: 100_000 };
	const storage = new MemorySessionStorage();
	const root = "/tmp/omp-image-rescue";
	const file = `${root}/${crypto.randomUUID()}.jsonl`;
	storage.writeTextSync(file, "");
	const manager = await SessionManager.open(file, root, storage, { suppressBreadcrumb: true });
	const image = {
		type: "image" as const,
		mimeType: "image/png",
		data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6mQAAAABJRU5ErkJggg==",
	};
	const firstId = manager.appendMessage({
		role: "user", content: [{ type: "text", text: "A".repeat(5_000) }, { ...image }], timestamp: 1,
	});
	manager.appendMessage({ role: "user", content: "B".repeat(1_000_000), timestamp: 2 });
	const tailId = manager.appendMessage({
		role: "user", content: [{ type: "text", text: "tail" }, { ...image }], timestamp: 3,
	});
	const auth = createInMemoryAuthStorage();
	auth.setRuntimeApiKey("openai", "isolated-no-provider");
	const settings = Settings.isolated({
		"requirements.enabled": false,
		"snapcompact.shape": "8x13-bw",
		"compaction.methodOrder": ["snapcompact"],
		"compaction.keepUserMessages": true,
		"compaction.keepFirstLimit": "messages:1",
		"compaction.keepLastLimit": "messages:0",
		"compaction.keepRecentUserMessagesLimit": "messages:0",
		"compaction.keepUserMessagesHeuristic": false,
		"compaction.keepUserMessagesClassifierFilter": false,
		"compaction.pruneLongUserMessages": "no",
		"compaction.autoContinue": false,
		"compaction.asyncEnabled": false,
		"compaction.keepRecentTokens": 1,
		"compaction.reserveTokens": 8192,
		"compaction.thresholdTokens": 80_000,
	});
	const agent = new Agent({
		initialState: { model, messages: manager.buildSessionContext().messages, systemPrompt: [], tools: [] },
		streamFn: () => { throw Error("No provider call is allowed in local image rescue"); },
	});
	const session = new AgentSession({ agent, sessionManager: manager, settings, modelRegistry: new ModelRegistry(auth) });
	try {
		const complete = Promise.withResolvers<void>();
		const unsubscribe = session.subscribe(event => {
			if (event.type !== "auto_compaction_end") return;
			if (!event.result || event.aborted) complete.reject(Error(JSON.stringify(event)));
			else complete.resolve();
		});
		const assistant = {
			role: "assistant" as const,
			content: [{ type: "text" as const, text: "Controlled turn complete." }],
			api: model.api, provider: model.provider, model: model.id, stopReason: "stop" as const,
			usage: {
				input: 90_000, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 90_001,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp: 4,
		};
		agent.emitExternalEvent({ type: "message_end", message: assistant });
		agent.emitExternalEvent({ type: "agent_end", messages: [assistant] });
		await complete.promise;
		unsubscribe();
		await session.waitForIdle();
		assert(manager.getBranch().some(entry => entry.type === "compaction"));
		const first = manager.getEntry(firstId);
		const tail = manager.getEntry(tailId);
		assert(first?.type === "message" && "content" in first.message && Array.isArray(first.message.content));
		assert(tail?.type === "message" && "content" in tail.message && Array.isArray(tail.message.content));
		assert.equal(first.message.content.filter(block => block.type === "image").length, 1);
		assert.equal(tail.message.content.filter(block => block.type === "image").length, 0,
			"Automatic rescue must still remove unselected images");
		const wire = convertToLlm(agent.state.messages);
		assert.equal(wire.flatMap(message => Array.isArray(message.content) ? message.content : [])
			.filter(block => block.type === "image" && block.data === image.data).length, 1);
		const removed = await session.dropImages();
		assert.equal(removed.removed, 1, "Explicit image removal overrides preservation");
		assert.equal(convertToLlm(agent.state.messages)
			.flatMap(message => Array.isArray(message.content) ? message.content : [])
			.filter(block => block.type === "image" && block.data === image.data).length, 0);
	} finally {
		await session.dispose();
		auth.close();
	}
}, 20_000);
