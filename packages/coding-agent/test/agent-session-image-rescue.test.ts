import assert from "node:assert/strict";
import { test } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, ImageContent, Model } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { TempDir } from "@oh-my-pi/pi-utils";
import { archiveSourceText, getPreservedArchive } from "@oh-my-pi/snapcompact";
import { ModelRegistry } from "../src/config/model-registry";
import { Settings } from "../src/config/settings";
import { AgentSession } from "../src/session/agent-session";
import type { AuthStorage } from "../src/session/auth-storage";
import { convertToLlm } from "../src/session/messages";
import { buildSessionContext } from "../src/session/session-context";
import { SessionManager } from "../src/session/session-manager";
import { MemorySessionStorage } from "../src/session/session-storage";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";

interface RescueFixture {
	model: Model;
	temp: TempDir;
	manager: SessionManager;
	image: ImageContent;
	firstId: string;
	tailId: string;
	auth: AuthStorage;
	settings: Settings;
	agent: Agent;
	session: AgentSession;
}

async function createRescueFixture(tailImage: boolean, middleCharacters = 1_000_000) {
	const bundled = getBundledModel("openai", "gpt-5.5");
	assert(bundled);
	const model = { ...bundled, tokenizer: undefined, contextWindow: 100_000 };
	const temp = TempDir.createSync();
	const storage = new MemorySessionStorage();
	const file = `${temp.path()}/${crypto.randomUUID()}.jsonl`;
	storage.writeTextSync(file, "");
	const manager = await SessionManager.open(file, temp.path(), storage, { suppressBreadcrumb: true });
	const image = {
		type: "image" as const,
		mimeType: "image/png",
		data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6mQAAAABJRU5ErkJggg==",
	};
	const firstId = manager.appendMessage({
		role: "user", content: [{ type: "text", text: "A".repeat(5_000) }, { ...image }], timestamp: 1,
	});
	manager.appendMessage({ role: "user", content: "B".repeat(middleCharacters), timestamp: 2 });
	const tailId = manager.appendMessage({
		role: "user",
		content: tailImage ? [{ type: "text", text: "tail" }, { ...image }] : "tail",
		timestamp: 3,
	});
	const auth = createInMemoryAuthStorage();
	auth.setRuntimeApiKey("openai", "isolated-no-provider");
	const settings = Settings.isolated({
		"requirements.enabled": false,
		"snapcompact.shape": "8x13-bw",
		"compaction.methodOrder": ["snapcompact"],
		"compaction.keepUserMessages": true,
		"compaction.keepFirstLimit": "messages:1",
		"compaction.keepLastLimit": "tokens:0",
		"compaction.keepRecentUserMessagesLimit": "tokens:0",
		"compaction.keepUserMessagesHeuristic": false,
		"compaction.keepUserMessagesClassifierFilter": false,
		"compaction.keepUserMessagesLlm": false,
		"compaction.pruneLongUserMessages": "no",
		"compaction.autoContinue": false,
		"compaction.asyncEnabled": false,
		"compaction.keepRecentTokens": 1,
		"compaction.reserveTokens": 8192,
		"compaction.thresholdTokens": 80_000,
	});
	const agent = new Agent({
		initialState: { model, messages: manager.buildSessionContext().messages, systemPrompt: [], tools: [] },
		streamFn: () => { throw Error("No provider call is allowed in local rescue"); },
	});
	const session = new AgentSession({ agent, sessionManager: manager, settings, modelRegistry: new ModelRegistry(auth) });
	return { model, temp, manager, image, firstId, tailId, auth, settings, agent, session };
}

async function disposeRescueFixture(fixture: RescueFixture): Promise<void> {
	await fixture.session.dispose();
	fixture.auth.close();
	fixture.temp.removeSync();
}

async function runAutomaticCompaction({ session, agent, model }: RescueFixture) {
	const complete = Promise.withResolvers<void>();
	const notices: string[] = [];
	let starts = 0;
	let ends = 0;
	const unsubscribe = session.subscribe(event => {
		if (event.type === "notice") notices.push(event.message);
		if (event.type === "auto_compaction_start") starts++;
		if (event.type !== "auto_compaction_end") return;
		ends++;
		if (!event.result || event.aborted) complete.reject(Error(JSON.stringify(event)));
		else complete.resolve();
	});
	try {
		const assistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "Controlled turn complete." }],
			api: model.api, provider: model.provider, model: model.id, stopReason: "stop",
			usage: {
				input: 90_000, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 90_001,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			// Must follow any real manual compaction used to prepare the rescue input.
			timestamp: Date.now(),
		};
		agent.emitExternalEvent({ type: "message_end", message: assistant });
		agent.emitExternalEvent({ type: "agent_end", messages: [assistant] });
		await complete.promise;
		await session.waitForIdle();
		return { starts, ends, notices };
	} finally {
		unsubscribe();
	}
}

function originalImageCount({ agent, image }: RescueFixture): number {
	return convertToLlm(agent.state.messages)
		.flatMap(message => Array.isArray(message.content) ? message.content : [])
		.filter(block => block.type === "image" && block.data === image.data).length;
}

test("automatic image rescue preserves selected originals but explicit image removal overrides preservation", async () => {
	const fixture = await createRescueFixture(true);
	const { manager, firstId, tailId, session } = fixture;
	try {
		await runAutomaticCompaction(fixture);
		const first = manager.getEntry(firstId);
		const tail = manager.getEntry(tailId);
		assert(first?.type === "message" && "content" in first.message && Array.isArray(first.message.content));
		assert(tail?.type === "message" && "content" in tail.message && Array.isArray(tail.message.content));
		assert.equal(first.message.content.filter(block => block.type === "image").length, 1);
		assert.equal(tail.message.content.filter(block => block.type === "image").length, 0,
			"Automatic rescue must still remove unselected images");
		assert.equal(originalImageCount(fixture), 1);
		assert.equal((await session.dropImages()).removed, 1, "Explicit image removal overrides preservation");
		assert.equal(originalImageCount(fixture), 0);
	} finally {
		await disposeRescueFixture(fixture);
	}
}, 20_000);

test("automatic frame rescue does not replace an archive with a physically larger text spill", async () => {
	const fixture = await createRescueFixture(false);
	try {
		const observed = await runAutomaticCompaction(fixture);
		const entries = fixture.manager.getBranch().filter(entry => entry.type === "compaction");
		assert.equal(entries.length, 1, "The nonreducing rescue must not append a second compaction");
		assert(entries[0]?.warning, "The existing insufficient-headroom warning remains visible");
		assert.equal(originalImageCount(fixture), 1);
		assert.equal(observed.starts, 1);
		assert.equal(observed.ends, 1);
	} finally {
		await disposeRescueFixture(fixture);
	}
}, 20_000);

test("automatic frame rescue accepts a genuine reduction above the headroom band without changing its source", async () => {
	const fixture = await createRescueFixture(false, 400_000);
	const { manager, session, settings, agent } = fixture;
	try {
		settings.set("snapcompact.shape", "silver16-bw");
		await session.compact(undefined, { mode: "snapcompact" });
		settings.set("snapcompact.shape", "8x13-bw");
		await runAutomaticCompaction(fixture);
		const entries = manager.getBranch().filter(entry => entry.type === "compaction");
		const before = entries.at(-2);
		const after = entries.at(-1);
		assert(before && after);
		const beforeArchive = getPreservedArchive(before.preserveData);
		const afterArchive = getPreservedArchive(after.preserveData);
		assert(beforeArchive && afterArchive);
		const countOptions = { excludeEncryptedReasoning: true } as const;
		const beforeTokens = agent.tokenizer.countMessages(buildSessionContext(manager.getBranch(before.id)).messages, countOptions);
		const afterTokens = agent.tokenizer.countMessages(agent.state.messages, countOptions);
		assert(afterArchive.frames.length < beforeArchive.frames.length);
		assert(afterTokens < beforeTokens, "A genuinely smaller reconstructed context must remain admissible");
		assert.equal(archiveSourceText(afterArchive), archiveSourceText(beforeArchive));
		assert(after.warning, "Reduction does not require clearing the maintenance headroom band");
		assert.equal(originalImageCount(fixture), 1);
	} finally {
		await disposeRescueFixture(fixture);
	}
}, 20_000);
