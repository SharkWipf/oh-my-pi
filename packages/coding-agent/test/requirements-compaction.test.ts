import assert from "node:assert/strict";
import { test } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { prepareCompaction } from "@oh-my-pi/pi-agent-core/compaction";
import { type Model, type ModelSpec, registerCustomApi, unregisterCustomApis } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import type { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { resolveMethodSettings } from "@oh-my-pi/pi-coding-agent/session/compaction-methods";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { archiveSourceText, getPreservedArchive } from "@oh-my-pi/snapcompact";
import { createAssistantMessage } from "./helpers/agent-session-setup";

test("compaction retains the pending delivered source without pinning its surrounding turn when user preservation is disabled", async () => {
	const registration = `requirements-compaction-${crypto.randomUUID()}`;
	const primary = getBundledModel("anthropic", "claude-sonnet-4-5");
	assert(primary);
	const review = buildModel({
		id: "held-review", name: "Held requirements extraction", api: registration,
		provider: "requirements-compaction-fixture", baseUrl: "http://127.0.0.1:1",
		reasoning: false, input: ["text"], contextWindow: 1_000_000, maxTokens: 8192,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	} as ModelSpec) as Model;
	const modelRegistry = {
		getAll: () => [primary, review], getAvailable: () => [primary, review],
		getApiKey: async () => "isolated-key", resolver: () => async () => "isolated-key",
	} as unknown as ModelRegistry;
	const createSettings = (enabled: boolean) => Settings.isolated({
		"requirements.enabled": enabled, "memory.backend": "off", "compaction.enabled": true,
		"compaction.methodOrder": ["snapcompact"], "compaction.asyncEnabled": false,
		"compaction.keepUserMessages": false, "compaction.keepRecentTokens": 1,
		"compaction.autoContinue": false, "todo.enabled": false, "skills.enabled": false,
	});
	const settings = createSettings(true);
	for (const role of ["requirements", "requirementsEvidence", "requirementsSanity"]) {
		settings.setModelRole(role, `${review.provider}/${review.id}`);
	}
	const manager = SessionManager.inMemory("/requirements-compaction-fixture");
	const agent = new Agent({
		getApiKey: () => "isolated-key",
		initialState: { model: primary, systemPrompt: [], messages: [], tools: [] },
		convertToLlm,
	});
	const session = new AgentSession({
		agent, sessionManager: manager, settings, modelRegistry,
		rebuildSystemPrompt: async () => ({ systemPrompt: [] }),
	});
	const controlManager = SessionManager.inMemory("/requirements-compaction-control");
	const controlAgent = new Agent({
		getApiKey: () => "isolated-key",
		initialState: { model: primary, systemPrompt: [], messages: [], tools: [] },
		convertToLlm,
	});
	const controlSession = new AgentSession({
		agent: controlAgent, sessionManager: controlManager, settings: createSettings(false), modelRegistry,
		rebuildSystemPrompt: async () => ({ systemPrompt: [] }),
	});
	const started = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	let requests = 0;
	registerCustomApi(review.api, (_model, context) => {
		const stream = new AssistantMessageEventStream();
		void (async () => {
			try {
				assert.equal(context.messages[0]?.role, "user");
				requests++;
				started.resolve();
				await release.promise;
				stream.fail(new Error("Controlled extraction released after compaction"));
			} catch (error) {
				stream.fail(error instanceof Error ? error : new Error(String(error)));
			}
		})();
		return stream;
	}, registration);
	try {
		// Bracket the pending source so ordinary retention omits it instead of retaining it at an archive edge.
		const appendDiscardableTurns = (start: number, end: number) => {
			for (let turn = start; turn < end; turn++) {
				manager.appendCustomMessageEntry("fixture-context", `old-turn-${turn}`, true, undefined, "agent", turn * 4);
				manager.appendMessage({
					...createAssistantMessage(""), stopReason: "toolUse",
					content: [{ type: "toolCall", id: `old-read-${turn}`, name: "read", arguments: { path: `/old/${turn}` } }],
				});
				manager.appendMessage({
					role: "toolResult", toolCallId: `old-read-${turn}`, toolName: "read",
					content: [{ type: "text", text: "Discardable older tool source." }],
					isError: false, timestamp: turn * 4 + 2,
				});
				manager.appendMessage(createAssistantMessage(`old-turn-${turn}-done ` + "Discardable older assistant text. ".repeat(4000)));
			}
		};
		appendDiscardableTurns(0, 6);
		await session.requirements.observeCommittedSources();
		const original = "Preserve the exact UTF-8 document.";
		const deliveryId = manager.appendMessage({
			role: "user", content: [{ type: "text", text: original }], producer: { type: "human" }, timestamp: 100,
		});
		manager.appendMessage({
			...createAssistantMessage(""), stopReason: "toolUse",
			content: [{ type: "toolCall", id: "pending-read", name: "read", arguments: { path: "/pending/document" } }],
		});
		manager.appendMessage({
			role: "toolResult", toolCallId: "pending-read", toolName: "read",
			content: [{ type: "text", text: "PENDING-TOOL-RESULT" }], isError: false, timestamp: 101,
		});
		manager.appendMessage(createAssistantMessage("PENDING-TURN-COMPLETE"));
		appendDiscardableTurns(6, 12);
		manager.appendCustomMessageEntry("fixture-context", "NEWEST-ORDINARY-TAIL", true, undefined, "agent", 102);
		agent.replaceMessages(manager.buildSessionContext().messages);
		const branch = manager.getBranch();
		const ordinary = prepareCompaction(branch, resolveMethodSettings(settings.getGroup("compaction"), "snapcompact"), primary, agent.tokenizer);
		assert(ordinary);
		assert(branch.findIndex(entry => entry.id === ordinary.firstKeptEntryId) > branch.findIndex(entry => entry.id === deliveryId));
		for (const entry of branch) controlManager.ingestReplicatedEntry(structuredClone(entry));
		controlAgent.replaceMessages(controlManager.buildSessionContext().messages);
		await controlSession.compact(undefined, { mode: "snapcompact" });
		const controlEntry = controlManager.getBranch().findLast(entry => entry.type === "compaction");
		assert(controlEntry);
		const controlArchive = getPreservedArchive(controlEntry.preserveData);
		assert(controlArchive);
		// The replay consumer includes imaged source text; raw provider text alone can miss the middle.
		const controlText = archiveSourceText(controlArchive) ?? "";
		assert.equal(controlText.split(original).length - 1, 0);
		await session.requirements.acceptDelivered(deliveryId);
		await started.promise;
		const frozen = session.requirements.snapshotApplicable();
		assert.deepEqual(session.requirements.pendingLiveSnapshot().entryIds, [deliveryId]);
		assert.equal(frozen.active.length, 0);
		const result = await session.compact(undefined, { mode: "snapcompact" });
		assert.equal(result.firstKeptEntryId, ordinary.firstKeptEntryId);
		const visible = await convertToLlm(agent.state.messages);
		const compactedEntry = manager.getBranch().findLast(entry => entry.type === "compaction");
		assert(compactedEntry);
		const archive = getPreservedArchive(compactedEntry.preserveData);
		assert(archive);
		const retainedText = archiveSourceText(archive) ?? "";
		assert.equal(retainedText.split(original).length - 1, 1);
		assert(!retainedText.includes("PENDING-TOOL-RESULT"));
		assert(!retainedText.includes("PENDING-TURN-COMPLETE"));
		assert(visible.some(message => typeof message.content === "string" ? message.content.includes("NEWEST-ORDINARY-TAIL") : message.content.some(block => block.type === "text" && block.text.includes("NEWEST-ORDINARY-TAIL"))));
		assert(!visible.some(message => message.role === "toolResult" && message.toolCallId === "pending-read"));
		assert.equal(requests, 1);
		assert.equal(session.requirements.snapshotApplicable().active.length, 0);
	} finally {
		session.requirements.cancelPending("Controlled compaction completed");
		release.resolve();
		await session.dispose();
		await controlSession.dispose();
		unregisterCustomApis(registration);
	}
}, 20_000);

test("disabled requirements do not pin genuine user history during ordinary compaction", async () => {
	const registration = `requirements-disabled-compaction-${crypto.randomUUID()}`;
	const bundled = getBundledModel("anthropic", "claude-sonnet-4-5");
	assert(bundled);
	const model = { ...bundled, api: registration };
	const modelRegistry = {
		getAll: () => [model], getAvailable: () => [model],
		getApiKey: async () => "isolated-key", resolver: () => async () => "isolated-key",
	} as unknown as ModelRegistry;
	const settings = Settings.isolated({
		"requirements.enabled": false, "memory.backend": "off", "compaction.enabled": true,
		"compaction.methodOrder": ["snapcompact"], "compaction.asyncEnabled": false,
		"compaction.keepUserMessages": false, "compaction.keepRecentTokens": 1,
		"compaction.autoContinue": false, "todo.enabled": false, "skills.enabled": false,
	});
	for (const role of ["requirements", "requirementsEvidence", "requirementsSanity"]) {
		settings.setModelRole(role, `${model.provider}/${model.id}`);
	}
	const manager = SessionManager.inMemory("/requirements-disabled-compaction-fixture");
	const agent = new Agent({
		getApiKey: () => "isolated-key",
		initialState: { model, systemPrompt: [], messages: [], tools: [] },
		convertToLlm,
	});
	const session = new AgentSession({
		agent, sessionManager: manager, settings, modelRegistry,
		rebuildSystemPrompt: async () => ({ systemPrompt: [] }),
	});
	let requests = 0;
	registerCustomApi(model.api, () => {
		requests++;
		const stream = new AssistantMessageEventStream();
		stream.fail(new Error("Disabled requirements and local compaction must not call a provider"));
		return stream;
	}, registration);
	try {
		const original = "OLD-GENUINE-ORIGINAL\n" + "Earlier source detail. ".repeat(20_000);
		const oldDeliveryId = manager.appendMessage({
			role: "user", content: [{ type: "text", text: original }], producer: { type: "human" }, timestamp: 1,
		});
		manager.appendMessage(createAssistantMessage("Completed the older turn."));
		const recent = "RECENT-GENUINE-ORIGINAL";
		manager.appendMessage({
			role: "user", content: [{ type: "text", text: recent }], producer: { type: "human" }, timestamp: 2,
		});
		agent.replaceMessages(manager.buildSessionContext().messages);
		const ordinary = prepareCompaction(manager.getBranch(), resolveMethodSettings(settings.getGroup("compaction"), "snapcompact"), model, agent.tokenizer);
		assert(ordinary);
		const result = await session.compact(undefined, { mode: "snapcompact" });
		assert.equal(result.firstKeptEntryId, ordinary.firstKeptEntryId);
		assert.notEqual(result.firstKeptEntryId, oldDeliveryId);
		const visible = await convertToLlm(agent.state.messages);
		assert.equal(visible.filter(message => message.role === "user" && JSON.stringify(message.content).includes(recent)).length, 1);
		assert.equal(requests, 0);
	} finally {
		await session.dispose();
		unregisterCustomApis(registration);
	}
}, 20_000);
