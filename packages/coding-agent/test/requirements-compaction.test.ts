import assert from "node:assert/strict";
import { test } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { getCompactionSourceRepresentation, prepareCompaction } from "@oh-my-pi/pi-agent-core/compaction";
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
import { createAssistantMessage } from "./helpers/agent-session-setup";

test("compaction retains the complete pending original turn while extraction is in flight and user preservation is disabled", async () => {
	const registration = `requirements-compaction-${crypto.randomUUID()}`;
	const primary = getBundledModel("anthropic", "claude-sonnet-4-5");
	assert(primary);
	const review = buildModel({
		id: "held-review", name: "Held requirements extraction", api: registration,
		provider: "requirements-compaction-fixture", baseUrl: "http://127.0.0.1:1",
		reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 8192,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	} as ModelSpec) as Model;
	const modelRegistry = {
		getAll: () => [primary, review], getAvailable: () => [primary, review],
		getApiKey: async () => "isolated-key", resolver: () => async () => "isolated-key",
	} as unknown as ModelRegistry;
	const settings = Settings.isolated({
		"requirements.enabled": true, "memory.backend": "off", "compaction.enabled": true,
		"compaction.methodOrder": ["snapcompact"], "compaction.asyncEnabled": false,
		"compaction.keepUserMessages": false, "compaction.keepRecentTokens": 1,
		"compaction.autoContinue": false, "todo.enabled": false, "skills.enabled": false,
	});
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
		// Synthetic context supplies older turns without inventing unresolved authored sources.
		for (let turn = 0; turn < 6; turn++) {
			manager.appendCustomMessageEntry("fixture-context", `old-turn-${turn}`, true, undefined, "agent", turn * 4);
			manager.appendMessage({
				...createAssistantMessage(""), stopReason: "toolUse",
				content: [{ type: "toolCall", id: `old-read-${turn}`, name: "read", arguments: { path: `/old/${turn}` } }],
			});
			manager.appendMessage({
				role: "toolResult", toolCallId: `old-read-${turn}`, toolName: "read",
				content: [{ type: "text", text: "Discardable older tool source. ".repeat(4000) }],
				isError: false, timestamp: turn * 4 + 2,
			});
			manager.appendMessage(createAssistantMessage(`old-turn-${turn}-done`));
		}
		await session.requirements.observeCommittedSources();
		const original = "Preserve the exact UTF-8 document.";
		const captureId = await manager.captureRequirementsInput(original);
		const deliveryId = manager.appendMessage({
			role: "user", content: [{ type: "text", text: original }], sourceCaptureId: captureId, timestamp: 100,
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
		manager.appendCustomMessageEntry("fixture-context", "NEWEST-ORDINARY-TAIL", true, undefined, "agent", 102);
		agent.replaceMessages(manager.buildSessionContext().messages);
		const branch = manager.getBranch();
		const ordinary = prepareCompaction(branch, resolveMethodSettings(settings.getGroup("compaction"), "snapcompact"), primary, agent.tokenizer);
		assert(ordinary);
		assert(branch.findIndex(entry => entry.id === ordinary.firstKeptEntryId) > branch.findIndex(entry => entry.id === deliveryId));
		await session.requirements.scheduleCommittedSources();
		await started.promise;
		const frozen = session.requirements.snapshotApplicable();
		assert.deepEqual(session.requirements.pendingSourceVisibility(frozen).entryIds, [deliveryId]);
		assert.equal(frozen.active.length, 0);
		const result = await session.compact(undefined, { mode: "snapcompact" });
		assert.equal(result.firstKeptEntryId, deliveryId);
		assert.notEqual(result.firstKeptEntryId, captureId);
		const visible = agent.state.messages;
		assert.equal(visible.filter(message => message.role === "user" && JSON.stringify(message.content).includes(original)).length, 1);
		assert(visible.some(message => message.role === "assistant" && message.content.some(block => block.type === "toolCall" && block.id === "pending-read")));
		assert(visible.some(message => message.role === "toolResult" && message.toolCallId === "pending-read"));
		assert(visible.some(message => message.role === "assistant" && JSON.stringify(message.content).includes("PENDING-TURN-COMPLETE")));
		const representation = getCompactionSourceRepresentation(result.preserveData);
		assert(representation);
		assert(!representation.coverage.some(run => run.entryId === deliveryId || run.entryId === captureId));
		assert.equal(requests, 1);
		assert.equal(session.requirements.snapshotApplicable().active.length, 0);
	} finally {
		session.requirements.cancelPending("Controlled compaction completed");
		release.resolve();
		await session.dispose();
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
		const oldCaptureId = await manager.captureRequirementsInput(original);
		const oldDeliveryId = manager.appendMessage({
			role: "user", content: [{ type: "text", text: original }], sourceCaptureId: oldCaptureId, timestamp: 1,
		});
		manager.appendMessage(createAssistantMessage("Completed the older turn."));
		const recent = "RECENT-GENUINE-ORIGINAL";
		const recentCaptureId = await manager.captureRequirementsInput(recent);
		manager.appendMessage({
			role: "user", content: [{ type: "text", text: recent }], sourceCaptureId: recentCaptureId, timestamp: 2,
		});
		agent.replaceMessages(manager.buildSessionContext().messages);
		const ordinary = prepareCompaction(manager.getBranch(), resolveMethodSettings(settings.getGroup("compaction"), "snapcompact"), model, agent.tokenizer);
		assert(ordinary);
		const result = await session.compact(undefined, { mode: "snapcompact" });
		assert.equal(result.firstKeptEntryId, ordinary.firstKeptEntryId);
		assert.notEqual(result.firstKeptEntryId, oldDeliveryId);
		const representation = getCompactionSourceRepresentation(result.preserveData);
		assert(representation);
		assert(representation.coverage.some(run => run.entryId === oldDeliveryId));
		assert(!agent.state.messages.some(message => message.role === "user" && JSON.stringify(message.content).includes("OLD-GENUINE-ORIGINAL")));
		assert.equal(agent.state.messages.filter(message => message.role === "user" && JSON.stringify(message.content).includes(recent)).length, 1);
		assert.equal(requests, 0);
	} finally {
		await session.dispose();
		unregisterCustomApis(registration);
	}
}, 20_000);
