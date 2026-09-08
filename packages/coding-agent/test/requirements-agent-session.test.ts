import { afterEach, expect, test } from "bun:test";
import { readFile, unlink, writeFile } from "node:fs/promises";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import {
	type AssistantMessage,
	type Context,
	type ImageContent,
	type Model,
	type ModelSpec,
	registerCustomApi,
	type SimpleStreamOptions,
	unregisterCustomApis,
} from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import type { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AgentStorage } from "@oh-my-pi/pi-coding-agent/session/agent-storage";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { FileSessionStorage } from "@oh-my-pi/pi-coding-agent/session/session-storage";
import {
	getInlineFrameAccounting,
	getInlineTextAccounting,
	SnapcompactInlineTransformer,
} from "@oh-my-pi/pi-coding-agent/session/snapcompact-inline";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage } from "./helpers/agent-session-setup";

const SOURCE = "requirements-agent-session-regressions";
const BASE = "Base runtime instructions; configured context files remain enabled.";
const ORDINARY_TRANSFORM = "Configured extension transform remains active.";
const sessions = new Set<AgentSession>();

interface ProviderCall {
	context: Context;
	sessionId: string | undefined;
}

interface Harness {
	agent: Agent;
	session: AgentSession;
	manager: SessionManager;
	calls: ProviderCall[];
	dispatchedModels: Model[];
	model: Model;
	modelRegistry: ModelRegistry;
	settings: Settings;
	ordinaryTransform: { calls: number; requirements: string[][] };
}

function model(api: string, id: string, contextWindow = 32768, vision = false): Model {
	return buildModel({
		id,
		name: id,
		api,
		provider: "requirements-agent-fixture",
		baseUrl: "http://127.0.0.1:1",
		reasoning: false,
		input: vision ? ["text", "image"] : ["text"],
		contextWindow,
		maxTokens: 8192,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	} as ModelSpec) as Model;
}

function registerBoundary(
	api: string,
	respond: (context: Context, options?: SimpleStreamOptions) => AssistantMessage | Promise<AssistantMessage>,
	onDispatch?: (model: Model) => void,
): void {
	const provider = createMockModel({
		handler: async (context, options) => {
			const message = await respond(context, options);
			return {
				content: message.content.filter(block => block.type === "text" || block.type === "toolCall"),
				stopReason: message.stopReason,
			};
		},
	});
	registerCustomApi(
		api,
		(model, context, options) => {
			onDispatch?.(model);
			return provider.stream(model, context, options);
		},
		SOURCE,
	);
}

function createHarness(
	options: {
		tool?: AgentTool;
		respond?: (context: Context, call: number) => AssistantMessage;
		reuse?: Harness;
		contextWindow?: number;
		vision?: boolean;
		dialect?: "glm";
		physical?: (context: Context, model: Model) => Context | Promise<Context>;
		manager?: SessionManager;
		agentStorage?: AgentStorage;
	} = {},
): Harness {
	const calls: ProviderCall[] = [];
	const dispatchedModels: Model[] = [];
	const primary =
		options.reuse?.model ??
		model(`requirements-primary-${crypto.randomUUID()}`, "primary", options.contextWindow, options.vision);
	const sanity = model(`requirements-sanity-${crypto.randomUUID()}`, "sanity");
	registerBoundary(
		primary.api,
		(context, streamOptions) => {
			calls.push({
				context: {
					...context,
					systemPrompt: context.systemPrompt?.slice(),
					messages: structuredClone(context.messages),
				},
				sessionId: streamOptions?.sessionId,
			});
			return (
				options.respond?.(context, calls.length) ?? createAssistantMessage("Completed the requested finite action.")
			);
		},
		model => {
			dispatchedModels.push(model);
		},
	);
	registerBoundary(sanity.api, context => {
		const message = context.messages[0];
		if (message?.role !== "user" || typeof message.content === "string" || message.content[0]?.type !== "text") {
			throw new Error("Expected candidate-only sanity payload");
		}
		const payload = JSON.parse(message.content[0].text) as { candidates: { id: string }[] };
		return createAssistantMessage(
			JSON.stringify({
				candidates: payload.candidates.map(candidate => ({
					id: candidate.id,
					decision: "pass",
					reason: "Finite literal requirement accepted by controlled independent reviewer",
				})),
			}),
		);
	});
	const modelRegistry = {
		getAll: () => [primary, sanity],
		getAvailable: () => [primary, sanity],
		getApiKey: async () => "isolated-key",
		resolver: () => async () => "isolated-key",
	} as unknown as ModelRegistry;
	const settings = Settings.isolated(
		{
			"requirements.enabled": true,
			"memory.backend": "off",
			"compaction.enabled": false,
			"todo.enabled": false,
			"skills.enabled": false,
			"compaction.reserveTokens": 0,
		},
		{ storage: options.agentStorage },
	);
	settings.setModelRole("requirementsSanity", `${sanity.provider}/${sanity.id}`);
	const manager = options.manager ?? SessionManager.inMemory("/requirements-agent-session-fixture");
	const ordinaryTransform = options.reuse?.ordinaryTransform ?? { calls: 0, requirements: [] };
	const agent =
		options.reuse?.agent ??
		new Agent({
			getApiKey: () => "isolated-key",
			initialState: {
				model: primary,
				systemPrompt: [BASE],
				tools: options.tool ? [options.tool] : [],
				messages: [],
			},
			convertToLlm,
			dialect: options.dialect,
			transformProviderContext: async (context, currentModel) => {
				ordinaryTransform.calls++;
				ordinaryTransform.requirements.push(requirementStatements({ context, sessionId: undefined }));
				const prepared = { ...context, systemPrompt: [...(context.systemPrompt ?? []), ORDINARY_TRANSFORM] };
				return options.physical ? options.physical(prepared, currentModel) : prepared;
			},
		});
	if (options.reuse) agent.reset();
	const session = new AgentSession({
		agent,
		sessionManager: manager,
		settings,
		modelRegistry,
		toolRegistry: options.tool ? new Map([[options.tool.name, options.tool]]) : undefined,
		rebuildSystemPrompt: async () => ({ systemPrompt: [BASE] }),
		getMemoryRecoveryContextFiles: () => ["fixture/AGENTS.md"],
	});
	sessions.add(session);
	return {
		agent,
		session,
		manager,
		calls,
		dispatchedModels,
		model: primary,
		modelRegistry,
		settings,
		ordinaryTransform,
	};
}

async function capture(harness: Harness, text: string) {
	const entryId = harness.manager.appendMessage({ role: "user", content: [{ type: "text", text }], timestamp: Date.now(), producer: { type: "human" } });
	await harness.session.requirements.observeCommittedSources();
	const descriptor = harness.manager.getRequirementsSource(entryId);
	if (!descriptor) throw new Error("Committed original source not indexed");
	const { source } = await harness.session.requirements.inspectSource(descriptor.key);
	return { source, captureId: entryId, entryId };
}

async function adopt(harness: Harness, text: string) {
	const captured = await capture(harness, text);
	await harness.session.requirements.applyOperatorAction({
		kind: "literal-adopt",
		sourceKey: captured.source.key,
		unitId: captured.source.units[0]!.id,
		scope: { kind: "session", sessionId: harness.manager.getSessionId(), epoch: captured.source.epoch },
	});
	expect(harness.session.requirements.snapshotApplicable().active.map(item => item.statement)).toContain(text);
	return captured;
}

function requirementStatements(call: ProviderCall): string[] {
	for (const segment of call.context.systemPrompt ?? []) {
		try {
			const start = segment.indexOf("{");
			if (start < 0) continue;
			const payload = JSON.parse(segment.slice(start)) as {
				requirements?: { statement: string }[];
			};
			if (Array.isArray(payload.requirements)) return payload.requirements.map(item => item.statement);
		} catch {
			// Ordinary system segments are not requirements payloads.
		}
	}
	return [];
}

afterEach(async () => {
	for (const session of sessions) await session.dispose();
	sessions.clear();
	unregisterCustomApis(SOURCE);
});

for (const actionKind of ["literal-adopt", "restore", "extract"] as const) {
	test("STOP during " + actionKind + " evidence preparation cannot dispatch or authorize a later stage", async () => {
		const harness = createHarness();
		const owner = harness.session.requirements;
		const text = "Preserve the bounded operator decision after a stopped review.";
		const captured = actionKind === "restore" ? await adopt(harness, text) : await capture(harness, text);
		const revisionId = owner.snapshotApplicable().active[0]?.id;
		if (revisionId) await owner.applyOperatorAction({ kind: "quarantine", revisionIds: [revisionId], reason: "Review before restoring" });
		const reviewer = harness.modelRegistry.getAll().find(candidate => candidate.id === "sanity")!;
		harness.settings.setModelRole("requirementsExtraction", reviewer.provider + "/" + reviewer.id);
		harness.settings.setModelRole("requirementsEvidence", reviewer.provider + "/" + reviewer.id);
		let modelCalls = 0;
		for (const candidate of harness.modelRegistry.getAll()) registerBoundary(candidate.api, () => {
			modelCalls++;
			throw new Error("A stopped requirements action dispatched a model");
		});
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const inspect = owner.inspectSource.bind(owner);
		owner.inspectSource = async (key, context) => {
			const resolved = await inspect(key, context);
			if (context) { entered.resolve(); await release.promise; }
			return resolved;
		};
		const operation = actionKind === "extract" ? owner.processPending(captured.source.key) : owner.applyOperatorAction(
			actionKind === "restore"
				? { kind: "restore", revisionIds: [revisionId!], literalUnitId: captured.source.units[0]!.id }
				: { kind: "literal-adopt", sourceKey: captured.source.key, unitId: captured.source.units[0]!.id,
					scope: { kind: "session", sessionId: harness.manager.getSessionId(), epoch: captured.source.epoch } },
		);
		const settled = operation.catch(error => error);
		try {
			await entered.promise;
			await harness.session.dispose();
			const stopped = owner.status({ includeLedger: true }).snapshot;
			release.resolve();
			expect(await settled).toBeInstanceOf(Error);
			expect(modelCalls).toBe(0);
			const after = owner.status({ includeLedger: true }).snapshot;
			expect(after.state.owners).toEqual(stopped.state.owners);
			expect(after.batches).toEqual(stopped.batches);
			expect(after.revisions).toEqual(stopped.revisions);
		} finally { release.resolve(); await settled; }
	});
}

test("STOP aborts the in-flight sanity stage and ignores its later successful response", async () => {
	const harness = createHarness();
	const owner = harness.session.requirements;
	const captured = await capture(harness, "Keep this rule inactive when its review is stopped.");
	const reviewer = harness.modelRegistry.getAll().find(candidate => candidate.id === "sanity")!;
	const entered = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	let stageSignal: AbortSignal | undefined;
	let modelCalls = 0;
	registerBoundary(reviewer.api, async (context, options) => {
		modelCalls++;
		stageSignal = options?.signal;
		entered.resolve();
		await release.promise;
		const content = context.messages[0]!.content;
		if (typeof content === "string" || content[0]?.type !== "text") throw new Error("Expected sanity input");
		const payload = JSON.parse(content[0].text) as { candidates: { id: string }[] };
		return createAssistantMessage(JSON.stringify({ candidates: payload.candidates.map(candidate => ({ id: candidate.id, decision: "pass", reason: "Accepted exact finite rule" })) }));
	});
	const settled = owner.applyOperatorAction({ kind: "literal-adopt", sourceKey: captured.source.key,
		unitId: captured.source.units[0]!.id,
		scope: { kind: "session", sessionId: harness.manager.getSessionId(), epoch: captured.source.epoch },
	}).catch(error => error);
	try {
		await entered.promise;
		expect(stageSignal?.aborted).toBe(false);
		await harness.session.dispose();
		expect(stageSignal?.aborted).toBe(true);
		release.resolve();
		expect(await settled).toBeInstanceOf(Error);
		expect(modelCalls).toBe(1);
		expect(owner.status({ includeLedger: true }).snapshot.revisions).toEqual([]);
	} finally { release.resolve(); await settled; }
});

test("literal adoption of unknown provenance creates new operator authority without laundering the original", async () => {
	const harness = createHarness();
	const text = "Keep the exact operand TLS_AES_256_GCM_SHA384 in the final result.";
	const entryId = harness.manager.appendMessage({ role: "user", content: text, timestamp: 1 });
	await harness.session.requirements.observeCommittedSources();
	const descriptor = harness.manager.getRequirementsSource(entryId)!;
	const selected = await harness.session.requirements.inspectSource(descriptor.key);
	expect(harness.session.requirements.snapshotApplicable().active).toEqual([]);
	const result = await harness.session.requirements.applyOperatorAction({
		kind: "literal-adopt", sourceKey: descriptor.key, unitId: selected.units[0]!.id,
		scope: { kind: "session", sessionId: harness.manager.getSessionId(), epoch: harness.manager.getRequirementsEpoch() },
	});
	expect(result).toMatchObject({ status: "accepted" });
	const active = harness.session.requirements.snapshotApplicable().active;
	expect(active.map(revision => revision.statement)).toEqual([text]);
	expect(active[0]!.sourceKey).not.toBe(descriptor.key);
	expect(active[0]!.referents).toEqual([{ sourceKey: descriptor.key, integrity: selected.source.integrity, unitId: selected.units[0]!.id }]);
	expect((await harness.session.requirements.inspectSource(descriptor.key)).source.origin.kind).toBe("unknown");
	await harness.session.prompt("Execute the bounded action.", { synthetic: true });
	await harness.session.waitForIdle();
	expect(requirementStatements(harness.calls[0]!)).toEqual([text]);
});

test("refreshes active requirements at every provider call, including a tool follow-up after operator publication", async () => {
	const entered = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	const tool: AgentTool = {
		name: "inspect_boundary",
		label: "Inspect boundary",
		description: "Read the finite controlled boundary result.",
		parameters: type({}),
		execute: async () => {
			entered.resolve();
			await release.promise;
			return { content: [{ type: "text", text: "Boundary inspected." }] };
		},
	};
	const harness = createHarness({
		tool,
		respond: (_context, call) =>
			call === 1
				? {
						...createAssistantMessage(""),
						stopReason: "toolUse",
						content: [{ type: "toolCall", id: "inspect-1", name: tool.name, arguments: {} }],
					}
				: createAssistantMessage("Inspection complete."),
	});
	const first = "Keep UTF-8 bytes unchanged in the exported result.";
	const second = "Include the exact checksum in the final response.";
	await adopt(harness, first);
	const run = harness.session.prompt("Inspect the boundary.", { synthetic: true });
	try {
		await Promise.race([
			entered.promise,
			run.then(() => {
				throw new Error(`Turn ended before tool execution: ${JSON.stringify(harness.agent.state.messages)}`);
			}),
		]);
		await adopt(harness, second);
	} finally {
		release.resolve();
	}
	await run;
	await harness.session.waitForIdle();
	expect(harness.calls).toHaveLength(2);
	expect(requirementStatements(harness.calls[0]!)).toEqual([first]);
	expect(requirementStatements(harness.calls[1]!)).toEqual([first, second]);
	expect(
		harness.calls[1]!.context.messages.some(
			message => message.role === "toolResult" && message.toolCallId === "inspect-1",
		),
	).toBe(true);
});

test("unregisters the disposed requirements owner when the same Agent is adopted by a new session", async () => {
	const first = createHarness();
	const oldRequirement = "The original session must use the cyan report header.";
	await adopt(first, oldRequirement);
	await first.session.prompt("Finish this session.", { synthetic: true });
	await first.session.waitForIdle();
	expect(requirementStatements(first.calls[0]!)).toEqual([oldRequirement]);
	await first.session.dispose();
	sessions.delete(first.session);
	const disposedReceipt = first.session.requirements.status().receipt;
	expect(first.ordinaryTransform.calls).toBe(1);

	const second = createHarness({ reuse: first });
	const newRequirement = "The replacement session must use the amber report header.";
	await adopt(second, newRequirement);
	await second.session.prompt("Begin the replacement session.", { synthetic: true });
	await second.session.waitForIdle();
	expect(second.calls).toHaveLength(1);
	expect(requirementStatements(second.calls[0]!)).toEqual([newRequirement]);
	expect(JSON.stringify(second.calls[0]!.context)).not.toContain(oldRequirement);
	expect(first.session.requirements.status().receipt).toEqual(disposedReceipt);
	expect(second.ordinaryTransform.calls).toBe(2);
	const system = second.calls[0]!.context.systemPrompt!;
	expect(system.filter(part => part === ORDINARY_TRANSFORM)).toEqual([ORDINARY_TRANSFORM]);
	expect(second.ordinaryTransform.requirements).toEqual([[oldRequirement], [newRequirement]]);
});

test("future bypass preserves contaminated history and warns; clean recovery rotates state and retains only selected original identity", async () => {
	const harness = createHarness();
	const sameText = "Keep only the explicitly selected original input.";
	const unselectedTwin = await adopt(harness, sameText);
	const wanted = await capture(harness, sameText);
	const unwanted = await capture(harness, "This unselected human input must not enter the clean retry.");
	const poison = "POISONED_ASSISTANT_TRANSCRIPT_DO_NOT_REPLAY";
	const contaminated = createAssistantMessage(poison);
	harness.manager.appendMessage(contaminated);
	harness.agent.appendMessage(contaminated);
	const notices: string[] = [];
	harness.session.subscribe(event => {
		if (event.type === "notice") notices.push(event.message);
	});
	const previousIdentity = harness.session.sessionId;
	let providerClosed = false;
	harness.session.providerSessionState.set("controlled-history", {
		close: () => {
			providerClosed = true;
		},
	});

	harness.session.bypassRequirementsForRun();
	expect(JSON.stringify(harness.agent.state.messages)).toContain(poison);
	expect(harness.session.sessionId).toBe(previousIdentity);
	expect(providerClosed).toBe(false);

	await harness.session.prompt("Continue with future-only bypass.", { synthetic: true });
	await harness.session.waitForIdle();
	const bypassedCall = harness.calls.at(-1)!;
	expect(requirementStatements(bypassedCall)).toEqual([]);
	expect(JSON.stringify(bypassedCall.context)).toContain(poison);
	expect(bypassedCall.sessionId).toBe(previousIdentity);

	await harness.session.retryWithoutMemory([wanted.source.key]);
	expect(harness.settings.get("memory.backend")).toBe("off");
	expect(harness.settings.get("autolearn.enabled")).toBe(false);
	expect(harness.settings.get("requirements.enabled")).toBe(false);
	expect(harness.session.sessionId).not.toBe(previousIdentity);
	expect(providerClosed).toBe(true);
	expect(harness.session.providerSessionState.size).toBe(0);
	const retained = harness.agent.state.messages;
	expect(retained).toHaveLength(1);
	expect(retained[0]).toMatchObject({ role: "user", content: [{ type: "text", text: sameText }] });
	expect(JSON.stringify(retained)).not.toContain(poison);
	expect(JSON.stringify(retained)).not.toContain("unselected human input");
	const currentSources = (await harness.manager.getRequirementsSources()).sources;
	expect(currentSources.map(source => source.key)).toEqual([wanted.source.key]);
	expect(currentSources.map(source => source.key)).not.toContain(unwanted.source.key);
	expect(currentSources.map(source => source.key)).not.toContain(unselectedTwin.source.key);

	await harness.session.prompt("Perform the clean retry.", { synthetic: true });
	await harness.session.waitForIdle();
	const retry = harness.calls.at(-1)!;
	expect(requirementStatements(retry)).toEqual([]);
	expect(JSON.stringify(retry.context)).not.toContain(poison);
	expect(JSON.stringify(retry.context)).not.toContain("unselected human input");
	expect(JSON.stringify(retry.context)).toContain("Keep only the explicitly selected original input.");
	expect(retry.sessionId).toBe(harness.session.sessionId);
});

test("independent side contexts never inherit execution requirements", async () => {
	const harness = createHarness();
	const requirement = "Preserve the source spelling exactly in the main result.";
	const captured = await adopt(harness, requirement);
	await harness.session.prompt("Execute the main request.", { synthetic: true });
	await harness.session.waitForIdle();
	const lastMainReceipt = harness.session.requirements.status().receipt;
	const sideMessages = [{ role: "user" as const, content: "Independent side input.", timestamp: 0 }];
	const raw = await harness.agent.buildSideRequestContext(sideMessages, ["Independent side instructions."]);
	expect(requirementStatements({ context: raw, sessionId: undefined })).toEqual([]);
	expect(JSON.stringify(raw)).not.toContain(requirement);
	expect(JSON.stringify(raw)).not.toContain(captured.source.key);
	expect(raw.systemPrompt).toEqual(["Independent side instructions.", ORDINARY_TRANSFORM]);
	expect(harness.session.requirements.status().receipt).toEqual(lastMainReceipt);
});

function denseRequirement(): string {
	return (
		"Preserve every exact identifier in this required output: " +
		Array.from({ length: 8000 }, (_, index) => `w${(index * 7919) % 100000}`).join(" ")
	);
}

test("capacity uses actual rendered requirements frames rather than rejecting oversized original text", async () => {
	const renderer = new SnapcompactInlineTransformer({
		renderSystemPrompt: "all",
		renderToolResults: false,
		shape: "6x12-dim",
	});
	let rendered: Context | undefined;
	const harness = createHarness({
		contextWindow: 12000,
		vision: true,
		physical: async (context, currentModel) => {
			rendered = await renderer.transform(context, currentModel);
			return rendered;
		},
	});
	const requirement = denseRequirement();
	await adopt(harness, requirement);
	harness.agent.appendMessage({ role: "user", content: "Read all prepared instructions.", timestamp: 0 });
	await harness.session.prompt("Produce the exact required output.", { synthetic: true });
	await harness.session.waitForIdle();
	const receipt = harness.session.requirements.status().receipt!;
	const capacity = receipt.capacity!;
	expect(harness.agent.tokenizer.countTokens(requirement)).toBeGreaterThan(
		capacity.contextWindow - capacity.reserveTokens,
	);
	expect(harness.calls).toHaveLength(1);
	expect(receipt.phase).toBe("prepared");
	expect(capacity.provenance).toBe("estimated");
	expect(capacity.irreducibleTokens!).toBeLessThanOrEqual(capacity.contextWindow - capacity.reserveTokens);
	const frames: ImageContent[] = [];
	for (const message of rendered!.messages) {
		if (typeof message.content === "string") continue;
		for (const block of message.content) if (block.type === "image") frames.push(block);
	}
	const facts = frames.map(frame => getInlineFrameAccounting(frame));
	expect(facts.map(fact => fact?.owner)).toEqual(["system", "system"]);
	expect(capacity.irreducibleTokens!).toBeGreaterThanOrEqual(
		facts.reduce((sum, fact) => sum + fact!.estimatedTokens, 0),
	);
	for (const frame of frames) expect(Buffer.from(frame.data, "base64").subarray(1, 4).toString()).toBe("PNG");
	expect(
		harness.calls[0]!.context.messages.flatMap(message =>
			typeof message.content === "string" ? [] : message.content.filter(block => block.type === "image"),
		),
	).toHaveLength(frames.length);
});

test("irreducible over-budget requirements refuse the provider without contaminating independent side contexts", async () => {
	const harness = createHarness({ contextWindow: 12000 });
	await adopt(harness, denseRequirement());
	await harness.session.prompt("Produce the exact required output.", { synthetic: true });
	await harness.session.waitForIdle();
	const receipt = harness.session.requirements.status().receipt!;
	expect(harness.calls).toEqual([]);
	expect(receipt.phase).toBe("refused");
	expect(receipt.capacity!.irreducibleTokens!).toBeGreaterThan(
		receipt.capacity!.contextWindow - receipt.capacity!.reserveTokens,
	);
	const side = await harness.agent.buildSideRequestContext(
		[{ role: "user", content: "Independent candidate only.", timestamp: 0 }],
		["Independent reviewer instructions."],
	);
	expect(side.systemPrompt).toEqual(["Independent reviewer instructions.", ORDINARY_TRANSFORM]);
	expect(requirementStatements({ context: side, sessionId: undefined })).toEqual([]);
	expect(harness.session.requirements.status().receipt).toEqual(receipt);
});

test("large ordinary history and tool output are not misclassified as irreducible requirements input", async () => {
	const harness = createHarness({ contextWindow: 12000 });
	const requirement = "Keep the final response finite and preserve UTF-8.";
	await adopt(harness, requirement);
	const history = denseRequirement();
	harness.agent.appendMessage({ role: "user", content: "Inspect the original data.", timestamp: 0 });
	harness.agent.appendMessage({
		...createAssistantMessage(history),
		content: [
			{ type: "text", text: history },
			{ type: "toolCall", id: "historical-read", name: "read", arguments: {} },
		],
		stopReason: "toolUse",
	});
	harness.agent.appendMessage({
		role: "toolResult",
		toolCallId: "historical-read",
		toolName: "read",
		content: [{ type: "text", text: history }],
		isError: false,
		timestamp: 1,
	});
	await harness.session.prompt("Continue from ordinary history.", { synthetic: true });
	await harness.session.waitForIdle();
	expect(harness.calls).toHaveLength(1);
	const receipt = harness.session.requirements.status().receipt!;
	expect(harness.agent.tokenizer.countTokens(JSON.stringify(harness.calls[0]!.context.messages))).toBeGreaterThan(
		receipt.capacity!.contextWindow,
	);
	expect(receipt.phase).toBe("prepared");
	expect(receipt.capacity!.irreducibleTokens!).toBeLessThan(
		receipt.capacity!.contextWindow - receipt.capacity!.reserveTokens,
	);
	expect(requirementStatements(harness.calls[0]!)).toEqual([requirement]);
});

test("quarantine while a physical transform awaits rejects the stale request and allows a fresh retry", async () => {
	const entered = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	const harness = createHarness({
		physical: async context => {
			entered.resolve();
			await release.promise;
			return context;
		},
	});
	await adopt(harness, "Use the cyan status indicator.");
	const revision = harness.session.requirements.snapshotApplicable().active[0]!;
	const run = harness.session.prompt("Show the status.", { synthetic: true });
	try {
		await entered.promise;
		await harness.session.requirements.applyOperatorAction({
			kind: "quarantine",
			revisionIds: [revision.id],
			reason: "Operator paused this decision before send",
		});
	} finally {
		release.resolve();
	}
	await run;
	await harness.session.waitForIdle();
	expect(harness.calls).toEqual([]);
	expect(harness.session.requirements.status().receipt?.phase).toBe("refused");
	await harness.session.prompt("Retry with the current decision state.", { synthetic: true });
	await harness.session.waitForIdle();
	expect(harness.calls).toHaveLength(1);
	expect(requirementStatements(harness.calls[0]!)).toEqual([]);
});

test("a changed emitted requirements frame reports unknown capacity instead of reusing stale raster pricing", async () => {
	const renderer = new SnapcompactInlineTransformer({
		renderSystemPrompt: "all",
		renderToolResults: false,
		shape: "6x12-dim",
	});
	let mutated = false;
	const harness = createHarness({
		contextWindow: 12000,
		vision: true,
		physical: async (context, currentModel) => {
			const rendered = await renderer.transform(context, currentModel);
			for (const message of rendered.messages) {
				if (typeof message.content === "string") continue;
				const frame = message.content.find(
					block => block.type === "image" && getInlineFrameAccounting(block)?.owner === "system",
				);
				if (frame?.type !== "image") continue;
				// Still decodes to a valid PNG, but no longer matches the identity-attested bytes.
				frame.data += "\n";
				mutated = true;
				break;
			}
			return rendered;
		},
	});
	await adopt(harness, denseRequirement());
	harness.agent.appendMessage({ role: "user", content: "Read all prepared instructions.", timestamp: 0 });
	await harness.session.prompt("Produce the exact required output.", { synthetic: true });
	await harness.session.waitForIdle();
	expect(mutated).toBe(true);
	expect(harness.calls).toHaveLength(1);
	expect(harness.session.requirements.status().receipt?.capacity).toMatchObject({
		provenance: "unknown",
		irreducibleTokens: null,
	});
});

test("inband final accounting charges the emitted catalog once and retains owned system frames through encoded tool history", async () => {
	const renderer = new SnapcompactInlineTransformer({
		renderSystemPrompt: "all",
		renderToolResults: false,
		shape: "6x12-dim",
	});
	const parameters = type({ value: "string" });
	const tool: AgentTool<typeof parameters> = {
		name: "catalog_probe",
		label: "Catalog probe",
		description: "Return the requested boundary value and the original source data.",
		parameters,
		execute: async (_id, args) => ({
			content: [{ type: "text", text: `CATALOG_RESULT:${args.value} ${denseRequirement()}` }],
		}),
	};
	const sentPrices: number[] = [];
	const harness = createHarness({
		contextWindow: 12000,
		vision: true,
		dialect: "glm",
		tool,
		physical: (context, currentModel) => renderer.transform(context, currentModel),
		respond: (context, call) => {
			let price = harness.agent.tokenizer.countTokens(context.systemPrompt ?? []);
			for (const message of context.messages) {
				if (typeof message.content === "string") continue;
				for (const block of message.content) {
					if (block.type === "image") {
						const fact = getInlineFrameAccounting(block);
						if (fact && fact.owner !== "tool") price += fact.estimatedTokens;
					} else if (block.type === "text") {
						const fact = getInlineTextAccounting(block);
						if (fact && fact.owner !== "tool") price += harness.agent.tokenizer.countTokens(block.text);
					}
				}
			}
			sentPrices.push(price);
			return createAssistantMessage(
				call === 1
					? "<tool_call>catalog_probe\n<arg_key>value</arg_key>\n<arg_value>accepted</arg_value>\n</tool_call>"
					: "Boundary result received.",
			);
		},
	});
	const finalRequests: { model: Model; capacity: number | null | undefined }[] = [];
	harness.agent.addBeforeModelCall((_context, _signal, request) => {
		finalRequests.push({
			model: request.model,
			capacity: harness.session.requirements.status().receipt?.capacity?.irreducibleTokens,
		});
	});
	await adopt(harness, denseRequirement());
	harness.agent.appendMessage({ role: "user", content: "Read all prepared instructions.", timestamp: 0 });
	await harness.session.prompt("Inspect the catalog boundary.", { synthetic: true });
	await harness.session.waitForIdle();
	expect(harness.calls).toHaveLength(2);
	expect(finalRequests.map(request => request.capacity)).toEqual(sentPrices);
	for (let index = 0; index < harness.calls.length; index++) {
		expect(finalRequests[index]!.model).toBe(harness.dispatchedModels[index]!);
		const context = harness.calls[index]!.context;
		expect(context.tools).toBeUndefined();
		expect((context.systemPrompt ?? []).join("\n").match(/"name":"catalog_probe"/g)).toHaveLength(1);
		expect(
			context.messages.flatMap(message =>
				typeof message.content === "string" ? [] : message.content.filter(block => block.type === "image"),
			),
		).toHaveLength(2);
	}
	const followup = harness.calls[1]!.context;
	expect(followup.messages.some(message => message.role === "toolResult")).toBe(false);
	const encodedResults = followup.messages
		.filter(message => message.role === "user")
		.flatMap(message =>
			typeof message.content === "string"
				? [message.content]
				: message.content.filter(block => block.type === "text").map(block => block.text),
		)
		.join("\n");
	expect(encodedResults).toContain(`CATALOG_RESULT:accepted ${denseRequirement()}`);
	expect(harness.agent.tokenizer.countTokens(encodedResults)).toBeGreaterThan(
		harness.session.requirements.status().receipt!.capacity!.contextWindow,
	);
	expect(harness.session.requirements.status().receipt?.capacity?.provenance).toBe("estimated");
});

test("inband tool catalog overflow refuses the provider after encoding even when preencoding requirements fit", async () => {
	const tool: AgentTool = {
		name: "large_catalog",
		label: "Large catalog",
		description: denseRequirement(),
		parameters: type({}),
		execute: async () => ({ content: [{ type: "text", text: "Catalog inspected." }] }),
	};
	let beforeEncodingTokens = 0;
	const harness = createHarness({
		contextWindow: 12000,
		dialect: "glm",
		tool,
		physical: context => {
			beforeEncodingTokens = harness.agent.tokenizer.countTokens(context.systemPrompt ?? []);
			return context;
		},
	});
	await adopt(harness, "Keep the final output finite.");
	await harness.session.prompt("Inspect the available catalog.", { synthetic: true });
	await harness.session.waitForIdle();
	const receipt = harness.session.requirements.status().receipt!;
	const capacity = receipt.capacity!;
	expect(beforeEncodingTokens).toBeLessThan(capacity.contextWindow - capacity.reserveTokens);
	expect(capacity.irreducibleTokens!).toBeGreaterThan(capacity.contextWindow - capacity.reserveTokens);
	expect(receipt.phase).toBe("refused");
	expect(harness.calls).toEqual([]);
});

test("deleting a foreign original while physical preparation awaits refuses the stale provider request", async () => {
	using temp = TempDir.createSync("requirements-foreign-send-race-");
	// Volatile captures retain inline bodies; the independent clone suppresses terminal
	// breadcrumbs while its real file adapter reads only this temporary journal.
	const manager = SessionManager.inMemory(temp.path(), new FileSessionStorage()).cloneCurrentSession({
		persist: false,
	});
	const entered = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	let prepared: string[] = [];
	const harness = createHarness({
		manager,
		physical: async context => {
			prepared = requirementStatements({ context, sessionId: undefined });
			entered.resolve();
			await release.promise;
			return context;
		},
	});
	try {
		const text = "Global rule: preserve the exact original identifier in every result.";
		const captured = await capture(harness, text);
		const originalPath = temp.join("original.jsonl");
		await writeFile(
			originalPath,
			[manager.getHeader(), ...manager.getEntries()].map(entry => JSON.stringify(entry)).join("\n") + "\n",
		);
		await manager.setSessionFile(originalPath);
		await harness.session.requirements.observeCommittedSources();
		await harness.session.requirements.applyOperatorAction({
			kind: "literal-adopt",
			sourceKey: captured.source.key,
			unitId: captured.source.units[0]!.id,
			scope: { kind: "global" },
		});
		await manager.newSession();
		await harness.session.requirements.observeCommittedSources();
		expect(harness.session.requirements.snapshotApplicable().active.map(revision => revision.statement)).toEqual([
			text,
		]);
		const run = harness.session.prompt("Use the applicable global rule.", { synthetic: true });
		try {
			await Promise.race([
				entered.promise,
				run.then(() => {
					throw new Error("Request ended before its physical preparation boundary");
				}),
			]);
			expect(prepared).toEqual([text]);
			await unlink(originalPath);
		} finally {
			release.resolve();
		}
		await run;
		await harness.session.waitForIdle();
		expect(harness.calls).toEqual([]);
		expect(harness.session.requirements.status().receipt?.phase).toBe("refused");
		await harness.session.requirements.refreshCurrentEvidence();
		expect(harness.session.requirements.snapshotApplicable().active).toEqual([]);
		expect(
			harness.session.requirements.status({ includeLedger: true }).snapshot.sources.find(source => source.key === captured.source.key)
				?.state,
		).toBe("orphaned");
	} finally {
		release.resolve();
		await harness.session.dispose();
		sessions.delete(harness.session);
	}
});

test("warm branch reconciliation checks foreign requirements published while source enumeration awaits", async () => {
	using temp = TempDir.createSync("requirements-warm-publication-race-");
	const storage = await AgentStorage.open(temp.join("agent.db"));
	const openHarness = async (name: string, text: string) => {
		// Persist inline captures in isolated journals without touching the shared body depot.
		const seed = SessionManager.inMemory(temp.path());
		const entryId = seed.appendMessage({ role: "user", content: text, timestamp: 1, producer: { type: "human" } });
		const journalPath = temp.join(`${name}.jsonl`);
		await writeFile(
			journalPath,
			[seed.getHeader(), ...seed.getEntries()].map(entry => JSON.stringify(entry)).join("\n") + "\n",
		);
		await seed.close();
		const harness = createHarness({
			manager: await SessionManager.open(journalPath, temp.path()),
			agentStorage: storage,
		});
		await harness.session.requirements.observeCommittedSources();
		const descriptor = harness.manager.getRequirementsSource(entryId)!;
		const { source } = await harness.session.requirements.inspectSource(descriptor.key);
		return { harness, source, entryId };
	};
	const localSource = await openHarness("local", "Keep this local unresolved source visible.");
	const foreignSource = await openHarness("foreign", "Global rule: retain the original identifier.");
	const local = localSource.harness;
	const foreign = foreignSource.harness;
	const release = Promise.withResolvers<void>();
	try {
		const common = local.manager.getLeafId()!;
		const branch = local.manager.appendCustomEntry("warm-branch", { name: "A" });
		local.manager.branch(common);
		local.manager.appendCustomEntry("warm-branch", { name: "B" });
		await local.session.requirements.observeCommittedSources();
		const entered = Promise.withResolvers<void>();
		const iterate = local.manager.iterateRequirementsSources.bind(local.manager);
		local.manager.iterateRequirementsSources = async function* (...args) {
			const epoch = yield* iterate(...args);
			entered.resolve();
			await release.promise;
			return epoch;
		};
		local.session.requirements.cancelPending("Switch warm branch");
		local.manager.branch(branch);
		const observing = local.session.requirements.observeCommittedSources();
		try {
			await entered.promise;
			const publication = await foreign.session.requirements.applyOperatorAction({
				kind: "literal-adopt",
				sourceKey: foreignSource.source.key,
				unitId: foreignSource.source.units[0]!.id,
				scope: { kind: "global" },
			});
			expect(publication).toMatchObject({ status: "accepted" });
			expect(local.session.requirements.snapshotApplicable().active.map(revision => revision.sourceKey)).toEqual([
				foreignSource.source.key,
			]);
			await unlink(temp.join("foreign.jsonl"));
		} finally {
			release.resolve();
			await observing;
		}
		const applicable = local.session.requirements.snapshotApplicable();
		expect(applicable.active).toEqual([]);
		expect(applicable.coverageGaps.find(source => source.key === foreignSource.source.key)?.state).toBe("orphaned");
		expect(applicable.ledgerCoverage.byState.pending).toBe(1);
		expect(local.session.requirements.pendingLiveSnapshot().entryIds).toEqual([]);
	} finally {
		release.resolve();
		for (const harness of [local, foreign]) {
			await harness.session.dispose();
			sessions.delete(harness.session);
		}
		AgentStorage.close();
	}
});

test("only accepted live delivery acquires a frozen hold; disable and bypass never repin history", async () => {
	const harness = createHarness();
	const cold = await capture(harness, "Cold original remains explicit backfill.");
	const owner = harness.session.requirements;
	expect(owner.pendingLiveSnapshot().entryIds).toEqual([]);
	await owner.acceptDelivered(cold.entryId);
	const protectedSource = owner.pendingLiveSnapshot();
	expect(protectedSource.entryIds).toEqual([cold.entryId]);
	harness.settings.override("requirements.enabled", false);
	expect(owner.pendingLiveSnapshot().entryIds).toEqual([]);
	expect(protectedSource.entryIds).toEqual([cold.entryId]);
	harness.settings.override("requirements.enabled", true);
	expect(owner.pendingLiveSnapshot().entryIds).toEqual([]);
	owner.setRecoveryMode("bypass");
	expect(owner.pendingLiveSnapshot().entryIds).toEqual([]);
	expect(owner.status({ includeLedger: true }).snapshot.sources.map(source => source.key)).toContain(cold.source.key);
});

test("cold catalog never blocks an ordinary provider request or acquires a live hold", async () => {
	const manager = SessionManager.inMemory("/requirements-cold-catalog");
	manager.appendMessage({ role: "user", content: "Cold requirement remains inspectable.", timestamp: 1, producer: { type: "human" } });
	const harness = createHarness({ manager });
	const entered = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	const iterate = manager.iterateRequirementsSources.bind(manager);
	manager.iterateRequirementsSources = async function* (...args) { entered.resolve(); await release.promise; return yield* iterate(...args); };
	try {
		await harness.session.prompt("Perform the bounded action.", { synthetic: true });
		await entered.promise;
		expect(harness.calls).toHaveLength(1);
		expect(harness.session.requirements.pendingLiveSnapshot().entryIds).toEqual([]);
	} finally { release.resolve(); }
	await harness.session.requirements.observeCommittedSources();
	expect(harness.session.requirements.status({ includeLedger: true }).snapshot.sources).toHaveLength(1);
});

test("disabled automatic observation leaves cold coverage lazy but explicit observation preserves all originals", async () => {
	const harness = createHarness();
	harness.settings.override("requirements.enabled", false);
	const ids: string[] = [];
	for (const text of ["Keep the first original.", "Keep the second original."]) {
		ids.push(
			harness.manager.appendMessage({ role: "user", content: [{ type: "text", text }], timestamp: 1, producer: { type: "human" } }),
		);
	}
	await harness.session.requirements.observeCommittedSources();
	harness.session.requirements.prepareFragment();
	expect(harness.session.requirements.status().sourceCatalog).toBe("unobserved");
	expect(harness.session.requirements.status({ includeLedger: true }).snapshot.sources).toEqual([]);
	await harness.session.requirements.observeCommittedSources(true);
	const status = harness.session.requirements.status({ includeLedger: true });
	expect(status.sourceCatalog).toBe("last-observed");
	expect(status.snapshot.sources.flatMap(source => source.locators.map(locator => locator.entryId))).toEqual(ids);
	expect(status.snapshot.sources.map(source => source.state)).toEqual(["pending", "pending"]);
	expect(status.snapshot.revisions).toEqual([]);
	expect(status.receipt?.coverageComplete).toBe(false);
});

test("enabling catalogs historical coverage without any model call or pending-live hold", async () => {
	const harness = createHarness();
	harness.settings.override("requirements.enabled", false);
	for (const text of ["Older unresolved requirement.", "Current committed requirement."]) {
		harness.manager.appendMessage({ role: "user", content: [{ type: "text", text }], timestamp: 1, producer: { type: "human" } });
	}
	harness.settings.override("requirements.enabled", true);
	while (
		harness.session.requirements.status().sourceCatalog !== "last-observed" ||
		harness.session.requirements.status().running
	)
		await new Promise<void>(resolve => setImmediate(resolve));
	const status = harness.session.requirements.status({ includeLedger: true });
	expect(status.snapshot.sources.map(source => source.state)).toEqual(["pending", "pending"]);
	expect(harness.session.requirements.pendingLiveSnapshot().entryIds).toEqual([]);
	expect(harness.calls).toEqual([]);
});

test("a cold observation canceled by session replacement cannot install abandoned source coverage", async () => {
	const harness = createHarness();
	const manager = harness.manager;
	manager.appendMessage({ role: "user", content: "Abandoned session original.", timestamp: 1, producer: { type: "human" } });
	const entered = Promise.withResolvers<void>();
	const release = Promise.withResolvers<void>();
	const iterate = manager.iterateRequirementsSources.bind(manager);
	let hold = true;
	manager.iterateRequirementsSources = async function* (...args) {
		const sources = iterate(...args);
		const first = await sources.next();
		if (hold) {
			hold = false;
			entered.resolve();
			await release.promise;
		}
		if (first.done) return first.value;
		yield first.value;
		return yield* sources;
	};
	const observing = harness.session.requirements.observeCommittedSources();
	let currentEntry: string;
	try {
		await entered.promise;
		harness.session.requirements.cancelPending("Session replaced during cold observation");
		await manager.newSession();
		currentEntry = manager.appendMessage({ role: "user", content: "Current session original.", timestamp: 2, producer: { type: "human" } });
	} finally {
		release.resolve();
	}
	await observing;
	const status = harness.session.requirements.status({ includeLedger: true });
	expect(status.sourceCatalog).toBe("last-observed");
	expect(status.snapshot.sources.flatMap(source => source.locators.map(locator => locator.entryId))).toEqual([currentEntry]);
	expect(harness.session.requirements.pendingLiveSnapshot().entryIds).toEqual([]);
});

test("quarantine survives unavailable original bytes and reappearance until explicit fresh restore", async () => {
	using temp = TempDir.createSync("requirements-original-restore-");
	const journalPath = temp.join("session.jsonl");
	const manager = await SessionManager.open(journalPath, temp.path(), new FileSessionStorage(), { initialCwd: temp.path(), suppressBreadcrumb: true });
	const harness = createHarness({ manager });
	const text = "Keep the original technical spelling exactly.";
	const captured = await adopt(harness, text);
	const revision = harness.session.requirements.snapshotApplicable().active[0]!;
	await manager.flush();
	const original = await readFile(journalPath);
	await harness.session.requirements.applyOperatorAction({ kind: "quarantine", revisionIds: [revision.id], reason: "Operator requested review" });
	await unlink(journalPath);
	await expect(harness.session.requirements.applyOperatorAction({ kind: "restore", revisionIds: [revision.id], literalUnitId: captured.source.units[0]!.id })).rejects.toThrow("unavailable");
	expect(await harness.session.requirements.applyOperatorAction({ kind: "inspect", revisionId: revision.id })).toMatchObject({ lifecycle: "quarantined" });
	await writeFile(journalPath, original);
	await harness.session.requirements.refreshCurrentEvidence();
	expect(harness.session.requirements.snapshotApplicable().active).toEqual([]);
	expect(await harness.session.requirements.applyOperatorAction({ kind: "restore", revisionIds: [revision.id], literalUnitId: captured.source.units[0]!.id })).toEqual([revision.id]);
	await harness.session.prompt("Use the freshly restored rule.", { synthetic: true });
	await harness.session.waitForIdle();
	expect(requirementStatements(harness.calls[0]!)).toEqual([text]);
});

test("historical unknown coverage is aggregated while a bounded branch request reaches the actual provider", async () => {
	using temp = TempDir.createSync("requirements-historical-provider-");
	const seed = SessionManager.inMemory(temp.path());
	const historical = 4096;
	for (let index = 0; index < historical; index++) seed.appendMessage({ role: "user", content: "Unknown historical input " + index, timestamp: index });
	const common = seed.getLeafId()!;
	const sibling = seed.appendMessage({ role: "user", content: "Abandoned sibling request", timestamp: historical + 1, producer: { type: "human" } });
	seed.branch(common);
	const current = { role: "user" as const, content: "Perform only this bounded branch action.", timestamp: historical + 2, producer: { type: "human" as const } };
	const entry = seed.appendMessage(current);
	const journalPath = temp.join("historical.jsonl");
	await writeFile(journalPath, [seed.getHeader(), ...seed.getEntries()].map(entry => JSON.stringify(entry)).join("\n") + "\n");
	await seed.close();
	const storage = await AgentStorage.open(temp.join("agent.db"));
	const manager = await SessionManager.open(journalPath, temp.path(), new FileSessionStorage(), { suppressBreadcrumb: true });
	const harness = createHarness({ manager, agentStorage: storage });
	try {
		await harness.session.requirements.observeCommittedSources();
		manager.branch(sibling);
		await harness.session.requirements.observeCommittedSources();
		manager.branch(entry);
		await harness.session.requirements.observeCommittedSources();
		expect(harness.session.requirements.pendingLiveSnapshot().entryIds).toEqual([]);
		harness.agent.replaceMessages([current]);
		await harness.session.prompt("Execute the bounded current action.", { synthetic: true });
		await harness.session.waitForIdle();
		expect(harness.calls, JSON.stringify({ receipt: harness.session.requirements.status().receipt, messages: harness.agent.state.messages })).toHaveLength(1);
		const request = JSON.stringify(harness.calls[0]!.context);
		expect(request).toContain(current.content);
		expect(request).not.toContain("Abandoned sibling request");
		expect(request).not.toContain("Unknown historical input");
		expect(harness.agent.tokenizer.countTokens(request)).toBeLessThan(harness.model.contextWindow! - harness.model.maxTokens!);
		const status = harness.session.requirements.status({ includeLedger: true });
		expect(status.snapshot.revisions).toEqual([]);
		expect(status.snapshot.batches).toEqual([]);
		expect(status.applicable.ledgerCoverage.byState.unsupported).toBe(historical);
		expect(status.applicable.ledgerCoverage.byState.pending).toBe(2);
		expect(status.receipt?.coverageComplete).toBe(false);
		expect(status.snapshot.sources.map(source => source.locators[0]!.entryId)).toContain(entry);
		expect(status.snapshot.sources.map(source => source.locators[0]!.entryId)).toContain(sibling);
		const first = (await harness.session.requirements.inspectSource(status.snapshot.sources[0]!.key)).source;
		const last = await harness.session.requirements.inspectSource(status.snapshot.sources[historical - 1]!.key);
		expect(first.state).toBe("unsupported");
		expect(last.units[0]!.text).toBe("Unknown historical input " + (historical - 1));
		const originalJournal = await readFile(journalPath, "utf8");
		const firstNewline = originalJournal.indexOf("\n");
		const header = JSON.parse(originalJournal.slice(0, firstNewline));
		await writeFile(journalPath, JSON.stringify({ ...header, id: crypto.randomUUID() }) + originalJournal.slice(firstNewline));
		await expect(harness.session.requirements.inspectSource(first.key)).rejects.toThrow("unavailable");
		await writeFile(journalPath, originalJournal);
		expect((await manager.observeRequirementsEvidence([first]))[0]!.integrity).toBe(first.integrity);
		await unlink(journalPath);
		expect((await manager.observeRequirementsEvidence([first]))[0]!.integrity).toBeNull();
		await expect(harness.session.requirements.inspectSource(first.key)).rejects.toThrow("unavailable");
	} finally {
		await harness.session.dispose(); sessions.delete(harness.session); AgentStorage.close();
	}
});
