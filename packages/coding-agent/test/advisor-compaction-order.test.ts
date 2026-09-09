import { afterEach, describe, expect, it, vi } from "bun:test";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import { createMockModel, type MockHandler, registerMockApi } from "@oh-my-pi/pi-ai/providers/mock";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { TempDir } from "@oh-my-pi/pi-utils";
import type { AdvisorSeverity } from "../src/advisor/advise-tool";
import { createAssistantMessage, createInMemoryAuthStorage } from "./helpers/agent-session-setup";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
	for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
	vi.restoreAllMocks();
});
const NOTE = "Stop repeating the failed operation; inspect its error and change the approach.";

function harness(
	options: {
		enabled?: boolean;
		severity?: AdvisorSeverity;
		noAdvice?: boolean;
		autoCompact?: boolean;
		summarize?: MockHandler;
		primary?: MockHandler;
		advise?: MockHandler;
		tools?: AgentTool[];
	} = {},
) {
	const temp = TempDir.createSync("@pi-advisor-guidance-");
	const auth = createInMemoryAuthStorage();
	auth.setRuntimeApiKey("anthropic", "test-key");
	registerMockApi();
	const trace: string[] = [];
	const primary = createMockModel({
		handler:
			options.primary ??
			(() => {
				trace.push("primary");
				return { content: ["primary complete"] };
			}),
	});
	let emitted = false;
	const advisor = createMockModel({
		provider: "anthropic",
		id: "advisor-model",
		handler:
			options.advise ??
			(() => {
				trace.push("review");
				if (!emitted && !options.noAdvice) {
					emitted = true;
					return {
						content: [
							{
								type: "toolCall",
								name: "advise",
								arguments: { note: NOTE, severity: options.severity ?? "nit" },
							},
						],
					};
				}
				return { content: ["review complete"] };
			}),
	});
	const summary = createMockModel({
		provider: "anthropic",
		id: "claude-sonnet-4-5",
		contextWindow: 100_000,
		handler:
			options.summarize ??
			(() => {
				trace.push("compact");
				return { content: ["COMPACTED HISTORY"] };
			}),
	});
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: { model: summary, systemPrompt: ["Test"], tools: options.tools ?? [] },
		convertToLlm,
		streamFn: primary.stream,
	});
	const settings = Settings.isolated({
		"advisor.compactBeforeGuidance": options.enabled ?? true,
		"advisor.syncBacklog": "1",
		"compaction.enabled": options.autoCompact ?? true,
		"compaction.asyncEnabled": true,
		"compaction.methodOrder": ["soft"],
		"compaction.thresholdPercent": 50,
		"compaction.keepRecentTokens": 500,
		"compaction.autoContinue": false,
		"contextPromotion.enabled": false,
		"todo.reminders": false,
	});
	settings.setModelRole("advisor", "anthropic/advisor-model");
	const manager = SessionManager.create(temp.path(), temp.path());
	const registry = new ModelRegistry(auth, temp.join("models.yml"));
	vi.spyOn(registry, "getApiKey").mockResolvedValue("test-key");
	vi.spyOn(registry, "getAvailable").mockReturnValue([summary, advisor]);
	const session = new AgentSession({
		agent,
		sessionManager: manager,
		settings,
		modelRegistry: registry,
		advisorTools: [],
		advisorStreamFn: advisor.stream,
		sideStreamFn: summary.stream,
	});
	const oldUser = { role: "user" as const, content: "OLD SOURCE ".repeat(4_000), timestamp: 1 };
	const oldAssistant = createAssistantMessage("old response ".repeat(4_000));
	manager.appendMessage(oldUser);
	manager.appendMessage(oldAssistant);
	agent.appendMessage(oldUser);
	agent.appendMessage(oldAssistant);
	session.setAdvisorEnabled(true);
	cleanups.push(async () => {
		await session.dispose();
		auth.close();
		await temp.remove();
	});
	async function run() {
		await session.prompt("Recover this stalled task");
		await session.waitForIdle();
	}
	return { session, agent, settings, manager, primary, advisor, summary, trace, run };
}

describe("compaction before accepted advisor guidance", () => {
	for (const severity of ["nit", "concern", "blocker"] as const) {
		it(`forces low-pressure compaction before ${severity} reaches the primary`, async () => {
			const h = harness({ severity });
			await h.run();
			const delivered = h.primary.calls.find(call => JSON.stringify(call.context.messages).includes(NOTE));
			expect(delivered).toBeDefined();
			expect(JSON.stringify(delivered!.context.messages).includes("COMPACTED HISTORY")).toBe(true);
			expect(h.manager.getBranch().filter(entry => entry.type === "compaction")).toHaveLength(1);
			expect(JSON.stringify(h.summary.calls[0].context.messages)).not.toContain(NOTE);
			expect(h.trace.indexOf("review")).toBeLessThan(h.trace.indexOf("compact"));
		});
	}

	it("does not force maintenance when disabled or when no guidance was accepted", async () => {
		const disabled = harness({ enabled: false, severity: "blocker" });
		await disabled.run();
		expect(disabled.primary.calls.some(call => JSON.stringify(call.context.messages).includes(NOTE))).toBe(true);
		expect(disabled.summary.calls).toHaveLength(0);
		const silent = harness({ noAdvice: true });
		await silent.run();
		expect(silent.summary.calls).toHaveLength(0);
		expect(silent.primary.calls).toHaveLength(1);
	});

	it("forces requested guidance recovery even with automatic pressure maintenance disabled", async () => {
		const h = harness({ autoCompact: false });
		await h.run();
		expect(h.manager.getBranch().filter(entry => entry.type === "compaction")).toHaveLength(1);
		expect(JSON.stringify(h.primary.calls.at(-1)?.context.messages)).toContain(NOTE);
		expect(JSON.stringify(h.primary.calls.at(-1)?.context.messages)).toContain("COMPACTED HISTORY");
	});

	it("drops held guidance when its advisor source is replaced during compaction", async () => {
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const h = harness({
			severity: "blocker",
			summarize: async () => {
				started.resolve();
				await release.promise;
				return { content: ["COMPACTED HISTORY"] };
			},
		});
		const running = h.run();
		await started.promise;
		h.session.setAdvisorEnabled(false);
		h.session.setAdvisorEnabled(true);
		release.resolve();
		await running;
		expect(h.primary.calls.some(call => JSON.stringify(call.context.messages).includes(NOTE))).toBe(false);
	});

	it("does not inject or commit guidance recovery after user cancellation", async () => {
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const h = harness({
			severity: "blocker",
			summarize: async () => {
				started.resolve();
				await release.promise;
				return { content: ["COMPACTED HISTORY"] };
			},
		});
		const running = h.run();
		await started.promise;
		const aborted = h.session.abort();
		release.resolve();
		await Promise.all([running, aborted]);
		expect(h.primary.calls.some(call => JSON.stringify(call.context.messages).includes(NOTE))).toBe(false);
		expect(h.manager.getBranch().some(entry => entry.type === "compaction")).toBe(false);
	});

	it("interrupts a live wait, compacts paired history, then delivers the blocker", async () => {
		const started = Promise.withResolvers<void>();
		const interrupted = Promise.withResolvers<void>();
		let primaryCalls = 0;
		const h = harness({
			severity: "blocker",
			primary: () =>
				++primaryCalls === 1
					? { content: [{ type: "toolCall", id: "waiting", name: "wait", arguments: {} }] }
					: { content: ["recovered"] },
			tools: [
				{
					name: "wait",
					label: "Wait",
					description: "Wait for interrupt",
					parameters: type({}),
					intent: "omit",
					interruptible: true,
					execute: async (_id, _args, signal) => {
						started.resolve();
						if (signal?.aborted) interrupted.resolve();
						else signal?.addEventListener("abort", () => interrupted.resolve(), { once: true });
						await interrupted.promise;
						return { content: [{ type: "text", text: "WAIT_INTERRUPTED" }] };
					},
				},
			],
		});
		h.session.setInterruptMode("immediate");
		const running = h.run();
		try {
			await started.promise;
			await h.session.getAdvisorAgent()!.prompt("Inspect the stalled wait and advise the primary.");
			await running;
			const delivered = h.primary.calls.find(call => JSON.stringify(call.context.messages).includes(NOTE));
			expect(JSON.stringify(delivered?.context.messages)).toContain("COMPACTED HISTORY");
			expect(JSON.stringify(delivered?.context.messages)).toContain("WAIT_INTERRUPTED");
			expect(JSON.stringify(h.summary.calls[0].context.messages)).not.toContain(NOTE);
			expect(h.manager.getBranch().filter(entry => entry.type === "compaction")).toHaveLength(1);
		} finally {
			interrupted.resolve();
		}
	});

	it("compacts before guidance accepted after the primary has gone idle", async () => {
		const h = harness({ noAdvice: true });
		await h.run();
		let emitted = false;
		h.advisor.fallback = () => {
			if (emitted) return { content: ["review complete"] };
			emitted = true;
			return { content: [{ type: "toolCall", name: "advise", arguments: { note: NOTE, severity: "blocker" } }] };
		};
		await h.session.getAdvisorAgent()!.prompt("Deliver the newly discovered blocker.");
		await h.session.waitForIdle();
		const delivered = h.primary.calls.find(call => JSON.stringify(call.context.messages).includes(NOTE));
		expect(JSON.stringify(delivered?.context.messages)).toContain("COMPACTED HISTORY");
		expect(h.manager.getBranch().filter(entry => entry.type === "compaction")).toHaveLength(1);
	});

	it("does not carry held guidance into a new conversation", async () => {
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const h = harness({
			severity: "blocker",
			summarize: async () => {
				started.resolve();
				await release.promise;
				return { content: ["COMPACTED HISTORY"] };
			},
		});
		const original = h.manager.getSessionId();
		const running = h.run();
		await started.promise;
		const transition = h.session.newSession();
		release.resolve();
		await Promise.all([running, transition]);
		expect(h.manager.getSessionId()).not.toBe(original);
		expect(JSON.stringify(h.agent.state.messages)).not.toContain(NOTE);
		expect(h.primary.calls.some(call => JSON.stringify(call.context.messages).includes(NOTE))).toBe(false);
	});

	it("delivers deferred nit guidance at a continuing tool boundary without interrupting the tool", async () => {
		let primaryCalls = 0;
		let aborted = false;
		const h = harness({
			severity: "nit",
			primary: () =>
				++primaryCalls === 1
					? { content: [{ type: "toolCall", id: "work", name: "work", arguments: {} }] }
					: { content: ["recovered"] },
			tools: [
				{
					name: "work",
					label: "Work",
					description: "Complete foreground work",
					parameters: type({}),
					intent: "omit",
					execute: async (_id, _args, signal) => {
						aborted = signal?.aborted ?? false;
						return { content: [{ type: "text", text: "WORK_COMPLETED" }] };
					},
				},
			],
		});
		await h.run();
		expect(aborted).toBe(false);
		const delivered = h.primary.calls.find(call => JSON.stringify(call.context.messages).includes(NOTE));
		expect(JSON.stringify(delivered?.context.messages)).toContain("COMPACTED HISTORY");
		expect(JSON.stringify(delivered?.context.messages)).toContain("WORK_COMPLETED");
		expect(h.manager.getBranch().filter(entry => entry.type === "compaction")).toHaveLength(1);
	});
});
