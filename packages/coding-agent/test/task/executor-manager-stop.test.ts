import { beforeAll, expect, it } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { ExtensionRuntime, loadExtensionFromFactory } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/loader";
import { ExtensionRunner } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/runner";
import { IrcBus } from "@oh-my-pi/pi-coding-agent/irc/bus";
import { AgentHubOverlayComponent } from "@oh-my-pi/pi-coding-agent/modes/components/agent-hub";
import { SessionObserverRegistry } from "@oh-my-pi/pi-coding-agent/modes/session-observer-registry";
import { initTheme } from "@oh-my-pi/pi-coding-agent/modes/theme/theme";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { runSubagentFollowUpTurn } from "@oh-my-pi/pi-coding-agent/task/executor";
import { EventBus } from "@oh-my-pi/pi-coding-agent/utils/event-bus";
import { untilAborted } from "@oh-my-pi/pi-utils";
import { createInMemoryAuthStorage } from "../helpers/agent-session-setup";

beforeAll(() => initTheme());

it.each(["hub", "lifecycle"] as const)(
	"%s kill cancels live provider work before shutdown hooks finish",
	async entry => {
		AgentLifecycleManager.resetGlobalForTests();
		AgentRegistry.resetGlobalForTests();
		const firstRequest = Promise.withResolvers<void>();
		const connectionClosed = Promise.withResolvers<void>();
		const responseBodies = new Set<ReadableStreamDefaultController<Uint8Array>>();
		const shutdownEntered = Promise.withResolvers<void>();
		const shutdownGate = Promise.withResolvers<void>();
		let shutdownCalls = 0;
		let requests = 0;
		const server = Bun.serve({
			hostname: "127.0.0.1",
			port: 0,
			fetch(request) {
				requests++;
				request.signal.addEventListener("abort", () => connectionClosed.resolve(), { once: true });
				firstRequest.resolve();
				return new Response(
					new ReadableStream<Uint8Array>({
						start(controller) {
							responseBodies.add(controller);
							controller.enqueue(
								new TextEncoder().encode(
									`data: ${JSON.stringify({
										id: `manager-stop-${requests}`,
										object: "chat.completion.chunk",
										created: 1,
										model: "gpt-4o",
										choices: [
											{ index: 0, delta: { role: "assistant", content: "working" }, finish_reason: null },
										],
									})}\n\n`,
								),
							);
						},
					}),
					{ headers: { "Content-Type": "text/event-stream" } },
				);
			},
		});
		const model = {
			...getBundledModel("openai", "gpt-4o")!,
			baseUrl: `${server.url}v1`,
			api: "openai-completions" as const,
		};
		const settings = Settings.isolated({ "compaction.enabled": false, "retry.enabled": false });
		const sessionManager = SessionManager.inMemory();
		const authStorage = createInMemoryAuthStorage();
		authStorage.setRuntimeApiKey("openai", "controlled-stop-key");
		const modelRegistry = new ModelRegistry(authStorage, undefined, {
			ignoreLocalModelConfig: true,
			settings,
			cacheDbPath: ":memory:",
		});
		const runtime = new ExtensionRuntime();
		const extension = await loadExtensionFromFactory(
			api =>
				api.on("session_shutdown", async () => {
					shutdownCalls++;
					shutdownEntered.resolve();
					await shutdownGate.promise;
				}),
			process.cwd(),
			new EventBus(),
			runtime,
			"controlled-stop-shutdown",
		);
		const session = new AgentSession({
			agent: new Agent({
				initialState: { model, systemPrompt: ["Controlled cancellation"], messages: [], tools: [] },
				getApiKey: () => "controlled-stop-key",
			}),
			sessionManager,
			settings,
			modelRegistry,
			extensionRunner: new ExtensionRunner([extension], runtime, process.cwd(), sessionManager, modelRegistry),
		});
		const registry = AgentRegistry.global();
		const lifecycle = AgentLifecycleManager.global();
		const id = "controlled-manager-stop-worker";
		const ref = registry.register({
			id,
			displayName: "task",
			kind: "sub",
			parentId: "Main",
			session,
			status: "idle",
		});
		const unsubscribe = registry.syncSessionStatus(id, session);
		lifecycle.adopt(id, { idleTtlMs: 0 }, ref);
		const hub = new AgentHubOverlayComponent({
			settings,
			observers: new SessionObserverRegistry(),
			hubKeys: [],
			onDone: () => {},
			requestRender: () => {},
			registry,
			lifecycle,
			irc: new IrcBus(registry),
		});
		const turn = runSubagentFollowUpTurn({
			id,
			agent: { name: "task", description: "controlled", systemPrompt: "test", source: "bundled" },
			message: "Wait for the controlled provider",
			maxRuntimeMs: 3_000,
		});
		try {
			await Promise.race([
				firstRequest.promise,
				turn.then(result => {
					throw new Error(`Worker ended before its provider request: ${result.error ?? result.stderr}`);
				}),
			]);
			const release = entry === "lifecycle" ? lifecycle.release(id, ref, { tombstone: true }) : undefined;
			if (entry === "hub") hub.handleInput("x");
			await shutdownEntered.promise;
			// This must close while the real session_shutdown extension is still blocked.
			await untilAborted(AbortSignal.timeout(1_000), () => connectionClosed.promise);
			const disposal = session.dispose();
			expect(session.dispose()).toBe(disposal);
			shutdownGate.resolve();
			const outcome = await turn;
			await release;
			await disposal;
			expect(shutdownCalls).toBe(1);

			expect(outcome.aborted).toBe(true);
			expect(outcome.exitCode).toBe(1);
			expect(ref.status).toBe("aborted");
			expect(session.isStreaming).toBe(false);
			expect(requests).toBe(1);
			// A delayed caller holding the old session must not start another request.
			expect(await session.prompt("Late callback after manager kill")).toBe(false);
			expect(requests).toBe(1);
		} finally {
			try {
				shutdownGate.resolve();
				for (const body of responseBodies) {
					try {
						body.close();
					} catch {
						// A cancelled HTTP response may already have closed its stream.
					}
				}
				await server.stop(true);
				await session.abort();
				await session.dispose();
				await turn.catch(() => {});
				hub.dispose();
				unsubscribe();
				registry.unregister(id, ref);
			} finally {
				await authStorage.close();
			}
		}
	},
	10_000,
);
