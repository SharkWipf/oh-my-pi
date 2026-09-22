import { describe, expect, it } from "bun:test";
import { Agent, type AgentMessage, type StreamFn } from "@oh-my-pi/pi-agent-core";
import type { Message } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { untilAborted } from "@oh-my-pi/pi-utils";
import { createAssistantMessage } from "./helpers/agent-session-setup";

function createSession(sideStreamFn: StreamFn, convertToLlm?: (messages: AgentMessage[]) => Promise<Message[]>) {
	const session = new AgentSession({
		agent: new Agent({
			initialState: {
				model: getBundledModel("anthropic", "claude-sonnet-4-5")!,
				systemPrompt: ["Test"],
				messages: [],
				tools: [],
			},
		}),
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: { resolver: () => async () => "test-key" } as never,
		sideStreamFn,
		convertToLlm,
	});
	session.setPlanModeState({ enabled: true, planFilePath: "/tmp/ephemeral-cancellation-plan" });
	return session;
}

function finishReply(stream: AssistantMessageEventStream, text: string): void {
	const message = createAssistantMessage(text);
	stream.push({ type: "text_delta", contentIndex: 0, delta: text, partial: message });
	stream.push({ type: "done", reason: "stop", message });
}

describe("AgentSession ephemeral cancellation", () => {
	it("aborts the live HTTP request owned by an IRC auto-reply", async () => {
		const received = Promise.withResolvers<void>();
		const disconnected = Promise.withResolvers<void>();
		let responseBody: ReadableStreamDefaultController<Uint8Array> | undefined;
		const server = Bun.serve({
			hostname: "127.0.0.1",
			port: 0,
			fetch(request) {
				request.signal.addEventListener("abort", () => disconnected.resolve(), { once: true });
				received.resolve();
				return new Response(
					new ReadableStream<Uint8Array>({
						start(controller) {
							responseBody = controller;
							controller.enqueue(new TextEncoder().encode("pending reply"));
						},
					}),
				);
			},
		});
		const session = createSession(async (_model, _context, options) => {
			const response = await fetch(server.url, { signal: options?.signal });
			const text = await response.text();
			const stream = new AssistantMessageEventStream();
			finishReply(stream, text);
			return stream;
		});
		try {
			await session.deliverIrcMessage(
				{ id: "http-reply", from: "ephemeral-parent", to: "ephemeral-child", body: "Status?", ts: Date.now() },
				{ expectsReply: true },
			);
			await received.promise;
			await session.abort();
			// Real socket closure cannot use fake timers; this deadline only bounds a broken transport.
			await untilAborted(AbortSignal.timeout(1_000), () => disconnected.promise);
			await session.waitForIrcReplies();
		} finally {
			try {
				responseBody?.close();
			} catch {
				// The request's abort may already have closed its response stream.
			}
			await server.stop(true);
			await session.waitForIrcReplies();
			await session.dispose();
		}
	});

	it("does not deliver a late successful provider reply after the session was aborted", async () => {
		const started = Promise.withResolvers<void>();
		const stream = new AssistantMessageEventStream();
		const session = createSession(() => {
			started.resolve();
			return stream;
		});
		const parent = createSession(() => {
			throw new Error("An incoming reply must not start a provider request in plan mode");
		});
		const registry = AgentRegistry.global();
		const parentId = "ephemeral-late-reply-parent";
		const parentRef = registry.register({
			id: parentId,
			displayName: parentId,
			kind: "main",
			status: "idle",
			session: parent,
		});
		try {
			await session.deliverIrcMessage(
				{ id: "late-reply", from: parentId, to: "ephemeral-child", body: "Status?", ts: Date.now() },
				{ expectsReply: true },
			);
			await started.promise;
			await session.abort();
			// Simulate a custom transport finishing successfully despite cancellation.
			finishReply(stream, "I am still working after stop");
			await session.waitForIrcReplies();
			expect(
				parent.messages.filter(message => message.role === "custom" && message.customType === "irc:incoming"),
			).toEqual([]);
		} finally {
			stream.end();
			registry.unregister(parentId, parentRef);
			await session.dispose();
			await parent.dispose();
		}
	});

	it("does not start a provider request when conversion finishes after disposal", async () => {
		const converting = Promise.withResolvers<void>();
		const converted = Promise.withResolvers<Message[]>();
		let requests = 0;
		const session = createSession(
			() => {
				requests++;
				const stream = new AssistantMessageEventStream();
				finishReply(stream, "Late request");
				return stream;
			},
			() => {
				converting.resolve();
				return converted.promise;
			},
		);
		const reply = session.runEphemeralTurn({ promptText: "Status?" }).then(
			() => "completed",
			() => "cancelled",
		);
		try {
			await converting.promise;
			session.beginDispose();
			converted.resolve([]);
			expect(await reply).toBe("cancelled");
			expect(requests).toBe(0);
			await expect(session.runEphemeralTurn({ promptText: "Another request" })).rejects.toThrow();
			expect(requests).toBe(0);
		} finally {
			converted.resolve([]);
			await reply;
			await session.dispose();
		}
	});
});

it("does not resume a pending prompt or accept new prompts after disposal starts", async () => {
	const requestedKey = Promise.withResolvers<void>();
	const key = Promise.withResolvers<string>();
	let providerRequests = 0;
	const session = new AgentSession({
		agent: new Agent({
			initialState: {
				model: getBundledModel("anthropic", "claude-sonnet-4-5")!,
				systemPrompt: ["Test"],
				messages: [],
				tools: [],
			},
			streamFn: () => {
				providerRequests++;
				const stream = new AssistantMessageEventStream();
				finishReply(stream, "Late prompt answer");
				return stream;
			},
		}),
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: {
			getApiKey: () => {
				requestedKey.resolve();
				return key.promise;
			},
			resolver: () => async () => "test-key",
		} as never,
	});
	const pending = session.prompt("Prompt waiting on credentials");
	try {
		await requestedKey.promise;
		session.beginDispose();
		key.resolve("test-key");
		await pending;
		expect(await session.prompt("Late callback prompt")).toBe(false);
		expect(providerRequests).toBe(0);
	} finally {
		key.resolve("test-key");
		await pending;
		await session.dispose();
	}
});
