import { afterEach, beforeAll, describe, expect, it, vi } from "bun:test";
import type { AssistantMessage, AssistantMessageEvent } from "@oh-my-pi/pi-ai";
import { resetSettingsForTest, Settings, settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AssistantMessageComponent } from "@oh-my-pi/pi-coding-agent/modes/components/assistant-message";
import { ToolExecutionComponent } from "@oh-my-pi/pi-coding-agent/modes/components/tool-execution";
import { Composer } from "@oh-my-pi/pi-coding-agent/modes/composer";
import { EventController } from "@oh-my-pi/pi-coding-agent/modes/controllers/event-controller";
import { initTheme } from "@oh-my-pi/pi-coding-agent/modes/theme/theme";
import type { AgentSessionEvent } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import type { Component } from "@oh-my-pi/pi-tui";
import { VirtualRenderScheduler } from "../../../../tui/test/virtual-render-scheduler";
import { VirtualTerminal } from "../../../../tui/test/virtual-terminal";
import { createInteractiveModeContext } from "../../helpers/interactive-mode-context";

beforeAll(async () => {
	await initTheme();
});

function makeStreamingMessage(content: AssistantMessage["content"]): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-5",
		stopReason: "stop",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		timestamp: Date.now(),
	};
}

// Components the controller mounts during a dispatch (pending tool previews).
// Sealed in afterEach so their spinner intervals never outlive the test file.
const mountedComponents: Component[] = [];

function createFixture(streamingMessage: AssistantMessage) {
	const streamingComponent = new AssistantMessageComponent();
	const ctx = createInteractiveModeContext({ streamingComponent, streamingMessage });
	const addChild = ctx.chatContainer.addChild.bind(ctx.chatContainer);
	vi.spyOn(ctx.chatContainer, "addChild").mockImplementation(child => {
		mountedComponents.push(child);
		addChild(child);
	});

	const controller = new EventController(ctx);
	return { controller, ctx };
}

describe("EventController finalizes assistant block when tool-call args stream", () => {
	afterEach(() => {
		for (const component of mountedComponents.splice(0)) {
			if (component instanceof ToolExecutionComponent) component.seal();
		}
		resetSettingsForTest();
		vi.restoreAllMocks();
	});

	it.each(["thinking", "text"] as const)("keeps late %s growth visible and releases closed prose to scrollback", async kind => {
		await Settings.init({ inMemory: true, overrides: { "display.smoothStreaming": false } });
		vi.useFakeTimers();
		const terminal = new VirtualTerminal(100, kind === "text" ? 8 : 24);
		const scheduler = new VirtualRenderScheduler();
		const composer = new Composer({ terminal, tuiOptions: { renderScheduler: scheduler } });
		let listener: ((event: AgentSessionEvent) => void) | undefined;
		const ctx = createInteractiveModeContext({
			ui: composer.ui,
			hideThinkingBlock: false,
			proseOnlyThinking: false,
			session: { isStreaming: true, subscribe: callback => { listener = callback; return () => {}; } },
		});
		const controller = new EventController(ctx);
		controller.subscribeToAgent();
		composer.setRuntimeChildren([ctx.chatContainer, composer.editor]);
		composer.start({ playWelcomeIntro: false });
		const flush = async () => {
			vi.advanceTimersByTime(34);
			for (let i = 0; i < 12; i++) await Promise.resolve();
			await scheduler.settle(terminal);
		};
		const send = (event: AssistantMessageEvent & { partial: AssistantMessage }) => {
			listener?.({ type: "message_update", message: event.partial, assistantMessageEvent: event });
		};
		const tool = { type: "toolCall", id: "interleaved", name: "write", arguments: { path: "/fixture.ts", content: Array.from({ length: 40 }, (_, i) => `const row${i} = ${i};`).join("\n") } } as const;
		const message = (text: string, later: boolean) => makeStreamingMessage([
			kind === "thinking" ? { type: "thinking", thinking: text } : { type: "text", text },
			...(later ? [kind === "thinking" ? { type: "text" as const, text: "VISIBLE_ANSWER" } : tool] : []),
		]);
		try {
			listener?.({ type: "message_start", message: makeStreamingMessage([]) });
			await flush();
			for (const [text, later] of [["Planning", false], ["Planning", true], ["Planning next step", true]] as const) {
				send({ type: kind === "thinking" ? "thinking_delta" : "text_delta", contentIndex: 0, delta: text, partial: message(text, later) });
				await flush();
			}
			const prose = `Planning next step\n\n${Array.from({ length: 45 }, (_, i) => `RETAINED_PROSE_${i}\n\n`).join("")}Still growing`;
			const partial = message(prose, true);
			send({ type: kind === "thinking" ? "thinking_delta" : "text_delta", contentIndex: 0, delta: prose, partial });
			await flush();
			if (kind === "thinking") {
				// The early paragraph must already be reachable in native history,
				// not lost above a clipped live tail after a stale publication froze.
				expect(terminal.getScrollBuffer().join("\n")).toContain("RETAINED_PROSE_0");
			}
			send({ type: kind === "thinking" ? "thinking_end" : "text_end", contentIndex: 0, content: prose, partial });
			// A later delta in the same coalescing window must not erase the end.
			const withTool = kind === "thinking" ? { ...partial, content: [...partial.content, tool] } : partial;
			if (kind === "thinking") send({ type: "text_end", contentIndex: 1, content: "VISIBLE_ANSWER", partial });
			send({ type: "toolcall_delta", contentIndex: withTool.content.length - 1, delta: "", partial: withTool });
			await flush();
			const screen = terminal.getScrollBuffer().join("\n");
			expect(screen).toContain("RETAINED_PROSE_0");
			expect(screen).toContain("RETAINED_PROSE_44");
			expect(screen.indexOf("RETAINED_PROSE_0")).toBeLessThan(screen.indexOf("RETAINED_PROSE_44"));
		} finally {
			controller.dispose();
			composer.stop();
			ctx.chatContainer.dispose();
			vi.useRealTimers();
		}
	});

	it("emits the per-turn usage row with the turn's local timestamp at message_end", async () => {
		await Settings.init({ inMemory: true, cwd: process.cwd() });
		settings.set("display.showTokenUsage", true);
		// Fixed local wall-clock time; single-digit fields exercise zero-padding.
		const timestamp = new Date(2026, 0, 2, 3, 4, 5).getTime();
		const message: AssistantMessage = {
			...makeStreamingMessage([{ type: "text", text: "done" }]),
			usage: {
				input: 1234,
				output: 7,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 1241,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			timestamp,
		};
		const { controller } = createFixture(message);
		await controller.handleEvent({ type: "message_end", message } as Extract<
			AgentSessionEvent,
			{ type: "message_end" }
		>);
		const row = mountedComponents.at(-1) as unknown as { render(width: number): string[] } | undefined;
		expect(row).toBeDefined();
		expect(row?.render(120).join("\n")).toContain("2026-01-02 03:04:05");
	});
});
describe("EventController finalizes orphaned post-tool assistant segments", () => {
	afterEach(() => {
		for (const component of mountedComponents.splice(0)) {
			if (component instanceof ToolExecutionComponent) component.seal();
		}
		resetSettingsForTest();
		vi.restoreAllMocks();
	});

	// Regression: post-tool assistant segments are created unfinalized at
	// message_update and finalized only at message_end. A dropped message_end
	// (mid-stream throw, superseded attempt) used to leave the segment active
	// forever — one unfinalized block at the transcript frontier blocks history
	// retirement, so every later block degraded to its one-line live allocation.
	it("finalizes a segment whose message_end never fired at the next message_start", async () => {
		await Settings.init({ inMemory: true, cwd: process.cwd() });
		const message = makeStreamingMessage([
			{ type: "toolCall", id: "tc-seg", name: "write", arguments: { file_path: "/tmp/c.ts", content: "z" } },
			{ type: "text", text: "post-tool commentary" },
		]);
		const { controller, ctx } = createFixture(message);
		await controller.handleEvent({
			type: "message_update",
			message,
			assistantMessageEvent: { type: "text_delta", contentIndex: 1, delta: "post-tool commentary", partial: message },
		} as Extract<AgentSessionEvent, { type: "message_update" }>);
		const segment = ctx.chatContainer.children.find(child => child instanceof AssistantMessageComponent);
		expect(segment).toBeInstanceOf(AssistantMessageComponent);
		expect((segment as AssistantMessageComponent).isTranscriptBlockFinalized()).toBe(false);

		await controller.handleEvent({
			type: "message_start",
			message: makeStreamingMessage([]),
		} as Extract<AgentSessionEvent, { type: "message_start" }>);
		expect((segment as AssistantMessageComponent).isTranscriptBlockFinalized()).toBe(true);
		controller.dispose();
	});
});
