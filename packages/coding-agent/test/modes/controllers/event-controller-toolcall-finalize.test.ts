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

	it.each(["thinking", "text"] as const)(
		"keeps late %s growth visible and releases closed prose to scrollback",
		async kind => {
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
				session: {
					isStreaming: true,
					subscribe: callback => {
						listener = callback;
						return () => {};
					},
				},
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
			const tool = {
				type: "toolCall",
				id: "interleaved",
				name: "write",
				arguments: {
					path: "/fixture.ts",
					content: Array.from({ length: 40 }, (_, i) => `const row${i} = ${i};`).join("\n"),
				},
			} as const;
			const message = (text: string, later: boolean) =>
				makeStreamingMessage([
					kind === "thinking" ? { type: "thinking", thinking: text } : { type: "text", text },
					...(later ? [kind === "thinking" ? { type: "text" as const, text: "VISIBLE_ANSWER" } : tool] : []),
				]);
			try {
				listener?.({ type: "message_start", message: makeStreamingMessage([]) });
				await flush();
				for (const [text, later] of [
					["Planning", false],
					["Planning", true],
					["Planning next step", true],
				] as const) {
					send({
						type: kind === "thinking" ? "thinking_delta" : "text_delta",
						contentIndex: 0,
						delta: text,
						partial: message(text, later),
					});
					await flush();
				}
				const prose = `Planning next step\n\n${Array.from({ length: 45 }, (_, i) => `RETAINED_PROSE_${i}\n\n`).join("")}Still growing`;
				const partial = message(prose, true);
				send({
					type: kind === "thinking" ? "thinking_delta" : "text_delta",
					contentIndex: 0,
					delta: prose,
					partial,
				});
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
		},
	);

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
describe("EventController retains local tool and post-tool bodies in native history", () => {
	afterEach(() => {
		resetSettingsForTest();
		vi.restoreAllMocks();
	});

	async function createTerminalFixture() {
		await Settings.init({ inMemory: true, overrides: { "display.smoothStreaming": false } });
		vi.useFakeTimers();
		const terminal = new VirtualTerminal(100, 16);
		const scheduler = new VirtualRenderScheduler();
		const composer = new Composer({ terminal, tuiOptions: { renderScheduler: scheduler } });
		const ctx = createInteractiveModeContext({
			ui: composer.ui,
			hideThinkingBlock: false,
			proseOnlyThinking: false,
			toolOutputExpanded: true,
			session: { isStreaming: true },
		});
		const controller = new EventController(ctx);
		composer.setRuntimeChildren([ctx.chatContainer, composer.editor]);
		composer.start({ playWelcomeIntro: false });
		const flush = async () => {
			vi.advanceTimersByTime(34);
			for (let i = 0; i < 12; i++) await Promise.resolve();
			await scheduler.settle(terminal);
		};
		const send = async (event: AgentSessionEvent) => {
			await controller.handleEvent(event);
			await flush();
		};
		return {
			terminal,
			flush,
			send,
			update: (event: AssistantMessageEvent & { partial: AssistantMessage }) =>
				send({ type: "message_update", message: event.partial, assistantMessageEvent: event }),
			history: () => terminal.getScrollBuffer().slice(0, -terminal.rows).join("\n"),
			close: () => {
				controller.dispose();
				composer.stop();
				ctx.chatContainer.dispose();
				vi.useRealTimers();
			},
		};
	}

	function expectOrderedOnce(history: string, markers: string[]) {
		let previous = -1;
		for (const marker of markers) {
			expect(history.split(marker).length - 1, marker + " in native history:\n" + history).toBe(1);
			const position = history.indexOf(marker);
			expect(position).toBeGreaterThan(previous);
			previous = position;
		}
	}

	it("retires canonical local Eval and Todo results across later messages without settling a running tool on resize", async () => {
		const fixture = await createTerminalFixture();
		const { terminal, send, update } = fixture;
		const retained: string[] = [];
		const completeMessage = async (index: number) => {
			const prefix = "TURN_" + index + "_";
			const prose = ["FIRST", "MIDDLE", "LAST"].map(suffix => prefix + suffix);
			const result = ["RESULT_FIRST", "RESULT_MIDDLE", "RESULT_LAST"].map(suffix => prefix + suffix);
			const postTool = index === 0 ? ["TURN_0_POST_TOOL"] : [];
			const tool = {
				type: "toolCall" as const,
				id: "canonical-" + index,
				name: index % 2 ? "todo" : "eval",
				arguments:
					index % 2 ? { op: "view" } : { language: "js", title: "Local cell " + index, code: "display(42)" },
			};
			await send({ type: "message_start", message: makeStreamingMessage([]) });
			const preview = makeStreamingMessage([
				{ type: "text", text: prose.join("\n\n") },
				{ ...tool, id: "provisional-" + index },
				...postTool.map(text => ({ type: "text" as const, text })),
			]);
			await update({ type: "toolcall_delta", contentIndex: 1, delta: "", partial: preview });
			// The final canonical identity arrives only at message_end, never in a final update.
			await send({
				type: "message_end",
				message: makeStreamingMessage([preview.content[0]!, tool, ...preview.content.slice(2)]),
			});
			await send({ type: "tool_execution_start", toolCallId: tool.id, toolName: tool.name, args: tool.arguments });
			await send({
				type: "tool_execution_end",
				toolCallId: tool.id,
				toolName: tool.name,
				isError: false,
				result: { content: [{ type: "text", text: result.join("\n") }] },
			});
			retained.push(...prose, ...result, ...postTool);
		};
		try {
			await completeMessage(0);
			for (let i = 1; i <= 12; i++) await completeMessage(i);
			// Recent cards may still occupy the viewport; earlier complete turns must be native history.
			expectOrderedOnce(fixture.history(), retained.slice(0, -12));
			expectOrderedOnce(terminal.getScrollBuffer().join("\n"), retained);
			const initialTurns = retained.slice();

			const running = {
				type: "toolCall" as const,
				id: "still-running",
				name: "eval",
				arguments: { language: "js", title: "Long local cell", code: "await work()" },
			};
			await send({ type: "message_start", message: makeStreamingMessage([]) });
			await send({ type: "message_end", message: makeStreamingMessage([running]) });
			await send({
				type: "tool_execution_start",
				toolCallId: running.id,
				toolName: running.name,
				args: running.arguments,
			});
			await send({
				type: "tool_execution_update",
				toolCallId: running.id,
				toolName: running.name,
				args: running.arguments,
				partialResult: { content: [{ type: "text", text: "LIVE_BEFORE_RESIZE" }] },
			});
			terminal.resize(76, 20);
			await fixture.flush();
			await send({
				type: "tool_execution_update",
				toolCallId: running.id,
				toolName: running.name,
				args: running.arguments,
				partialResult: { content: [{ type: "text", text: "LIVE_AFTER_RESIZE" }] },
			});
			expect(terminal.getViewport().join("\n")).toContain("LIVE_AFTER_RESIZE");
			expect(fixture.history()).not.toContain("LIVE_AFTER_RESIZE");
			expect(terminal.getScrollBuffer().join("\n")).not.toContain("LIVE_BEFORE_RESIZE");
			const completed = ["COMPLETED_FIRST", "COMPLETED_MIDDLE", "COMPLETED_LAST"];
			await send({
				type: "tool_execution_end",
				toolCallId: running.id,
				toolName: running.name,
				isError: false,
				result: { content: [{ type: "text", text: completed.join("\n") }] },
			});
			await completeMessage(13);
			await completeMessage(14);
			expectOrderedOnce(fixture.history(), [...initialTurns, ...completed]);
			expect(terminal.getScrollBuffer().join("\n")).not.toContain("LIVE_AFTER_RESIZE");
		} finally {
			fixture.close();
		}
	});

	it.each(["text", "thinking"] as const)(
		"keeps indexed post-tool %s revisable until explicit ends release complete native history",
		async kind => {
			const fixture = await createTerminalFixture();
			const { send, update, terminal } = fixture;
			const toolA = {
				type: "toolCall" as const,
				id: "tool-a",
				name: "eval",
				arguments: { language: "js", code: "display(1)" },
			};
			const toolB = { ...toolA, id: "tool-b", arguments: { language: "js", code: "display(2)" } };
			const proseBlock = (text: string): AssistantMessage["content"][number] =>
				kind === "text" ? { type: "text", text } : { type: "thinking", thinking: text };
			const child =
				kind === "text"
					? { type: "thinking" as const, thinking: "CHILD_INITIAL" }
					: { type: "text" as const, text: "CHILD_INITIAL" };
			const partial = (text: string, later: boolean) =>
				makeStreamingMessage([toolA, proseBlock(text), ...(later ? [child, toolB] : [])]);
			const delta = (text: string, later: boolean) =>
				update({
					type: kind === "text" ? "text_delta" : "thinking_delta",
					contentIndex: 1,
					delta: text,
					partial: partial(text, later),
				});
			try {
				await send({ type: "message_start", message: makeStreamingMessage([]) });
				await update({
					type: "toolcall_delta",
					contentIndex: 0,
					delta: "",
					partial: makeStreamingMessage([toolA]),
				});
				await send({
					type: "tool_execution_start",
					toolCallId: toolA.id,
					toolName: toolA.name,
					args: toolA.arguments,
				});
				await send({
					type: "tool_execution_end",
					toolCallId: toolA.id,
					toolName: toolA.name,
					isError: false,
					result: { content: [{ type: "text", text: "TOOL_A_RESULT" }] },
				});
				await delta("SEGMENT_INITIAL", false);
				await delta("SEGMENT_INITIAL", true);
				await delta("SEGMENT_INITIAL REVISED_AFTER_CHILD", true);
				expect(terminal.getScrollBuffer().join("\n")).toContain("REVISED_AFTER_CHILD");
				const rows = Array.from({ length: 36 }, (_, i) => "SEGMENT_ROW_" + String(i).padStart(2, "0"));
				const prose = ["SEGMENT_INITIAL REVISED_AFTER_CHILD", ...rows, "SEGMENT_FINAL"].join("\n\n");
				await delta(prose, true);
				const final = partial(prose, true);
				final.content[2] =
					kind === "text"
						? { type: "thinking", thinking: "CHILD_INITIAL CHILD_FINAL" }
						: { type: "text", text: "CHILD_INITIAL CHILD_FINAL" };
				await update({
					type: kind === "text" ? "thinking_delta" : "text_delta",
					contentIndex: 2,
					delta: " CHILD_FINAL",
					partial: final,
				});
				// Full-message indices 1 and 2 must close local segment children 0 and 1.
				await update({
					type: kind === "text" ? "text_end" : "thinking_end",
					contentIndex: 1,
					content: prose,
					partial: final,
				});
				await update({
					type: kind === "text" ? "thinking_end" : "text_end",
					contentIndex: 2,
					content: "CHILD_INITIAL CHILD_FINAL",
					partial: final,
				});
				await send({
					type: "tool_execution_start",
					toolCallId: toolB.id,
					toolName: toolB.name,
					args: toolB.arguments,
				});
				await send({
					type: "tool_execution_end",
					toolCallId: toolB.id,
					toolName: toolB.name,
					isError: false,
					result: {
						content: [{ type: "text", text: Array.from({ length: 30 }, (_, i) => "TOOL_B_ROW_" + i).join("\n") }],
					},
				});
				// No message_end: explicit child ends and the later tool must release prose now.
				expectOrderedOnce(fixture.history(), [
					"TOOL_A_RESULT",
					"REVISED_AFTER_CHILD",
					...rows,
					"SEGMENT_FINAL",
					"CHILD_FINAL",
				]);
			} finally {
				fixture.close();
			}
		},
	);
});
