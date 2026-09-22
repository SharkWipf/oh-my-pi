import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "bun:test";
import { type } from "@oh-my-pi/omptype";
import type { AgentTool } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, ToolCall, ToolResultMessage, Usage } from "@oh-my-pi/pi-ai";
import { resetSettingsForTest, Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { ChatTranscriptBuilder } from "@oh-my-pi/pi-tui/chat/chat-transcript-builder";
import { ReadToolGroupComponent } from "@oh-my-pi/pi-tui/chat/read-tool-group";
import { EventController } from "@oh-my-pi/pi-coding-agent/modes/controllers/event-controller";
import { initTheme } from "@oh-my-pi/pi-tui/theme";
import { UiHelpers } from "@oh-my-pi/pi-coding-agent/modes/utils/ui-helpers";
import type { AgentSessionEvent } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { type Component, TERMINAL } from "@oh-my-pi/pi-tui";
import { createInteractiveModeContext } from "./helpers/interactive-mode-context";

const TOOL_CALL_A_ID = "toolu_mixed_text_order_a";
const TOOL_CALL_B_ID = "toolu_mixed_text_order_b";
const INTRO_MARKER = "INTRO TEXT BEFORE FIRST TOOL";
const TOOL_RESULT_A_MARKER = "TOOL RESULT FROM FIRST TOOL";
const MIDDLE_MARKER = "MIDDLE TEXT BETWEEN TOOL CALLS";
const TOOL_RESULT_B_MARKER = "TOOL RESULT FROM SECOND TOOL";
const FINAL_MARKER = "FINAL ANSWER AFTER SECOND TOOL";
const HIDDEN_BASH_COMMAND_MARKER = "HIDDEN BASH COMMAND MARKER";
const HIDDEN_BASH_FAILURE_MARKER = "HIDDEN BASH FAILURE MARKER";
const HIDDEN_READ_PATH_MARKER = "hidden-tool-activity.ts";

function zeroUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function assistantMessage(content: AssistantMessage["content"]): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "cursor",
		provider: "cursor",
		model: "cursor-model",
		stopReason: "stop",
		usage: zeroUsage(),
		timestamp: 1,
	};
}

function lineContaining(lines: string[], marker: string): number {
	const index = lines.findIndex(line => line.includes(marker));
	if (index === -1) {
		throw new Error(`Rendered transcript did not contain ${marker}:\n${lines.join("\n")}`);
	}
	return index;
}

function createFixture(
	hideToolActivity = false,
	toolByName: (name: string) => AgentTool | undefined = () => undefined,
) {
	let hasDisplayableThinkingContent = false;
	const ctx = createInteractiveModeContext({
		session: { getToolByName: toolByName },
		hideToolActivity,
		noteDisplayableThinkingContent: vi.fn((message: AssistantMessage) => {
			const hasThinking = message.content.some(
				content => content.type === "thinking" && content.thinking.trim() !== "",
			);
			if (!hasThinking || hasDisplayableThinkingContent) return false;
			hasDisplayableThinkingContent = true;
			return true;
		}),
		lastAssistantUsage: zeroUsage(),
	});
	ctx.chatContainer.setToolActivityVisible(!hideToolActivity);

	return { controller: new EventController(ctx), chatContainer: ctx.chatContainer, ctx };
}

describe("EventController mixed assistant text/tool rendering", () => {
	beforeAll(async () => {
		await initTheme(false);
	});

	beforeEach(async () => {
		resetSettingsForTest();
		await Settings.init({ inMemory: true, overrides: { "display.smoothStreaming": false } });
	});

	afterEach(() => {
		vi.restoreAllMocks();
		resetSettingsForTest();
	});

	it("finalizes and removes an orphaned streaming component on the next message_start", async () => {
		// Regression: a stream that died between message_start and message_end
		// (transport drop, hook throw) left its component live in the transcript.
		// One unfinalized block at the retirement frontier blocks history commits
		// for everything after it, so the whole transcript tail stayed in the
		// mutable viewport in pressure mode (no separators, compacted blocks).
		const { controller, chatContainer } = createFixture();

		await controller.handleEvent({ type: "message_start", message: assistantMessage([]) } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);
		const partial = assistantMessage([{ type: "thinking", thinking: "**dead attempt**" }]);
		await controller.handleEvent({
			type: "message_update",
			message: partial,
			assistantMessageEvent: { type: "thinking_delta", contentIndex: 0, delta: "**dead attempt**", partial },
		} as Extract<AgentSessionEvent, { type: "message_update" }>);
		const orphan = chatContainer.children.at(-1) as Component & {
			isTranscriptBlockFinalized(): boolean;
		};
		expect(orphan.isTranscriptBlockFinalized()).toBe(false);

		// Retry attempt streams a fresh message without the dead one ever ending.
		await controller.handleEvent({ type: "message_start", message: assistantMessage([]) } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);

		expect(chatContainer.children).not.toContain(orphan);
		expect(orphan.isTranscriptBlockFinalized()).toBe(true);
	});

	it("renders assistant text segments in order around two tool results from one mixed message", async () => {
		const { controller, chatContainer } = createFixture();
		const toolCallA: ToolCall = {
			type: "toolCall",
			id: TOOL_CALL_A_ID,
			name: "contract_probe_a",
			arguments: { value: "a" },
		};
		const toolCallB: ToolCall = {
			type: "toolCall",
			id: TOOL_CALL_B_ID,
			name: "contract_probe_b",
			arguments: { value: "b" },
		};
		const started = assistantMessage([]);
		const withFirstToolCall = assistantMessage([{ type: "text", text: INTRO_MARKER }, toolCallA]);
		const withSecondToolCall = assistantMessage([
			{ type: "text", text: INTRO_MARKER },
			toolCallA,
			{ type: "text", text: MIDDLE_MARKER },
			toolCallB,
		]);
		const completed = assistantMessage([
			{ type: "text", text: INTRO_MARKER },
			toolCallA,
			{ type: "text", text: MIDDLE_MARKER },
			toolCallB,
			{ type: "text", text: FINAL_MARKER },
		]);

		await controller.handleEvent({ type: "message_start", message: started } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);
		await controller.handleEvent({
			type: "message_update",
			message: withFirstToolCall,
			assistantMessageEvent: {
				type: "toolcall_end",
				contentIndex: 1,
				toolCall: toolCallA,
				partial: withFirstToolCall,
			},
		} as Extract<AgentSessionEvent, { type: "message_update" }>);
		await controller.handleEvent({
			type: "message_update",
			message: withSecondToolCall,
			assistantMessageEvent: {
				type: "toolcall_end",
				contentIndex: 3,
				toolCall: toolCallB,
				partial: withSecondToolCall,
			},
		} as Extract<AgentSessionEvent, { type: "message_update" }>);
		const liveLines = chatContainer.render(120).map(line => Bun.stripANSI(line));
		expect(lineContaining(liveLines, INTRO_MARKER)).toBeLessThan(lineContaining(liveLines, MIDDLE_MARKER));
		await controller.handleEvent({
			type: "tool_execution_start",
			toolCallId: TOOL_CALL_A_ID,
			toolName: "contract_probe_a",
			args: { value: "a" },
		} as Extract<AgentSessionEvent, { type: "tool_execution_start" }>);
		await controller.handleEvent({
			type: "tool_execution_end",
			toolCallId: TOOL_CALL_A_ID,
			toolName: "contract_probe_a",
			result: { content: [{ type: "text", text: TOOL_RESULT_A_MARKER }] },
			isError: false,
		} as Extract<AgentSessionEvent, { type: "tool_execution_end" }>);
		await controller.handleEvent({
			type: "tool_execution_start",
			toolCallId: TOOL_CALL_B_ID,
			toolName: "contract_probe_b",
			args: { value: "b" },
		} as Extract<AgentSessionEvent, { type: "tool_execution_start" }>);
		await controller.handleEvent({
			type: "tool_execution_end",
			toolCallId: TOOL_CALL_B_ID,
			toolName: "contract_probe_b",
			result: { content: [{ type: "text", text: TOOL_RESULT_B_MARKER }] },
			isError: false,
		} as Extract<AgentSessionEvent, { type: "tool_execution_end" }>);
		await controller.handleEvent({ type: "message_end", message: completed } as Extract<
			AgentSessionEvent,
			{ type: "message_end" }
		>);

		const lines = chatContainer.render(120).map(line => Bun.stripANSI(line));
		const introLine = lineContaining(lines, INTRO_MARKER);
		const toolResultALine = lineContaining(lines, TOOL_RESULT_A_MARKER);
		const middleLine = lineContaining(lines, MIDDLE_MARKER);
		const toolResultBLine = lineContaining(lines, TOOL_RESULT_B_MARKER);
		const finalLine = lineContaining(lines, FINAL_MARKER);

		expect(introLine).toBeLessThan(toolResultALine);
		expect(toolResultALine).toBeLessThan(middleLine);
		expect(lines.filter(line => line.includes(MIDDLE_MARKER))).toHaveLength(1);
		expect(middleLine).toBeLessThan(toolResultBLine);
		expect(toolResultBLine).toBeLessThan(finalLine);
	});

	it("uses the canonical mounted-tool renderer for prefixed calls live and after transcript rebuild", async () => {
		const githubTool: AgentTool = {
			name: "github",
			label: "GitHub",
			description: "GitHub test tool",
			parameters: type({}),
			execute: async () => ({ content: [] }),
		};
		const toolByName = (name: string) => (name === "github" || name === "xd://github" ? githubTool : undefined);
		const toolCall: ToolCall = {
			type: "toolCall",
			id: "toolu_prefixed_github",
			name: "xd://github",
			arguments: { op: "repo_view", repo: "can1357/oh-my-pi" },
		};
		const streaming = assistantMessage([toolCall]);

		const live = createFixture(false, toolByName);
		await live.controller.handleEvent({ type: "message_start", message: assistantMessage([]) } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);
		await live.controller.handleEvent({
			type: "message_update",
			message: streaming,
			assistantMessageEvent: {
				type: "toolcall_end",
				contentIndex: 0,
				toolCall,
				partial: streaming,
			},
		} as Extract<AgentSessionEvent, { type: "message_update" }>);
		expect(Bun.stripANSI(live.chatContainer.render(120).join("\n"))).toContain("GitHub Repo can1357/oh-my-pi");

		const executionOnly = createFixture(false, toolByName);
		await executionOnly.controller.handleEvent({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		} as Extract<AgentSessionEvent, { type: "tool_execution_start" }>);
		expect(Bun.stripANSI(executionOnly.chatContainer.render(120).join("\n"))).toContain(
			"GitHub Repo can1357/oh-my-pi",
		);

		const rebuilt = createFixture(false, toolByName);
		const rebuiltHelpers = new UiHelpers(rebuilt.ctx);
		rebuilt.ctx.addMessageToChat = (message, options) => rebuiltHelpers.addMessageToChat(message, options);
		rebuiltHelpers.renderSessionContext({
			messages: [streaming],
			models: {},
			injectedTtsrRules: [],
			mode: "none",
		});
		expect(Bun.stripANSI(rebuilt.chatContainer.render(120).join("\n"))).toContain("GitHub Repo can1357/oh-my-pi");

		// Canonicalization is presentation-only; provider replay keeps the wire spelling.
		expect(toolCall.name).toBe("xd://github");
	});

	it("keeps assistant text streaming while hiding bash failures and grouped read activity", async () => {
		const { controller, chatContainer } = createFixture(true);
		const bashCall: ToolCall = {
			type: "toolCall",
			id: TOOL_CALL_A_ID,
			name: "bash",
			arguments: { command: `printf '${HIDDEN_BASH_COMMAND_MARKER}'` },
		};
		const readCall: ToolCall = {
			type: "toolCall",
			id: TOOL_CALL_B_ID,
			name: "read",
			arguments: { path: HIDDEN_READ_PATH_MARKER },
		};
		const started = assistantMessage([]);
		const streaming = assistantMessage([
			{ type: "text", text: INTRO_MARKER },
			bashCall,
			{ type: "text", text: MIDDLE_MARKER },
			readCall,
			{ type: "text", text: FINAL_MARKER },
		]);

		await controller.handleEvent({ type: "message_start", message: started } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);
		await controller.handleEvent({
			type: "message_update",
			message: streaming,
			assistantMessageEvent: {
				type: "toolcall_end",
				contentIndex: 3,
				toolCall: readCall,
				partial: streaming,
			},
		} as Extract<AgentSessionEvent, { type: "message_update" }>);
		await controller.handleEvent({
			type: "tool_execution_start",
			toolCallId: TOOL_CALL_A_ID,
			toolName: "bash",
			args: bashCall.arguments,
		} as Extract<AgentSessionEvent, { type: "tool_execution_start" }>);
		await controller.handleEvent({
			type: "tool_execution_end",
			toolCallId: TOOL_CALL_A_ID,
			toolName: "bash",
			result: { content: [{ type: "text", text: HIDDEN_BASH_FAILURE_MARKER }] },
			isError: true,
		} as Extract<AgentSessionEvent, { type: "tool_execution_end" }>);
		await controller.handleEvent({
			type: "tool_execution_start",
			toolCallId: TOOL_CALL_B_ID,
			toolName: "read",
			args: readCall.arguments,
		} as Extract<AgentSessionEvent, { type: "tool_execution_start" }>);
		await controller.handleEvent({
			type: "tool_execution_end",
			toolCallId: TOOL_CALL_B_ID,
			toolName: "read",
			result: { content: [{ type: "text", text: "read result must stay hidden" }] },
			isError: false,
		} as Extract<AgentSessionEvent, { type: "tool_execution_end" }>);
		await controller.handleEvent({ type: "message_end", message: streaming } as Extract<
			AgentSessionEvent,
			{ type: "message_end" }
		>);

		const rendered = Bun.stripANSI(chatContainer.render(120).join("\n"));
		expect(rendered).toContain(INTRO_MARKER);
		expect(rendered).toContain(MIDDLE_MARKER);
		expect(rendered).toContain(FINAL_MARKER);
		expect(rendered).not.toContain(HIDDEN_BASH_COMMAND_MARKER);
		expect(rendered).not.toContain(HIDDEN_BASH_FAILURE_MARKER);
		expect(rendered).not.toContain(HIDDEN_READ_PATH_MARKER);
	});

	it("keeps late inter-read text between separate read results instead of grouping across its empty start", async () => {
		const { controller, chatContainer, ctx } = createFixture();
		const firstPath = "first-delayed-read.ts";
		const secondPath = "second-delayed-read.ts";
		const readA: ToolCall = {
			type: "toolCall",
			id: TOOL_CALL_A_ID,
			name: "read",
			arguments: { path: firstPath },
		};
		const readB: ToolCall = {
			type: "toolCall",
			id: TOOL_CALL_B_ID,
			name: "read",
			arguments: { path: secondPath },
		};
		const intro = { type: "text" as const, text: INTRO_MARKER };
		try {
			await controller.handleEvent({ type: "message_start", message: assistantMessage([]) });
			const firstRead = assistantMessage([intro, readA]);
			await controller.handleEvent({
				type: "message_update",
				message: firstRead,
				assistantMessageEvent: { type: "toolcall_end", contentIndex: 1, toolCall: readA, partial: firstRead },
			});
			const emptyMiddle = assistantMessage([intro, readA, { type: "text", text: "" }]);
			await controller.handleEvent({
				type: "message_update",
				message: emptyMiddle,
				assistantMessageEvent: { type: "text_start", contentIndex: 2, partial: emptyMiddle },
			});
			const bothReads = assistantMessage([...emptyMiddle.content, readB]);
			await controller.handleEvent({
				type: "message_update",
				message: bothReads,
				assistantMessageEvent: { type: "toolcall_end", contentIndex: 3, toolCall: readB, partial: bothReads },
			});
			for (const read of [readA, readB]) {
				await controller.handleEvent({
					type: "tool_execution_start",
					toolCallId: read.id,
					toolName: read.name,
					args: read.arguments,
				});
				await controller.handleEvent({
					type: "tool_execution_end",
					toolCallId: read.id,
					toolName: read.name,
					isError: false,
					result: { content: [{ type: "text", text: "file contents" }] },
				});
			}
			// The separator acquires visible content only after both read cards exist.
			const middle = assistantMessage([intro, readA, { type: "text", text: MIDDLE_MARKER }, readB]);
			await controller.handleEvent({
				type: "message_update",
				message: middle,
				assistantMessageEvent: { type: "text_end", contentIndex: 2, content: MIDDLE_MARKER, partial: middle },
			});
			const completed = assistantMessage([...middle.content, { type: "text", text: FINAL_MARKER }]);
			await controller.handleEvent({
				type: "message_update",
				message: completed,
				assistantMessageEvent: { type: "text_end", contentIndex: 4, content: FINAL_MARKER, partial: completed },
			});
			await controller.handleEvent({ type: "message_end", message: completed });

			const replay = new ChatTranscriptBuilder({ ui: ctx.ui, cwd: process.cwd(), requestRender: () => {} });
			try {
				replay.rebuild([
					{ type: "message", id: "assistant", parentId: null, timestamp: "2026-09-15", message: completed },
					...[readA, readB].map(read => ({
						type: "message" as const,
						id: read.id,
						parentId: null,
						timestamp: "2026-09-15",
						message: {
							role: "toolResult" as const,
							toolCallId: read.id,
							toolName: read.name,
							content: [{ type: "text" as const, text: "file contents" }],
							isError: false,
							timestamp: 1,
						},
					})),
				]);
				for (const container of [chatContainer, replay.container]) {
					const lines = container.render(120).map(line => Bun.stripANSI(line));
					const transcript = lines.join("\n");
					const markers = [INTRO_MARKER, firstPath, MIDDLE_MARKER, secondPath, FINAL_MARKER];
					let previous = -1;
					for (const marker of markers) {
						expect(transcript.split(marker).length - 1, marker).toBe(1);
						const position = lineContaining(lines, marker);
						expect(position).toBeGreaterThan(previous);
						previous = position;
					}
				}
			} finally {
				replay.dispose();
			}
		} finally {
			controller.dispose();
			chatContainer.dispose();
		}
	});

	it("does not recreate a completed grouped read when later thinking arrives", async () => {
		const { controller, chatContainer, ctx } = createFixture();
		const readCall: ToolCall = {
			type: "toolCall",
			id: "read-completed-stable",
			name: "read",
			arguments: { path: "stable-completed-read.ts" },
		};
		const withRead = assistantMessage([{ type: "thinking", thinking: "planning the read" }, readCall]);
		const withLaterThinking = assistantMessage([
			{ type: "thinking", thinking: "planning the read" },
			readCall,
			{ type: "thinking", thinking: "more reasoning after the read finished" },
		]);

		await controller.handleEvent({ type: "message_start", message: assistantMessage([]) } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);
		await controller.handleEvent({
			type: "message_update",
			message: withRead,
			assistantMessageEvent: { type: "toolcall_end", contentIndex: 1, toolCall: readCall, partial: withRead },
		} as Extract<AgentSessionEvent, { type: "message_update" }>);
		await controller.handleEvent({
			type: "tool_execution_end",
			toolCallId: readCall.id,
			toolName: "read",
			result: { content: [{ type: "text", text: "file contents" }] },
			isError: false,
		} as Extract<AgentSessionEvent, { type: "tool_execution_end" }>);
		expect(ctx.pendingTools.size).toBe(0);
		expect(chatContainer.children.filter(child => child instanceof ReadToolGroupComponent)).toHaveLength(1);

		await controller.handleEvent({
			type: "message_update",
			message: withLaterThinking,
			assistantMessageEvent: { type: "thinking_delta", delta: "more", contentIndex: 2, partial: withLaterThinking },
		} as Extract<AgentSessionEvent, { type: "message_update" }>);

		const groups = chatContainer.children.filter(child => child instanceof ReadToolGroupComponent);
		expect(groups).toHaveLength(1);
		expect(ctx.pendingTools.size).toBe(0);
		expect(Bun.stripANSI(chatContainer.render(120).join("\n"))).toContain("stable-completed-read.ts");
	});

	it("settles a grouped read whose result arrives before the streamed card", async () => {
		const { controller, chatContainer, ctx } = createFixture();
		const readCall: ToolCall = {
			type: "toolCall",
			id: "read-result-before-card",
			name: "read",
			arguments: { path: "result-before-card.ts" },
		};
		const streaming = assistantMessage([readCall]);

		await controller.handleEvent({ type: "message_start", message: assistantMessage([]) } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);
		await controller.handleEvent({
			type: "tool_execution_end",
			toolCallId: readCall.id,
			toolName: "read",
			result: { content: [{ type: "text", text: "held read body" }] },
			isError: false,
		} as Extract<AgentSessionEvent, { type: "tool_execution_end" }>);
		expect(chatContainer.children.filter(child => child instanceof ReadToolGroupComponent)).toHaveLength(0);
		expect(ctx.pendingTools.size).toBe(0);

		await controller.handleEvent({
			type: "message_update",
			message: streaming,
			assistantMessageEvent: { type: "toolcall_end", contentIndex: 0, toolCall: readCall, partial: streaming },
		} as Extract<AgentSessionEvent, { type: "message_update" }>);

		const groups = chatContainer.children.filter(child => child instanceof ReadToolGroupComponent);
		expect(groups).toHaveLength(1);
		expect(ctx.pendingTools.size).toBe(0);
		expect(Bun.stripANSI(groups[0]!.render(120).join("\n"))).toContain("result-before-card.ts");
	});

	it("finalizes closed inter-tool reasoning so multiple completed greps can retire", async () => {
		const { controller, chatContainer, ctx } = createFixture();
		const greps = [1, 2, 3].map(n => ({
			call: {
				type: "toolCall" as const,
				id: `grep-mixed-${n}`,
				name: "grep",
				arguments: { pattern: `pattern-${n}` },
			},
			result: `GREP_RESULT_${n}_UNIQUE`,
			thinking: `REASONING_AFTER_GREP_${n}`,
		}));

		await controller.handleEvent({ type: "message_start", message: assistantMessage([]) } as Extract<
			AgentSessionEvent,
			{ type: "message_start" }
		>);

		const content: AssistantMessage["content"] = [{ type: "thinking", thinking: "REASONING_BEFORE_TOOLS" }];
		for (const grep of greps) {
			const preceding = assistantMessage([...content]);
			await controller.handleEvent({
				type: "message_update",
				message: preceding,
				assistantMessageEvent: {
					type: "thinking_end",
					contentIndex: content.length - 1,
					content: "",
					partial: preceding,
				},
			} as Extract<AgentSessionEvent, { type: "message_update" }>);
			content.push(grep.call);
			const withTool = assistantMessage([...content]);
			await controller.handleEvent({
				type: "message_update",
				message: withTool,
				assistantMessageEvent: {
					type: "toolcall_end",
					contentIndex: content.length - 1,
					toolCall: grep.call,
					partial: withTool,
				},
			} as Extract<AgentSessionEvent, { type: "message_update" }>);
			await controller.handleEvent({
				type: "tool_execution_end",
				toolCallId: grep.call.id,
				toolName: "grep",
				result: { content: [{ type: "text", text: grep.result }] },
				isError: false,
			} as Extract<AgentSessionEvent, { type: "tool_execution_end" }>);
			content.push({ type: "thinking", thinking: grep.thinking });
			const withThinking = assistantMessage([...content]);
			await controller.handleEvent({
				type: "message_update",
				message: withThinking,
				assistantMessageEvent: {
					type: "thinking_delta",
					delta: grep.thinking,
					contentIndex: content.length - 1,
					partial: withThinking,
				},
			} as Extract<AgentSessionEvent, { type: "message_update" }>);
		}

		expect(ctx.pendingTools.size).toBe(0);

		const flushed = Bun.stripANSI(chatContainer.peekFlushBatch(120)?.rows.join("\n") ?? "");
		expect(flushed).toContain("GREP_RESULT_1_UNIQUE");
		expect(flushed).toContain("GREP_RESULT_2_UNIQUE");
	});

	for (const arrival of ["early", "buffered"] as const) {
		it(`renders inline images from ${arrival} held read results`, async () => {
			const protocol = Object.getOwnPropertyDescriptor(TERMINAL, "imageProtocol")!;
			Object.defineProperty(TERMINAL, "imageProtocol", { value: null });
			try {
				const { controller, chatContainer, ctx } = createFixture();
				ctx.settings.set("terminal.showImages", true);
				const readCall: ToolCall = {
					type: "toolCall",
					id: `read-image-${arrival}`,
					name: "read",
					arguments: { path: "pixel.png" },
				};
				const result: ToolResultMessage = {
					role: "toolResult",
					toolCallId: readCall.id,
					toolName: "read",
					content: [
						{
							type: "image",
							mimeType: "image/png",
							data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
						},
					],
					isError: false,
					timestamp: 1,
				};
				if (arrival === "buffered") {
					ctx.session.agent.getPendingToolResults = () => [result];
					controller.resetTranscriptAnchors();
				}
				await controller.handleEvent({ type: "message_start", message: assistantMessage([]) });
				if (arrival === "early") {
					await controller.handleEvent({
						type: "tool_execution_end",
						toolCallId: readCall.id,
						toolName: "read",
						result,
						isError: false,
					});
				}
				const message = assistantMessage([{ type: "text", text: "Inspecting the sample image." }, readCall]);
				const update: Extract<AgentSessionEvent, { type: "message_update" }> = {
					type: "message_update",
					message,
					assistantMessageEvent: { type: "toolcall_end", contentIndex: 1, toolCall: readCall, partial: message },
				};
				await controller.handleEvent(update);
				await controller.handleEvent(update);
				const rendered = Bun.stripANSI(chatContainer.render(120).join("\n"));
				expect(rendered.match(/\[Image: image\/png\]/g)).toHaveLength(1);
				expect(ctx.pendingTools.size).toBe(0);
			} finally {
				Object.defineProperty(TERMINAL, "imageProtocol", protocol);
			}
		});
	}
});
