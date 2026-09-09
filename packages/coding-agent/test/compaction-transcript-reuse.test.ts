import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { Message } from "@oh-my-pi/pi-ai";
import { resetSettingsForTest, Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { TranscriptContainer } from "@oh-my-pi/pi-coding-agent/modes/components/transcript-container";
import { Composer } from "@oh-my-pi/pi-coding-agent/modes/composer";
import { initTheme } from "@oh-my-pi/pi-coding-agent/modes/theme/theme";
import type { InteractiveModeContext } from "@oh-my-pi/pi-coding-agent/modes/types";
import { UiHelpers } from "@oh-my-pi/pi-coding-agent/modes/utils/ui-helpers";
import { VirtualRenderScheduler } from "../../tui/test/virtual-render-scheduler";
import { VirtualTerminal } from "../../tui/test/virtual-terminal";

function buildContext(composer: Composer): InteractiveModeContext {
	return {
		chatContainer: new TranscriptContainer(),
		transcriptMessageComponents: new WeakMap(),
		pendingTools: new Map(),
		getUserMessageText: (message: Message) =>
			message.role === "user" && typeof message.content === "string" ? message.content : "",
		viewSession: {
			getToolByName: () => undefined,
			hasBuiltInTool: () => true,
			sessionManager: { putBlobSync: () => "unused", getCwd: () => "/fixture" },
		},
		ui: composer.ui,
		settings: { get: () => false },
		effectiveHideThinkingBlock: false,
		proseOnlyThinking: true,
	} as unknown as InteractiveModeContext;
}

beforeAll(async () => {
	await Settings.init({ inMemory: true });
	await initTheme(false);
});
afterAll(() => resetSettingsForTest());

describe("post-compaction transcript reuse", () => {
	it("replays retained messages and tool details without waiting for another paint", async () => {
		const terminal = new VirtualTerminal(100, 24);
		const scheduler = new VirtualRenderScheduler();
		const composer = new Composer({
			terminal,
			tuiOptions: { renderScheduler: scheduler },
			welcome: { version: "test", modelName: "fixture", providerName: "test" },
		});
		const ctx = buildContext(composer);
		const helpers = new UiHelpers(ctx);
		ctx.addMessageToChat = helpers.addMessageToChat.bind(helpers);
		const messages: AgentMessage[] = [{ role: "user", content: "RETAINED_USER_MESSAGE", timestamp: 1 }];
		for (let index = 0; index < 8; index++) {
			messages.push(
				{
					role: "assistant",
					content: [
						{ type: "text", text: `RETAINED_ASSISTANT_${index}` },
						{
							type: "toolCall",
							id: `eval-${index}`,
							name: "eval",
							arguments: { language: "js", title: `Retained cell ${index}`, code: `display("RESULT_${index}")` },
						},
					],
					api: "anthropic-messages",
					provider: "anthropic",
					model: "test",
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "toolUse",
					timestamp: index + 2,
				},
				{
					role: "toolResult",
					toolCallId: `eval-${index}`,
					toolName: "eval",
					content: [{ type: "text", text: `RESULT_${index}\nSecond result line\nThird result line` }],
					isError: false,
					timestamp: index + 2,
				},
			);
		}
		const context = { messages, thinkingLevel: "off" as const, models: {}, injectedTtsrRules: [], mode: "none" as const };
		try {
			helpers.renderSessionContext(context);
			composer.setRuntimeChildren([ctx.chatContainer, composer.editor]);
			composer.start({ playWelcomeIntro: false });
			await scheduler.settle(terminal);

			// Manual compaction rebuilds retained history, then requests a
			// scrollback-clearing replay. It must be complete in the first paint.
			ctx.chatContainer.clear();
			helpers.renderSessionContext(context, { reuseSettledComponents: true });
			composer.beginHistoryReplay();
			const frame = composer.renderFrame({ columns: 100, rows: 24 });
			const transcript = Bun.stripANSI([...(frame.history?.rows ?? []), ...frame.viewport].join("\n"));
			expect(transcript).toContain("RETAINED_USER_MESSAGE");
			for (let index = 0; index < 8; index++) {
				expect(transcript).toContain(`RETAINED_ASSISTANT_${index}`);
				expect(transcript).toContain(`RESULT_${index}`);
			}
		} finally {
			composer.stop();
		}
	});
});
