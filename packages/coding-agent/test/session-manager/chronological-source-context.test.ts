import { describe, expect, it } from "bun:test";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import { remapCompactionSourceRepresentation } from "@oh-my-pi/pi-agent-core/compaction/source";
import type { SourceLayoutPart, SourceRepresentation } from "@oh-my-pi/pi-ai/compaction-source";
import { getSourceOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import * as snapcompact from "@oh-my-pi/snapcompact";
import { makeAssistantMessage } from "./helpers";

function user(session: SessionManager, content: string): string {
	return session.appendMessage({ role: "user", content, timestamp: 1 });
}

function representation(layout: SourceLayoutPart[]): SourceRepresentation {
	return { version: 1, coverage: [], layout };
}

function userContents(messages: AgentMessage[]): unknown[] {
	return messages.filter(message => message.role === "user").map(message => message.content);
}

describe("chronological committed source context", () => {
	it("retains equal-content source IDs separately and unions partial selection with the untouched ordinary suffix", async () => {
		using dir = TempDir.createSync("@pi-chronological-source-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			const first = user(session, "identical");
			const second = user(session, "identical");
			const ordinary = user(session, "ordinary: αβ\n  preserve every byte\t!");
			session.appendCompaction("recap", undefined, ordinary, 1000, {
				method: "soft",
				preserveData: {
					sourceRepresentation: representation([
						{ kind: "source", entryId: first, order: 0 },
						{ kind: "source", entryId: second, order: 1 },
						{ kind: "source", entryId: ordinary, order: 2, spans: [{ blockIndex: 0, start: 2, end: 8 }] },
					]),
				},
			});
			const context = session.buildSessionContext({ diagnostics: true });
			expect(userContents(context.messages)).toEqual(["identical", "identical", "ordinary: αβ\n  preserve every byte\t!"]);
			expect(context.messageSourceIds?.filter(Boolean)).toEqual([first, second, ordinary]);
			expect(context.messages[0]?.role).toBe("compactionSummary");
			const ordinaryMessage = context.messages.find((_, index) => context.messageSourceIds?.[index] === ordinary);
			const ordinaryEntry = session.getEntry(ordinary);
			if (ordinaryEntry?.type !== "message") throw new Error("Expected original ordinary source");
			expect(ordinaryMessage).toEqual(ordinaryEntry.message);
		} finally {
			await session.close();
		}
	});

	it("replays accepted original coordinates rather than expanded delivery after durable reload", async () => {
		using dir = TempDir.createSync("@pi-original-source-reload-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			const text = "SELECTED_LAST_ORIGINAL";
			const selected = session.appendMessage({ role: "user", content: "EXPANDED_LAST_BODY", timestamp: 1,
				compactionOverride: "keep", originalSubmission: { text: `/keep ${text}`, compactionOverride: "keep" } });
			const ordinary = user(session, "ordinary");
			session.appendCompaction("recap", undefined, ordinary, 1000, { method: "soft", preserveData: {
				sourceRepresentation: representation([{ kind: "source", entryId: selected, order: 0, projection: "original",
					spans: [{ blockIndex: 0, start: 0, end: text.length }] }]),
			} });
			expect(userContents(session.buildSessionContext().messages)).toEqual([text, "ordinary"]);
			await session.ensureOnDisk();
			await session.flush();
			const reopened = await SessionManager.open(session.getSessionFile()!);
			try { expect(userContents(reopened.buildSessionContext().messages)).toEqual([text, "ordinary"]); }
			finally { await reopened.close(); }
		} finally { await session.close(); }
	});

	it("keeps original text and images beside the untouched ordinary expanded projection", async () => {
		const session = SessionManager.inMemory();
		try {
			const image = { type: "image" as const, data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=", mimeType: "image/png" };
			const delivered = [{ type: "text" as const, text: "EXPANDED_WITH_IMAGE" }, image];
			const selected = session.appendMessage({ role: "user", content: delivered, timestamp: 1,
				originalSubmission: { text: "raw input", images: [image] } });
			session.appendCompaction("recap", undefined, selected, 1000, { method: "soft", preserveData: {
				sourceRepresentation: representation([
					{ kind: "source", entryId: selected, order: 0, projection: "original", spans: [{ blockIndex: 0, start: 0, end: 9 }] },
					{ kind: "original-image", entryId: selected, order: 0, projection: "original", blockIndex: 1, currentBlockIndex: 1 },
				]),
			} });
			const context = session.buildSessionContext();
			expect(userContents(context.messages)).toEqual([[{ type: "text", text: "raw input" }, image], delivered]);
			const imageOrigins = context.messages.filter(message => message.role === "user")
				.flatMap(message => Array.isArray(message.content) ? message.content.filter(block => block.type === "image") : [])
				.map(block => getSourceOrigin(block));
			expect(imageOrigins.map(origin => origin?.kind === "source" ? origin.parts.map(part => part.projection) : undefined))
				.toEqual([["original"], [undefined]]);
			expect(userContents(session.buildSessionContext({ transcript: true }).messages)).toEqual([delivered]);
		} finally { await session.close(); }
	});

	it("unions overlapping selected ranges without duplicating source bytes or dropping ordinary source", async () => {
		using dir = TempDir.createSync("@pi-chronological-spans-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			const selected = user(session, "0123456789");
			const ordinary = user(session, "ordinary");
			session.appendCompaction("recap", undefined, ordinary, 1000, {
				method: "handoff",
				preserveData: {
					sourceRepresentation: representation([
						{ kind: "source", entryId: selected, order: 0, spans: [{ blockIndex: 0, start: 0, end: 5 }] },
						{ kind: "source", entryId: selected, order: 0, spans: [{ blockIndex: 0, start: 3, end: 10 }] },
					]),
				},
			});
			expect(userContents(session.buildSessionContext().messages)).toEqual(["0123456789", "ordinary"]);
		} finally {
			await session.close();
		}
	});

	it("does not resurrect sibling or pre-clear source references", async () => {
		using dir = TempDir.createSync("@pi-chronological-boundary-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			const beforeClear = user(session, "forbidden pre-clear");
			const boundary = session.appendResetBoundary();
			const sibling = user(session, "forbidden sibling");
			session.branch(boundary);
			const selected = user(session, "active selected");
			const ordinary = user(session, "active ordinary");
			session.appendCompaction("recap", undefined, ordinary, 1000, {
				method: "soft",
				preserveData: {
					sourceRepresentation: representation([
						{ kind: "source", entryId: beforeClear, order: 0 },
						{ kind: "source", entryId: sibling, order: 2 },
						{ kind: "source", entryId: selected, order: 2 },
					]),
				},
			});
			const context = session.buildSessionContext({ diagnostics: true });
			expect(userContents(context.messages)).toEqual(["active selected", "active ordinary"]);
			expect(context.messageSourceIds?.filter(Boolean)).toEqual([selected, ordinary]);
			session.appendResetBoundary();
			const newest = user(session, "new conversation");
			expect(session.buildSessionContext({ diagnostics: true }).messageSourceIds).toEqual([newest]);
		} finally {
			await session.close();
		}
	});

	it("hydrates original images in source block order and preserves the committed output across disk reload", async () => {
		using dir = TempDir.createSync("@pi-chronological-images-");
		const session = SessionManager.create(dir.path(), dir.path());
		let reopened: SessionManager | undefined;
		try {
			const content = [
				{ type: "text" as const, text: "before" },
				{ type: "image" as const, data: "Zmlyc3Q=", mimeType: "image/png" },
				{ type: "text" as const, text: "between" },
				{ type: "image" as const, data: "c2Vjb25k", mimeType: "image/png" },
				{ type: "text" as const, text: "after" },
			];
			const selected = session.appendMessage({ role: "user", content, timestamp: 1 });
			const ordinary = user(session, "ordinary");
			const sourceRepresentation = representation([
				{ kind: "source", entryId: selected, order: 0, spans: [{ blockIndex: 0, start: 0, end: 6 }] },
				{ kind: "original-image", entryId: selected, order: 0, blockIndex: 7, currentBlockIndex: 1 },
				{ kind: "source", entryId: selected, order: 0, spans: [{ blockIndex: 2, start: 0, end: 7 }] },
				{ kind: "original-image", entryId: selected, order: 0, blockIndex: 9, currentBlockIndex: 3 },
				{ kind: "source", entryId: selected, order: 0, spans: [{ blockIndex: 4, start: 0, end: 5 }] },
			]);
			session.appendCompaction("recap", undefined, ordinary, 1000, {
				method: "soft",
				preserveData: { sourceRepresentation },
			});
			session.appendMessage(makeAssistantMessage());
			const context = session.buildSessionContext({ diagnostics: true });
			expect(userContents(context.messages)).toEqual([content, "ordinary"]);
			await session.flush();
			reopened = await SessionManager.open(session.getSessionFile()!);
			expect(reopened.buildSessionContext({ diagnostics: true })).toEqual(context);
			expect(reopened.buildSessionContext().messages).toEqual(context.messages);
		} finally {
			await reopened?.close();
			await session.close();
		}
	});

	it("counts only wholly missing source entries in actual leading, internal and trailing gaps", async () => {
		using dir = TempDir.createSync("@pi-chronological-gaps-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			user(session, "leading omission");
			session.appendCustomEntry("not-source", { count: 900 });
			const first = user(session, "first selected");
			user(session, "internal omission one");
			session.appendThinkingLevelChange("high");
			user(session, "internal omission two");
			const partial = user(session, "abcdefghij");
			user(session, "trailing omission");
			session.appendCompaction("recap", undefined, "", 1000, {
				method: "soft",
				preserveData: {
					sourceRepresentation: representation([
						{ kind: "source", entryId: first, order: 2 },
						{ kind: "source", entryId: partial, order: 6, spans: [{ blockIndex: 0, start: 0, end: 3 }] },
					]),
				},
			});
			const context = session.buildSessionContext({ diagnostics: true });
			expect(context.messages.slice(1).map(message => {
				if (message.role === "custom" && message.customType === "compaction-source-gap") {
					return message.details;
				}
				return message.role;
			})).toEqual([{ wholeMessages: 1 }, "user", { wholeMessages: 2 }, "user", { wholeMessages: 1 }]);
			expect(context.messageSourceIds?.filter(Boolean)).toEqual([first, partial]);
		} finally {
			await session.close();
		}
	});

	it("keeps a complete assistant-tool atom contiguous between whole-message gaps", async () => {
		using dir = TempDir.createSync("@pi-chronological-atom-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			user(session, "omitted before atom");
			const assistant = session.appendMessage({
				...makeAssistantMessage(),
				content: [{ type: "toolCall", id: "call", name: "read", arguments: { path: "file" } }],
				stopReason: "toolUse",
			});
			const result = session.appendMessage({
				role: "toolResult", toolCallId: "call", toolName: "read",
				content: [{ type: "text", text: "source bytes" }], isError: false, timestamp: 1,
			});
			user(session, "omitted after atom");
			session.appendCompaction("recap", undefined, "", 1000, {
				method: "soft",
				preserveData: { sourceRepresentation: representation([
					{ kind: "source", entryId: assistant, order: 1 },
					{ kind: "source", entryId: result, order: 2 },
				]) },
			});
			const context = session.buildSessionContext();
			expect(context.messages.map(message => message.role)).toEqual([
				"compactionSummary", "custom", "assistant", "toolResult", "custom",
			]);
			const emittedAssistant = context.messages[2];
			if (emittedAssistant?.role !== "assistant") throw new Error("Expected assistant atom");
			expect(emittedAssistant.content).toEqual([{ type: "toolCall", id: "call", name: "read", arguments: { path: "file" } }]);
		} finally {
			await session.close();
		}
	});


	it("uses the full committed sparse layout across repeated compaction instead of reopening the firstKept interval", async () => {
		using dir = TempDir.createSync("@pi-chronological-repeat-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			const selected = user(session, "old selected");
			user(session, "old omitted between sparse members");
			const formerTail = user(session, "former ordinary tail");
			const firstRepresentation: SourceRepresentation = {
				...representation([
					{ kind: "source", entryId: selected, order: 0 },
					{ kind: "source", entryId: formerTail, order: 2 },
				]),
				throughEntryId: formerTail,
			};
			session.appendCompaction("first recap", undefined, formerTail, 1000, {
				method: "soft", preserveData: { sourceRepresentation: firstRepresentation },
			});
			const next = user(session, "new source");
			expect(userContents(session.buildSessionContext().messages)).toEqual([
				"old selected", "former ordinary tail", "new source",
			]);
			const nextRepresentation: SourceRepresentation = {
				...representation([
					{ kind: "source", entryId: selected, order: 0 },
					{ kind: "source", entryId: next, order: 4 },
				]),
				throughEntryId: next,
			};
			session.appendCompaction("second recap", undefined, selected, 1000, {
				method: "soft", preserveData: { sourceRepresentation: nextRepresentation },
			});
			const after = user(session, "after second compaction");
			const context = session.buildSessionContext({ diagnostics: true });
			expect(userContents(context.messages)).toEqual(["old selected", "new source", "after second compaction"]);
			expect(context.messageSourceIds?.filter(Boolean)).toEqual([selected, next, after]);
		} finally {
			await session.close();
		}
	});


	it("maps snap text and original images to actual emitted blocks without replaying archived source", async () => {
		using dir = TempDir.createSync("@pi-chronological-snap-");
		const session = SessionManager.create(dir.path(), dir.path());
		let reopened: SessionManager | undefined;
		try {
			const image = { type: "image" as const, data: "c291cmNl", mimeType: "image/png" };
			const archived = session.appendMessage({
				role: "user", timestamp: 1,
				content: [{ type: "text", text: "left" }, image, { type: "text", text: "right" }],
			});
			const ordinary = user(session, "ordinary");
			const sourceRepresentation: SourceRepresentation = {
				version: 1,
				throughEntryId: ordinary,
				coverage: [
					{ entryId: archived, order: 0, snapshot: { blockIndex: 0, start: 0, end: 4 },
						current: { blockIndex: 0, start: 0, end: 4 }, normalized: { start: 0, end: 4 }, status: "exact-current" },
					{ entryId: archived, order: 0, snapshot: { blockIndex: 2, start: 0, end: 5 },
						current: { blockIndex: 2, start: 0, end: 5 }, normalized: { start: 4, end: 9 }, status: "exact-current" },
				],
				layout: [
					{ kind: "text", range: { start: 0, end: 4 } },
					{ kind: "original-image", entryId: archived, order: 0, blockIndex: 7, currentBlockIndex: 1 },
					{ kind: "text", range: { start: 4, end: 9 } },
					{ kind: "source", entryId: ordinary, order: 1 },
				],
			};
			session.appendCompaction("snap recap", undefined, ordinary, 1000, {
				method: "snapcompact",
				preserveData: {
					sourceRepresentation,
					[snapcompact.PRESERVE_KEY]: { frames: [], totalChars: 9, truncatedChars: 0, text: "leftright" },
				},
			});
			session.appendMessage(makeAssistantMessage());
			const context = session.buildSessionContext({ diagnostics: true });
			expect(userContents(context.messages)).toEqual(["ordinary"]);
			const sourceBlocks = context.sourceLocations?.filter(location => location.layoutIndex < 3).map(location => {
				const message = context.messages[location.messageIndex];
				if (message?.role !== "compactionSummary" || location.blockIndex === undefined) {
					throw new Error("Expected an actual archive block location");
				}
				return { layoutIndex: location.layoutIndex, block: message.blocks?.[location.blockIndex] };
			});
			expect(sourceBlocks).toEqual([
				{ layoutIndex: 0, block: { type: "text", text: "left" } },
				{ layoutIndex: 1, block: image },
				{ layoutIndex: 2, block: { type: "text", text: "right" } },
			]);
			await session.flush();
			reopened = await SessionManager.open(session.getSessionFile()!);
			expect(reopened.buildSessionContext({ diagnostics: true })).toEqual(context);
			expect(reopened.buildSessionContext().messages).toEqual(context.messages);
		} finally {
			await reopened?.close();
			await session.close();
		}
	});


	it("does not replay an explicitly deleted image when its former block index now names another image", async () => {
		using dir = TempDir.createSync("@pi-chronological-image-delete-");
		const session = SessionManager.create(dir.path(), dir.path());
		try {
			const survivingImage = { type: "image" as const, data: "c3Vydml2b3I=", mimeType: "image/png" };
			const content = [{ type: "text" as const, text: "caption" }, survivingImage];
			const selected = session.appendMessage({ role: "user", content, timestamp: 1 });
			const ordinary = user(session, "ordinary");
			const preserveData = remapCompactionSourceRepresentation({ sourceRepresentation: representation([
				{ kind: "source", entryId: selected, order: 0, spans: [{ blockIndex: 0, start: 0, end: 7 }] },
				{ kind: "original-image", entryId: selected, order: 0, blockIndex: 1, currentBlockIndex: 1 },
				{ kind: "original-image", entryId: selected, order: 0, blockIndex: 2, currentBlockIndex: 2 },
			]) }, [{ entryId: selected, blocks: [
				{ oldBlockIndex: 1, newBlockIndex: null },
				{ oldBlockIndex: 2, newBlockIndex: 1 },
			] }]);
			session.appendCompaction("recap", undefined, ordinary, 1000, { method: "soft", preserveData });
			expect(userContents(session.buildSessionContext().messages)).toEqual([content, "ordinary"]);
		} finally {
			await session.close();
		}
	});

	it("rehydrates custom-message images and spans without changing the source role or attribution", async () => {
		using dir = TempDir.createSync("@pi-chronological-custom-");
		const session = SessionManager.create(dir.path(), dir.path());
		let reopened: SessionManager | undefined;
		try {
			const image = { type: "image" as const, data: "YQ==", mimeType: "image/png" };
			const archived = session.appendCustomMessageEntry("human-note", [{ type: "text", text: "head" }, image], true, { tag: "manual" }, "user", 1);
			const selected = session.appendCustomMessageEntry("human-note", [{ type: "text", text: "0123456789" }, image], true, { tag: "manual" }, "user", 2);
			const ordinary = user(session, "ordinary");
			const sourceRepresentation: SourceRepresentation = {
				version: 1, throughEntryId: ordinary,
				coverage: [{ entryId: archived, order: 0, snapshot: { blockIndex: 0, start: 0, end: 4 }, current: { blockIndex: 0, start: 0, end: 4 }, normalized: { start: 0, end: 4 }, status: "exact-current" }],
				layout: [
					{ kind: "text", range: { start: 0, end: 4 } },
					{ kind: "original-image", entryId: archived, order: 0, blockIndex: 7, currentBlockIndex: 1 },
					{ kind: "source", entryId: selected, order: 1, spans: [{ blockIndex: 0, start: 2, end: 8 }, { blockIndex: 1, start: 0, end: 0 }] },
					{ kind: "source", entryId: ordinary, order: 2 },
				],
			};
			session.appendCompaction("recap", undefined, ordinary, 1000, {
				method: "snapcompact",
				preserveData: { sourceRepresentation, [snapcompact.PRESERVE_KEY]: { frames: [], text: "head", totalChars: 4, truncatedChars: 0 } },
			});
			session.appendMessage(makeAssistantMessage());
			const context = session.buildSessionContext({ diagnostics: true });
			const message = context.messages.find((_, index) => context.messageSourceIds?.[index] === selected);
			if (message?.role !== "custom") throw new Error("Expected original custom source role");
			expect(message.attribution).toBe("user");
			expect(message.details).toEqual({ tag: "manual" });
			expect(message.content).toEqual([{ type: "text", text: "[truncated]234567[truncated]" }, image]);
			const summary = context.messages[0];
			if (summary?.role !== "compactionSummary") throw new Error("Expected actual archive wrapper");
			expect(summary.blocks).toEqual([{ type: "text", text: "head" }, image]);
			await session.flush();
			reopened = await SessionManager.open(session.getSessionFile()!);
			expect(reopened.buildSessionContext({ diagnostics: true })).toEqual(context);
		} finally {
			await reopened?.close();
			await session.close();
		}
	});

});