import { expect, test } from "bun:test";
import {
	type CompactionEntry,
	DEFAULT_COMPACTION_SETTINGS,
	prepareCompaction,
	type SessionEntry,
} from "../src/compaction";
import { materializeCompactionSourceMessage } from "../src/compaction/source";
import { Tokenizer } from "../src/tokenizer";
import { createUserMessage } from "./helpers";

test("repeated compaction retains the actual sparse source stream and unions partial ordinary coverage", () => {
	const tokenizer = new Tokenizer();
	tokenizer.countMessage = () => 10;
	const user = (index: number): SessionEntry => ({
		type: "message",
		id: `u${index}`,
		parentId: null,
		timestamp: "2026-01-01T00:00:00.000Z",
		message: createUserMessage(`abcdefgh${index}`),
	});
	const original = createUserMessage("abcdefgh2");
	const entries = Array.from({ length: 6 }, (_, index) => user(index));
	const first = prepareCompaction(
		entries,
		{ ...DEFAULT_COMPACTION_SETTINGS, keepRecentTokens: 20 },
		undefined,
		tokenizer,
		{
			selectedSources: [
				{ entryId: "u0", order: 0, message: createUserMessage("abcdefgh0") },
				{ entryId: "u2", order: 2, message: original, spans: [{ blockIndex: 0, start: 4, end: 6 }] },
			],
		},
	)!;
	const previous: CompactionEntry = {
		type: "compaction",
		id: "c",
		parentId: "u5",
		timestamp: "2026-01-01T00:00:00.000Z",
		summary: "Synthetic prior aggregate",
		firstKeptEntryId: first.firstKeptEntryId,
		tokensBefore: 0,
		preserveData: first.sourcePreserveData,
	};
	const repeated = prepareCompaction(
		[...entries, previous, user(6)],
		{ ...DEFAULT_COMPACTION_SETTINGS, keepRecentTokens: 40 },
		undefined,
		tokenizer,
		{ selectedSources: [{ entryId: "u2", order: 2, message: original, spans: [{ blockIndex: 0, start: 0, end: 2 }] }] },
	)!;

	// Former P is now real ordinary history. The intervening omitted u1/u3 never
	// become input again merely because the new cut reaches the old sparse u2.
	expect(repeated.sourcesToSummarize?.map(source => source.entryId)).toEqual(["u0"]);
	expect(repeated.firstKeptEntryId).toBe("u2");
	expect(repeated.recentSources?.map(source => source.entryId)).toEqual(["u2", "u4", "u5", "u6"]);
	expect(repeated.recentMessages[0]).toMatchObject({ content: "[truncated]ef[truncated]" });
	expect(repeated.recentSources?.[0]).toMatchObject({
		message: { content: "abcdefgh2" },
		spans: [{ blockIndex: 0, start: 4, end: 6 }],
	});
	const representation = repeated.sourcePreserveData!.sourceRepresentation;
	expect(representation.throughEntryId).toBe("u6");
	expect(representation.layout.flatMap(part => part.kind === "source" ? [part.entryId] : [])).toEqual([
		"u2", "u4", "u5", "u6",
	]);
	const retained = representation.layout[0];
	if (retained.kind !== "source") throw new Error("Expected retained source");
	expect(retained.contribution).toBeUndefined();
	expect(representation.coverage.filter(run => run.entryId === "u2").map(run => ({
		span: run.current,
		contribution: run.contribution,
	}))).toEqual([
		{ span: { blockIndex: 0, start: 0, end: 2 }, contribution: "selected-user" },
		{ span: { blockIndex: 0, start: 4, end: 6 }, contribution: "ordinary" },
	]);
	expect(materializeCompactionSourceMessage(original, retained.spans)).toMatchObject({
		content: "ab[truncated]ef[truncated]",
	});
});

test("pending delivered users join selection without expanding the ordinary retained tail", () => {
	const tokenizer = new Tokenizer();
	tokenizer.countMessage = () => 10;
	const entries: SessionEntry[] = Array.from({ length: 8 }, (_, index) => ({
		type: "message", id: "u" + index, parentId: index ? "u" + (index - 1) : null,
		timestamp: "2026-09-08T00:00:00.000Z", message: createUserMessage("Original " + index),
	}));
	const settings = { ...DEFAULT_COMPACTION_SETTINGS, keepRecentTokens: 20 };
	const ordinary = prepareCompaction(entries, settings, undefined, tokenizer)!;
	const pending = prepareCompaction(entries, settings, undefined, tokenizer, { pendingSourceEntryIds: new Set(["u0"]) })!;
	expect(pending.firstKeptEntryId).toBe(ordinary.firstKeptEntryId);
	expect(pending.recentMessages).toEqual(ordinary.recentMessages);
	const retained = pending.sourcePreserveData!.sourceRepresentation.layout.flatMap(part => part.kind === "source" ? [part.entryId] : []);
	expect(retained).toEqual(["u0", "u6", "u7"]);
});

test("selected source intervals materialize the original submission rather than expanded delivery", () => {
	const tokenizer = new Tokenizer();
	tokenizer.countMessage = () => 10;
	const original = createUserMessage("/keep SELECTED_LAST_ORIGINAL");
	const entries: SessionEntry[] = Array.from({ length: 4 }, (_, index) => ({
		type: "message", id: "u" + index, parentId: index ? "u" + (index - 1) : null,
		timestamp: "2026-09-08T00:00:00.000Z", message: createUserMessage(index === 0 ? "EXPANDED_LAST_BODY" : "tail " + index),
	}));
	const selection = { selectedSources: [{ entryId: "u0", order: 0, message: original, spans: [{ blockIndex: 0, start: 6, end: 28 }] }] };
	const settings = { ...DEFAULT_COMPACTION_SETTINGS, keepRecentTokens: 20 };
	const preparation = prepareCompaction(entries, settings, undefined, tokenizer, selection)!;
	const source = preparation.selectedSources![0]!;
	expect(materializeCompactionSourceMessage(source.message, source.spans)).toMatchObject({ content: "[truncated]SELECTED_LAST_ORIGINAL" });
	const pending = prepareCompaction(entries, settings, undefined, tokenizer, { ...selection, pendingSourceEntryIds: new Set(["u0"]) })!.selectedSources![0]!;
	expect(materializeCompactionSourceMessage(pending.message, pending.spans)).toMatchObject({ content: "/keep SELECTED_LAST_ORIGINAL" });
});

test("ordinary delivery and selected original remain distinct adjacent projections across compaction", async () => {
	const snap = await import("@oh-my-pi/snapcompact");
	const tokenizer = new Tokenizer();
	tokenizer.countMessage = () => 10;
	const original = { role: "user" as const, content: "SELECTED_LAST_ORIGINAL", timestamp: 0 };
	const delivered = { role: "user" as const, content: "EXPANDED_LAST_BODY with longer ordinary context", timestamp: 0 };
	const entry = (id: string, message = createUserMessage(id)): SessionEntry => ({ type: "message", id, parentId: null, timestamp: "2026-09-08T00:00:00.000Z", message });
	const entries = [entry("old"), entry("mixed", delivered), entry("tail")];
	const selection = { originalSourceMessage: () => original, selectedSources: [{ entryId: "mixed", order: 1, projection: "original" as const, message: original, spans: [{ blockIndex: 0, start: 0, end: original.content.length }] }] };
	const preparation = prepareCompaction(entries, { ...DEFAULT_COMPACTION_SETTINGS, keepRecentTokens: 20 }, undefined, tokenizer, selection)!;
	const sourceParts = preparation.sourcePreserveData!.sourceRepresentation.layout.filter(part => part.kind === "source");
	const materialized = sourceParts.map(part => materializeCompactionSourceMessage(part.projection ? original : part.entryId === "mixed" ? delivered : createUserMessage(part.entryId), part.spans));
	expect(materialized.map(message => message && "content" in message ? message.content : undefined)).toEqual([delivered.content, original.content, "tail"]);
	const result = await snap.compact(preparation);
	const representation = result.preserveData!.sourceRepresentation as NonNullable<typeof preparation.sourcePreserveData>["sourceRepresentation"];
	expect(representation.layout.filter(part => part.kind === "source").map(part => [part.entryId, part.projection])).toEqual([["mixed", undefined], ["mixed", "original"], ["tail", undefined]]);
	const previous: CompactionEntry = { type: "compaction", id: "compact", parentId: "tail", timestamp: "2026-09-08T00:00:00.000Z", summary: result.summary, firstKeptEntryId: result.firstKeptEntryId, tokensBefore: 0, preserveData: result.preserveData };
	const repeated = prepareCompaction([...entries, previous, entry("new")], { ...DEFAULT_COMPACTION_SETTINGS, keepRecentTokens: 10 }, undefined, tokenizer, { originalSourceMessage: () => original })!;
	expect(repeated.sourcesToSummarize!.filter(source => source.entryId === "mixed").map(source => [source.projection, "content" in source.message ? source.message.content : undefined])).toEqual([[undefined, delivered.content], ["original", original.content]]);
	const archived = await snap.compact(repeated);
	const archive = snap.getPreservedArchive(archived.preserveData)!;
	expect(archive.text).toContain(delivered.content);
	expect(archive.text).toContain(original.content);
	const coverage = (archived.preserveData!.sourceRepresentation as typeof representation).coverage.filter(run => run.entryId === "mixed");
	for (const run of coverage) {
		const raw = run.projection ? original.content : delivered.content;
		expect(archive.text!.slice(run.normalized!.start, run.normalized!.end)).toBe(raw.slice(run.current!.start, run.current!.end));
	}
});
