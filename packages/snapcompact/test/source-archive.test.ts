import { describe, expect, it } from "bun:test";
import type { ImageContent, Message } from "@oh-my-pi/pi-ai";
import type { SourceMessage, SourceRepresentation } from "@oh-my-pi/pi-ai/compaction-source";
import * as snap from "../src";

const shape: snap.Shape = {
	font: "8x8", cellWidth: 8, cellHeight: 8, lineRepeat: 1,
	variant: "bw", frameSize: 128, frameTokenEstimate: 100,
};
const user = (entryId: string, order: number, content: string): SourceMessage<Message> => ({
	entryId, order, message: { role: "user", content, timestamp: order },
});
function prepare(sources: SourceMessage<Message>[], selectedSources = sources): snap.CompactionPreparation<Message> {
	return {
		firstKeptEntryId: "tail", tokensBefore: 0, fileOps: snap.createFileOps(),
		messagesToSummarize: sources.map(source => source.message), turnPrefixMessages: [],
		sourcesToSummarize: sources, turnPrefixSources: [], recentSources: [], selectedSources,
	};
}
function unpack(result: snap.CompactionResult) {
	const archive = snap.getPreservedArchive(result.preserveData)!;
	const representation = result.preserveData!.sourceRepresentation as SourceRepresentation;
	return { archive, representation };
}

describe("chronological source archive", () => {
	it("unions one long entry across a split normalization unit and preserves the map after reload", async () => {
		const options = { shape: { ...shape, frameSize: 64 }, maxFrames: 1 };
		// The leading cut keeps only '-' in the first case; the trailing cut keeps only '>' in the second.
		for (const raw of ["A".repeat(57) + "→" + "B".repeat(250), "A".repeat(257) + "→" + "B".repeat(63)]) {
			const source = user("long", 0, raw);
			const expected = snap.normalize(snap.serializeConversation([source.message]), { shape: options.shape });
			const ordinary = await snap.compact(prepare([source], []), options);
			const selected = await snap.compact(prepare([source]), options);
			expect(unpack(selected).archive.truncatedChars).toBe(0);
			const repeated = await snap.compact({ ...prepare([], [source]), previousSummary: ordinary.summary,
				previousPreserveData: JSON.parse(JSON.stringify(ordinary.preserveData)) }, options);
			for (const result of [selected, repeated]) {
				const { archive, representation } = unpack(result);
				expect(archive.text).toBe(expected);
				expect(archive.frames).toHaveLength(1);
				const rawPositions = new Set<number>();
				for (const run of representation.coverage) {
					expect(run.status).toBe("exact-current");
					const current = run.current!;
					for (let i = current.start; i < current.end; i++) rawPositions.add(i);
					const normalized = snap.normalize(raw.slice(current.start, current.end), { shape: options.shape });
					const unit = run.normalizedUnit;
					expect(archive.text!.slice(run.normalized!.start, run.normalized!.end)).toBe(unit ? normalized.slice(unit.start, unit.end) : normalized);
				}
				expect([...rawPositions].sort((a, b) => a - b)).toEqual(Array.from({ length: raw.length }, (_, i) => i));
			}
		}
	});

	it("restores complete admitted tool content without replacing historical source bytes", async () => {
		const atomicGroup = { id: "a", entryIds: ["a", "r"] };
		const assistant: SourceMessage<Message> = { entryId: "a", order: 0, atomicGroup, message: {
			role: "assistant", api: "mock", provider: "mock", model: "mock", timestamp: 0, stopReason: "stop",
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			content: [{ type: "toolCall", id: "c", name: "read", arguments: { query: "Q".repeat(1000) } }],
		} };
		const result: SourceMessage<Message> = { entryId: "r", order: 1, atomicGroup, message: {
			role: "toolResult", toolCallId: "c", toolName: "read", timestamp: 1, isError: false,
			content: [{ type: "text", text: "HISTORICAL" + "R".repeat(4000) }],
		} };
		const ordinary = unpack(await snap.compact(prepare([assistant, result], []), { shape, maxFrames: 1 }));
		const selectedResult = await snap.compact(prepare([assistant, result]), { shape, maxFrames: 1 });
		const selected = unpack(selectedResult);
		expect(selected.archive.text).toContain("Q".repeat(1000));
		expect(selected.archive.text).toContain("HISTORICAL" + "R".repeat(4000));
		for (const run of ordinary.representation.coverage) if (run.current && run.current.end > run.current.start) {
			expect(selected.representation.coverage.some(next => next.entryId === run.entryId && next.current?.blockIndex === run.current!.blockIndex && next.current.start <= run.current!.start && next.current.end >= run.current!.end)).toBe(true);
		}
		const prior = { ...selectedResult.preserveData, sourceRepresentation: { ...selected.representation,
			coverage: selected.representation.coverage.map(run => ({ ...run, current: undefined, status: "historical-not-current" as const })),
		} };
		const current: SourceMessage<Message> = { ...result, message: { ...result.message as Extract<Message, { role: "toolResult" }>, content: [{ type: "text", text: "CURRENT REPLACEMENT" }] } };
		const repeated = unpack(await snap.compact({ ...prepare([], [assistant, current]), previousPreserveData: prior }, { shape, maxFrames: 1 }));
		const baseline = unpack(await snap.compact({ ...prepare([], []), previousPreserveData: prior }, { shape, maxFrames: 1 }));
		expect(repeated.archive.text!.startsWith(baseline.archive.text!)).toBe(true);
		expect(repeated.archive.text).toContain("CURRENT REPLACEMENT");
		expect(repeated.representation.coverage.some(run => run.entryId === "r" && run.status === "historical-not-current")).toBe(true);
	});

	it("retains original raw offsets through sparse selection and Unicode expansion on the next compaction", async () => {
		const source = { ...user("sparse", 0, "AB→CDXYZ"), spans: [{ blockIndex: 0, start: 0, end: 3 }, { blockIndex: 0, start: 6, end: 8 }] };
		const first = await snap.compact(prepare([source]), { shape });
		const initial = unpack(first);
		expect(initial.archive.text).toBe("¶user:AB->[truncated]YZ");
		const coverage = initial.representation.coverage.filter(run => run.current);
		expect(coverage.map(run => run.current)).toEqual([
			{ blockIndex: 0, start: 0, end: 2 }, { blockIndex: 0, start: 2, end: 3 }, { blockIndex: 0, start: 6, end: 8 },
		]);
		const sameText = user("different-id", 1, "AB→CDXYZ");
		const repeated = unpack(await snap.compact({ ...prepare([sameText], []), previousPreserveData: first.preserveData }, { shape }));
		expect(repeated.archive.text).toContain("AB->[truncated]YZ");
		expect(repeated.archive.text).toContain("AB->CDXYZ");
		expect(new Set(repeated.representation.coverage.map(run => run.entryId))).toEqual(new Set(["sparse", "different-id"]));
	});

	it("attributes an added original image independently of ordinary text and resets its next-archive ownership", async () => {
		const image: ImageContent = { type: "image", data: (await snap.render("OWNED SOURCE IMAGE", shape)).data, mimeType: "image/png" };
		const source: SourceMessage<Message> = { entryId: "mixed", order: 0, message: {
			role: "user", timestamp: 0, content: [{ type: "text", text: "ordinary before" }, image, { type: "text", text: "ordinary after" }],
		} };
		const selected = { ...source, spans: [{ blockIndex: 1, start: 0, end: 1 }] };
		const first = await snap.compact(prepare([source], [selected]), { shape });
		const frozen = unpack(first);
		expect(frozen.representation.coverage.filter(run => run.normalized).map(run => run.contribution)).toEqual(["ordinary", "ordinary"]);
		const imageRuns = frozen.representation.coverage.filter(run => !run.normalized);
		expect(imageRuns).toEqual([{ entryId: "mixed", order: 0, snapshot: { blockIndex: 1, start: 0, end: 1 },
			current: { blockIndex: 1, start: 0, end: 1 }, status: "exact-current", contribution: "selected-user" }]);
		const repeated = unpack(await snap.compact({ ...prepare([], [selected]), previousPreserveData: first.preserveData }, { shape }));
		expect(repeated.representation.coverage.filter(run => !run.normalized)).toEqual(imageRuns.map(run => ({ ...run, contribution: "ordinary" })));
		for (const archive of [frozen, repeated]) {
			expect(snap.historyBlocks(archive.archive, { sourceRepresentation: archive.representation, resolveSourceImage: () => image }).filter(block => block.type === "image")).toEqual([image]);
			expect(JSON.stringify(archive.representation)).not.toContain(image.data);
		}
	});

	it("keeps a tool-result original image inside the standard folded atomic conversation", async () => {
		const atomicGroup = { id: "assistant", entryIds: ["assistant", "result"] };
		const assistant: SourceMessage<Message> = { entryId: "assistant", order: 0, atomicGroup, message: {
			role: "assistant", api: "mock", provider: "mock", model: "mock", timestamp: 0, stopReason: "stop",
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			content: [{ type: "toolCall", id: "call", name: "read", arguments: {} }, { type: "text", text: "ASSISTANT AFTER" }],
		} };
		const image: ImageContent = { type: "image", data: (await snap.render("SOURCE IMAGE", shape)).data, mimeType: "image/png" };
		const result: SourceMessage<Message> = { entryId: "result", order: 1, atomicGroup, message: {
			role: "toolResult", toolCallId: "call", toolName: "read", timestamp: 1, isError: false,
			content: [{ type: "text", text: "BEFORE IMAGE" }, image, { type: "text", text: "AFTER IMAGE" }],
		} };
		const emitted = unpack(await snap.compact(prepare([], [assistant, result]), { shape }));
		const blocks = snap.historyBlocks(emitted.archive, { sourceRepresentation: emitted.representation, resolveSourceImage: () => image });
		expect(blocks.map(block => block.type === "image" ? "IMAGE" : block.text)).toEqual([
			"¶call:read()\n<out>\nBEFORE IMAGE", "IMAGE", "AFTER IMAGE\n</out>\n¶ai:ASSISTANT AFTER",
		]);
		expect(emitted.representation.coverage.filter(run => run.entryId === "result" && run.normalized).map(run => run.current?.blockIndex)).toEqual([0, 2]);
		expect(emitted.representation.coverage.filter(run => run.entryId === "result" && !run.normalized)).toEqual([{ entryId: "result", order: 1, atomicGroup,
			snapshot: { blockIndex: 1, start: 0, end: 1 }, current: { blockIndex: 1, start: 0, end: 1 }, status: "exact-current", contribution: "manual-nonuser" }]);
	});

	it("counts only wholly absent source IDs and does not call raster overflow an omission", async () => {
		const sources = Array.from({ length: 20 }, (_, i) => user(`u${i}`, i, `ENTRY${i} `.repeat(50).trimEnd()));
		const ordinary = unpack(await snap.compact(prepare(sources, []), { shape: { ...shape, frameSize: 64 }, maxFrames: 1 }));
		const present = new Set(ordinary.representation.coverage.map(run => run.entryId));
		const absent = sources.filter(source => !present.has(source.entryId)).length;
		expect(absent).toBeGreaterThan(0);
		expect(ordinary.representation.layout.reduce((sum, part) => sum + (part.kind === "gap" ? part.wholeMessages ?? 0 : 0), 0)).toBe(absent);
		const selected = unpack(await snap.compact(prepare(sources), { shape: { ...shape, frameSize: 64 }, maxFrames: 1 }));
		expect(selected.representation.layout.filter(part => part.kind === "gap")).toEqual([]);
		expect(selected.archive.truncatedChars).toBe(0);
		expect(snap.historyBlocks(selected.archive, { sourceRepresentation: selected.representation }).some(block => block.type === "text" && block.text.startsWith("[continued]"))).toBe(true);
	});
	it("removes a closed sparse gap without dropping fuller ordinary spans", async () => {
		const source = user("sparse", 0, "ABCDEFGHIJ");
		const sparse = { ...source, spans: [{ blockIndex: 0, start: 0, end: 2 }, { blockIndex: 0, start: 8, end: 10 }] };
		const first = await snap.compact(prepare([sparse], []), { shape });
		const filled = unpack(await snap.compact({ ...prepare([], [source]), previousPreserveData: first.preserveData }, { shape }));
		expect(filled.archive.text).toBe("¶user:ABCDEFGHIJ");
		const partial = { ...source, spans: [{ blockIndex: 0, start: 3, end: 5 }] };
		const overlap = unpack(await snap.compact(prepare([source], [partial]), { shape }));
		expect(overlap.archive.text).toBe("¶user:ABCDEFGHIJ");
	});

	it("unions selected intervals into a partially retained recent source", async () => {
		const source = user("recent", 3, "ABCDEFGHIJ");
		const ordinary = { ...source, spans: [{ blockIndex: 0, start: 0, end: 6 }] };
		const selected = { ...source, spans: [{ blockIndex: 0, start: 4, end: 10 }] };
		const result = unpack(await snap.compact({ ...prepare([user("old", 0, "history")], [selected]), recentSources: [ordinary] }, { shape }));
		expect(result.representation.layout.find(part => part.kind === "source" && part.entryId === "recent")).toMatchObject({ spans: [{ blockIndex: 0, start: 0, end: 10 }] });
	});

	it("keeps vanilla pagination with zero selection and reduces reconstructed rescue history", async () => {
		const source = user("long", 0, "Source history. ".repeat(300));
		const mapped = prepare([source], []);
		const { sourcesToSummarize: _sources, turnPrefixSources: _prefix, recentSources: _recent, selectedSources: _selected, ...plain } = mapped;
		const options = { shape, maxFrames: 1 };
		const vanilla = unpack(await snap.compact(plain, options));
		const ordinary = unpack(await snap.compact(mapped, options));
		expect(ordinary.archive).toEqual(vanilla.archive);
		const selected = await snap.compact(prepare([source]), options);
		const rescue = unpack(await snap.compact({ ...prepare([], []), previousPreserveData: selected.preserveData }, options));
		const physical = (result: ReturnType<typeof unpack>) => snap.historyBlocks(result.archive, { sourceRepresentation: result.representation }).reduce((sum, block) => sum + (block.type === "text" ? block.text.length : 100), 0);
		expect(physical(rescue)).toBeLessThan(physical(unpack(selected)));
		expect(rescue.archive.text).toBe(ordinary.archive.text);
	});

});
