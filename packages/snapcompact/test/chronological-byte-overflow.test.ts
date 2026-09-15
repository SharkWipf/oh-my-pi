import { expect, test } from "bun:test";
import type { Message } from "@oh-my-pi/pi-ai";
import type { SourceMessage, SourceRepresentation } from "@oh-my-pi/pi-ai/compaction-source";
import { compact, createFileOps, getPreservedArchive, type Shape } from "../src/snapcompact";

test("first measured PNG-byte nonfit spills every later admitted span instead of packing newer frames into holes", async () => {
	const shape: Shape = { font: "8x8", cellWidth: 8, cellHeight: 8, lineRepeat: 1, variant: "bw", frameSize: 128, frameTokenEstimate: 100 };
	const sources: SourceMessage<Message>[] = Array.from({ length: 10 }, (_, order) => ({
		entryId: `u${order}`, order,
		message: { role: "user", content: `${String.fromCharCode(65 + order)}${order} `.repeat(220), timestamp: order },
	}));
	const input = {
		firstKeptEntryId: "tail", messagesToSummarize: sources.map(source => source.message), turnPrefixMessages: [],
		sourcesToSummarize: sources, turnPrefixSources: [], recentSources: [], selectedSources: sources,
		tokensBefore: 0, fileOps: createFileOps(),
	};
	const full = await compact(input, { shape, maxFrames: 3 });
	const fullArchive = getPreservedArchive(full.preserveData)!;
	expect(fullArchive.frames).toHaveLength(3);
	const byteLimit = fullArchive.frames[0]!.data.length + fullArchive.frames[1]!.data.length - 1;
	const spilled = await compact(input, { shape, maxFrames: 3, maxFrameDataBytes: byteLimit });
	const archive = getPreservedArchive(spilled.preserveData)!;
	expect(archive.frames).toHaveLength(1);
	expect(archive.frames[0]!.data).toBe(fullArchive.frames[0]!.data);
	expect(archive.text).toBe(fullArchive.text);
	const representation = spilled.preserveData!.sourceRepresentation as SourceRepresentation;
	const imaged = representation.layout.findIndex(part => part.kind === "frame");
	expect(imaged).toBeGreaterThanOrEqual(0);
	expect(representation.layout.slice(imaged + 1).some(part => part.kind === "frame")).toBe(false);
	expect(representation.layout.slice(imaged + 1).some(part => part.kind === "text")).toBe(true);
});
