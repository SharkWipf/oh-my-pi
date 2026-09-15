import { describe, expect, test } from "bun:test";
import {
	getCompactionSourceRepresentation,
	remapCompactionSourceRepresentation,
	type SourceRepresentation,
	unionSourceRanges,
} from "../src/compaction/source";

function capturedSource(entryId = "user"): SourceRepresentation {
	return {
		version: 1,
		coverage: [{
			entryId, order: 0,
			snapshot: { blockIndex: 1, start: 0, end: 6 },
			current: { blockIndex: 1, start: 0, end: 6 },
			normalized: { start: 20, end: 26 },
			status: "exact-current",
		}],
		layout: [
			{ kind: "original-image", entryId, order: 0, blockIndex: 0, currentBlockIndex: 0 },
			{ kind: "frame", frameIndex: 0, range: { start: 20, end: 26 } },
		],
	};
}

describe("compaction source continuity", () => {
	test("replacement bytes are missing current coverage while historical pixels keep their source positions", () => {
		const captured = capturedSource();
		const preserve = { sourceRepresentation: captured, snapcompact: { text: "ABCDEF", frames: [{ data: "unchanged" }] } };
		const result = remapCompactionSourceRepresentation(preserve, [{
			entryId: "user",
			blocks: [
				{ oldBlockIndex: 0, newBlockIndex: null },
				{ oldBlockIndex: 1, newBlockIndex: 0, textEdits: [{ start: 2, end: 4, replacementLength: 3 }] },
			],
		}]);
		const mapped = getCompactionSourceRepresentation(result)!;
		const source = "ABxyZEF";
		const surviving = mapped.coverage.flatMap(run => run.current ? [source.slice(run.current.start, run.current.end)] : []);
		expect(surviving).toEqual(["AB", "EF"]);
		expect(mapped.coverage.map(run => run.normalized)).toEqual([
			{ start: 20, end: 22 }, { start: 22, end: 24 }, { start: 24, end: 26 },
		]);
		expect(mapped.coverage[1].status).toBe("historical-not-current");
		expect(mapped.layout).toEqual([{ kind: "frame", frameIndex: 0, range: { start: 20, end: 26 } }]);
		expect(result?.snapcompact).toBe(preserve.snapcompact);
		expect(captured.coverage[0].current).toEqual({ blockIndex: 1, start: 0, end: 6 });
	});

	test("insertion and subsequent edit follow current coordinates without crediting inserted content", () => {
		const inserted = remapCompactionSourceRepresentation({ sourceRepresentation: capturedSource() }, [{
			entryId: "user", blocks: [{ oldBlockIndex: 1, newBlockIndex: 1, textEdits: [{ start: 2, end: 2, replacementLength: 3 }] }],
		}]);
		const replaced = remapCompactionSourceRepresentation(inserted, [{
			entryId: "user", blocks: [{ oldBlockIndex: 1, newBlockIndex: 1, textEdits: [{ start: 6, end: 7, replacementLength: 1 }] }],
		}]);
		const mapped = getCompactionSourceRepresentation(replaced)!;
		const currentSource = "ABxyzCQEF";
		expect(mapped.coverage.flatMap(run => run.current ? [currentSource.slice(run.current.start, run.current.end)] : [])).toEqual(["AB", "C", "EF"]);
		expect(mapped.coverage.filter(run => run.status === "historical-not-current").map(run => run.snapshot)).toEqual([{ blockIndex: 1, start: 3, end: 4 }]);
	});

	test("an unavailable positional map cannot claim exact current coverage or alter another source ID", () => {
		const one = capturedSource("one");
		const two = capturedSource("two");
		const result = remapCompactionSourceRepresentation({ sourceRepresentation: {
			...one, coverage: [...one.coverage, ...two.coverage], layout: [...one.layout, ...two.layout],
		} }, [{ entryId: "one" }]);
		const mapped = getCompactionSourceRepresentation(result)!;
		expect(mapped.coverage[0].status).toBe("unknown");
		expect(mapped.coverage[0].current).toBeUndefined();
		expect(mapped.coverage[1]).toEqual(two.coverage[0]);
	});

	test("partial ordinary and selected ranges form a complete union without an artificial omission", () => {
		expect(unionSourceRanges([{ start: 4, end: 8 }, { start: 0, end: 4 }, { start: 2, end: 6 }])).toEqual([{ start: 0, end: 8 }]);
		expect(unionSourceRanges([{ start: 0, end: 2 }, { start: 5, end: 8 }])).toEqual([{ start: 0, end: 2 }, { start: 5, end: 8 }]);
	});
});
