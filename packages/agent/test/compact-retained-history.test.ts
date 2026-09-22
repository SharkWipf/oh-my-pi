import { describe, expect, test } from "bun:test";
import {
	type CompactionEntry,
	DEFAULT_COMPACTION_SETTINGS,
	prepareCompaction,
	type SessionEntry,
	type SessionMessageEntry,
} from "@oh-my-pi/pi-agent-core/compaction";
import { withOpenAiRemoteCompactionPreserveData } from "@oh-my-pi/pi-agent-core/compaction/openai";
import { createUserMessage } from "./helpers";

const timestamp = new Date(0).toISOString();
const settings = { ...DEFAULT_COMPACTION_SETTINGS, keepRecentTokens: 1 };

function user(id: string): SessionMessageEntry {
	return { type: "message", id, parentId: null, timestamp, message: createUserMessage(id) };
}

function compacted(id: string, firstKeptEntryId: string): CompactionEntry {
	return { type: "compaction", id, parentId: null, timestamp, firstKeptEntryId, summary: id, tokensBefore: 0 };
}

describe("repeated compaction retained history", () => {
	test("includes the prior unsummarized tail in the next summary input", () => {
		const folded = user("already summarized request");
		const retained = user("previously retained request");
		const fresh = user("new request to summarize");
		const recent = user("most recent request");
		const entries: SessionEntry[] = [folded, retained, compacted("first summary", retained.id), fresh, recent];

		const preparation = prepareCompaction(entries, settings);

		expect(preparation?.messagesToSummarize).toEqual([retained.message, fresh.message]);
		expect(preparation?.recentMessages).toEqual([recent.message]);
	});

	test("includes in-flight native suffix without resummarizing replay-covered originals", () => {
		const covered = user("request captured by native replay");
		const inFlight = user("request appended while native compaction ran");
		const fresh = user("new request to summarize");
		const recent = user("most recent request");
		const previous = compacted("native summary", covered.id);
		previous.providerReplayThroughEntryId = covered.id;
		previous.preserveData = withOpenAiRemoteCompactionPreserveData(undefined, {
			provider: "openai",
			replacementHistory: [{ type: "compaction", encrypted_content: "opaque" }],
			compactionItem: { type: "compaction", encrypted_content: "opaque" },
		});
		const entries: SessionEntry[] = [covered, inFlight, previous, fresh, recent];

		const preparation = prepareCompaction(entries, settings);

		expect(preparation?.messagesToSummarize).toEqual([inFlight.message, fresh.message]);
		expect(preparation?.recentMessages).toEqual([recent.message]);
	});
});
