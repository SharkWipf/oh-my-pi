import { describe, expect, it } from "bun:test";
import type { AssistantMessage } from "../src/types";
import { getOpenAIResponsesHistoryPayload } from "../src/utils";
import { bindMessageSource, exportItemOrigins, importItemOrigins } from "../src/utils/source-origin";

function assistant(): AssistantMessage {
	return {
		role: "assistant", content: [{ type: "text", text: "native answer" }],
		api: "openai-responses", provider: "openai", model: "test", stopReason: "stop", timestamp: 1,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		providerPayload: {
			type: "openaiResponsesHistory", provider: "openai", dt: true,
			items: [{ type: "message", content: [{ type: "output_text", text: "native answer" }] }],
			contentBlocks: [{ itemIndex: 0, contentIndex: 0 }],
		},
	};
}

describe("read-only native source binding", () => {
	it("keeps journal bytes unchanged while exporting bound replay provenance", () => {
		const message = assistant();
		const before = JSON.stringify(message);
		bindMessageSource(message, "source-entry", 4);
		const payload = getOpenAIResponsesHistoryPayload(message.providerPayload, "openai")!;
		expect(exportItemOrigins(payload.items)).toEqual([{ kind: "source", parts: [expect.objectContaining({
			entryId: "source-entry", order: 4, blockIndex: 0, sourceLength: 13,
		})] }]);
		expect(JSON.stringify(message)).toBe(before);
	});

	it("retains live bindings over absent-map placeholders but accepts explicit invalidation", () => {
		const message = assistant();
		const original = message.providerPayload!;
		if (original.type !== "openaiResponsesHistory") throw new Error("Expected Responses payload");
		original.origins = [{ kind: "unknown", reason: "legacy-map-absent" }];
		const before = JSON.stringify(message);
		bindMessageSource(message, "source-entry", 4);
		const payload = getOpenAIResponsesHistoryPayload(message.providerPayload, "openai")!;
		expect(exportItemOrigins(payload.items)[0]).toMatchObject({ kind: "source", parts: [{ entryId: "source-entry" }] });
		expect(JSON.stringify(message)).toBe(before);
		importItemOrigins(payload.items, [{ kind: "unknown", reason: "externally-mutated" }]);
		expect(exportItemOrigins(payload.items)).toEqual([{ kind: "unknown", reason: "externally-mutated" }]);
	});
});
