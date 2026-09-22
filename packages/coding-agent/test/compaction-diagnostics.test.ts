import { expect, test } from "bun:test";
import { IMAGE_TOKEN_ESTIMATE, Tokenizer } from "@oh-my-pi/pi-agent-core";
import type { Context } from "@oh-my-pi/pi-ai";
import { setSourceOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";
import { FRAME_TOKEN_ESTIMATE } from "@oh-my-pi/snapcompact";
import { buildCompactionDiagnostics, type CompactionDiagnosticsInput } from "../src/session/compaction-diagnostics";

function inventory(input: Partial<CompactionDiagnosticsInput>) {
	return buildCompactionDiagnostics({
		messages: [],
		tokenizer: new Tokenizer(),
		fixedCosts: { systemPromptTokens: 0, toolsTokens: 0, systemContextTokens: 0, skillsTokens: 0 },
		model: { provider: "openai", id: "test" },
		method: "uncompacted",
		settings: {},
		target: {},
		before: { tokens: null, basis: "unknown", description: "Not observed" },
		...input,
	});
}

test("native image references and typed screenshot outputs remain priced without pixel bytes", () => {
	const context: Context = {
		messages: [
			{
				role: "user",
				content: "",
				timestamp: 1,
				providerPayload: {
					type: "openaiResponsesHistory",
					provider: "openai",
					items: [
						{ type: "input_image", file_id: "image-reference", detail: "auto" },
						{
							type: "computer_call_output",
							call_id: "computer",
							output: { type: "computer_screenshot", file_id: "screenshot-reference" },
						},
						{ type: "image_generation_call", result: "aW1hZ2U=" },
						{ type: "compaction", encrypted_content: "opaque" },
					],
				},
			},
		],
	};
	const facts = inventory({ preparedContext: context });
	const images = facts.rows.filter(row => row.kind === "original-image");
	expect(images.map(row => row.quantity.tokens)).toEqual([
		IMAGE_TOKEN_ESTIMATE,
		IMAGE_TOKEN_ESTIMATE,
		IMAGE_TOKEN_ESTIMATE,
	]);
	expect(images.map(row => row.counts.images)).toEqual([1, 1, 1]);
	expect(facts.total.tokens).toBe(3 * IMAGE_TOKEN_ESTIMATE);
	expect(facts.rows.some(row => row.kind === "native" && row.quantity.tokens === null)).toBe(true);
	expect(
		facts.distribution.ordinary.tokens! + facts.distribution.addedUser.tokens! + facts.distribution.shared.tokens!,
	).toBe(facts.total.tokens!);
});

test("neutral original-image identity wins over summary-role fallback while a mixed archive frame is charged once", () => {
	const original = { type: "image" as const, mimeType: "image/png", data: "original" };
	setSourceOrigin(original, {
		kind: "source",
		parts: [{ entryId: "selected", order: 0, blockIndex: 0, coverage: "full", representation: "original-image" }],
	});
	const frame = { type: "image" as const, mimeType: "image/png", data: "frame" };
	const facts = inventory({
		method: "snapcompact",
		messages: [
			{ role: "compactionSummary", summary: "", tokensBefore: 9000, timestamp: 1, blocks: [original, frame] },
		],
		sourceRepresentation: {
			version: 1,
			throughEntryId: "ordinary",
			layout: [{ kind: "frame", frameIndex: 0, range: { start: 0, end: 10 } }],
			coverage: [
				{
					entryId: "selected",
					order: 0,
					snapshot: { blockIndex: 1, start: 0, end: 5 },
					normalized: { start: 0, end: 5 },
					status: "exact-current",
					contribution: "selected-user",
				},
				{
					entryId: "ordinary",
					order: 1,
					snapshot: { blockIndex: 0, start: 0, end: 5 },
					normalized: { start: 5, end: 10 },
					status: "exact-current",
					contribution: "ordinary",
				},
			],
		},
		sourceLocations: [{ messageIndex: 0, blockIndex: 1, layoutIndex: 0 }],
	});
	expect(facts.rows.filter(row => row.kind === "original-image").map(row => row.quantity.tokens)).toEqual([
		IMAGE_TOKEN_ESTIMATE,
	]);
	const frames = facts.rows.filter(row => row.kind === "frame");
	expect(frames.map(row => row.quantity.tokens)).toEqual([FRAME_TOKEN_ESTIMATE]);
	expect(frames[0]!.contributions).toEqual(["selected-user", "ordinary"]);
	expect(facts.rows.reduce((sum, row) => sum + (row.quantity.tokens ?? 0), 0)).toBe(facts.total.tokens!);
});

test("same-entry original and delivered projections keep separate physical ownership", () => {
	const delivered = {
		role: "user" as const,
		content: [{ type: "text" as const, text: "expanded delivery" }],
		timestamp: 1,
	};
	const original = {
		role: "user" as const,
		content: [{ type: "text" as const, text: "/original command" }],
		timestamp: 1,
	};
	const originalProjection = { projection: "original" as const };
	const sourceRepresentation: NonNullable<CompactionDiagnosticsInput["sourceRepresentation"]> = {
		version: 1,
		coverage: [
			{
				entryId: "same",
				order: 0,
				snapshot: { blockIndex: 0, start: 0, end: delivered.content[0]!.text.length },
				current: { blockIndex: 0, start: 0, end: delivered.content[0]!.text.length },
				status: "exact-current",
				contribution: "ordinary",
			},
			{
				entryId: "same",
				order: 0,
				snapshot: { blockIndex: 0, start: 0, end: original.content[0]!.text.length },
				current: { blockIndex: 0, start: 0, end: original.content[0]!.text.length },
				status: "exact-current",
				contribution: "selected-user",
				...originalProjection,
			},
		],
		layout: [
			{
				kind: "source",
				entryId: "same",
				order: 0,
				spans: [{ blockIndex: 0, start: 0, end: delivered.content[0]!.text.length }],
			},
			{
				kind: "source",
				entryId: "same",
				order: 0,
				spans: [{ blockIndex: 0, start: 0, end: original.content[0]!.text.length }],
				...originalProjection,
			},
		],
	};
	for (const form of ["reconstructed", "prepared-ranges", "prepared-whole"] as const) {
		const prepared = form !== "reconstructed";
		if (prepared)
			for (const [message, projection] of [
				[delivered, {}],
				[original, originalProjection],
			] as const) {
				setSourceOrigin(message.content[0]!, {
					kind: "source",
					parts: [
						{
							entryId: "same",
							order: 0,
							blockIndex: 0,
							coverage: "full",
							representation: "native",
							sourceLength: message.content[0]!.text.length,
							...projection,
						},
					],
				});
			}
		if (form === "prepared-whole")
			for (const [index, layout] of sourceRepresentation.layout.entries()) {
				if (layout.kind === "source") layout.contribution = index === 0 ? "ordinary" : "selected-user";
			}
		const facts = inventory({
			method: "snapcompact",
			messages: [delivered, original],
			sourceRepresentation,
			...(prepared
				? { preparedContext: { messages: [delivered, original] } }
				: {
						sourceLocations: [
							{ messageIndex: 0, layoutIndex: 0 },
							{ messageIndex: 1, layoutIndex: 1 },
						],
					}),
		});
		expect(facts.rows.filter(row => row.kind === "text").map(row => row.contributions)).toEqual([
			["ordinary"],
			["selected-user"],
		]);
		expect(facts.distribution.ordinary.tokens).toBe(new Tokenizer().countMessage(delivered));
		expect(facts.distribution.addedUser.tokens).toBe(new Tokenizer().countMessage(original));
		expect(facts.distribution.shared.tokens).toBe(0);
	}
});

test("Anthropic replay inventories aggregate summary, opaque state and file metadata without charging its fallback wrapper", () => {
	const payload = {
		type: "anthropicCompaction" as const,
		provider: "anthropic",
		content: "Native summary",
		encryptedContent: "opaque-state",
		filesText: "<read-files>README</read-files>",
	};
	const message = {
		role: "user" as const,
		timestamp: 1,
		content: "Fallback wrapper must not be priced as native replay",
		providerPayload: payload,
	};
	const retained = { role: "user" as const, timestamp: 2, content: "Retained original" };
	const tokenizer = new Tokenizer();
	const facts = inventory({
		model: { provider: "anthropic", id: "native" },
		method: "anthropic-native",
		preparedContext: { messages: [message, retained] },
	});
	const replay = facts.rows.filter(row => row.coverage === "aggregate");
	expect(replay.map(row => row.kind)).toEqual(["summary", "native", "summary"]);
	expect(replay.map(row => row.quantity.tokens)).toEqual([
		tokenizer.countTokens(payload.content),
		null,
		tokenizer.countTokens(payload.filesText),
	]);
	expect(replay.map(row => row.sourceIds)).toEqual([undefined, undefined, undefined]);
	expect(replay.reduce((sum, row) => sum + row.counts.blocks, 0)).toBe(2);
	expect(replay.reduce((sum, row) => sum + row.counts.messages, 0)).toBe(1);
	expect(facts.total.tokens).toBe(
		tokenizer.countTokens(payload.content) +
			tokenizer.countTokens(payload.filesText) +
			tokenizer.countMessage(retained),
	);
	expect(facts.warning).toBeDefined();
	const fallback = inventory({ preparedContext: { messages: [message] } });
	expect(fallback.total.tokens).toBe(tokenizer.countMessage(message));
	expect(fallback.rows.some(row => row.quantity.tokens === null)).toBe(false);
});

test("Anthropic assistant replay counts its continuing text alongside native summary state", () => {
	const tokenizer = new Tokenizer();
	const message = {
		role: "assistant" as const,
		content: [{ type: "text" as const, text: "Continuing answer" }],
		api: "anthropic-messages" as const,
		provider: "anthropic",
		model: "native",
		stopReason: "stop" as const,
		timestamp: 1,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		providerPayload: {
			type: "anthropicCompaction" as const,
			provider: "anthropic",
			content: "Native summary",
			encryptedContent: "opaque",
			filesText: "not emitted on assistant payload",
		},
	};
	const facts = inventory({
		model: { provider: "anthropic", id: "native" },
		preparedContext: { messages: [message] },
	});
	expect(facts.total.tokens).toBe(tokenizer.countTokens("Native summary") + tokenizer.countMessage(message));
	expect(facts.rows.filter(row => row.coverage !== "fixed").map(row => row.kind)).toEqual([
		"summary",
		"native",
		"text",
	]);
	expect(facts.rows.reduce((sum, row) => sum + row.counts.messages, 0)).toBe(1);
});
