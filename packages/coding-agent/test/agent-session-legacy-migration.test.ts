import assert from "node:assert/strict";
import { test } from "bun:test";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { prepareCompaction } from "@oh-my-pi/pi-agent-core/compaction";
import { storeCompactionV2PreserveData } from "@oh-my-pi/pi-agent-core/compaction/compaction-v2-streaming";
import { bindMessageSource, getSourceOrigin } from "@oh-my-pi/pi-ai/utils/source-origin";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { compact as compactArchive, getPreservedArchive, resolveShape } from "@oh-my-pi/snapcompact";
import { TempDir } from "@oh-my-pi/pi-utils";
import { ModelRegistry } from "../src/config/model-registry";
import { Settings } from "../src/config/settings";
import { AgentSession } from "../src/session/agent-session";
import { resolveMethodSettings } from "../src/session/compaction-methods";
import { convertToLlm } from "../src/session/messages";
import { SessionManager } from "../src/session/session-manager";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";

test("next-real legacy migration preserves installed history and reports unavailable original sources", async () => {
	const dir = TempDir.createSync("@legacy-migration-");
	const bundled = getBundledModel("openai", "gpt-5.5");
	assert(bundled);
	const model = { ...bundled, tokenizer: undefined, contextWindow: 100_000 };
	const auth = createInMemoryAuthStorage();
	auth.setRuntimeApiKey("openai", "isolated-no-provider");
	const settings = Settings.isolated({
		"requirements.enabled": false,
		"snapcompact.shape": "8x13-bw",
		"compaction.methodOrder": ["snapcompact"],
		"compaction.keepUserMessages": false,
		"compaction.keepFirstLimit": "tokens:0",
		"compaction.keepLastLimit": "tokens:0",
		"compaction.keepRecentUserMessagesLimit": "tokens:0",
		"compaction.keepUserMessagesHeuristic": false,
		"compaction.keepUserMessagesClassifierFilter": false,
		"compaction.keepUserMessagesLlm": false,
		"compaction.pruneLongUserMessages": "no",
		"compaction.autoContinue": false,
		"compaction.asyncEnabled": false,
		"compaction.keepRecentTokens": 1,
		"compaction.reserveTokens": 8192,
		"compaction.thresholdTokens": 80_000,
	});
	const seed = SessionManager.create(dir.path(), dir.path());
	const noProvider = () => { throw new Error("Local prerequisite failure must not call a provider"); };
	try {
		for (let index = 0; index < 8; index++) {
			seed.appendMessage({ role: "user", content: `${index}: ${"Original archived history. ".repeat(300)}`, timestamp: index });
		}
		seed.appendMessage({ role: "user", content: "Original retained tail", timestamp: 9 });
		const seedAgent = new Agent({ initialState: { model, messages: seed.buildSessionContext().messages, systemPrompt: [], tools: [] }, streamFn: noProvider });
		const preparation = prepareCompaction(seed.getBranch(), resolveMethodSettings(settings.getGroup("compaction"), "snapcompact"), model, seedAgent.tokenizer);
		assert(preparation);
		const rendered = await compactArchive(preparation, { convertToLlm, model, shape: resolveShape(model, "8x13-bw"), maxFrames: 3 });
		const legacyArchive = structuredClone(rendered.preserveData);
		assert(legacyArchive);
		delete legacyArchive.sourceRepresentation;
		assert(getPreservedArchive(legacyArchive));

		const image = { type: "image" as const, mimeType: "image/png", data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6mQAAAABJRU5ErkJggg==" };
		const imageMessage = { role: "user" as const, content: [image], timestamp: 10 };
		const imageSourceId = seed.appendMessage(imageMessage);
		bindMessageSource(imageMessage, imageSourceId, 0);
		const imageOrigin = getSourceOrigin(imageMessage);
		assert(imageOrigin?.kind === "source" && imageOrigin.parts[0]?.representation === "original-image");
		// A controlled imported window: its PNG has known source identity; its old aggregate does not.
		const compactionItem = { type: "compaction", encrypted_content: "controlled-imported-legacy-item" };
		const legacyNative = storeCompactionV2PreserveData({
			compactionItem,
			replacementHistory: [{ type: "message", role: "user", content: [{ type: "input_image", image_url: `data:image/png;base64,${image.data}` }] }, compactionItem],
			replacementOrigins: [imageOrigin, { kind: "unknown", reason: "imported legacy aggregate" }],
			usedTokens: 1000,
			retainedImageCount: 1,
		}, model);

		for (const kind of ["source-boundary", "original-image"] as const) {
			for (const trigger of ["manual", "automatic"] as const) {
				const manager = SessionManager.create(dir.path(), dir.path());
				const boundary = kind === "original-image"
					? manager.appendMessage({ role: "user", content: "Available original retained tail", timestamp: 9 })
					: rendered.firstKeptEntryId;
				const missingId = kind === "original-image" ? imageSourceId : boundary;
				const importedId: string = manager.appendCompaction(rendered.summary, rendered.shortSummary, boundary, rendered.tokensBefore, {
					method: kind === "original-image" ? "remote" : "snapcompact",
					preserveData: structuredClone(kind === "original-image" ? legacyNative : legacyArchive),
					details: rendered.details,
				});
				assert.equal(manager.getEntry(missingId), undefined);
				for (let turn = 0; turn < 8; turn++) {
					manager.appendMessage({ role: "user", content: `Fresh turn ${turn}: ${"ordinary new history ".repeat(2000)}`, timestamp: 11 + turn });
				}
				manager.appendMessage({ role: "user", content: "Fresh ordinary tail", timestamp: 20 });
				await manager.flush();
				const installedEntry: string | undefined = JSON.stringify(manager.getEntry(importedId));
				const installedContext = JSON.stringify(convertToLlm(manager.buildSessionContext().messages));
				const agent = new Agent({ initialState: { model, messages: manager.buildSessionContext().messages, systemPrompt: [], tools: [] }, streamFn: noProvider });
				const session = new AgentSession({ agent, sessionManager: manager, settings, modelRegistry: new ModelRegistry(auth) });
				let automaticError: string | undefined;
				const unsubscribe = session.subscribe(event => {
					if (event.type === "auto_compaction_end") automaticError = event.errorMessage;
				});
				try {
					await session.preparePreservedMessages();
					session.getPreservedMessageSelection();
					assert.equal(JSON.stringify(manager.getEntry(importedId)), installedEntry);
					assert.equal(JSON.stringify(convertToLlm(agent.state.messages)), installedContext);
					let manualError: unknown;
					if (trigger === "manual") {
						try { await session.compact(undefined, { mode: "snapcompact" }); }
						catch (error) { manualError = error; }
						assert(manualError instanceof Error);
					} else {
						await session.runIdleCompaction();
					}
					const diagnostic = manualError instanceof Error ? manualError.message : automaticError;
					assert(typeof diagnostic === "string" && diagnostic.includes(missingId), `${kind}/${trigger} must identify the unavailable durable source`);
					if (kind === "original-image") assert.match(diagnostic, /image.*block 0/i);
					assert.equal(manager.getBranch().filter(entry => entry.type === "compaction").length, 1);
					assert.equal(JSON.stringify(manager.getEntry(importedId)), installedEntry);
					assert.equal(JSON.stringify(convertToLlm(agent.state.messages)), installedContext);
				} finally {
					unsubscribe();
					await session.dispose();
				}
			}
		}
	} finally {
		await seed.close();
		auth.close();
		dir.removeSync();
	}
}, 30_000);
