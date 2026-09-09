import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { AgentStorage } from "../src/session/agent-storage";
import { getApplicableRequirements, requirementsSourceKey, RequirementsStore } from "../src/requirements/store";
import type {
	RequirementsAuthority,
	RequirementsBatch,
	RequirementsOperation,
	RequirementsReview,
	RequirementsRestoreReceipt,
	RequirementsSource,
} from "../src/requirements/types";
import { composeProviderRequirements, type RequirementsApplicableSnapshot } from "../src/session/session-requirements";

const directories: string[] = [];
function reviewed(operations: RequirementsOperation[], operationIds: string[]): RequirementsReview {
	const candidates = operationIds.map(id => ({
		id,
		decision: "pass" as const,
		reason: "Faithful to the complete original unit",
	}));
	return {
		evidence: {
			model: "isolated-evidence",
			format: "json",
			outcome: "accepted",
			candidates,
			obligations: operations.map((operation, index) => ({
				id: `obligation-${index}`,
				kind: operation.kind,
				requirementId: operation.requirementId,
				predecessorRevisionIds: operation.predecessorRevisionIds,
				statement: operation.statement,
				sourceUnitIds: operation.evidence.map(item => item.unitId),
				operationIds: [operationIds[index]],
				applicableRevisionIds: [],
				adoptedUnitIds: [],
				decision: "pass",
				reason: "Independent obligation covered by this candidate",
			})),
		},
		sanity: { model: "candidate-only", format: "json", outcome: "accepted", candidates: structuredClone(candidates) },
	};
}
afterEach(() => {
	AgentStorage.close();
	for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

async function fixture() {
	const directory = mkdtempSync(join(tmpdir(), "omp-requirements-reducer-"));
	directories.push(directory);
	const dbPath = join(directory, "agent.db");
	let storage = await AgentStorage.open(dbPath);
	const authority: RequirementsAuthority = { ownerSessionId: "owner", branchId: "leaf", epoch: 0, generation: 0 };
	storage.authorizeRequirementsOwner(authority);
	function capture(
		id: string,
		body: string | string[],
		parent: RequirementsSource | null = null,
		owner = "owner",
		referenceOnly = false,
	) {
		const journalPath = join(directory, `${id}.journal`);
		writeFileSync(journalPath, typeof body === "string" ? body : JSON.stringify(body));
		const integrity = createHash("sha256").update(readFileSync(journalPath)).digest("hex");
		const original = { journalId: owner, entryId: id };
		const units: RequirementsSource["units"] = (typeof body === "string" ? [body] : body).map((text, index) => ({
			id: index === 0 ? "text" : `text${index}`,
			kind: "text",
			byteLength: Buffer.byteLength(text),
			sha256: createHash("sha256").update(text).digest("hex"),
		}));
		const source: RequirementsSource = {
			key: requirementsSourceKey(original),
			original,
			integrity,
			parentKey: parent?.key ?? null,
			ownerSessionId: owner,
			branchId: "leaf",
			epoch: 0,
			origin: { kind: referenceOnly ? "assistant" : "human" },
			referenceOnly: referenceOnly ? true : undefined,
			locators: [{ sessionId: owner, journalPath, entryId: id }],
			units,
			durable: true,
			state: "pending",
		};
		storage.intakeRequirementsSource(source);
		return source;
	}
	function verified() {
		return Object.fromEntries(
			storage
				.getRequirementsSnapshot()
				.sources.map(source => [
					source.key,
					createHash("sha256").update(readFileSync(source.locators[0].journalPath!)).digest("hex"),
				]),
		);
	}
	function proposal(
		source: RequirementsSource,
		statement: string,
		target?: string,
		predecessors: string[] = [],
		auth = authority,
	): RequirementsBatch {
		const operation: RequirementsOperation = {
			kind: target ? "change" : "add",
			requirementId: target,
			statement,
			scope: { kind: "project", projectId: "project" },
			evidence: [
				{
					sourceKey: source.key,
					integrity: source.integrity,
					unitId: "text",
				},
			],
			predecessorRevisionIds: predecessors,
		};
		const operationIds = [randomUUID()];
		return {
			operationIds,
			id: randomUUID(),
			sourceKey: source.key,
			sourceIntegrity: source.integrity,
			extractionVersion: "test",
			reviewRevision: randomUUID(),
			manifest: source.units,
			readHeads: storage.getRequirementsConsumptionSnapshot({ projectId: "project", sessionId: "owner", epoch: 0, branchId: "leaf" }).readHeads,
			readSourceIntegrities: verified(),
			authority: auth,
			operations: [operation],
			review: reviewed([operation], operationIds),
			status: "reviewed",
		};
	}
	function publish(batch: RequirementsBatch) {
		const id = storage.saveRequirementsBatch(batch);
		return storage.publishRequirementsBatch(id, batch.authority, verified());
	}
	return {
		directory,
		dbPath,
		get storage() {
			return storage;
		},
		context: { projectId: "project", sessionId: "owner", epoch: 0, branchId: "leaf" },
		authority,
		capture,
		proposal,
		publish,
		verified,
		receipt(
			revisionIds: string[],
			generation: number,
			store: AgentStorage | RequirementsStore = storage,
		): RequirementsRestoreReceipt {
			const operations = revisionIds.map(id => store.getRequirementsRevision(id)!);
			const evidence = operations.flatMap(operation => [
				...operation.evidence,
				...(operation.referents ?? []),
				...(operation.relations ?? []).flatMap(relation => relation.evidence),
			]);
			return {
				id: randomUUID(), actor: "operator", revisionIds, operationIds: revisionIds, generation,
				publicationRevision: store.getRequirementsState().publicationRevision,
				readHeads: store.getRequirementsConsumptionSnapshot(this.context).readHeads,
				sourceIntegrities: Object.fromEntries(evidence.map(item => [item.sourceKey, item.integrity])),
				review: reviewed(operations, revisionIds),
			};
		},
		async restart() {
			AgentStorage.close();
			storage = await AgentStorage.open(dbPath);
		},
		applicable() {
			return getApplicableRequirements(storage.getRequirementsSnapshot(), {
				sessionId: "owner",
				epoch: 0,
				projectId: "project",
			});
		},
	};
}

test("reverse corrections survive restart; stale relationships cannot CAS-rebase, gaps/backfill and unordered sessions preserve authority", async () => {
	const f = await fixture();
	const root = f.capture("root", "Use default indentation.");
	const initial = f.publish(f.proposal(root, "default"));
	expect(initial.status).toBe("accepted");
	const firstRevision = f.storage.getRequirementsSnapshot().revisions[0];
	const tabs = f.capture("tabs", "Use tabs.", root);
	const spaces = f.capture("spaces", "Correction: use spaces.", tabs);
	const newerEarly = f.proposal(spaces, "spaces", firstRevision.requirementId, [firstRevision.id]);
	f.storage.saveRequirementsBatch(newerEarly);
	await f.restart();
	expect(f.publish(f.proposal(tabs, "tabs", firstRevision.requirementId, [firstRevision.id])).status).toBe("accepted");
	expect(f.storage.publishRequirementsBatch(newerEarly.id, f.authority, f.verified()).status).toBe("stale");
	const tabsRevision = f.storage.getRequirementsSnapshot().revisions.find(revision => revision.statement === "tabs")!;
	const repaired = f.proposal(spaces, "spaces", firstRevision.requirementId, [tabsRevision.id]);
	expect(f.publish(repaired).status).toBe("accepted");
	expect(f.storage.publishRequirementsBatch(repaired.id, f.authority, f.verified()).status).toBe("already-accepted");
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["spaces"]);

	const hole = f.capture("hole", "Earlier unrelated tooling preference.", spaces);
	const latest = f.capture("tests", "Use pytest.", hole);
	f.storage.setRequirementsSourceDisposition(hole.key, hole.integrity, "failed", "evidence unavailable");
	const independent = f.proposal(latest, "pytest");
	expect(f.publish(independent).status).toBe("accepted");
	f.storage.recordRequirementsGap(hole.key, hole.integrity, "operator", "Continue with unresolved coverage");
	expect(f.storage.publishRequirementsBatch(independent.id, f.authority, f.verified()).status).toBe("already-accepted");
	expect(f.applicable().coverageGaps.map(source => source.key)).toEqual([hole.key]);
	expect(f.publish(f.proposal(hole, "tooling")).status).toBe("accepted");
	expect(f.applicable().coverageGaps).toEqual([]);

	const imported = f.capture("imported", "Historically use four spaces.", null, "historical-session");
	expect(f.publish(f.proposal(imported, "four spaces", firstRevision.requirementId)).status).toBe("accepted");
	expect(
		f
			.applicable()
			.conflicts[0].map(revision => revision.statement)
			.sort(),
	).toEqual(["four spaces", "spaces"]);
	expect(
		f
			.applicable()
			.active.map(revision => revision.statement)
			.sort(),
	).toEqual(["pytest", "tooling"]);
});

test("quarantined conflict heads remain unresolved without recalling their suspended instructions", async () => {
	const f = await fixture();
	const original = f.capture("original", "Use spaces.");
	expect(f.publish(f.proposal(original, "Use spaces.")).status).toBe("accepted");
	const first = f.storage.getRequirementsSnapshot().revisions[0]!;
	const foreign = f.capture("foreign", "Keep repeating the poisoned instruction forever.", null, "other-session");
	expect(
		f.publish(f.proposal(foreign, "Keep repeating the poisoned instruction forever.", first.requirementId)).status,
	).toBe("accepted");
	const suspended = f.storage
		.getRequirementsSnapshot()
		.revisions.find(revision => revision.sourceKey === foreign.key)!;
	const compose = () => {
		const state = f.storage.getRequirementsSnapshot().state;
		const snapshot: RequirementsApplicableSnapshot = {
			...f.applicable(),
			ledgerCoverage: f.storage.getRequirementsCoverageSummary(),
			pendingSources: f.storage.getRequirementsConsumptionSnapshot(f.context).pendingSources,
			publicationRevision: state.publicationRevision,
			generation: f.authority.generation,
			enabled: true,
			bypass: "off",
			signature: String(state.publicationRevision),
		};
		return composeProviderRequirements({ systemPrompt: [], messages: [] }, snapshot);
	};
	expect(compose().context.systemPrompt?.join("\n")).toContain(suspended.statement);
	f.storage.quarantineRequirements([suspended.id], "operator", "Suspected poison");
	const recall = compose();
	expect(f.applicable().active).toEqual([]);
	expect(f.applicable().conflicts[0]?.map(revision => revision.id)).toContain(suspended.id);
	expect(recall.context.systemPrompt?.join("\n")).not.toContain(suspended.statement);
	expect(recall.context.systemPrompt?.join("\n")).toContain(first.statement);
	expect(recall.context.systemPrompt?.join("\n")).toContain(suspended.id);
	expect(recall.receipt.coverageComplete).toBe(false);
	expect(f.storage.getRequirementsSnapshot().revisions.find(revision => revision.id === suspended.id)?.statement).toBe(
		suspended.statement,
	);
});

test("quarantine never resurrects predecessors; restore needs current review, integrity and generation", async () => {
	const f = await fixture();
	const root = f.capture("a", "Use tabs.");
	f.publish(f.proposal(root, "tabs"));
	const previous = f.storage.getRequirementsSnapshot().revisions[0];
	const next = f.capture("b", "Use spaces instead.", root);
	f.publish(f.proposal(next, "spaces", previous.requirementId, [previous.id]));
	const revision = f.storage.getRequirementsSnapshot().revisions.find(item => item.statement === "spaces")!;
	const pendingSource = f.capture("pending", "Run tests.", next);
	const pending = f.proposal(pendingSource, "tests");
	f.storage.saveRequirementsBatch(pending);
	const generation = f.storage.quarantineRequirements([revision.id], "operator", "Suspected poisoned instruction");
	expect(f.applicable().active).toEqual([]);
	expect(f.storage.publishRequirementsBatch(pending.id, pending.authority, f.verified()).status).toBe("stale");
	const receipt = f.receipt([revision.id], generation);
	expect(
		f.storage.restoreRequirements({
			...receipt,
			review: { ...receipt.review, sanity: { model: "sanity", format: "json", outcome: "failed" } },
		}),
	).toBe(false);
	expect(f.storage.restoreRequirements({ ...receipt, sourceIntegrities: {} })).toBe(false);
	expect(f.storage.restoreRequirements(receipt)).toBe(true);
	expect(f.storage.restoreRequirements(receipt)).toBe(true);
	expect(f.applicable().active.map(item => item.statement)).toEqual(["spaces"]);
	await f.restart();
	expect(f.storage.getRequirementsSnapshot().state.restoreReceipts.map(item => item.id)).toEqual([receipt.id]);
});

test("journal rewrite invalidates frozen output and missing provenance before recall", async () => {
	const f = await fixture();
	const source = f.capture("rewrite", "éKeep spaces.");
	const batch = f.proposal(source, "spaces");
	f.storage.saveRequirementsBatch(batch);
	writeFileSync(source.locators[0].journalPath!, "changed bytes");
	expect(f.storage.publishRequirementsBatch(batch.id, f.authority, f.verified()).status).toBe("stale");
	f.storage.reconcileRequirementsSources([{ key: source.key, integrity: null }]);
	expect(f.storage.getRequirementsSnapshot().sources[0].state).toBe("orphaned");
	expect(f.storage.getRequirementsSnapshot().batches[0].status).toBe("stale");
	expect(f.applicable().active).toEqual([]);
});

test("separate processes publish unrelated scoped heads without a global completion fence", async () => {
	const f = await fixture();
	const a = f.capture("one", "First instruction.");
	const b = f.capture("two", "Other instruction.", null, "other");
	const first = f.proposal(a, "first");
	const second = f.proposal(b, "second");
	f.storage.saveRequirementsBatch(first);
	f.storage.saveRequirementsBatch(second);
	const beforePublication = f.storage.getRequirementsConsumptionSnapshot(f.context);
	const script = `import {AgentStorage} from ${JSON.stringify(join(import.meta.dir, "../src/session/agent-storage.ts"))}; const [path,id,authority,verified]=process.argv.slice(1); const db=await AgentStorage.open(path); console.log(JSON.stringify(db.publishRequirementsBatch(id,JSON.parse(authority),JSON.parse(verified)))); AgentStorage.close();`;
	const outputs = await Promise.all(
		[first, second].map(async batch => {
			const process = Bun.spawn(
				["bun", "-e", script, f.dbPath, batch.id, JSON.stringify(batch.authority), JSON.stringify(f.verified())],
				{ stdout: "pipe", stderr: "pipe" },
			);
			const output = await new Response(process.stdout).text();
			const errors = await new Response(process.stderr).text();
			expect(await process.exited, errors).toBe(0);
			return JSON.parse(output).status;
		}),
	);
	expect(outputs).toEqual(["accepted", "accepted"]);
	expect(f.applicable().active.map(item => item.statement).sort()).toEqual(["first", "second"]);
	const afterPublication = f.storage.getRequirementsConsumptionSnapshot(f.context);
	expect(beforePublication.state.publicationRevision).toBe(0);
	expect(beforePublication.revisions).toEqual([]);
	expect(afterPublication.state.publicationRevision).toBe(2);
	expect(afterPublication.revisions.map(item => item.statement).sort()).toEqual(["first", "second"]);
	expect(afterPublication.sources.map(source => source.key)).toContain(afterPublication.revisions[0].sourceKey);

});

test("reviewed historical source edges preserve newer heads and reject cycles atomically", async () => {
	const f = await fixture();
	const root = f.capture("root", "Use tabs.");
	f.publish(f.proposal(root, "tabs"));
	const first = f.storage.getRequirementsSnapshot().revisions[0];
	const next = f.capture("next", "Use spaces.", root);
	f.publish(f.proposal(next, "spaces", first.requirementId, [first.id]));
	const old = f.capture("old", "Historical indentation.", null, "imported-journal");
	const backfill = f.proposal(old, "historical", first.requirementId);
	backfill.operations[0].relations = [
		{ predecessorSourceKey: old.key, successorSourceKey: root.key, evidence: backfill.operations[0].evidence },
	];
	expect(f.publish(backfill).status).toBe("accepted");
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["spaces"]);
	const cycle = f.capture("cycle", "Reverse the relationship.");
	const bad = f.proposal(cycle, "invalid", first.requirementId);
	bad.operations[0].relations = [
		{ predecessorSourceKey: next.key, successorSourceKey: old.key, evidence: bad.operations[0].evidence },
	];
	const before = f.storage.getRequirementsSnapshot().state.publicationRevision;
	expect(f.publish(bad).status).toBe("rejected");
	expect(f.storage.getRequirementsSnapshot().state.publicationRevision).toBe(before);
	expect(f.storage.getRequirementsSnapshot().sources.find(source => source.key === cycle.key)?.state).toBe("pending");
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["spaces"]);
});


test("retries replace unfinished work while accepted lineage and failed restore reviews remain durable", async () => {
	const f = await fixture();
	const source = f.capture("pending", "Keep this instruction.");
	const first = f.proposal(source, "one");
	f.storage.saveRequirementsBatch(first);
	const second = f.proposal(source, "two");
	f.storage.saveRequirementsBatch(second);
	await f.restart();
	expect(f.storage.getRequirementsSnapshot().batches.map(batch => batch.id)).toEqual([second.id]);
	expect(f.storage.publishRequirementsBatch(first.id, f.authority, f.verified()).status).toBe("rejected");
	expect(f.storage.publishRequirementsBatch(second.id, f.authority, f.verified()).status).toBe("accepted");
	const revision = f.storage.getRequirementsSnapshot().revisions[0];
	const generation = f.storage.quarantineRequirements([revision.id], "operator", "review needed");
	const rejected = f.receipt([revision.id], generation);
	rejected.review.sanity!.outcome = "rejected";
	expect(f.storage.restoreRequirements(rejected)).toBe(false);
	await f.restart();
	expect(f.storage.getRequirementsSnapshot().state.restoreReviews[revision.id].receipt.review.sanity?.outcome).toBe(
		"rejected",
	);
	expect(
		f.storage.restoreRequirements({
			...rejected,
			id: randomUUID(),
			review: { ...rejected.review, sanity: { ...rejected.review.sanity!, outcome: "accepted" } },
		}),
	).toBe(true);
	expect(f.storage.getRequirementsSnapshot().state.restoreReviews[revision.id].receipt.review.sanity?.outcome).toBe(
		"rejected",
	);
	expect(f.applicable().active.map(item => item.statement)).toEqual(["two"]);
});

test("missing-source reconciliation is idempotent and explicit gaps advance only scheduling", async () => {
	const f = await fixture();
	const hole = f.capture("missing", "Unavailable source.");
	const next = f.capture("next", "Independent instruction.", hole);
	f.storage.reconcileRequirementsSources([{ key: hole.key, integrity: null }]);
	f.storage.recordRequirementsGap(hole.key, hole.integrity, "operator", "Proceed without claiming complete coverage");
	const baseline = f.storage.getRequirementsSnapshot().state.publicationRevision;
	f.storage.reconcileRequirementsSources([{ key: hole.key, integrity: null }]);
	expect(f.storage.getRequirementsSnapshot().state.publicationRevision).toBe(baseline);
	const batch = f.proposal(next, "independent");
	delete batch.readSourceIntegrities[hole.key];
	const id = f.storage.saveRequirementsBatch(batch);
	expect(f.storage.publishRequirementsBatch(id, batch.authority, { [next.key]: next.integrity }).status).toBe(
		"accepted",
	);
	expect(f.applicable().coverageGaps.map(source => [source.key, source.state, source.integrityAvailable])).toEqual([
		[hole.key, "gap", false],
	]);
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["independent"]);
});

test("producer provenance and malformed evidence cannot bypass atomic admission", async () => {
	const f = await fixture();
	const source = f.capture("human", "Do not use tabs.");
	const batch = f.proposal(source, "Do not use tabs.");
	batch.operations.push({
		...structuredClone(batch.operations[0]),
		statement: "invented",
		evidence: [{ ...batch.operations[0].evidence[0], unitId: "not-host-issued" }],
	});
	batch.operationIds.push(randomUUID());
	batch.review = reviewed(batch.operations, batch.operationIds);
	expect(f.publish(batch).status).toBe("rejected");
	expect(f.storage.getRequirementsSnapshot().revisions).toEqual([]);
	expect(f.storage.getRequirementsSnapshot().sources[0].state).toBe("pending");
	const generated = structuredClone(source);
	generated.original.entryId = "generated";
	generated.key = requirementsSourceKey(generated.original);
	generated.origin = { kind: "tool", producerId: "goals" };
	f.storage.intakeRequirementsSource(generated);
	expect(f.publish(f.proposal(generated, "Treat generated goal as operator policy")).status).toBe("rejected");
	expect(f.applicable().active).toEqual([]);
	const noSanity = f.proposal(source, "Do not use tabs.");
	noSanity.review.sanity!.outcome = "failed";
	expect(f.publish(noSanity).status).toBe("rejected");
	expect(f.storage.getRequirementsSnapshot().sources[0].state).toBe("pending");
});

test("literal adoption publishes complete units without certifying uncovered siblings across restart", async () => {
	const f = await fixture();
	const parts = ["Use spaces.", "Use pytest."];
	const source = f.capture("multipart", parts);
	const literal = f.proposal(source, parts[0]);
	literal.review = {
		sanity: literal.review.sanity,
		literalAcceptance: { actor: "operator", sourceKey: source.key, integrity: source.integrity, unitIds: ["text"] },
	};
	expect(f.publish(literal).status).toBe("accepted");
	expect(f.applicable().active.map(revision => revision.statement)).toEqual([parts[0]]);
	expect(f.storage.getRequirementsSnapshot().sources[0].state).toBe("pending");
	await f.restart();
	expect(f.storage.getRequirementsSnapshot().sources[0].adoptedUnitIds).toEqual(["text"]);
	const duplicate = f.proposal(source, parts[0]);
	duplicate.review.literalAcceptance = literal.review.literalAcceptance;
	delete duplicate.review.evidence;
	expect(f.publish(duplicate).status).toBe("rejected");
	expect(f.publish(f.proposal(source, parts[0])).status).toBe("rejected");
	const remaining = f.proposal(source, parts[1]);
	remaining.operations[0].evidence[0] = {
		sourceKey: source.key,
		integrity: source.integrity,
		unitId: "text1",
	};
	remaining.review = {
		sanity: remaining.review.sanity,
		literalAcceptance: { actor: "operator", sourceKey: source.key, integrity: source.integrity, unitIds: ["text1"] },
	};
	expect(f.publish(remaining).status).toBe("accepted");
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(parts);
	expect(f.storage.getRequirementsSnapshot().sources[0].state).toBe("complete");
});

test("literal unit exception retains unsupported sibling coverage and still requires fresh sanity", async () => {
	const f = await fixture();
	const source = f.capture("parts", ["Keep this policy.", "opaque attachment"]);
	writeFileSync(
		source.locators[0].journalPath!,
		JSON.stringify(["Keep this policy.", { type: "opaque", data: "opaque attachment" }]),
	);
	const changedIntegrity = f.verified()[source.key];
	const units = structuredClone(source.units);
	units[1].kind = "opaque";
	units[1].unsupportedReason = "Unsupported attachment";
	f.storage.reconcileRequirementsSources([{ key: source.key, integrity: changedIntegrity, units }]);
	source.integrity = changedIntegrity;
	source.units = units;
	const literal = f.proposal(source, "Keep this policy.");
	literal.readSourceIntegrities = { [source.key]: changedIntegrity };
	literal.review = {
		sanity: { ...literal.review.sanity!, outcome: "failed" },
		literalAcceptance: { actor: "operator", sourceKey: source.key, integrity: source.integrity, unitIds: ["text"] },
	};
	let id = f.storage.saveRequirementsBatch(literal);
	expect(f.storage.publishRequirementsBatch(id, literal.authority, literal.readSourceIntegrities).status).toBe(
		"rejected",
	);
	literal.review.sanity!.outcome = "accepted";
	id = f.storage.saveRequirementsBatch(literal);
	expect(f.storage.publishRequirementsBatch(id, literal.authority, literal.readSourceIntegrities).status).toBe(
		"accepted",
	);
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["Keep this policy."]);
	expect(f.storage.getRequirementsSnapshot().sources[0].state).toBe("pending");
	expect(f.storage.getRequirementsSnapshot().sources[0].units[1].unsupportedReason).toBe("Unsupported attachment");
});

test("adopted non-human evidence has no coverage authority and reopens its human decision when unavailable", async () => {
	const f = await fixture();
	const referent = f.capture("option", "Option two uses UTF-8.", null, "owner", true);
	expect(f.applicable().coverageGaps).toEqual([]);
	expect(() => f.publish(f.proposal(referent, "Use UTF-8."))).toThrow();
	expect(() => f.storage.recordRequirementsGap(referent.key, referent.integrity, "operator", "Not a source")).toThrow();
	const acceptance = f.capture("acceptance", "Use option two.");
	const proposal = f.proposal(acceptance, "Use UTF-8.");
	proposal.operations[0].referents = [{
		sourceKey: referent.key,
		integrity: referent.integrity,
		unitId: "text",
	}];
	proposal.operations[0].relations = [{
		predecessorSourceKey: referent.key,
		successorSourceKey: acceptance.key,
		evidence: proposal.operations[0].evidence,
	}];
	expect(f.publish(proposal).status).toBe("rejected");
	delete proposal.operations[0].relations;
	expect(f.publish(proposal).status).toBe("accepted");
	await f.restart();
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["Use UTF-8."]);
	expect(f.applicable().coverageGaps).toEqual([]);
	f.storage.reconcileRequirementsSources([{ key: referent.key, integrity: null }]);
	expect(f.applicable().active).toEqual([]);
	expect(f.applicable().coverageGaps.map(source => [source.key, source.state])).toEqual([[acceptance.key, "pending"]]);
	f.storage.reconcileRequirementsSources([{ key: referent.key, integrity: referent.integrity }]);
	await f.restart();
	expect(f.applicable().active.map(item => item.statement)).toEqual(["Use UTF-8."]);
	expect(f.applicable().coverageGaps).toEqual([]);
});

test("bulk intake preserves accepted coverage and atomically reconciles changed referents", async () => {
	const f = await fixture();
	const referent = f.capture("bulk-referent", "Use UTF-8.", null, "owner", true);
	const acceptance = f.capture("bulk-acceptance", "Use that encoding.");
	const proposal = f.proposal(acceptance, "Use UTF-8.");
	proposal.operations[0].referents = [{
		sourceKey: referent.key,
		integrity: referent.integrity,
		unitId: "text",
	}];
	expect(f.publish(proposal).status).toBe("accepted");
	const moved = { ...acceptance, locators: [...acceptance.locators, { sessionId: "fork", entryId: "copy" }] };
	f.storage.intakeRequirementsSources([referent, moved, moved]);
	f.storage.reconcileRequirementsSources([{ key: referent.key, integrity: null }]);
	expect(f.applicable().active).toEqual([]);
	f.storage.intakeRequirementsSources([{ ...referent, units: [], integrityAvailable: false }, moved]);
	expect(f.applicable().active).toEqual([]);
	f.storage.intakeRequirementsSource(referent);
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["Use UTF-8."]);
	await f.restart();
	const before = f.storage.getRequirementsSnapshot();
	expect(before.sources.find(source => source.key === acceptance.key)?.locators).toEqual(moved.locators);
	expect(f.applicable().active.map(revision => revision.statement)).toEqual(["Use UTF-8."]);
	expect(f.applicable().coverageGaps).toEqual([]);
	const original = { journalId: "owner", entryId: "bulk-new" };
	const added = { ...referent, original, key: requirementsSourceKey(original) };
	const changed = { ...referent, integrity: "changed", units: [{ ...referent.units[0], sha256: "changed" }] };
	const malformed = { ...acceptance, units: [{ ...acceptance.units[0], byteLength: -1 }] };
	expect(() => f.storage.intakeRequirementsSources([added, malformed])).toThrow();
	expect(f.storage.getRequirementsSnapshot()).toEqual(before);
	expect(() => f.storage.intakeRequirementsSources([added, changed, malformed])).toThrow();
	expect(f.storage.getRequirementsSnapshot()).toEqual(before);
	f.storage.intakeRequirementsSources([added, changed]);
	await f.restart();
	expect(f.applicable().active).toEqual([]);
	expect(f.applicable().coverageGaps.map(source => [source.key, source.state])).toEqual([[acceptance.key, "pending"]]);
	const duplicateOriginal = { journalId: "owner", entryId: "bulk-duplicate" };
	const duplicate = { ...referent, original: duplicateOriginal, key: requirementsSourceKey(duplicateOriginal) };
	const revision = f.storage.getRequirementsSnapshot().state.publicationRevision;
	f.storage.intakeRequirementsSources([duplicate, { ...duplicate, integrity: "new-integrity" }]);
	const after = f.storage.getRequirementsSnapshot();
	expect(after.sources.filter(source => source.key === duplicate.key).map(source => source.integrity)).toEqual(["new-integrity"]);
	expect(after.state.publicationRevision).toBe(revision + 1);
});

test("owner changes preserve restore history and fence late publication across SQLite connections", async () => {
	const f = await fixture();
	const source = f.capture("owner-policy", "Keep this policy.");
	expect(f.publish(f.proposal(source, "Keep this policy.")).status).toBe("accepted");
	const revision = f.storage.getRequirementsSnapshot().revisions[0];
	const generation = f.storage.quarantineRequirements([revision.id], "operator", "Review policy");
	const receipt = f.receipt([revision.id], generation);
	expect(f.storage.restoreRequirements(receipt)).toBe(true);
	const authority = { ...f.authority, branchId: "new-leaf", epoch: 1, generation };
	f.storage.authorizeRequirementsOwner(authority);
	const pendingSource = f.capture("owner-pending", "Run checks.");
	const pending = f.proposal(pendingSource, "Run checks.", undefined, [], authority);
	f.storage.saveRequirementsBatch(pending);
	const other = { ...authority, ownerSessionId: "other" };
	const script =
		"import {AgentStorage} from " + JSON.stringify(join(import.meta.dir, "../src/session/agent-storage.ts")) +
		"; const [path,other]=process.argv.slice(1); const storage=await AgentStorage.open(path); " +
		"storage.authorizeRequirementsOwner(JSON.parse(other)); storage.invalidateRequirementsOwner('owner'); AgentStorage.close();";
	const child = Bun.spawn(["bun", "-e", script, f.dbPath, JSON.stringify(other)], { stdout: "pipe", stderr: "pipe" });
	const errors = await new Response(child.stderr).text();
	expect(await child.exited, errors).toBe(0);
	expect(() => f.storage.authorizeRequirementsOwner(f.authority)).toThrow();
	expect(f.storage.publishRequirementsBatch(pending.id, authority, f.verified()).status).toBe("stale");
	f.storage.authorizeRequirementsOwner(authority);
	f.storage.invalidateRequirementsOwner("absent-owner");
	await f.restart();
	expect(f.storage.getRequirementsSnapshot().state.owners).toEqual({ owner: authority, other });
	expect(f.storage.restoreRequirements(receipt)).toBe(true);
	expect(f.applicable().active.map(item => item.statement)).toEqual(["Keep this policy."]);
	expect(f.publish(f.proposal(pendingSource, "Run checks.", undefined, [], authority)).status).toBe("accepted");
});

test("unchanged observations preserve frozen reviews; locator moves and same-byte restoration retain evidence fences", async () => {
	const f = await fixture();
	const referent = f.capture("observed-reference", "Use UTF-8.", null, "owner", true);
	const source = f.capture("observed-acceptance", "Use that encoding.");
	const batch = f.proposal(source, "Use UTF-8.");
	batch.operations[0].referents = [{
		sourceKey: referent.key,
		integrity: referent.integrity,
		unitId: "text",
	}];
	f.storage.saveRequirementsBatch(batch);
	const unchanged = [referent, source].map(item => ({
		key: item.key,
		integrity: item.integrity,
		locators: structuredClone(item.locators),
		units: structuredClone(item.units),
	}));
	f.storage.reconcileRequirementsSources([...unchanged, unchanged[0], { key: "unknown", integrity: null }]);
	expect(f.storage.publishRequirementsBatch(batch.id, f.authority, f.verified()).status).toBe("accepted");
	const movedPath = join(f.directory, "moved-reference.journal");
	writeFileSync(movedPath, readFileSync(referent.locators[0].journalPath!));
	rmSync(referent.locators[0].journalPath!);
	const moved = [{ ...referent.locators[0], journalPath: movedPath }];
	f.storage.reconcileRequirementsSources([{ key: referent.key, integrity: referent.integrity, locators: moved }]);
	await f.restart();
	expect(f.verified()[referent.key]).toBe(referent.integrity);
	expect(f.applicable().active.map(item => item.statement)).toEqual(["Use UTF-8."]);
	const pendingSource = f.capture("observed-pending", "Run checks.");
	const pending = f.proposal(pendingSource, "Run checks.");
	f.storage.saveRequirementsBatch(pending);
	f.storage.reconcileRequirementsSources([
		{ key: referent.key, integrity: referent.integrity, locators: moved },
		{ key: referent.key, integrity: null },
		{ key: referent.key, integrity: referent.integrity, locators: moved, units: referent.units },
	]);
	await f.restart();
	expect(f.applicable().active.map(item => item.statement)).toEqual(["Use UTF-8."]);
	expect(f.applicable().coverageGaps.map(item => item.key)).toEqual([pendingSource.key]);
	expect(f.storage.getRequirementsSnapshot().revisions[0].lifecycle).toBe("accepted");
	expect(f.storage.publishRequirementsBatch(pending.id, f.authority, f.verified()).status).toBe("accepted");
	expect(f.storage.getRequirementsSource(source.key)?.state).toBe("complete");
});





test("settled applicable IDs include fulfilled withdrawal heads but exclude conflict, quarantine and predecessors", async () => {
	const f = await fixture();
	const original = f.capture("settled-original", "Use tabs.");
	expect(f.publish(f.proposal(original, "Use tabs.")).status).toBe("accepted");
	const first = f.storage.getRequirementsSnapshot().revisions[0];
	const withdrawal = f.capture("settled-withdrawal", "Withdraw the indentation policy.", original);
	const batch = f.proposal(withdrawal, "Withdraw the indentation policy.", first.requirementId, [first.id]);
	batch.operations[0].kind = "withdraw";
	batch.review = reviewed(batch.operations, batch.operationIds);
	expect(f.publish(batch).status).toBe("accepted");
	const withdrawn = f.storage.getRequirementsSnapshot().revisions.find(item => item.batchId === batch.id)!;
	for (const storage of [f.storage, new RequirementsStore(f.storage.getRequirementsSnapshot())]) {
		const view = storage.getRequirementsConsumptionSnapshot(f.context);
		expect(view.applicable.active).toEqual([]);
		expect(view.applicableRevisionIds).toEqual([withdrawn.id]);
		storage.quarantineRequirements([withdrawn.id], "operator", "Review withdrawn effect");
		expect(storage.getRequirementsConsumptionSnapshot(f.context).applicableRevisionIds).toEqual([]);
	}
	const authority = { ...f.authority, generation: f.storage.getRequirementsSnapshot().state.generation };
	f.storage.authorizeRequirementsOwner(authority);
	const conflictSource = f.capture("settled-conflict", "Use spaces.", null, "foreign");
	const conflict = f.proposal(conflictSource, "Use spaces.", first.requirementId, [], authority);
	expect(f.publish(conflict).status).toBe("accepted");
	const view = f.storage.getRequirementsConsumptionSnapshot(f.context);
	expect(view.applicable.conflicts[0].map(item => item.id)).toContain(withdrawn.id);
	expect(view.applicableRevisionIds).toEqual([]);
});

test("temporary original loss preserves quarantine through exact recovery without bypassing fresh restore gates", async () => {
	const f = await fixture();
	const source = f.capture("suspended-original", "Keep this instruction suspended until reviewed.");
	const batch = f.proposal(source, "Keep this instruction suspended until reviewed.");
	expect(f.publish(batch).status).toBe("accepted");
	const revision = f.storage.getRequirementsSnapshot().revisions[0];
	const bytes = readFileSync(source.locators[0].journalPath!);
	for (const storage of [f.storage, new RequirementsStore(f.storage.getRequirementsSnapshot())]) {
		const generation = storage.quarantineRequirements([revision.id], "operator", "Suspected poisoned source");
		const receipt = () => f.receipt([revision.id], generation, storage);
		rmSync(source.locators[0].journalPath!);
		storage.reconcileRequirementsSources([{ key: source.key, integrity: null }]);
		const unavailable = storage.getRequirementsSnapshot();
		expect(unavailable.revisions[0].lifecycle).toBe("quarantined");
		expect(unavailable.revisions[0].quarantine?.reason).toBe("Suspected poisoned source");
		expect(unavailable.sources[0].state).toBe("orphaned");
		expect(storage.getRequirementsConsumptionSnapshot(f.context).applicable.active).toEqual([]);
		const stale = receipt();
		expect(storage.restoreRequirements(stale)).toBe(false);
		writeFileSync(source.locators[0].journalPath!, bytes);
		const integrity = createHash("sha256").update(readFileSync(source.locators[0].journalPath!)).digest("hex");
		storage.reconcileRequirementsSources([{ key: source.key, integrity, units: source.units }]);
		expect(storage.getRequirementsSnapshot().revisions[0].lifecycle).toBe("quarantined");
		expect(storage.restoreRequirements({ ...receipt(), sourceIntegrities: {} })).toBe(false);
		expect(storage.restoreRequirements({ ...receipt(), review: { ...batch.review, sanity: { model: "sanity", format: "json", outcome: "failed" } } })).toBe(false);
		expect(storage.restoreRequirements(receipt())).toBe(true);
		expect(storage.getRequirementsConsumptionSnapshot(f.context).applicable.active.map(item => item.id)).toEqual([revision.id]);
		storage.quarantineRequirements([revision.id], "operator", "Recheck changed source");
		storage.reconcileRequirementsSources([{ key: source.key, integrity: null }]);
		storage.reconcileRequirementsSources([{ key: source.key, integrity: "changed-bytes", units: source.units }]);
		expect(storage.getRequirementsRevision(revision.id)?.lifecycle).toBe("quarantined");
		expect(storage.getRequirementsRevision(revision.id)?.availability).toBe("changed");
	}
});


test("metadata-only historical intake cannot replace a rich accepted manifest", async () => {
	const f = await fixture();
	const source = f.capture("rich-original", ["Keep this policy.", "Original image identity"]);
	source.units[1].kind = "image";
	source.units[1].locator = "blob:original-image";
	f.storage.intakeRequirementsSource(source);
	expect(f.publish(f.proposal(source, "Keep this policy.")).status).toBe("accepted");
	const metadata = { ...source, units: [], integrityAvailable: false, origin: { kind: "unknown" as const }, state: "orphaned" as const };
	for (const store of [f.storage, new RequirementsStore(f.storage.getRequirementsSnapshot())]) {
		store.intakeRequirementsSource(metadata);
		expect(store.getRequirementsSource(source.key)?.units).toEqual(source.units);
		expect(store.getRequirementsConsumptionSnapshot(f.context).applicable.active.map(item => item.statement)).toEqual(["Keep this policy."]);
	}
	await f.restart();
	expect(f.storage.getRequirementsSource(source.key)?.units).toEqual(source.units);
	expect(f.storage.getRequirementsSource(source.key)?.origin.kind).toBe("human");
});

test("cold unavailable ancestry does not block an unrelated live addition", async () => {
	const f = await fixture();
	const ancestor = f.capture("cold-ancestor", "Historical unresolved decision.");
	f.storage.reconcileRequirementsSources([{ key: ancestor.key, integrity: null }]);
	const source = f.capture("fresh-unrelated", "Use pytest.", ancestor);
	const batch = f.proposal(source, "Use pytest.");
	batch.readSourceIntegrities = { [source.key]: source.integrity };
	expect(f.publish(batch).status).toBe("accepted");
	expect(f.storage.getRequirementsConsumptionSnapshot(f.context).applicable.active.map(item => item.statement)).toEqual(["Use pytest."]);
	expect(f.storage.getRequirementsSource(ancestor.key)?.state).toBe("orphaned");
	expect(f.storage.getRequirementsCoverageSummary().byState.orphaned).toBe(1);
});

test("scoped clear survives reopen, blocks historical backfill and admits new live authority", async () => {
	const f = await fixture();
	const original = f.capture("clear-original", "Use tabs.");
	expect(f.publish(f.proposal(original, "Use tabs.")).status).toBe("accepted");
	const old = f.storage.getRequirementsConsumptionSnapshot(f.context).applicable.active[0];
	const historical = f.capture("clear-backfill", "Use spaces.");
	const outside = f.capture("clear-other-project", "Use pytest.");
	const outsideBatch = f.proposal(outside, "Use pytest.");
	outsideBatch.operations[0].scope = { kind: "project", projectId: "other" };
	expect(f.publish(outsideBatch).status).toBe("accepted");
	const generation = f.storage.clearRequirements({ kind: "project", projectId: "project" }, "operator", "Remove poisoned policies");
	const authority = { ...f.authority, generation };
	f.storage.authorizeRequirementsOwner(authority);
	await f.restart();
	expect(f.applicable().active).toEqual([]);
	expect(f.storage.getRequirementsConsumptionSnapshot({ ...f.context, projectId: "other" }).applicable.active.map(item => item.statement)).toEqual(["Use pytest."]);
	expect(f.publish(f.proposal(historical, "Use spaces.", undefined, [], authority)).status).toBe("rejected");
	expect(f.storage.restoreRequirements(f.receipt([old.id], generation))).toBe(false);
	const live = f.capture("clear-new-delivery", "Use four spaces.");
	live.authorityGeneration = generation;
	f.storage.intakeRequirementsSource(live);
	expect(f.publish(f.proposal(live, "Use four spaces.", undefined, [], authority)).status).toBe("accepted");
	expect(f.applicable().active.map(item => item.statement)).toEqual(["Use four spaces."]);
});

test("exact operator withdrawal remains historical through evidence return and quarantine recovery", async () => {
	const f = await fixture();
	const source = f.capture("withdraw-exact", "Keep this policy.");
	expect(f.publish(f.proposal(source, "Keep this policy.")).status).toBe("accepted");
	const revision = f.applicable().active[0];
	const generation = f.storage.withdrawRequirements([revision.id], "operator", "No longer wanted");
	f.storage.reconcileRequirementsSources([{ key: source.key, integrity: null }]);
	f.storage.reconcileRequirementsSources([{ key: source.key, integrity: source.integrity }]);
	await f.restart();
	expect(f.storage.getRequirementsRevision(revision.id)?.lifecycle).toBe("historical");
	expect(f.storage.getRequirementsRevision(revision.id)?.withdrawal).toEqual({ actor: "operator", reason: "No longer wanted", generation });
	expect(f.applicable().active).toEqual([]);
	expect(f.storage.restoreRequirements(f.receipt([revision.id], generation))).toBe(false);
});

test("persisted global pass cannot admit incomplete or misidentified independent review mappings", async () => {
	const f = await fixture();
	const cases: [string, (batch: RequirementsBatch) => void][] = [
		["missing candidate decision", batch => { batch.review.sanity!.candidates = []; }],
		["uncertain candidate despite global pass", batch => { batch.review.evidence!.candidates![0].decision = "uncertain"; }],
		["candidate has no independent obligation", batch => { batch.review.evidence!.obligations = []; }],
		["operation ID does not join the reviewed candidate", batch => { batch.operationIds[0] = "different-candidate"; }],
		["obligation maps to unknown candidate", batch => { batch.review.evidence!.obligations![0].operationIds = ["unknown-candidate"]; }],
		["obligation cites unknown applicable head", batch => { batch.review.evidence!.obligations![0].applicableRevisionIds = ["unknown-head"]; }],
	];
	for (const [name, corrupt] of cases) {
		const source = f.capture(name, "Use pytest.");
		const batch = f.proposal(source, "Use pytest.");
		corrupt(batch);
		const id = f.storage.saveRequirementsBatch(batch);
		await f.restart();
		expect(f.storage.publishRequirementsBatch(id, f.authority, f.verified()).status, name).toBe("rejected");
		expect(f.storage.getRequirementsSource(source.key)?.state, name).toBe("pending");
	}
	expect(f.applicable().active).toEqual([]);
	const source = f.capture("review-repair", "Use pytest.");
	expect(f.publish(f.proposal(source, "Use pytest.")).status).toBe("accepted");
	expect(f.applicable().active.map(item => item.statement)).toEqual(["Use pytest."]);
});

