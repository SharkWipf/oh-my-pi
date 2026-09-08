import type { Settings } from "../config/settings";
import type { AgentSession } from "../session/agent-session";
import { SessionManager } from "../session/session-manager";
import type { RequirementsScope } from "./types";

/** Preserve only explicitly selected dependents before the ordinary journal deletion. */
export async function confirmRequirementsJournalDeletion(
	settings: Settings,
	journalPath: string,
	choose: (title: string, options: string[]) => Promise<string | undefined>,
): Promise<boolean> {
	if (!settings.get("requirements.enabled")) return true;
	const storage = settings.getStorage();
	if (!storage) return true;
	const dependents = storage.getRequirementsJournalDependents(journalPath);
	if (!dependents.revisions.length) return true;
	const retain = "Retain required original evidence, then delete session";
	const withdraw = "Withdraw dependent requirements, then delete session";
	const choice = await choose(
		`This session supplies evidence for ${dependents.revisions.length} current requirements.\n${renderRequirementsData(dependents.revisions.map(({ id, statement, scope }) => ({ id, statement, scope })))}\nOrdinary memory banks are not deleted.`,
		["Cancel", retain, withdraw],
	);
	if (choice !== retain && choice !== withdraw) return false;
	const current = storage.getRequirementsJournalDependents(journalPath);
	if (JSON.stringify(current) !== JSON.stringify(dependents)) {
		throw new Error("Requirements dependencies changed while confirming deletion; inspect and confirm again.");
	}
	if (choice === withdraw) {
		storage.withdrawRequirements(dependents.revisions.map(revision => revision.id), "operator:session-delete", "Explicit withdrawal before deleting original session evidence");
		return true;
	}
	const manager = await SessionManager.open(journalPath, undefined, undefined, { suppressBreadcrumb: true });
	try {
		const retained = [];
		for (const source of dependents.sources) retained.push(await manager.retainRequirementsEvidence(source));
		storage.reconcileRequirementsSources(retained.map(source => ({ key: source.key, integrity: source.integrity, locators: source.locators, units: source.units })));
	} finally {
		await manager.close();
	}
	return true;
}

export const REQUIREMENTS_COMMANDS = [
	{ name: "requirements", description: "Living requirements, coverage holes, models and actual injection receipt" },
	{ name: "requirements list", description: "List active, pending, inactive or quarantined records: [state] [search text]" },
	{ name: "requirements coverage", description: "Inspect durable source and requirement coverage across sessions and branches" },
	{ name: "requirements inspect", description: "Inspect a revision: <revision-id>" },
	{ name: "requirements source", description: "Jump to / retrieve original evidence: <source-key>" },
	{ name: "requirements retry", description: "Retry or backfill owner work: [source-key]" },
	{
		name: "requirements evidence",
		description: "Supply host-resolved evidence: <source-key> <referent-source-key...>",
	},
	{
		name: "requirements adopt",
		description: "Manual complete-unit adoption: <source-key> <unit-id> <scope-json> --confirm",
	},
	{ name: "requirements gap", description: "Continue with unresolved coverage: <source-key> <reason> --confirm" },
	{ name: "requirements correct", description: "Submit source-backed correction: <revision-id> <new decision>" },
	{ name: "requirements withdraw", description: "Submit explicit withdrawal: <revision-id> <reason>" },
	{ name: "requirements scope", description: "Submit explicit scope decision: <revision-id> <scope and reason>" },
	{
		name: "requirements quarantine",
		description: "Suspend revisions, preserving evidence: <id,id...> <reason> --confirm",
	},
	{ name: "requirements restore", description: "Request fresh evidence and sanity review: <id,id...>" },
	{
		name: "requirements clear",
		description: "Explicit scoped withdrawal: <session|project|global|all> <reason> --confirm",
	},
	{
		name: "requirements bypass-future",
		description: "Stop only future V2 requirements use; ordinary memory/history remain: --confirm",
	},
	{
		name: "requirements retry-clean",
		description: "Open a fresh session without learned memory; retain ONLY selected originals: [source-key...] --confirm",
	},
	{ name: "requirements help", description: "All actions, confirmation syntax and safety limits" },
];

/** Render derived data as inert text, without changing the authoritative stored evidence. */
export function renderRequirementsData(value: unknown): string {
	return (JSON.stringify(value, null, 2) ?? "null").replace(
		/[<>&`\u007f-\u009f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/g,
		char => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
	);
}

export function requirementsContextText(session: AgentSession): string {
	const status = session.requirements.status();
	const receipt = status.receipt;
	const coverage = status.applicable.ledgerCoverage;
	const unresolved = coverage.total - coverage.referenceOnly - coverage.byState.complete;
	return [
		"Requirements — /memory requirements",
		`State: ${status.enabled ? "enabled" : "disabled"}; work ${status.running ? "running" : "idle"}; accepted active: ${status.applicable.active.length}; conflict groups: ${status.applicable.conflicts.length}; relevant coverage holes: ${status.applicable.coverageGaps.length}; current pending inputs: ${status.applicable.pendingSources.length}`,
		`Original source catalog: ${status.sourceCatalog}. Catalog metadata is last-observed, not proof that every original remains available. Consumed requirements evidence is freshly verified; cataloging is separate from extraction.`,
		`Ledger-wide coverage (all known sessions and branches): ${coverage.total} source records; ${coverage.referenceOnly} referents; dispositions ${renderRequirementsData(coverage.byState)}. ${status.sourceCatalog === "last-observed" && unresolved === 0 ? "Known source processing complete; not semantic completeness." : "Coverage incomplete; history not fully cataloged or unresolved source dispositions remain."}`,
		"Historical coverage is an inventory, not active instructions or a demand to resolve all history before work. Full ledger: /memory requirements coverage; exact original: /memory requirements source <source-key>; explicit processing: /memory requirements retry [source-key].",
		`Total active token estimate: ${status.activeTokens ?? "unknown"} (${status.tokenProvenance}); accepted records remain inspectable when recall is suspended.`,
		`Last request receipt: ${receipt?.phase ?? "unknown"}; actual injected: ${receipt?.phase === "sent" ? receipt.revisionIds.length : "unknown"}; ${receipt?.phase === "sent" ? "sent" : "not confirmed sent"} tokens: ${receipt?.tokens ?? "unknown"} (${receipt?.tokenProvenance ?? "unknown"}). This receipt is not a promise about the next request.`,
		...(receipt?.capacity
			? [
					`Request capacity: context window ${receipt.capacity.contextWindow} tokens; irreducible prompt ${receipt.capacity.irreducibleTokens ?? "unknown"} tokens; reserve ${receipt.capacity.reserveTokens} tokens (${receipt.capacity.provenance}). Unknown capacity is not proof of overflow.`,
				]
			: []),
		"Settings: /settings → Memory → Living Requirements; /model → Requirements Extractor / Evidence / Sanity",
		`V2 requirements bypass: ${status.bypass}. ${status.bypass === "off" ? "" : "Requirements recall intentionally suspended; not complete recall. Ordinary memory and earlier transcript/provider history remain."} AGENTS, rules, skills and context files remain independent and are not certified safe.`,
		`Context files still enabled: ${renderRequirementsData(session.memoryRecoveryContextFiles)}`,
		"Coverage describes processed evidence, not proof that every natural-language requirement was found.",
		"Capacity overflow: select a larger-context model, explicitly narrow scope, or approve staged work; no hidden truncation.",
		`Model routes and capabilities: ${renderRequirementsData(status.models)}`,
		...(status.error ? [`Unresolved: ${renderRequirementsData(status.error)}`] : []),
	].join("\n");
}

function help(): string {
	return [
		...REQUIREMENTS_COMMANDS.map(command => `/memory ${command.name} — ${command.description}`),
		"IDs are exact owner-assigned IDs shown in status/inspect/source; no text matching or implicit source selection.",
		'adopt scope-json example: {"kind":"project","projectId":"/exact/project/path"}. Use scope identities from status.',
		"Manual adoption accepts the COMPLETE source unit, never an extracted substring; fresh candidate-only sanity is still required.",
		"Status/source shows adoptedUnitIds. Adopting one unit leaves sibling units uncovered; use Retry, supply evidence, adopt another complete unit, or explicitly continue with a gap.",
		"Correct, withdraw and scope create explicit operator sources and enter the normal review pipeline; they do not self-certify acceptance.",
		"Gap passage leaves coverage visibly incomplete. Restore leaves rejected/unavailable/uncertain revisions suspended.",
		"Requirements bypass is sticky for this run; restart to resume V2. It does not disable ordinary memory tools, erase prior injections or clean provider history.",
		"Retry-clean opens a new session and provider root with backend off, autolearn off and V2 disabled. No prior transcript/summary is implicitly carried. Already submitted remote writes cannot be retracted.",
		"Start without learned memory before startup: omp --start-without-memory. AGENTS, rules, skills and context files still apply independently.",
	].join("\n");
}

/** Shared operator-command path for TUI, ACP, RPC and print hosts. Jobs belong to the session owner. */
export async function executeRequirementsCommand(session: AgentSession, argumentText: string): Promise<string> {
	const match = /^(\S+)?\s*([\s\S]*)$/.exec(argumentText.trim());
	const verb = match?.[1]?.toLowerCase() ?? "status";
	let rest = match?.[2] ?? "";
	const confirmed = /(?:^|\s)--confirm$/.test(rest);
	if (confirmed) rest = rest.replace(/(?:^|\s)--confirm$/, "").trim();
	const first = /^(\S+)\s*([\s\S]*)$/.exec(rest);
	const id = first?.[1];
	const tail = first?.[2] ?? "";
	const need = (condition: unknown, usage: string): void => {
		if (!condition) throw new Error(`Usage: /memory requirements ${usage}`);
	};
	const owner = session.requirements;
	switch (verb) {
		case "help":
			return help();
		case "status":
			return `${requirementsContextText(session)}\n\n${renderRequirementsData(owner.status())}\n\n${help()}`;
		case "list": {
			const state = id ?? "active";
			need(["active", "pending", "inactive", "quarantined"].includes(state), "list [active|pending|inactive|quarantined] [search text]");
			const status = owner.status();
			const activeIds = new Set(status.applicable.active.map(revision => revision.id));
			const records = state === "active" ? status.applicable.active
				: state === "pending" ? status.applicable.pendingSources
				: owner.status({ includeLedger: true }).snapshot.revisions.filter(revision => state === "quarantined"
					? revision.lifecycle === "quarantined"
					: revision.lifecycle !== "quarantined" && !activeIds.has(revision.id));
			const search = tail.toLocaleLowerCase();
			const matching = search ? records.filter(record => JSON.stringify(record).toLocaleLowerCase().includes(search)) : records;
			return `${state}: ${matching.length} matching of ${records.length} records\n${renderRequirementsData(matching)}`;
		}
		case "coverage":
			return `${requirementsContextText(session)}\n\n${renderRequirementsData(owner.status({ includeLedger: true }))}\n\n${help()}`;
		case "inspect":
			need(id && !tail, "inspect <revision-id>");
			return renderRequirementsData(await owner.applyOperatorAction({ kind: "inspect", revisionId: id! }));
		case "source":
		case "jump":
		case "retrieve":
			need(id && !tail, "source <source-key>");
			return renderRequirementsData(await owner.inspectSource(id!));
		case "retry":
			need(!tail, "retry [source-key]");
			await owner.applyOperatorAction({ kind: "retry", sourceKey: id });
			break;
		case "evidence":
			need(id && tail, "evidence <source-key> <referent-source-key...>");
			await owner.applyOperatorAction({ kind: "evidence", sourceKey: id!, referentKeys: tail.split(/\s+/) });
			break;
		case "adopt": {
			const unit = /^(\S+)\s+([\s\S]+)$/.exec(tail);
			need(id && unit && confirmed, "adopt <source-key> <unit-id> <scope-json> --confirm");
			const scope: RequirementsScope = JSON.parse(unit![2]);
			await owner.applyOperatorAction({ kind: "literal-adopt", sourceKey: id!, unitId: unit![1], scope });
			break;
		}
		case "gap":
			need(id && tail && confirmed, "gap <source-key> <reason> --confirm");
			await owner.applyOperatorAction({ kind: "gap", sourceKey: id!, reason: tail });
			break;
		case "correct":
		case "withdraw":
		case "scope":
			need(id && tail, `${verb} <revision-id> <decision/reason>`);
			await owner.applyOperatorAction({ kind: "inspect", revisionId: id! });
			await owner.applyOperatorAction({
				kind: "decision",
				targetRevisionIds: [id!],
				text: `Explicit operator requirements ${verb} decision for revision ${id}:\n${tail}`,
			});
			break;
		case "quarantine":
			need(id && tail && confirmed, "quarantine <revision-id,revision-id...> <reason> --confirm");
			await owner.applyOperatorAction({ kind: "quarantine", revisionIds: id!.split(","), reason: tail });
			break;
		case "restore":
			need(id && !tail, "restore <revision-id,revision-id...>");
			await owner.applyOperatorAction({ kind: "restore", revisionIds: id!.split(",") });
			break;
		case "clear":
			need(
				id && ["session", "project", "global", "all"].includes(id) && tail && confirmed,
				"clear <session|project|global|all> <reason> --confirm",
			);
			await owner.applyOperatorAction({
				kind: "clear",
				scope: id as "session" | "project" | "global" | "all",
				reason: tail,
			});
			break;
		case "bypass-future":
			need(confirmed && !rest, "bypass-future --confirm");
			session.bypassRequirementsForRun();
			return `Future V2 requirements bypassed. Ordinary memory tools, existing transcript and provider history remain; this is NOT a clean session.\n${requirementsContextText(session)}`;
		case "retry-clean": {
			need(confirmed, "retry-clean [original-source-key...] --confirm");
			const result = await session.retryWithoutMemory(rest ? rest.split(/\s+/) : []);
			return `Fresh session opened with learned memory off; only explicitly selected originals retained. AGENTS, rules, skills and context files remain independent. Already submitted remote writes cannot be retracted.\n${renderRequirementsData(result)}\n${requirementsContextText(session)}`;
		}
		default:
			throw new Error(help());
	}
	return `Operator action recorded; consult status for pending review versus accepted publication.\n${requirementsContextText(session)}`;
}
