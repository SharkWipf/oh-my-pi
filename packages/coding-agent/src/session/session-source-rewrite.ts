import {
	remapCompactionSourceRepresentation,
	type SourceRewrite,
} from "@oh-my-pi/pi-agent-core/compaction/source";
import { remapNativeItemOrigins } from "@oh-my-pi/pi-ai/utils/source-origin";
import {
	buildPreservedUserMessageClassifierInputFromLookup,
	preservedUserMessageClassifierInputsEqual,
	projectPreservedUserMessageClassifierMessage,
} from "./preserve-user-messages-classifier";
import {
	decodePreservedUserMessageClassifications,
	INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
	packPreservedUserMessageClassifications,
	USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE,
} from "./preserved-message-settings";
import type { SessionEntry } from "./session-entries";

type EntryLookup = (id: string) => SessionEntry | undefined;
type ChildLookup = (id: string) => readonly SessionEntry[];

/** Follow only the auxiliary window a changed source can affect, including sibling branches. */
function collectTargets(
	entryId: string,
	getEntry: EntryLookup,
	childrenOf: ChildLookup,
	targets: Set<string>,
): void {
	const source = getEntry(entryId);
	if (source?.type !== "message") return;
	const projection = projectPreservedUserMessageClassifierMessage(source.message);
	if (!projection) return;
	const user = projection.role === "user";
	if (user) targets.add(entryId);
	const stack = childrenOf(entryId).map(entry => ({ id: entry.id, assistants: 0 }));
	while (stack.length > 0) {
		const cursor = stack.pop()!;
		const entry = getEntry(cursor.id);
		if (!entry || entry.type === "reset_boundary") continue;
		let assistants = cursor.assistants;
		if (entry.type === "message") {
			const neighbor = projectPreservedUserMessageClassifierMessage(entry.message);
			if (neighbor?.role === "user") {
				targets.add(entry.id);
				if (user) continue;
			} else if (neighbor?.role === "assistant" && !user && ++assistants === 2) {
				continue;
			}
		}
		for (const child of childrenOf(entry.id)) stack.push({ id: child.id, assistants });
	}
}

function ancestryContains(entry: SessionEntry, affected: ReadonlySet<string>, getEntry: EntryLookup): boolean {
	let parentId = entry.parentId;
	while (parentId !== null) {
		if (affected.has(parentId)) return true;
		const parent = getEntry(parentId);
		if (!parent) return false;
		parentId = parent.parentId;
	}
	return false;
}

/**
 * A cold, synchronous journal mutation: source bytes, current artifact coverage, and
 * every successful classifier shadow move together before the existing disk rewrite.
 * The callback changes only the source entries named by the positional maps.
 */
export function rewriteSessionSources(
	entries: readonly SessionEntry[],
	rewrites: readonly SourceRewrite[],
	applySourceRewrites: () => void,
	getEntry: EntryLookup,
	childrenOf: ChildLookup,
): readonly string[] {
	const originals = new Map<string, SessionEntry>();
	const targets = new Set<string>();
	for (const rewrite of rewrites) {
		const entry = getEntry(rewrite.entryId);
		if (entry && !originals.has(entry.id)) originals.set(entry.id, structuredClone(entry));
		collectTargets(rewrite.entryId, getEntry, childrenOf, targets);
	}
	const getOriginal: EntryLookup = id => originals.get(id) ?? getEntry(id);

	applySourceRewrites();

	for (const rewrite of rewrites) collectTargets(rewrite.entryId, getEntry, childrenOf, targets);
	const affected = new Set<string>();
	for (const id of targets) {
		if (!preservedUserMessageClassifierInputsEqual(
			buildPreservedUserMessageClassifierInputFromLookup(id, getOriginal),
			buildPreservedUserMessageClassifierInputFromLookup(id, getEntry),
		)) affected.add(id);
	}
	for (const entry of entries) {
		if (entry.type === "compaction") {
			entry.preserveData = remapCompactionSourceRepresentation(entry.preserveData, rewrites);
			const native = entry.preserveData?.openaiRemoteCompaction;
			if (native && typeof native === "object") {
				const replacementOrigins = "replacementOrigins" in native && Array.isArray(native.replacementOrigins)
					? remapNativeItemOrigins(native.replacementOrigins, rewrites) : undefined;
				const allUserOrigin = "allUserSources" in native && Array.isArray(native.allUserSources)
					? remapNativeItemOrigins([{ kind: "source", parts: native.allUserSources }], rewrites)[0] : undefined;
				if (replacementOrigins || allUserOrigin?.kind === "source") {
					entry.preserveData = { ...entry.preserveData, openaiRemoteCompaction: {
						...native,
						...(replacementOrigins && { replacementOrigins }),
						...(allUserOrigin?.kind === "source" && { allUserSources: allUserOrigin.parts }),
					} };
				}
			}
		} else if (entry.type === "message" && "providerPayload" in entry.message) {
			const payload = entry.message.providerPayload;
			if (payload?.type === "openaiResponsesHistory" && payload.origins) {
				payload.origins = remapNativeItemOrigins(payload.origins, rewrites);
			}
		}
		if (affected.size === 0 || entry.type !== "custom" || entry.customType !== USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE) {
			continue;
		}
		const decoded = decodePreservedUserMessageClassifications(entry.data);
		if (decoded.status === "valid") {
			const retained = decoded.classifications.filter(item => !affected.has(item.id));
			if (retained.length !== decoded.classifications.length) {
				entry.data = packPreservedUserMessageClassifications(retained);
			}
		} else if (ancestryContains(entry, affected, getEntry) || ancestryContains(entry, affected, getOriginal)) {
			entry.customType = INVALIDATED_USER_MESSAGE_CLASSIFICATION_CUSTOM_TYPE;
		}
	}
	return [...affected];
}
