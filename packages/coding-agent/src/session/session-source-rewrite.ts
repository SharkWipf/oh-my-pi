import {
	remapCompactionSourceRepresentation,
	type SourceRewrite,
} from "@oh-my-pi/pi-agent-core/compaction/source";
import { bindMessageSource, remapNativeItemOrigins } from "@oh-my-pi/pi-ai/utils/source-origin";
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
	let bindNativeIngress = false;
	for (const rewrite of rewrites) {
		const entry = getEntry(rewrite.entryId);
		if (entry?.type === "message" && entry.message.role === "assistant" && entry.message.providerPayload?.type === "openaiResponsesHistory" && entry.message.providerPayload.contentBlocks?.length) bindNativeIngress = true;
		if (entry && !originals.has(entry.id)) originals.set(entry.id, entry);
		collectTargets(rewrite.entryId, getEntry, childrenOf, targets);
	}
	if (bindNativeIngress) {
		// Ancestry depth is the frozen branch's source order, not the mixed-branch file position.
		const pending = entries.filter(entry => entry.parentId === null).map(entry => ({ entry, order: 0 }));
		while (pending.length) {
			const { entry, order } = pending.pop()!;
			if (originals.has(entry.id) && entry.type === "message" && entry.message.role === "assistant" && entry.message.providerPayload?.type === "openaiResponsesHistory" && entry.message.providerPayload.contentBlocks?.length) {
				// Consume construction correspondence before the first mutation can shift content positions.
				bindMessageSource(entry.message, entry.id, order);
			}
			for (const child of childrenOf(entry.id)) pending.push({ entry: child, order: order + 1 });
		}
	}
	for (const [id, entry] of originals) originals.set(id, structuredClone(entry));
	const getOriginal: EntryLookup = id => originals.get(id) ?? getEntry(id);

	applySourceRewrites();

	// Representation changes must not replace the accepted original with pruned
	// bytes. Reuse the already-frozen changed entries; ordinary unchanged sources
	// remain sparse and do not acquire a second body. Authored invalidation keeps
	// the old source unavailable and is not a representation update.
	for (const [id, originalEntry] of originals) {
		const entry = getEntry(id);
		if (entry?.type !== "message" || originalEntry.type !== "message" ||
			("requirementsInvalidated" in entry && entry.requirementsInvalidated)) continue;
		const message = entry.message;
		const original = originalEntry.message;
		if ((message.role !== "user" && message.role !== "custom") ||
			(original.role !== "user" && original.role !== "custom") || message.originalSubmission) continue;
		const content = original.content;
		message.originalSubmission = original.originalSubmission ?? {
			text: typeof content === "string" ? content : content.filter(block => block.type === "text").map(block => block.text).join(""),
			images: typeof content === "string" ? undefined : content.filter(block => block.type === "image"),
			imageLinks: original.imageLinks,
			compactionOverride: original.compactionOverride,
		};
	}

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
