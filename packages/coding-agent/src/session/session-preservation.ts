import {
	decodeCompactionMessageOverride,
	isPreservationAction,
	MESSAGE_OVERRIDE_CUSTOM_TYPE,
	migrateLegacyCompactionPin,
	type PreservationAction,
} from "./preserved-message-settings";
import type { PreservedMessageQuery } from "./preserved-messages";
import type { CustomEntry } from "./session-entries";
import type { SessionManager } from "./session-manager";

/** Cold preparation only: preserve legacy event identity/order before source-ID-addressed work. */
export async function ensurePreservedMessageStateOnDisk(manager: SessionManager): Promise<void> {
	const migrations: { entry: CustomEntry; data: { messageIds: string[]; state: PreservationAction } }[] = [];
	for (const entry of manager.getEntries()) {
		if (entry.type !== "custom") continue;
		if (entry.customType === "com.omp.compaction-preserved") {
			const data = migrateLegacyCompactionPin(entry.data);
			if (data) migrations.push({ entry, data });
		} else if (entry.customType === MESSAGE_OVERRIDE_CUSTOM_TYPE && entry.data &&
			typeof entry.data === "object" && !Array.isArray((entry.data as { messageIds?: unknown }).messageIds)) {
			const data = decodeCompactionMessageOverride(entry.data);
			if (data) migrations.push({ entry, data });
		}
	}
	if (migrations.length > 0) {
		await manager.rewriteEntries([], () => {
			for (const { entry, data } of migrations) {
				entry.customType = MESSAGE_OVERRIDE_CUSTOM_TYPE;
				entry.data = data;
			}
		});
	}
	await manager.ensureOnDisk();
	await manager.flush();
}

export interface PreservedMessageOverrideResetGroup {
	readonly id: string;
	readonly memberIds: readonly string[];
	readonly members: readonly {
		readonly sourceId: string;
		readonly state: PreservationAction;
		readonly revisionId: string | null;
	}[];
}

export interface PreservedMessageOverrideResetSnapshot {
	readonly sessionId: string;
	readonly resetId: string | null;
	readonly anchorId: string | null;
	readonly sourceCount: number;
	readonly groupCount: number;
	readonly groups: readonly PreservedMessageOverrideResetGroup[];
}

export interface SessionPreservationHost {
	readonly sessionManager: SessionManager;
	getQuery(): PreservedMessageQuery;
	/** Establish durable IDs/legacy migration and refresh the authoritative query. */
	preparePreservedMessages(): Promise<PreservedMessageQuery>;
	/** Changes on branch/reset/session transitions, not ordinary suffix appends. */
	ownership(): object;
	onChanged(messageIds: readonly string[]): void;
}

interface CommittedOverride {
	entryId: string;
	messageIds: string[];
	skipped: number;
}

interface ResetAction {
	ownership: object;
	committed?: CommittedOverride;
	complete: boolean;
}

/** Manual actions use the policy owner's complete atoms and the journal's existing write boundary. */
export class SessionPreservation {
	readonly #host: SessionPreservationHost;
	readonly #actions = new WeakMap<PreservedMessageOverrideResetSnapshot, ResetAction>();
	readonly #sets = new Map<string, { state: PreservationAction; snapshot: PreservedMessageOverrideResetSnapshot }>();
	#tail: Promise<unknown> = Promise.resolve();

	constructor(host: SessionPreservationHost) {
		this.#host = host;
	}

	async capturePreservedMessageOverrideReset(sourceIds?: readonly string[]): Promise<PreservedMessageOverrideResetSnapshot> {
		const query = this.#host.getQuery();
		const groups: PreservedMessageOverrideResetGroup[] = [];
		const snapshot = {
			sessionId: this.#host.sessionManager.getSessionId(),
			resetId: query.resetId,
			anchorId: this.#host.sessionManager.getLeafId(),
			sourceCount: 0,
			groupCount: 0,
			groups,
		};
		const action: ResetAction = { ownership: this.#host.ownership(), complete: false };
		const candidates = sourceIds === undefined ? query.getManualGroups({ nonAutoOnly: true }) : (function* () {
			const count = sourceIds.length;
			for (let index = 0; index < count; index++) {
				const group = query.getManualGroup(sourceIds[index]!);
				if (!group) throw new Error("This source is not a complete manageable message group.");
				yield group;
			}
		})();
		const seen = sourceIds === undefined ? undefined : new Set<string>();
		let deadline = performance.now() + 4;
		for (const group of candidates) {
			if (!seen?.has(group.id) && group.members.some(member => member.state !== "auto")) {
				seen?.add(group.id);
				groups.push(Object.freeze({
					id: group.id,
					memberIds: Object.freeze([...group.memberIds]),
					members: Object.freeze(group.members.map(member => Object.freeze({ ...member }))),
				}));
				snapshot.sourceCount += group.memberIds.length;
			}
			if (performance.now() >= deadline) {
				await new Promise<void>(resolve => setTimeout(resolve, 0));
				this.#validate(snapshot, action);
				deadline = performance.now() + 4;
			}
		}
		await this.#host.sessionManager.ensureOnDisk();
		await this.#host.sessionManager.flush();
		this.#validate(snapshot, action);
		snapshot.groupCount = groups.length;
		Object.freeze(groups);
		Object.freeze(snapshot);
		this.#actions.set(snapshot, action);
		return snapshot;
	}

	setPreservedMessageOverride(sourceId: string, state: PreservationAction): Promise<void> {
		if (!isPreservationAction(state)) return Promise.reject(new Error("Invalid manual preservation state."));
		let previous = this.#sets.get(sourceId);
		if (previous && this.#actions.get(previous.snapshot)?.ownership !== this.#host.ownership()) {
			this.#sets.delete(sourceId);
			previous = undefined;
		}
		let snapshot: PreservedMessageOverrideResetSnapshot;
		if (previous && previous.state === state) snapshot = previous.snapshot;
		else {
			const group = this.#host.getQuery().getManualGroup(sourceId);
			if (!group) return Promise.reject(new Error("This source is not a complete manageable message group."));
			if (!previous && group.members.every(member => member.state === state)) return Promise.resolve();
			snapshot = this.#capture([group]);
			this.#sets.set(sourceId, { state, snapshot });
		}
		return this.#enqueue(async () => {
			if (previous && previous.snapshot !== snapshot && this.#actions.get(previous.snapshot)?.committed) {
				await this.#apply(previous.snapshot, previous.state, false);
			}
			await this.#apply(snapshot, state, false);
			if (this.#sets.get(sourceId)?.snapshot === snapshot) this.#sets.delete(sourceId);
		});
	}

	resetPreservedMessageOverrides(snapshot: PreservedMessageOverrideResetSnapshot): Promise<{ reset: number; skipped: number }> {
		return this.#enqueue(() => this.#apply(snapshot, "auto", true));
	}

	#enqueue<T>(operation: () => Promise<T>): Promise<T> {
		const result = this.#tail.then(operation);
		this.#tail = result.catch(() => {});
		return result;
	}

	#capture(groups: readonly PreservedMessageOverrideResetGroup[]): PreservedMessageOverrideResetSnapshot {
		const frozenGroups = Object.freeze(groups.map(group => Object.freeze({
			id: group.id,
			memberIds: Object.freeze([...group.memberIds]),
			members: Object.freeze(group.members.map(member => Object.freeze({ ...member }))),
		})));
		const snapshot = Object.freeze({
			sessionId: this.#host.sessionManager.getSessionId(),
			resetId: this.#host.getQuery().resetId,
			anchorId: this.#host.sessionManager.getLeafId(),
			sourceCount: frozenGroups.reduce((count, group) => count + group.memberIds.length, 0),
			groupCount: frozenGroups.length,
			groups: frozenGroups,
		});
		this.#actions.set(snapshot, { ownership: this.#host.ownership(), complete: false });
		return snapshot;
	}

	#validate(snapshot: PreservedMessageOverrideResetSnapshot, action: ResetAction): void {
		const manager = this.#host.sessionManager;
		if (manager.getSessionId() !== snapshot.sessionId || this.#host.ownership() !== action.ownership ||
			this.#host.getQuery().resetId !== snapshot.resetId) {
			throw new Error("The session, branch, or clear boundary changed; reopen the manual action.");
		}
		let cursor = manager.getLeafId();
		while (cursor !== snapshot.anchorId) {
			if (cursor === null) throw new Error("The captured message branch is no longer active.");
			const entry = manager.getEntry(cursor);
			if (!entry) throw new Error("The captured message branch is no longer active.");
			cursor = entry.parentId;
		}
	}

	#targets(snapshot: PreservedMessageOverrideResetSnapshot, state: PreservationAction, skipNewer: boolean): {
		messageIds: string[]; skipped: number;
	} {
		const query = this.#host.getQuery();
		const messageIds: string[] = [];
		let skipped = 0;
		for (const captured of snapshot.groups) {
			const current = query.getManualGroup(captured.id);
			if (!current || current.memberIds.length !== captured.memberIds.length ||
				current.memberIds.some((id, index) => id !== captured.memberIds[index])) {
				throw new Error("The captured source group changed; reopen the manual action.");
			}
			if (skipNewer && current.members.some((member, index) => member.revisionId !== captured.members[index]?.revisionId)) {
				skipped += captured.memberIds.length;
				continue;
			}
			if (!current.members.every(member => member.state === state)) messageIds.push(...current.memberIds);
		}
		return { messageIds, skipped };
	}

	async #apply(snapshot: PreservedMessageOverrideResetSnapshot, state: PreservationAction, skipNewer: boolean): Promise<{ reset: number; skipped: number }> {
		const action = this.#actions.get(snapshot);
		if (!action) throw new Error("Unknown manual reset snapshot; capture the action again.");
		this.#validate(snapshot, action);
		if (action.complete || snapshot.sourceCount === 0) return { reset: 0, skipped: 0 };
		const manager = this.#host.sessionManager;
		if (action.committed) {
			// A failed post-commit flush retries the retained journal transition, never a second append.
			await manager.recoverPersistenceFromCurrentState();
			await manager.flush();
			this.#validate(snapshot, action);
		} else {
			await this.#host.preparePreservedMessages();
			await manager.ensureOnDisk();
			await manager.flush();
			this.#validate(snapshot, action);
			const targets = this.#targets(snapshot, state, skipNewer);
			if (targets.messageIds.length === 0) {
				action.complete = true;
				return { reset: 0, skipped: targets.skipped };
			}
			try {
				await manager.appendEntriesAtomically(() => {
					this.#validate(snapshot, action);
					const current = this.#targets(snapshot, state, skipNewer);
					if (current.messageIds.length === 0) return;
					const entryId = manager.appendCustomEntry(MESSAGE_OVERRIDE_CUSTOM_TYPE, { messageIds: current.messageIds, state });
					action.committed = { entryId, ...current };
				});
				await manager.flush();
			} catch (error) {
				const retained = this.#actions.get(snapshot)?.committed;
				if (retained && !manager.getEntry(retained.entryId)) action.committed = undefined;
				throw error;
			}
			this.#validate(snapshot, action);
		}
		action.complete = true;
		const committed = action.committed;
		if (!committed) return { reset: 0, skipped: this.#targets(snapshot, state, skipNewer).skipped };
		this.#host.onChanged(committed.messageIds);
		return { reset: committed.messageIds.length, skipped: committed.skipped };
	}
}
