import { Database, type Statement } from "bun:sqlite";
import * as fs from "node:fs";
import * as path from "node:path";
import {
	type AuthCredential,
	type AuthCredentialStore,
	isSqliteBusyError,
	SqliteAuthCredentialStore,
	type StoredAuthCredential,
} from "@oh-my-pi/pi-ai";
import {
	AsyncDrain,
	checkpointWal,
	getAgentDbPath,
	getDbBusyTimeoutMs,
	getStatsDbPath,
	isRecord,
	logger,
	postmortem,
} from "@oh-my-pi/pi-utils";
import type { RawSettings as Settings } from "../config/settings";
import {
	createEmptyRequirementsCoverageSummary,
	createEmptyRequirementsSnapshot,
	type RequirementsBackend,
	RequirementsStore,
} from "../requirements/store";
import type {
	RequirementsAuthority,
	RequirementsBatch,
	RequirementsConsumptionContext,
	RequirementsConsumptionSnapshot,
	RequirementsCoverageSummary,
	RequirementsObservation,
	RequirementsRestoreReceipt,
	RequirementsRelation,
	RequirementsRevision,
	RequirementsScope,
	RequirementsSnapshot,
	RequirementsSource,
	RequirementsSourceMetadata,
	RequirementsSourceState,
} from "../requirements/types";

/** Row shape for settings table queries */
type SettingsRow = {
	key: string;
	value: string;
};

/** Row shape for model_usage table queries */
type ModelUsageRow = {
	model_key: string;
	last_used_at: number;
};

/** Row shape for model_perf table queries */
type ModelPerfRow = {
	model_key: string;
	samples: number;
	output_tokens: number;
	gen_ms: number;
	ttft_samples: number;
	ttft_ms: number;
};

/** Row shape read from an `omp stats` messages table during backfill. */
type StatsMessageRow = {
	rowid: number;
	timestamp: number;
	provider: string;
	model: string;
	output_tokens: number;
	duration: number;
	ttft: number | null;
};

/** Per-model running sums accumulated during a backfill walk. */
type PerfAccum = {
	samples: number;
	outputTokens: number;
	genMs: number;
	ttftSamples: number;
	ttftMs: number;
};

/** One completed request's timing, folded into the per-model aggregates. */
export interface ModelPerfSample {
	/** Output tokens the provider reported for the turn. */
	outputTokens: number;
	/** Total request duration in milliseconds. */
	durationMs: number;
	/** Time to first token in milliseconds; omit when the provider did not report one. */
	ttftMs?: number;
}

/** Validated, insert-ready model_perf sample (see {@link normalizeModelPerfSample}). */
type ModelPerfInsert = {
	modelKey: string;
	outputTokens: number;
	durationMs: number;
	ttftSamples: 0 | 1;
	ttftMs: number;
};

/** Recency-weighted per-model performance averages. */
export interface ModelPerfStats {
	/** Decayed sample count backing the averages. */
	samples: number;
	/** Average output tokens/sec over the total request duration. */
	tps: number;
	/** Average time-to-first-token in milliseconds; null when no sample reported one. */
	ttftMs: number | null;
}

/**
 * Decay threshold for model_perf running sums: once a model accumulates this
 * many samples, each new sample first halves every aggregate, turning the
 * plain average into a recency-weighted one (provider speeds drift over time).
 */
const MODEL_PERF_DECAY_AT = 256;
/** meta-table marker set once historical stats.db rows have been imported into model_perf. */
const MODEL_PERF_BACKFILL_KEY = "model_perf_backfill";
/** Batch window for deferred model_perf writes; matches prompt-history's drain cadence. */
const MODEL_PERF_FLUSH_DELAY_MS = 100;
/** Backfill ignores stats.db history older than this; decay makes stale provider speeds worthless anyway. */
const MODEL_PERF_BACKFILL_MAX_AGE_MS = 90 * 86_400_000;
/** Rows fetched per synchronous backfill chunk — keeps per-chunk event-loop blocking under ~20ms even on cold I/O. */
const MODEL_PERF_BACKFILL_CHUNK = 2048;
/** Hard ceiling on rows scanned per backfill run, whatever the age cutoff admits — bounds total CPU on very high-volume databases (models only seen earlier than the newest N measurable rows get no backfill). */
const MODEL_PERF_BACKFILL_MAX_ROWS = 250_000;

/**
 * Validates one request timing and shapes it for the model_perf upsert.
 * Returns null for unmeasurable samples (no tokens, no duration). Out-of-range
 * TTFT (>= duration) is bogus latency data; the sample still measures throughput.
 */
function normalizeModelPerfSample(modelKey: string, sample: ModelPerfSample): ModelPerfInsert | null {
	const { outputTokens, durationMs } = sample;
	if (!Number.isFinite(outputTokens) || outputTokens <= 0) return null;
	if (!Number.isFinite(durationMs) || durationMs <= 0) return null;
	const ttftMs =
		sample.ttftMs !== undefined && Number.isFinite(sample.ttftMs) && sample.ttftMs > 0 && sample.ttftMs < durationMs
			? sample.ttftMs
			: undefined;
	return { modelKey, outputTokens, durationMs, ttftSamples: ttftMs !== undefined ? 1 : 0, ttftMs: ttftMs ?? 0 };
}

/** Current agent.db schema version; bump when schema changes require migration. */
export const SCHEMA_VERSION = 10;
const SQLITE_NOW_EPOCH = "CAST(strftime('%s','now') AS INTEGER)";

/** Singleton instances per database path */
const instances = new Map<string, AgentStorage>();
let cancelExitCleanup: (() => void) | undefined;

/**
 * Unified SQLite storage for agent settings, model usage, and auth credentials.
 * Delegates auth credential operations to AuthCredentialStore from @oh-my-pi/pi-ai.
 * Uses singleton pattern per database path; access via AgentStorage.open().
 */
export class AgentStorage {
	#db: Database;
	#authStore: AuthCredentialStore;

	#listSettingsStmt: Statement;
	#upsertModelUsageStmt: Statement;
	#listModelUsageStmt: Statement;
	#upsertModelPerfStmt: Statement;
	#listModelPerfStmt: Statement;
	#upsertCommandUsageStmt: Statement;
	#listCommandUsageStmt: Statement;
	#modelUsageCache: string[] | null = null;
	/** Only the real user db auto-imports stats.db history; custom paths (tests, embedding) opt in explicitly. */
	#autoPerfBackfill: boolean;
	/** One backfill *check* per process; the persistent gate is the meta marker. */
	#perfBackfillChecked = false;
	/** Coalesces per-turn perf samples into one deferred transaction off the turn's hot path. */
	#perfDrain = new AsyncDrain<ModelPerfInsert>(MODEL_PERF_FLUSH_DELAY_MS);
	#closing = false;
	#requirements!: RequirementsStore;
	#requirementsReady?: Promise<void>;

	private constructor(dbPath: string) {
		this.#autoPerfBackfill = dbPath === getAgentDbPath();
		this.#ensureDir(dbPath);
		try {
			this.#db = new Database(dbPath);
		} catch (err) {
			const dir = path.dirname(dbPath);
			const dirExists = fs.existsSync(dir);
			const errMsg = err instanceof Error ? err.message : String(err);
			throw new Error(
				`Failed to open agent database at '${dbPath}': ${errMsg}\n` +
					`Directory '${dir}' exists: ${dirExists}\n` +
					`Ensure the directory is writable and not corrupted.`,
			);
		}

		this.#initializeSchema();
		this.#requirements = new RequirementsStore(this.#requirementsBackend());
		this.#hardenPermissions(dbPath);

		// Create AuthCredentialStore with our open database
		this.#authStore = new SqliteAuthCredentialStore(this.#db);

		this.#listSettingsStmt = this.#db.prepare("SELECT key, value FROM settings");
		this.#upsertModelUsageStmt = this.#db.prepare(
			`INSERT INTO model_usage (model_key, last_used_at) VALUES (?, ${SQLITE_NOW_EPOCH}) ON CONFLICT(model_key) DO UPDATE SET last_used_at = ${SQLITE_NOW_EPOCH}`,
		);
		this.#listModelUsageStmt = this.#db.prepare(
			"SELECT model_key, last_used_at FROM model_usage ORDER BY last_used_at DESC",
		);
		// Recency-weighted upsert: past MODEL_PERF_DECAY_AT samples, every new
		// sample first halves the aggregates so old measurements fade out.
		this.#upsertModelPerfStmt = this.#db.prepare(
			`INSERT INTO model_perf (model_key, samples, output_tokens, gen_ms, ttft_samples, ttft_ms, updated_at)
VALUES (?1, 1, ?2, ?3, ?4, ?5, ${SQLITE_NOW_EPOCH})
ON CONFLICT(model_key) DO UPDATE SET
	samples = (CASE WHEN model_perf.samples >= ${MODEL_PERF_DECAY_AT} THEN model_perf.samples / 2 ELSE model_perf.samples END) + 1,
	output_tokens = (CASE WHEN model_perf.samples >= ${MODEL_PERF_DECAY_AT} THEN model_perf.output_tokens * 0.5 ELSE model_perf.output_tokens END) + excluded.output_tokens,
	gen_ms = (CASE WHEN model_perf.samples >= ${MODEL_PERF_DECAY_AT} THEN model_perf.gen_ms * 0.5 ELSE model_perf.gen_ms END) + excluded.gen_ms,
	ttft_samples = (CASE WHEN model_perf.samples >= ${MODEL_PERF_DECAY_AT} THEN model_perf.ttft_samples * 0.5 ELSE model_perf.ttft_samples END) + excluded.ttft_samples,
	ttft_ms = (CASE WHEN model_perf.samples >= ${MODEL_PERF_DECAY_AT} THEN model_perf.ttft_ms * 0.5 ELSE model_perf.ttft_ms END) + excluded.ttft_ms,
	updated_at = ${SQLITE_NOW_EPOCH}`,
		);
		this.#listModelPerfStmt = this.#db.prepare(
			"SELECT model_key, samples, output_tokens, gen_ms, ttft_samples, ttft_ms FROM model_perf",
		);
		this.#upsertCommandUsageStmt = this.#db.prepare(
			`INSERT INTO command_usage (name, count, last_used_at) VALUES (?, 1, ${SQLITE_NOW_EPOCH})
ON CONFLICT(name) DO UPDATE SET count = command_usage.count + 1, last_used_at = ${SQLITE_NOW_EPOCH}`,
		);
		this.#listCommandUsageStmt = this.#db.prepare("SELECT name, count FROM command_usage");
	}

	/**
	 * Creates tables if missing and migrates legacy settings.
	 * AuthCredentialStore handles auth_credentials and cache tables.
	 */
	#initializeSchema(): void {
		// Install the busy handler BEFORE any lock-taking statement (incl.
		// `PRAGMA journal_mode=WAL`, which acquires an exclusive lock during WAL
		// recovery). Without this, concurrent omp startups can crash here with
		// `SQLITE_BUSY` / `SQLITE_BUSY_RECOVERY`. See issue #2421. Headless
		// hosts bound the wait so lock contention cannot freeze the protocol
		// loop for the full interactive timeout.
		this.#db.run(`PRAGMA busy_timeout = ${getDbBusyTimeoutMs()}`);
		this.#db.run(`
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS model_usage (
	model_key TEXT PRIMARY KEY,
	last_used_at INTEGER NOT NULL DEFAULT (${SQLITE_NOW_EPOCH})
);

CREATE TABLE IF NOT EXISTS model_perf (
	model_key TEXT PRIMARY KEY,
	samples REAL NOT NULL DEFAULT 0,
	output_tokens REAL NOT NULL DEFAULT 0,
	gen_ms REAL NOT NULL DEFAULT 0,
	ttft_samples REAL NOT NULL DEFAULT 0,
	ttft_ms REAL NOT NULL DEFAULT 0,
	updated_at INTEGER NOT NULL DEFAULT (${SQLITE_NOW_EPOCH})
);

CREATE TABLE IF NOT EXISTS command_usage (
	name TEXT PRIMARY KEY,
	count INTEGER NOT NULL DEFAULT 0,
	last_used_at INTEGER NOT NULL DEFAULT (${SQLITE_NOW_EPOCH})
);

CREATE TABLE IF NOT EXISTS meta (
	key TEXT PRIMARY KEY,
	value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
`);

		const settingsInfo = this.#db.prepare("PRAGMA table_info(settings)").all() as Array<{ name?: string }>;
		const hasSettingsTable = settingsInfo.length > 0;
		const hasKey = settingsInfo.some(column => column.name === "key");
		const hasValue = settingsInfo.some(column => column.name === "value");

		if (!hasSettingsTable) {
			this.#db.run(`
CREATE TABLE settings (
	key TEXT PRIMARY KEY,
	value TEXT NOT NULL,
	updated_at INTEGER NOT NULL DEFAULT (${SQLITE_NOW_EPOCH})
);
`);
		} else if (!hasKey || !hasValue) {
			// Migrate v1 schema: single JSON blob in `data` column → per-key rows
			let legacySettings: Record<string, unknown> | null = null;
			const row = this.#db.prepare("SELECT data FROM settings WHERE id = 1").get() as { data?: string } | undefined;
			if (row?.data) {
				try {
					const parsed = JSON.parse(row.data);
					if (isRecord(parsed)) {
						legacySettings = parsed;
					} else {
						logger.warn("AgentStorage legacy settings invalid shape");
					}
				} catch (error) {
					logger.warn("AgentStorage failed to parse legacy settings", { error: String(error) });
				}
			}

			const migrate = this.#db.transaction((settings: Record<string, unknown> | null) => {
				this.#db.run("DROP TABLE settings");
				this.#db.run(`
CREATE TABLE settings (
	key TEXT PRIMARY KEY,
	value TEXT NOT NULL,
	updated_at INTEGER NOT NULL DEFAULT (${SQLITE_NOW_EPOCH})
);
`);
				if (settings) {
					const insert = this.#db.prepare(
						`INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ${SQLITE_NOW_EPOCH})`,
					);
					for (const [key, value] of Object.entries(settings)) {
						if (value === undefined) continue;
						const serialized = JSON.stringify(value);
						if (serialized === undefined) continue;
						insert.run(key, serialized);
					}
				}
			});

			migrate(legacySettings);
		}

		const versionRow = this.#db.prepare("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").get() as
			| { version?: number }
			| undefined;
		const schemaVersion = typeof versionRow?.version === "number" ? versionRow.version : 0;
		if (versionRow?.version !== undefined && versionRow.version !== SCHEMA_VERSION) {
			logger.warn("AgentStorage schema version mismatch", {
				current: versionRow.version,
				expected: SCHEMA_VERSION,
			});
		}
		if (schemaVersion < SCHEMA_VERSION) {
			this.#migrateSchema(schemaVersion);
		}
		this.#db.prepare("INSERT OR REPLACE INTO schema_version(version) VALUES (?)").run(SCHEMA_VERSION);
	}

	#migrateSchema(fromVersion: number): void {
		if (fromVersion < 4) {
			// v3 → v4: Add disabled column to auth_credentials (handled by AuthCredentialStore)
			// Nothing to do here - AuthCredentialStore will handle this migration
		}
		if (fromVersion < 5) {
			this.#migrateSchemaV4ToV5();
		}
		if (fromVersion < 6) {
			// v5 → v6: TPS switched from the post-TTFT decode window to total
			// request duration (hidden reasoning made decode-window rates bogus).
			// Purge the old aggregates and re-arm the stats.db backfill so
			// history is re-imported through the corrected fold.
			this.#db.run("DELETE FROM model_perf");
			this.#db.prepare("DELETE FROM meta WHERE key = ?").run(MODEL_PERF_BACKFILL_KEY);
		}
		if (fromVersion < 7) {
			this.#db.run(`
CREATE TABLE IF NOT EXISTS requirements_sources (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requirements_batches (id TEXT PRIMARY KEY, source_key TEXT NOT NULL, source_integrity TEXT NOT NULL, extraction_version TEXT NOT NULL, review_revision TEXT NOT NULL, body TEXT NOT NULL, UNIQUE(source_key, source_integrity, extraction_version, review_revision));
CREATE TABLE IF NOT EXISTS requirements_revisions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requirements_state (id INTEGER PRIMARY KEY CHECK(id = 1), body TEXT NOT NULL);
`);
		}
		if (fromVersion < 8) {
			this.#db.run(`
CREATE INDEX IF NOT EXISTS requirements_source_coverage ON requirements_sources(COALESCE(json_extract(body, '$.referenceOnly'), 0), json_extract(body, '$.state'));
CREATE INDEX IF NOT EXISTS requirements_pending_input ON requirements_sources(id) WHERE COALESCE(json_extract(body, '$.referenceOnly'), 0) = 0 AND json_extract(body, '$.state') != 'complete' AND json_extract(body, '$.origin.kind') IN ('human', 'sdk');
`);
		}

		if (fromVersion < 9) {
			this.#db.run(`
ALTER TABLE requirements_state RENAME TO requirements_state_legacy;
CREATE TABLE requirements_state (id INTEGER PRIMARY KEY CHECK(id = 1), publication_revision INTEGER NOT NULL, generation INTEGER NOT NULL);
INSERT INTO requirements_state VALUES(1, 0, 0);
UPDATE requirements_state SET publication_revision = COALESCE((SELECT json_extract(body, '$.publicationRevision') FROM requirements_state_legacy WHERE id = 1), 0), generation = COALESCE((SELECT json_extract(body, '$.generation') FROM requirements_state_legacy WHERE id = 1), 0);
CREATE TABLE requirements_owners (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE requirements_restore_receipts (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE requirements_restore_reviews (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE requirements_tombstones (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE requirements_suspensions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
INSERT INTO requirements_owners SELECT entry.key, entry.value FROM requirements_state_legacy, json_each(body, '$.owners') entry;
INSERT INTO requirements_restore_receipts SELECT json_extract(entry.value, '$.id'), entry.value FROM requirements_state_legacy, json_each(body, '$.restoreReceipts') entry;
INSERT INTO requirements_restore_reviews SELECT entry.key, entry.value FROM requirements_state_legacy, json_each(body, '$.restoreReviews') entry;
INSERT INTO requirements_tombstones SELECT json_array(json_extract(entry.value, '$.scope.kind'), CASE WHEN json_extract(entry.value, '$.scope.kind') = 'project' THEN json_extract(entry.value, '$.scope.projectId') END, CASE WHEN json_extract(entry.value, '$.scope.kind') IN ('session', 'task') THEN json_extract(entry.value, '$.scope.sessionId') END, CASE WHEN json_extract(entry.value, '$.scope.kind') IN ('session', 'task') THEN json_extract(entry.value, '$.scope.epoch') END), entry.value FROM requirements_state_legacy, json_each(body, '$.tombstones') entry;
DROP TABLE requirements_state_legacy;
CREATE TABLE requirements_heads (id TEXT PRIMARY KEY, requirement_id TEXT NOT NULL, scope_key TEXT NOT NULL, source_key TEXT NOT NULL);
CREATE INDEX requirements_heads_requirement ON requirements_heads(requirement_id);
CREATE INDEX requirements_heads_scope ON requirements_heads(scope_key);
CREATE TABLE requirements_dependencies (kind TEXT NOT NULL, dependent_id TEXT NOT NULL, source_key TEXT NOT NULL, PRIMARY KEY(kind, dependent_id, source_key));
CREATE INDEX requirements_dependencies_source ON requirements_dependencies(source_key, kind, dependent_id);
CREATE TABLE requirements_source_locators (source_key TEXT NOT NULL, journal_path TEXT NOT NULL, PRIMARY KEY(source_key, journal_path));
CREATE INDEX requirements_locator_journal ON requirements_source_locators(journal_path, source_key);
CREATE INDEX requirements_batch_source ON requirements_batches(source_key);
CREATE INDEX requirements_latest_batch ON requirements_batches(source_key, source_integrity);
CREATE INDEX requirements_revision_batch ON requirements_revisions(json_extract(body, '$.batchId'));
CREATE INDEX requirements_pending_owner ON requirements_sources(json_extract(body, '$.ownerSessionId'), json_extract(body, '$.epoch')) WHERE COALESCE(json_extract(body, '$.referenceOnly'), 0) = 0 AND json_extract(body, '$.state') != 'complete' AND json_extract(body, '$.origin.kind') IN ('human', 'sdk');
CREATE TABLE requirements_coverage (reference_only INTEGER NOT NULL, state TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(reference_only, state));
INSERT INTO requirements_coverage SELECT COALESCE(json_extract(body, '$.referenceOnly'), 0), json_extract(body, '$.state'), count(*) FROM requirements_sources GROUP BY 1, 2;
CREATE TRIGGER requirements_source_insert AFTER INSERT ON requirements_sources BEGIN
 INSERT INTO requirements_coverage VALUES(COALESCE(json_extract(NEW.body, '$.referenceOnly'), 0), json_extract(NEW.body, '$.state'), 1) ON CONFLICT(reference_only, state) DO UPDATE SET count = count + 1;
END;
CREATE TRIGGER requirements_source_delete AFTER DELETE ON requirements_sources BEGIN
 UPDATE requirements_coverage SET count = count - 1 WHERE reference_only = COALESCE(json_extract(OLD.body, '$.referenceOnly'), 0) AND state = json_extract(OLD.body, '$.state');
END;
CREATE TRIGGER requirements_source_update AFTER UPDATE OF body ON requirements_sources WHEN COALESCE(json_extract(OLD.body, '$.referenceOnly'), 0) != COALESCE(json_extract(NEW.body, '$.referenceOnly'), 0) OR json_extract(OLD.body, '$.state') != json_extract(NEW.body, '$.state') BEGIN
 UPDATE requirements_coverage SET count = count - 1 WHERE reference_only = COALESCE(json_extract(OLD.body, '$.referenceOnly'), 0) AND state = json_extract(OLD.body, '$.state');
 INSERT INTO requirements_coverage VALUES(COALESCE(json_extract(NEW.body, '$.referenceOnly'), 0), json_extract(NEW.body, '$.state'), 1) ON CONFLICT(reference_only, state) DO UPDATE SET count = count + 1;
END;
INSERT INTO meta(key, value) VALUES('requirements_normalizing', '0') ON CONFLICT(key) DO NOTHING;
`);
		}
		if (fromVersion < 10) {
			this.#db.run(`
CREATE TABLE requirements_relation_edges (revision_id TEXT NOT NULL, position INTEGER NOT NULL, successor_key TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(revision_id, position));
CREATE INDEX requirements_relations_successor ON requirements_relation_edges(successor_key);
DROP INDEX requirements_pending_input;
DROP INDEX requirements_pending_owner;
CREATE INDEX requirements_pending_input ON requirements_sources(id) WHERE COALESCE(json_extract(body, '$.referenceOnly'), 0) = 0 AND json_extract(body, '$.state') != 'complete' AND json_extract(body, '$.origin.kind') = 'human';
CREATE INDEX requirements_pending_owner ON requirements_sources(json_extract(body, '$.ownerSessionId'), json_extract(body, '$.epoch')) WHERE COALESCE(json_extract(body, '$.referenceOnly'), 0) = 0 AND json_extract(body, '$.state') != 'complete' AND json_extract(body, '$.origin.kind') = 'human';
CREATE TABLE requirements_legacy_sdk_sources (id TEXT PRIMARY KEY);
INSERT INTO requirements_legacy_sdk_sources SELECT id FROM requirements_sources WHERE json_extract(body, '$.origin.kind') = 'sdk';
UPDATE requirements_sources SET body = json_set(body, '$.origin.kind', 'unknown', '$.state', 'unsupported', '$.integrityAvailable', json('false'), '$.reason', 'Legacy SDK provenance requires operator adoption') WHERE id IN (SELECT id FROM requirements_legacy_sdk_sources);
`);
			if (fromVersion >= 9) this.#db.query("INSERT INTO meta(key, value) VALUES('requirements_relations_normalizing', '0') ON CONFLICT(key) DO NOTHING").run();
		}
	}

	#migrateSchemaV4ToV5(): void {
		const migrate = this.#db.transaction(() => {
			this.#db.run("ALTER TABLE settings RENAME TO settings_legacy");
			this.#db.run(`
CREATE TABLE settings (
	key TEXT PRIMARY KEY,
	value TEXT NOT NULL,
	updated_at INTEGER NOT NULL DEFAULT (${SQLITE_NOW_EPOCH})
);
`);
			this.#db.run(`
INSERT INTO settings (key, value, updated_at)
SELECT key, value, updated_at
FROM settings_legacy
`);
			this.#db.run("DROP TABLE settings_legacy");

			this.#db.run("ALTER TABLE model_usage RENAME TO model_usage_legacy");
			this.#db.run(`
CREATE TABLE model_usage (
	model_key TEXT PRIMARY KEY,
	last_used_at INTEGER NOT NULL DEFAULT (${SQLITE_NOW_EPOCH})
);
`);
			this.#db.run(`
INSERT INTO model_usage (model_key, last_used_at)
SELECT model_key, last_used_at
FROM model_usage_legacy
`);
			this.#db.run("DROP TABLE model_usage_legacy");
		});
		migrate();
	}

	/**
	 * Returns singleton instance for the given database path, creating if needed.
	 * Retries on the `SQLITE_BUSY` family (including `SQLITE_BUSY_RECOVERY`) with
	 * exponential backoff. See issue #2421.
	 * @param dbPath - Path to the SQLite database file (defaults to config path)
	 * @returns AgentStorage instance for the given path
	 */
	static async open(dbPath: string = getAgentDbPath()): Promise<AgentStorage> {
		const existing = instances.get(dbPath);
		if (existing) { await existing.#requirementsReady; return existing; }

		const maxRetries = 4;
		const baseDelayMs = 100;
		let lastError: Error | undefined;

		for (let attempt = 0; attempt < maxRetries; attempt++) {
			try {
				const storage = new AgentStorage(dbPath);
				// Exit-only: a keep-alive cleanup leaves the open handle valid for the
				// continuing process (Settings, MCP cache, callers hold it); the real
				// exit closes. Register before publishing so a real-exit-in-progress
				// late registration sees an empty map.
				cancelExitCleanup ??= postmortem.register("agent-storage", () => AgentStorage.close(), { exitOnly: true });
				instances.set(dbPath, storage);
				storage.#requirementsReady = storage.#normalizeRequirements().then(() => storage.#normalizeRequirementsRelations()).then(() => storage.#normalizeLegacyRequirementsSdk());
				try { await storage.#requirementsReady; } catch (error) { instances.delete(dbPath); storage.#db.close(); throw error; }
				return storage;
			} catch (err) {
				if (!isSqliteBusyError(err)) {
					throw err;
				}
				lastError = err instanceof Error ? err : new Error(String(err));
				if (attempt < maxRetries - 1) {
					await Bun.sleep(baseDelayMs * 2 ** attempt);
				}
			}
		}

		throw new Error(
			`Failed to open agent database at '${dbPath}' after ${maxRetries} attempts: ${lastError?.message}`,
			{ cause: lastError },
		);
	}

	/** Flushes deferred writes, closes every process-wide database, and permits reopening them. */
	static close(): void {
		for (const storage of instances.values()) storage.#close();
		instances.clear();
		cancelExitCleanup?.();
		cancelExitCleanup = undefined;
	}

	#close(): void {
		this.#closing = true;
		// Model-performance batches are synchronous once invoked, so this
		// persists them before finalizing their statements during process exit.
		void this.#perfDrain.flush();
		checkpointWal(this.#db);
		this.#listSettingsStmt.finalize();
		this.#upsertModelUsageStmt.finalize();
		this.#listModelUsageStmt.finalize();
		this.#upsertModelPerfStmt.finalize();
		this.#listModelPerfStmt.finalize();
		this.#upsertCommandUsageStmt.finalize();
		this.#listCommandUsageStmt.finalize();
		// SqliteAuthCredentialStore.close() finalizes its own statements and
		// closes the shared #db handle — must run after our statements finalize.
		this.#authStore.close();
	}

	/**
	 * Reads legacy settings persisted in the agent.db `settings` table.
	 * The canonical settings store is `config.yml`; this accessor only
	 * exists so the config loader can migrate values from older installs.
	 * @returns Settings object, or null if no settings are stored
	 */
	getSettings(): Settings | null {
		const rows = (this.#listSettingsStmt.all() as SettingsRow[]) ?? [];
		if (rows.length === 0) return null;
		const settings: Record<string, unknown> = {};
		for (const row of rows) {
			try {
				settings[row.key] = JSON.parse(row.value) as unknown;
			} catch (error) {
				logger.warn("AgentStorage failed to parse setting", {
					key: row.key,
					error: String(error),
				});
			}
		}
		return settings as Settings;
	}

	/**
	 * Records model usage, updating the last-used timestamp.
	 * @param modelKey - Model key in "provider/modelId" format
	 */
	recordModelUsage(modelKey: string): void {
		try {
			this.#upsertModelUsageStmt.run(modelKey);
			this.#modelUsageCache = null;
		} catch (error) {
			logger.warn("AgentStorage failed to record model usage", { modelKey, error: String(error) });
		}
	}

	/**
	 * Gets model keys ordered by most recently used.
	 * Results are cached until recordModelUsage is called.
	 * @returns Array of model keys ("provider/modelId") in MRU order
	 */
	getModelUsageOrder(): string[] {
		if (this.#modelUsageCache) {
			return this.#modelUsageCache;
		}
		try {
			const rows = this.#listModelUsageStmt.all() as ModelUsageRow[];
			this.#modelUsageCache = rows.map(row => row.model_key);
			return this.#modelUsageCache;
		} catch (error) {
			logger.warn("AgentStorage failed to get model usage order", { error: String(error) });
			return [];
		}
	}
	/**
	 * Records one slash-command invocation, bumping its usage count and
	 * last-used timestamp. Frequency-ranked autocomplete reads these counts.
	 * @param name - Canonical command name (e.g. "model", "skill:review")
	 */
	recordCommandUsage(name: string): void {
		try {
			this.#upsertCommandUsageStmt.run(name);
		} catch (error) {
			logger.warn("AgentStorage failed to record command usage", { name, error: String(error) });
		}
	}

	/**
	 * Gets slash-command usage counts keyed by canonical command name.
	 * @returns Command name → invocation count
	 */
	listCommandUsage(): Record<string, number> {
		try {
			const rows = this.#listCommandUsageStmt.all() as Array<{ name: string; count: number }>;
			const counts: Record<string, number> = {};
			for (const row of rows) counts[row.name] = row.count;
			return counts;
		} catch (error) {
			logger.warn("AgentStorage failed to list command usage", { error: String(error) });
			return {};
		}
	}

	/**
	 * Folds one completed request's timing into the model's perf aggregates.
	 * TPS is measured over the total request duration — not the post-TTFT
	 * decode window, which undercounts generation time (and so inflates the
	 * rate) when reasoning tokens are generated before the first visible
	 * token. Invalid samples (no tokens, no duration) are dropped.
	 *
	 * Deferred like prompt history: samples are batched and written in one
	 * transaction after {@link MODEL_PERF_FLUSH_DELAY_MS}, keeping SQLite off
	 * the turn-completion hot path. Fire-and-forget safe — flush failures are
	 * logged, never thrown; await the returned promise only to observe the flush.
	 * @param modelKey - Model key in "provider/modelId" format
	 */
	recordModelPerf(modelKey: string, sample: ModelPerfSample): Promise<void> {
		const row = normalizeModelPerfSample(modelKey, sample);
		if (!row) return Promise.resolve();
		return this.#perfDrain.push(row, rows => this.#flushModelPerf(rows));
	}

	#flushModelPerf(rows: ModelPerfInsert[]): void {
		// A close-triggered flush must persist only the queued live batch. Starting
		// the async stats import here could commit aggregates without its marker.
		if (!this.#closing) this.#kickModelPerfBackfill();
		try {
			this.#db.transaction((batch: ModelPerfInsert[]) => {
				for (const row of batch) this.#foldModelPerf(row);
			})(rows);
		} catch (error) {
			logger.warn("AgentStorage failed to record model perf", { error: String(error) });
		}
	}

	#foldModelPerf(row: ModelPerfInsert): void {
		this.#upsertModelPerfStmt.run(row.modelKey, row.outputTokens, row.durationMs, row.ttftSamples, row.ttftMs);
	}

	/**
	 * Returns recency-weighted TPS/TTFT averages for every model with recorded
	 * requests, keyed by "provider/modelId". Read by the /models browser.
	 * Also kicks the one-time background stats.db import; until it completes,
	 * models without live samples are simply absent.
	 */
	getModelPerf(): Map<string, ModelPerfStats> {
		this.#kickModelPerfBackfill();
		const stats = new Map<string, ModelPerfStats>();
		try {
			for (const row of this.#listModelPerfStmt.all() as ModelPerfRow[]) {
				if (row.gen_ms <= 0 || row.output_tokens <= 0) continue;
				stats.set(row.model_key, {
					samples: row.samples,
					tps: (row.output_tokens * 1000) / row.gen_ms,
					ttftMs: row.ttft_samples > 0 ? row.ttft_ms / row.ttft_samples : null,
				});
			}
		} catch (error) {
			logger.warn("AgentStorage failed to read model perf", { error: String(error) });
		}
		return stats;
	}

	/**
	 * One-time, non-blocking import of historical request timings from the
	 * `omp stats` database (`~/.omp/stats.db`) into model_perf. Fire-and-forget:
	 * the walk runs in bounded chunks with event-loop yields between them
	 * (bun:sqlite is synchronous — an unbounded scan here froze the TUI for
	 * ~30s on multi-million-row stats databases), and the persistent meta
	 * marker is only set on success so a crash or error retries next process.
	 * A missing stats.db leaves the marker unset so a later `omp stats` run
	 * still gets imported. No-op for non-default db paths.
	 */
	#kickModelPerfBackfill(): void {
		if (!this.#autoPerfBackfill || this.#perfBackfillChecked) return;
		this.#perfBackfillChecked = true;
		try {
			const marker = this.#db.prepare("SELECT value FROM meta WHERE key = ?").get(MODEL_PERF_BACKFILL_KEY);
			if (marker) return;
			const statsDbPath = getStatsDbPath();
			if (!fs.existsSync(statsDbPath)) return;
			void this.backfillModelPerfFromStats(statsDbPath)
				.then(imported => {
					this.#db
						.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)")
						.run(MODEL_PERF_BACKFILL_KEY, "complete");
					logger.info("AgentStorage imported model perf history from stats.db", { imported });
				})
				.catch(error => {
					logger.warn("AgentStorage model perf backfill failed", { error: String(error) });
				});
		} catch (error) {
			logger.warn("AgentStorage model perf backfill failed", { error: String(error) });
		}
	}

	/**
	 * Imports recent measurable request rows from an `omp stats` database
	 * (`messages` table) into the model_perf aggregates. Walks newest-first
	 * over the timestamp index in {@link MODEL_PERF_BACKFILL_CHUNK}-row chunks,
	 * yielding to the event loop between chunks, and keeps at most
	 * {@link MODEL_PERF_DECAY_AT} rows per model within the
	 * {@link MODEL_PERF_BACKFILL_MAX_AGE_MS} window — beyond either bound the
	 * live decay would erase the contribution anyway. Errored turns are
	 * excluded; aborted turns with reported usage count, matching live capture.
	 * Sums land in one additive transaction at the end, so concurrent live
	 * samples merge correctly regardless of order.
	 * @param statsDbPath - Path to a stats.db file; opened read-only
	 * @returns Number of rows folded in
	 * @throws When the stats db cannot be opened or queried
	 */
	async backfillModelPerfFromStats(statsDbPath: string): Promise<number> {
		const statsDb = new Database(statsDbPath, { readonly: true });
		try {
			statsDb.run(`PRAGMA busy_timeout = ${getDbBusyTimeoutMs()}`);
			const select = statsDb.prepare(
				`SELECT rowid, timestamp, provider, model, output_tokens, duration, ttft
FROM messages
WHERE (timestamp < ?1 OR (timestamp = ?1 AND rowid < ?2))
	AND timestamp >= ?3
	AND duration > 0 AND output_tokens > 0 AND stop_reason != 'error'
ORDER BY timestamp DESC, rowid DESC
LIMIT ?4`,
			);
			const cutoff = Date.now() - MODEL_PERF_BACKFILL_MAX_AGE_MS;
			const sums = new Map<string, PerfAccum>();
			let cursorTimestamp = Number.MAX_SAFE_INTEGER;
			let cursorRowid = Number.MAX_SAFE_INTEGER;
			let scanned = 0;
			let imported = 0;
			while (scanned < MODEL_PERF_BACKFILL_MAX_ROWS) {
				const chunk = Math.min(MODEL_PERF_BACKFILL_CHUNK, MODEL_PERF_BACKFILL_MAX_ROWS - scanned);
				const rows = select.all(cursorTimestamp, cursorRowid, cutoff, chunk) as StatsMessageRow[];
				if (rows.length === 0) break;
				scanned += rows.length;
				const last = rows[rows.length - 1];
				cursorTimestamp = last.timestamp;
				cursorRowid = last.rowid;
				for (const row of rows) {
					const key = `${row.provider}/${row.model}`;
					let accum = sums.get(key);
					if (accum && accum.samples >= MODEL_PERF_DECAY_AT) continue;
					const normalized = normalizeModelPerfSample(key, {
						outputTokens: row.output_tokens,
						durationMs: row.duration,
						ttftMs: row.ttft ?? undefined,
					});
					if (!normalized) continue;
					if (!accum) {
						accum = { samples: 0, outputTokens: 0, genMs: 0, ttftSamples: 0, ttftMs: 0 };
						sums.set(key, accum);
					}
					accum.samples += 1;
					accum.outputTokens += normalized.outputTokens;
					accum.genMs += normalized.durationMs;
					accum.ttftSamples += normalized.ttftSamples;
					accum.ttftMs += normalized.ttftMs;
					imported++;
				}
				if (rows.length < chunk) break;
				// Yield so a chunked walk never freezes the TUI (bun:sqlite is sync).
				await Bun.sleep(0);
			}
			if (sums.size > 0) {
				const upsert = this.#db.prepare(
					`INSERT INTO model_perf (model_key, samples, output_tokens, gen_ms, ttft_samples, ttft_ms, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ${SQLITE_NOW_EPOCH})
ON CONFLICT(model_key) DO UPDATE SET
	samples = model_perf.samples + excluded.samples,
	output_tokens = model_perf.output_tokens + excluded.output_tokens,
	gen_ms = model_perf.gen_ms + excluded.gen_ms,
	ttft_samples = model_perf.ttft_samples + excluded.ttft_samples,
	ttft_ms = model_perf.ttft_ms + excluded.ttft_ms,
	updated_at = ${SQLITE_NOW_EPOCH}`,
				);
				this.#db.transaction(() => {
					for (const [key, accum] of sums) {
						upsert.run(key, accum.samples, accum.outputTokens, accum.genMs, accum.ttftSamples, accum.ttftMs);
					}
				})();
			}
			return imported;
		} finally {
			statsDb.close();
		}
	}

	/**
	 * Checks if any auth credentials exist in storage.
	 * @returns True if at least one credential is stored
	 */
	hasAuthCredentials(): boolean {
		return this.#authStore.listAuthCredentials().length > 0;
	}

	/**
	 * Returns the underlying {@link AuthCredentialStore} so callers that need
	 * the lower-level pi-ai abstraction (e.g. `findAnthropicAuth(store)`) can
	 * reuse this storage's open database connection instead of opening their
	 * own.
	 */
	get authStore(): AuthCredentialStore {
		return this.#authStore;
	}

	/**
	 * Lists auth credentials, optionally filtered by provider.
	 * Only returns active (non-disabled) credentials by default.
	 * @param provider - Optional provider name to filter by
	 * @param includeDisabled - If true, includes disabled credentials
	 * @returns Array of stored credentials with their database IDs
	 */
	listAuthCredentials(provider?: string, includeDisabled = false): StoredAuthCredential[] {
		const credentials = this.#authStore.listAuthCredentials(provider);
		if (!includeDisabled) return credentials;

		const stmt = this.#db.prepare(
			provider
				? "SELECT id, provider, credential_type, data, disabled_cause FROM auth_credentials WHERE provider = ? ORDER BY id ASC"
				: "SELECT id, provider, credential_type, data, disabled_cause FROM auth_credentials ORDER BY id ASC",
		);
		const rows = (provider ? stmt.all(provider) : stmt.all()) as Array<{
			id: number;
			provider: string;
			credential_type: string;
			data: string;
			disabled_cause: string | null;
		}>;

		const results: StoredAuthCredential[] = [];
		for (const row of rows) {
			try {
				const parsed = JSON.parse(row.data);
				if (!parsed || typeof parsed !== "object") continue;

				let credential: AuthCredential;
				if (row.credential_type === "api_key" && typeof (parsed as { key?: unknown }).key === "string") {
					credential = { type: "api_key", key: (parsed as { key: string }).key };
				} else if (row.credential_type === "oauth") {
					credential = { type: "oauth", ...(parsed as Record<string, unknown>) } as AuthCredential;
				} else {
					continue;
				}

				results.push({ id: row.id, provider: row.provider, credential, disabledCause: row.disabled_cause });
			} catch {}
		}
		return results;
	}

	/**
	 * Atomically replaces all credentials for a provider.
	 * Useful for OAuth token refresh where old tokens should be discarded.
	 * @param provider - Provider name (e.g., "anthropic", "openai")
	 * @param credentials - New credentials to store
	 * @returns Array of newly stored credentials with their database IDs
	 */
	replaceAuthCredentialsForProvider(provider: string, credentials: AuthCredential[]): StoredAuthCredential[] {
		return this.#authStore.replaceAuthCredentialsForProvider(provider, credentials);
	}

	/**
	 * Updates an existing auth credential by ID.
	 * @param id - Database row ID of the credential to update
	 * @param credential - New credential data
	 */
	updateAuthCredential(id: number, credential: AuthCredential): void {
		this.#authStore.updateAuthCredential(id, credential);
	}

	/**
	 * Disables an auth credential by ID with a persisted cause.
	 * @param id - Database row ID of the credential to disable
	 * @param disabledCause - Human-readable cause stored with the disabled row
	 */
	deleteAuthCredential(id: number, disabledCause: string): void {
		this.#authStore.deleteAuthCredential(id, disabledCause);
	}

	/**
	 * Disables all auth credentials for a provider with a persisted cause.
	 * @param provider - Provider name whose credentials should be disabled
	 * @param disabledCause - Human-readable cause stored with the disabled rows
	 */
	deleteAuthCredentialsForProvider(provider: string, disabledCause: string): void {
		this.#authStore.deleteAuthCredentialsForProvider(provider, disabledCause);
	}

	/**
	 * Gets a cached value by key. Returns null if not found or expired.
	 */
	getCache(key: string): string | null {
		return this.#authStore.getCache(key);
	}

	/**
	 * Sets a cached value with expiry time (unix seconds).
	 */
	setCache(key: string, value: string, expiresAtSec: number): void {
		this.#authStore.setCache(key, value, expiresAtSec);
	}

	/**
	 * Deletes expired cache entries. Call periodically for cleanup.
	 */
	cleanExpiredCache(): void {
		this.#authStore.cleanExpiredCache();
	}

	/**
	 * Ensures the parent directory for the database file exists.
	 * @param dbPath - Path to the database file
	 */
	#ensureDir(dbPath: string): void {
		const dir = path.dirname(dbPath);
		try {
			fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
		} catch (err) {
			const code = (err as NodeJS.ErrnoException).code;
			// EEXIST is fine - directory already exists
			if (code !== "EEXIST") {
				throw new Error(`Failed to create agent storage directory '${dir}': ${code || err}`);
			}
		}
		// Verify directory was created
		if (!fs.existsSync(dir)) {
			throw new Error(`Agent storage directory '${dir}' does not exist after creation attempt`);
		}
	}

	#hardenPermissions(dbPath: string): void {
		const dir = path.dirname(dbPath);
		try {
			fs.chmodSync(dir, 0o700);
		} catch (error) {
			logger.warn("AgentStorage failed to chmod agent dir", { path: dir, error: String(error) });
		}

		if (!fs.existsSync(dbPath)) return;
		try {
			fs.chmodSync(dbPath, 0o600);
		} catch (error) {
			logger.warn("AgentStorage failed to chmod db file", { path: dbPath, error: String(error) });
		}
	}
	/** One resumable cooperative pass on a legacy database, before publishing its open handle. */
	async #normalizeRequirements(): Promise<void> {
		const marker = this.#db.query("SELECT value FROM meta WHERE key = 'requirements_normalizing'").get() as { value: string } | null;
		if (!marker) return;
		let progress: { phase: "sources" | "batches" | "revisions"; cursor: number } = marker.value === "0" ? { phase: "sources", cursor: 0 } : JSON.parse(marker.value);
		const backend = this.#requirementsBackend();
		for (;;) {
			const rows = this.#db.query("SELECT rowid AS cursor, body FROM requirements_" + progress.phase + " WHERE rowid > ? ORDER BY rowid LIMIT 128").all(progress.cursor) as { cursor: number; body: string }[];
			this.#db.transaction(() => {
				for (const item of rows) {
					if (progress.phase === "sources") this.#writeRequirementsLocators(JSON.parse(item.body) as RequirementsSource);
					else if (progress.phase === "batches") this.#writeRequirementsDependencies(JSON.parse(item.body) as RequirementsBatch);
					else {
						const revision = JSON.parse(item.body) as RequirementsRevision;
						this.#retireLegacyRequirementsAdmission(revision);
						backend.putRevision(revision);
						const heads = this.#requirementsHeadIds(undefined, revision.requirementId).map(id => this.getRequirementsRevision(id)!);
						const remaining: RequirementsRevision[] = [];
						let superseded = false;
						for (const head of heads) {
							if (head.id === revision.id) continue;
							if (this.#requirementsScopeKey(head.scope) !== this.#requirementsScopeKey(revision.scope)) { remaining.push(head); continue; }
							if (revision.predecessorRevisionIds.includes(head.id) || this.#legacyRequirementsPrecedes(head.sourceKey, revision.sourceKey, [...(head.relations ?? []), ...(revision.relations ?? [])])) continue;
							if (head.predecessorRevisionIds.includes(revision.id) || this.#legacyRequirementsPrecedes(revision.sourceKey, head.sourceKey, [...(head.relations ?? []), ...(revision.relations ?? [])])) superseded = true;
							remaining.push(head);
						}
						if (!superseded) remaining.push(revision);
						this.#setRequirementsHeads(revision.requirementId, remaining);
					}
					progress.cursor = item.cursor;
				}
				if (!rows.length && progress.phase !== "revisions") progress = { phase: progress.phase === "sources" ? "batches" : "revisions", cursor: 0 };
				this.#db.query("UPDATE meta SET value = ? WHERE key = 'requirements_normalizing'").run(JSON.stringify(progress));
			}).immediate();
			if (!rows.length && progress.phase === "revisions" && progress.cursor !== 0) break;
			if (!rows.length && progress.phase === "revisions" && !(this.#db.query("SELECT 1 FROM requirements_revisions WHERE rowid > ? LIMIT 1").get(progress.cursor))) break;
			await Bun.sleep(0);
		}
		this.#db.query("DELETE FROM meta WHERE key = 'requirements_normalizing'").run();
	}

	#legacyRequirementsPrecedes(predecessor: string, successor: string, relations: NonNullable<RequirementsRevision["relations"]>): boolean {
		const pending = [successor], seen = new Set<string>();
		while (pending.length) {
			const key = pending.pop()!;
			if (seen.has(key)) continue;
			seen.add(key);
			const source = this.getRequirementsSource(key);
			if (!source || source.integrityAvailable === false || source.state === "orphaned") continue;
			if (key === predecessor) return true;
			const parent = source.parentKey ? this.getRequirementsSource(source.parentKey) : undefined;
			if (parent && parent.ownerSessionId === source.ownerSessionId && parent.epoch === source.epoch) pending.push(parent.key);
			for (const relation of relations) if (relation.successorSourceKey === key) pending.push(relation.predecessorSourceKey);
		}
		return false;
	}

	/** Only explicit inspection materializes the retained ledger. Normal operations use indexed affected rows. */
	#requirementsBackend(): RequirementsBackend {
		return {
			transaction: run => this.#db.transaction(run).immediate(),
			state: (owner, receipt, revisions, scopes) => this.#readRequirementsState(owner, receipt, revisions, scopes),
			writeState: (before, after) => this.#writeRequirementsState(before, after),
			source: key => this.getRequirementsSource(key),
			batch: id => this.getRequirementsBatch(id),
			revision: id => this.getRequirementsRevision(id),
			putSource: source => { this.#putRequirementsRow("sources", source.key, source); this.#writeRequirementsLocators(source); },
			putBatch: batch => {
				this.#db.query("INSERT INTO requirements_batches(id, source_key, source_integrity, extraction_version, review_revision, body) VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET source_key=excluded.source_key, source_integrity=excluded.source_integrity, extraction_version=excluded.extraction_version, review_revision=excluded.review_revision, body=excluded.body")
					.run(batch.id, batch.sourceKey, batch.sourceIntegrity, batch.extractionVersion, batch.reviewRevision, JSON.stringify(batch));
				this.#writeRequirementsDependencies(batch);
			},
			putRevision: revision => {
				const { quarantine, ...row } = revision;
				this.#putRequirementsRow("revisions", revision.id, row);
				if (quarantine) this.#putRequirementsRow("suspensions", revision.id, quarantine);
				else this.#db.query("DELETE FROM requirements_suspensions WHERE id = ?").run(revision.id);
				this.#writeRequirementsDependencies(revision);
				this.#writeRequirementsRelations(revision);
			},
			deleteBatch: id => {
				this.#db.query("DELETE FROM requirements_dependencies WHERE kind = 'batch' AND dependent_id = ?").run(id);
				this.#db.query("DELETE FROM requirements_batches WHERE id = ?").run(id);
			},
			batchIdsForSource: key => this.#requirementsIds("SELECT id FROM requirements_batches WHERE source_key = ?", key),
			revisionIdsForBatch: id => this.#requirementsIds("SELECT id FROM requirements_revisions WHERE json_extract(body, '$.batchId') = ?", id),
			dependentIds: (key, kind) => this.#requirementsIds("SELECT dependent_id AS id FROM requirements_dependencies WHERE source_key = ? AND kind = ?", key, kind),
			relationsForSuccessor: key => (this.#db.query("SELECT body FROM requirements_relation_edges WHERE successor_key = ?").all(key) as { body: string }[]).map(row => JSON.parse(row.body) as RequirementsRelation),
			headIds: (context, requirementId) => this.#requirementsHeadIds(context, requirementId),
			setHeads: (id, rows) => this.#setRequirementsHeads(id, rows),
			pending: context => this.getRequirementsPendingSources(context),
			coverage: keys => this.getRequirementsCoverageSummary(keys),
			snapshot: () => this.#readRequirementsSnapshot(),
		};
	}

	#requirementsIds(sql: string, ...params: string[]): string[] {
		return (this.#db.query(sql).all(...params) as { id: string }[]).map(row => row.id);
	}

	#requirementsRow<T>(table: string, id: string): T | undefined {
		const row = this.#db.query("SELECT body FROM requirements_" + table + " WHERE id = ?").get(id) as { body: string } | null;
		return row ? JSON.parse(row.body) as T : undefined;
	}

	#putRequirementsRow(table: string, id: string, row: unknown): void {
		this.#db.query("INSERT INTO requirements_" + table + "(id, body) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET body = excluded.body").run(id, JSON.stringify(row));
	}

	#requirementsScopeKey(scope: RequirementsScope): string {
		return JSON.stringify([scope.kind, scope.kind === "project" ? scope.projectId ?? null : null, scope.kind === "session" || scope.kind === "task" ? scope.sessionId ?? null : null, scope.kind === "session" || scope.kind === "task" ? scope.epoch ?? null : null]);
	}

	#readRequirementsState(owner?: string, receipt?: string, revisions: readonly string[] = [], scopes: readonly RequirementsScope[] = []): RequirementsSnapshot["state"] {
		const scalar = this.#db.query("SELECT publication_revision AS publicationRevision, generation FROM requirements_state WHERE id = 1").get() as { publicationRevision: number; generation: number };
		const state: RequirementsSnapshot["state"] = { ...scalar, owners: {}, restoreReceipts: [], restoreReviews: {}, tombstones: [] };
		if (owner) { const row = this.#requirementsRow<RequirementsAuthority>("owners", owner); if (row) state.owners[owner] = row; }
		if (receipt) { const row = this.#requirementsRow<RequirementsRestoreReceipt>("restore_receipts", receipt); if (row) state.restoreReceipts.push(row); }
		for (const id of revisions) { const row = this.#requirementsRow<RequirementsSnapshot["state"]["restoreReviews"][string]>("restore_reviews", id); if (row) state.restoreReviews[id] = row; }
		for (const scope of scopes) { const row = this.#requirementsRow<NonNullable<RequirementsSnapshot["state"]["tombstones"]>[number]>("tombstones", this.#requirementsScopeKey(scope)); if (row) state.tombstones!.push(row); }
		return state;
	}

	#writeRequirementsState(before: RequirementsSnapshot["state"], after: RequirementsSnapshot["state"]): void {
		if (before.publicationRevision !== after.publicationRevision || before.generation !== after.generation)
			this.#db.query("UPDATE requirements_state SET publication_revision = ?, generation = ? WHERE id = 1").run(after.publicationRevision, after.generation);
		for (const id of Object.keys(before.owners)) if (!after.owners[id]) this.#db.query("DELETE FROM requirements_owners WHERE id = ?").run(id);
		for (const [id, row] of Object.entries(after.owners)) if (JSON.stringify(row) !== JSON.stringify(before.owners[id])) this.#putRequirementsRow("owners", id, row);
		const receipts = new Set(before.restoreReceipts.map(row => row.id));
		for (const row of after.restoreReceipts) if (!receipts.has(row.id)) this.#putRequirementsRow("restore_receipts", row.id, row);
		for (const [id, row] of Object.entries(after.restoreReviews)) if (JSON.stringify(row) !== JSON.stringify(before.restoreReviews[id])) this.#putRequirementsRow("restore_reviews", id, row);
		const old = new Map((before.tombstones ?? []).map(row => [this.#requirementsScopeKey(row.scope), JSON.stringify(row)]));
		for (const row of after.tombstones ?? []) {
			const id = this.#requirementsScopeKey(row.scope);
			if (old.get(id) !== JSON.stringify(row)) this.#putRequirementsRow("tombstones", id, row);
			old.delete(id);
		}
		for (const id of old.keys()) this.#db.query("DELETE FROM requirements_tombstones WHERE id = ?").run(id);
	}
	/** Accepted relation history remains authoritative after its revision ceases to be a head. */
	#writeRequirementsRelations(revision: RequirementsRevision): void {
		this.#db.query("DELETE FROM requirements_relation_edges WHERE revision_id = ?").run(revision.id);
		if (revision.lifecycle !== "accepted" || (revision.availability && revision.availability !== "available")) return;
		const put = this.#db.query("INSERT INTO requirements_relation_edges(revision_id, position, successor_key, body) VALUES(?, ?, ?, ?)");
		for (const [position, relation] of (revision.relations ?? []).entries()) put.run(revision.id, position, relation.successorSourceKey, JSON.stringify(relation));
	}
	/** Old slice citations and aggregate verdicts are history, never upgraded into whole-unit admission. */
	#retireLegacyRequirementsAdmission(revision: RequirementsRevision, force = false): boolean {
		if (revision.lifecycle === "historical") return false;
		const batch = this.getRequirementsBatch(revision.batchId);
		const evidence = [...revision.evidence, ...(revision.referents ?? []), ...(revision.relations ?? []).flatMap(relation => relation.evidence)];
		const ids = batch?.operationIds;
		const hasCandidateMap = (review: RequirementsBatch["review"]["sanity"]): boolean =>
			Array.isArray(ids) && Array.isArray(review?.candidates) && review.candidates.length === ids.length &&
			new Set(review.candidates.map(candidate => candidate.id)).size === ids.length && review.candidates.every(candidate => ids.includes(candidate.id));
		const legacy = force || evidence.some(unit => "start" in unit || "end" in unit) || !batch ||
			!Array.isArray(ids) || ids.length !== batch.operations.length || new Set(ids).size !== ids.length ||
			!hasCandidateMap(batch.review.sanity) || (!batch.review.literalAcceptance &&
				(!hasCandidateMap(batch.review.evidence) || !Array.isArray(batch.review.evidence?.obligations) ||
					batch.review.evidence.obligations.some(obligation => !Array.isArray(obligation.operationIds) || !Array.isArray(obligation.applicableRevisionIds))));
		if (!legacy) return false;
		const reason = "Legacy requirements admission needs fresh whole-unit extraction and independent review";
		revision.lifecycle = "historical";
		revision.availability = "changed";
		if (batch && batch.status !== "stale") {
			batch.status = "stale"; batch.reason = reason;
			this.#db.query("UPDATE requirements_batches SET body = ? WHERE id = ?").run(JSON.stringify(batch), batch.id);
		}
		const source = this.getRequirementsSource(revision.sourceKey);
		if (source && source.origin.kind === "human") { source.state = "pending"; source.reason = reason; this.#putRequirementsRow("sources", source.key, source); }
		return true;
	}


	async #normalizeRequirementsRelations(): Promise<void> {
		const marker = this.#db.query("SELECT value FROM meta WHERE key = 'requirements_relations_normalizing'").get() as { value: string } | null;
		if (!marker) return;
		let cursor = Number(marker.value);
		for (;;) {
			const rows = this.#db.query("SELECT rowid AS cursor, body FROM requirements_revisions WHERE rowid > ? ORDER BY rowid LIMIT 128").all(cursor) as { cursor: number; body: string }[];
			if (!rows.length) break;
			this.#db.transaction(() => {
				for (const row of rows) {
					const revision = JSON.parse(row.body) as RequirementsRevision;
					revision.quarantine = this.#requirementsRow("suspensions", revision.id);
					if (this.#retireLegacyRequirementsAdmission(revision)) this.#requirementsBackend().putRevision(revision);
					else this.#writeRequirementsRelations(revision);
					cursor = row.cursor;
				}
				this.#db.query("UPDATE meta SET value = ? WHERE key = 'requirements_relations_normalizing'").run(String(cursor));
			}).immediate();
			await Bun.sleep(0);
		}
		this.#db.query("DELETE FROM meta WHERE key = 'requirements_relations_normalizing'").run();
	}
	async #normalizeLegacyRequirementsSdk(): Promise<void> {
		if (!this.#db.query("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'requirements_legacy_sdk_sources'").get()) return;
		const backend = this.#requirementsBackend();
		for (;;) {
			const rows = this.#db.query("SELECT id FROM requirements_legacy_sdk_sources LIMIT 128").all() as { id: string }[];
			if (!rows.length) break;
			this.#db.transaction(() => {
				const ids = new Set<string>();
				for (const row of rows) for (const id of backend.dependentIds(row.id, "revision")) ids.add(id);
				for (const id of ids) { const revision = this.getRequirementsRevision(id)!; this.#retireLegacyRequirementsAdmission(revision, true); revision.availability = "unavailable"; backend.putRevision(revision); }
				for (const row of rows) this.#db.query("DELETE FROM requirements_legacy_sdk_sources WHERE id = ?").run(row.id);
			}).immediate();
			await Bun.sleep(0);
		}
		this.#db.run("DROP TABLE requirements_legacy_sdk_sources");
	}

	#writeRequirementsDependencies(row: RequirementsBatch | RequirementsRevision): void {
		const batch = "operations" in row;
		const kind = batch ? "batch" : "revision";
		const keys = new Set([row.sourceKey]);
		if (batch) for (const key of Object.keys(row.readSourceIntegrities)) keys.add(key);
		for (const operation of batch ? row.operations : [row]) {
			for (const evidence of operation.evidence) keys.add(evidence.sourceKey);
			for (const evidence of operation.referents ?? []) keys.add(evidence.sourceKey);
			for (const relation of operation.relations ?? []) {
				keys.add(relation.predecessorSourceKey); keys.add(relation.successorSourceKey);
				for (const evidence of relation.evidence) keys.add(evidence.sourceKey);
			}
		}
		this.#db.query("DELETE FROM requirements_dependencies WHERE kind = ? AND dependent_id = ?").run(kind, row.id);
		const insert = this.#db.query("INSERT INTO requirements_dependencies(kind, dependent_id, source_key) VALUES(?, ?, ?)");
		for (const key of keys) insert.run(kind, row.id, key);
	}

	#requirementsHeadIds(context?: RequirementsConsumptionContext, requirementId?: string): string[] {
		if (requirementId !== undefined) return this.#requirementsIds("SELECT id FROM requirements_heads WHERE requirement_id = ?", requirementId);
		if (!context) return this.#requirementsIds("SELECT id FROM requirements_heads");
		const scopes: RequirementsScope[] = [{ kind: "global" }, { kind: "session", sessionId: context.sessionId, epoch: context.epoch }, { kind: "task", sessionId: context.sessionId, epoch: context.epoch }];
		if (context.projectId) scopes.push({ kind: "project", projectId: context.projectId });
		const read = this.#db.query("SELECT id, source_key FROM requirements_heads WHERE scope_key = ?");
		return scopes.flatMap(scope => (read.all(this.#requirementsScopeKey(scope)) as { id: string; source_key: string }[])
			.filter(row => scope.kind === "global" || scope.kind === "project" || !context.sourceKeys || context.sourceKeys.has(row.source_key)).map(row => row.id));
	}

	#setRequirementsHeads(id: string, revisions: RequirementsRevision[]): void {
		const old = new Set(this.#requirementsIds("SELECT id FROM requirements_heads WHERE requirement_id = ?", id));
		const put = this.#db.query("INSERT INTO requirements_heads(id, requirement_id, scope_key, source_key) VALUES(?, ?, ?, ?) ON CONFLICT(id) DO NOTHING");
		for (const row of revisions) { if (!old.delete(row.id)) put.run(row.id, id, this.#requirementsScopeKey(row.scope), row.sourceKey); }
		for (const revisionId of old) this.#db.query("DELETE FROM requirements_heads WHERE id = ?").run(revisionId);
	}

	getRequirementsSourceMetadata(key: string): RequirementsSourceMetadata | undefined {
		const row = this.#db.query("SELECT id AS key, json_extract(body, '$.integrity') AS integrity, json_extract(body, '$.integrityAvailable') AS integrityAvailable, json_extract(body, '$.state') AS state, json_extract(body, '$.locators') AS locators FROM requirements_sources WHERE id = ?").get(key) as { key: string; integrity: string; integrityAvailable: number | null; state: RequirementsSourceState; locators: string } | null;
		return row ? { key: row.key, integrity: row.integrity, integrityAvailable: row.integrityAvailable === null ? undefined : row.integrityAvailable !== 0, state: row.state, locators: JSON.parse(row.locators) } : undefined;
	}
	#writeRequirementsLocators(source: RequirementsSource): void {
		this.#db.query("DELETE FROM requirements_source_locators WHERE source_key = ?").run(source.key);
		const put = this.#db.query("INSERT OR IGNORE INTO requirements_source_locators(source_key, journal_path) VALUES(?, ?)");
		for (const locator of source.locators) if (locator.journalPath) put.run(source.key, locator.journalPath);
	}

	getRequirementsJournalDependents(journalPath: string): { sources: RequirementsSource[]; revisions: RequirementsRevision[] } {
		return this.#db.transaction(() => {
			const edges = this.#db.query("SELECT DISTINCT d.source_key, h.id FROM requirements_source_locators l INDEXED BY requirements_locator_journal CROSS JOIN requirements_dependencies d INDEXED BY requirements_dependencies_source ON d.source_key = l.source_key AND d.kind = 'revision' CROSS JOIN requirements_heads h ON h.id = d.dependent_id CROSS JOIN requirements_revisions r ON r.id = h.id WHERE l.journal_path = ? AND json_extract(r.body, '$.lifecycle') != 'historical'").all(journalPath) as { source_key: string; id: string }[];
			return { sources: [...new Set(edges.map(row => row.source_key))].map(key => this.getRequirementsSource(key)!), revisions: [...new Set(edges.map(row => row.id))].map(id => this.getRequirementsRevision(id)!) };
		})();
	}


	getRequirementsSource(key: string): RequirementsSource | undefined { return this.#requirementsRow("sources", key); }
	getRequirementsBatch(id: string): RequirementsBatch | undefined { return this.#requirementsRow("batches", id); }
	getRequirementsRevision(id: string): RequirementsRevision | undefined {
		const row = this.#requirementsRow<RequirementsRevision>("revisions", id);
		if (row) row.quarantine = this.#requirementsRow("suspensions", id);
		return row;
	}
	getRequirementsLatestBatch(key: string, integrity?: string): RequirementsBatch | undefined {
		const row = (integrity === undefined
			? this.#db.query("SELECT body FROM requirements_batches WHERE source_key = ? ORDER BY rowid DESC LIMIT 1").get(key)
			: this.#db.query("SELECT body FROM requirements_batches WHERE source_key = ? AND source_integrity = ? ORDER BY rowid DESC LIMIT 1").get(key, integrity)) as { body: string } | null;
		return row ? JSON.parse(row.body) : undefined;
	}
	getRequirementsState() { return this.#requirements.getRequirementsState(); }
	getRequirementsPendingSources(context?: RequirementsConsumptionContext): RequirementsSource[] {
		const predicate = "COALESCE(json_extract(body, '$.referenceOnly'), 0) = 0 AND json_extract(body, '$.state') != 'complete' AND json_extract(body, '$.origin.kind') = 'human'";
		const rows = (context
			? this.#db.query("SELECT body FROM requirements_sources INDEXED BY requirements_pending_owner WHERE " + predicate + " AND json_extract(body, '$.ownerSessionId') = ? AND json_extract(body, '$.epoch') = ? ORDER BY rowid").all(context.sessionId, context.epoch)
			: this.#db.query("SELECT body FROM requirements_sources INDEXED BY requirements_pending_input WHERE " + predicate + " ORDER BY rowid").all()) as { body: string }[];
		return rows.map(row => JSON.parse(row.body) as RequirementsSource).filter(row => !context?.sourceKeys || context.sourceKeys.has(row.key));
	}
	getRequirementsCoverageSummary(keys?: ReadonlySet<string>): RequirementsCoverageSummary {
		const summary = createEmptyRequirementsCoverageSummary();
		if (keys) {
			const read = this.#db.query("SELECT COALESCE(json_extract(body, '$.referenceOnly'), 0) AS reference_only, json_extract(body, '$.state') AS state FROM requirements_sources WHERE id = ?");
			for (const key of keys) { const row = read.get(key) as { reference_only: number; state: RequirementsSourceState } | null; if (row) { summary.total++; if (row.reference_only) summary.referenceOnly++; else summary.byState[row.state]++; } }
		} else for (const row of this.#db.query("SELECT reference_only, state, count FROM requirements_coverage").all() as { reference_only: number; state: RequirementsSourceState; count: number }[]) {
			summary.total += row.count; if (row.reference_only) summary.referenceOnly += row.count; else summary.byState[row.state] += row.count;
		}
		return summary;
	}
	getRequirementsConsumptionSnapshot(context: RequirementsConsumptionContext): RequirementsConsumptionSnapshot { return this.#requirements.getRequirementsConsumptionSnapshot(context); }
	getRequirementsSnapshot(): RequirementsSnapshot { return this.#requirements.getRequirementsSnapshot(); }
	#readRequirementsSnapshot(): RequirementsSnapshot {
		const snapshot = createEmptyRequirementsSnapshot();
		snapshot.state = this.#readRequirementsState();
		snapshot.state.tombstones = (this.#db.query("SELECT body FROM requirements_tombstones").all() as { body: string }[]).map(row => JSON.parse(row.body));
		for (const table of ["owners", "restore_reviews"] as const) {
			const values = Object.fromEntries((this.#db.query("SELECT id, body FROM requirements_" + table).all() as { id: string; body: string }[]).map(row => [row.id, JSON.parse(row.body)]));
			if (table === "owners") snapshot.state.owners = values; else snapshot.state.restoreReviews = values;
		}
		snapshot.state.restoreReceipts = (this.#db.query("SELECT body FROM requirements_restore_receipts").all() as { body: string }[]).map(row => JSON.parse(row.body));
		for (const table of ["sources", "batches", "revisions"] as const) Object.assign(snapshot, { [table]: (this.#db.query("SELECT body FROM requirements_" + table).all() as { body: string }[]).map(row => JSON.parse(row.body)) });
		for (const revision of snapshot.revisions) revision.quarantine = this.#requirementsRow("suspensions", revision.id);
		return snapshot;
	}
	authorizeRequirementsOwner(authority: RequirementsAuthority): void { this.#requirements.authorizeRequirementsOwner(authority); }
	invalidateRequirementsOwner(ownerSessionId: string): void { this.#requirements.invalidateRequirementsOwner(ownerSessionId); }
	intakeRequirementsSource(source: RequirementsSource): void { this.intakeRequirementsSources([source]); }
	intakeRequirementsSources(sources: readonly RequirementsSource[]): void {
		for (const source of sources) if (!source.durable) throw new Error("Volatile source cannot be recorded as restart-durable requirements");
		if (sources.length) this.#requirements.intakeRequirementsSources(sources);
	}
	setRequirementsSourceDisposition(key: string, integrity: string, state: Exclude<RequirementsSourceState, "complete" | "gap">, reason: string): void { this.#requirements.setRequirementsSourceDisposition(key, integrity, state, reason); }
	recordRequirementsGap(key: string, integrity: string, actor: string, reason: string): void { this.#requirements.recordRequirementsGap(key, integrity, actor, reason); }
	saveRequirementsBatch(batch: RequirementsBatch): string { return this.#requirements.saveRequirementsBatch(batch); }
	publishRequirementsBatch(batchId: string, authority: RequirementsAuthority, verifiedIntegrities: Record<string, string>) { return this.#requirements.publishRequirementsBatch(batchId, authority, verifiedIntegrities); }
	reconcileRequirementsSources(observations: RequirementsObservation[]): void { this.#requirements.reconcileRequirementsSources(observations); }
	quarantineRequirements(revisionIds: string[], actor: string, reason: string): number { return this.#requirements.quarantineRequirements(revisionIds, actor, reason); }
	restoreRequirements(receipt: RequirementsRestoreReceipt): boolean { return this.#requirements.restoreRequirements(receipt); }
	withdrawRequirements(revisionIds: string[], actor: string, reason: string): number { return this.#requirements.withdrawRequirements(revisionIds, actor, reason); }
	clearRequirements(scope: RequirementsScope, actor: string, reason: string): number { return this.#requirements.clearRequirements(scope, actor, reason); }
}
