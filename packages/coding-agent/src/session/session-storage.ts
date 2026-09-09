import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import * as path from "node:path";
import { hasFsCode, isEnoent, logger, parseJsonlLenient, peekFileEnds, readLines, Snowflake, toError } from "@oh-my-pi/pi-utils";
import { overlayTitleSlotContent, parseTitleSlotLine, type SessionTitleUpdate, serializeTitleSlot } from "./session-title-slot";

const utf8Decoder = new TextDecoder("utf-8");

export interface SessionStorageStat {
	size: number;
	mtimeMs: number;
	mtime: Date;
	/** Physical filesystem identity, only when the backend actually provides it. */
	dev?: number;
	ino?: number;
	ctimeMs?: number;
	mode?: number;
}

export interface SessionStorageWriter {
	/**
	 * Append one newline-terminated line.
	 *
	 * File and memory storage apply the line synchronously before the returned
	 * promise settles, so a software crash after `append` returns (or after a
	 * fire-and-forget call begins) still sees the entry on disk / in body. No
	 * `fsync` — power loss may still drop the last page. Indexed backends update
	 * the local index immediately and queue the remote publish in call order.
	 *
	 * `line` MUST include the trailing newline.
	 */
	append(line: string): Promise<void>;
	/**
	 * Synchronous append when the backend can apply the line before return.
	 * File and memory implement this so {@link SessionManager} can latch the
	 * first write failure before the appending call returns (surfaced by a later
	 * flushSync/close/next append — the turn loop does not throw from append).
	 * Indexed backends update the local index immediately and queue remote I/O.
	 */
	appendSync?(line: string): void;
	/** Resolve once all queued appends complete. No fsync. */
	flush(): Promise<void>;
	/** Drain synchronously flushable queued work when the backend supports it. No fsync. */
	flushSync?(): void;
	/** False once close() has begun/finished. */
	isOpen(): boolean;
	close(): Promise<void>;
	getError(): Error | undefined;
}

/**
 * Optional guard applied by atomic writes and appends. The
 * backend MUST call `commitGuard()` synchronously immediately before it makes
 * the staged content visible at `path`. If it returns `false`, the staged
 * write is discarded and the target is left untouched. Backends MUST NOT
 * yield between calling the guard and publishing the write, so a concurrent
 * synchronous rewrite that took over cannot be overwritten by a stale body.
 */
export interface WriteTextAtomicOptions {
	commitGuard?: () => boolean;
}

export interface SessionStorage {
	ensureDirSync(dir: string): void;
	existsSync(path: string): boolean;
	writeTextSync(path: string, content: string): void;
	/**
	 * Update the current session title through the storage backend.
	 *
	 * File-like backends rewrite the fixed-width JSONL title slot; indexed
	 * backends can store the semantic title fields and synthesize the slot when
	 * reading.
	 */
	updateSessionTitle(path: string, update: SessionTitleUpdate): Promise<void>;
	statSync(path: string): SessionStorageStat;
	listFilesSync(dir: string, pattern: string): string[];

	exists(path: string): Promise<boolean>;
	readText(path: string): Promise<string>;
	/** Read the requested UTF-8 byte windows from the head and tail of the file. */
	readTextSlices(path: string, prefixBytes: number, suffixBytes: number): Promise<[string, string]>;
	writeText(path: string, content: string): Promise<void>;
	writeTextAtomic(path: string, content: string, options?: WriteTextAtomicOptions): Promise<void>;
	/** Append a complete suffix atomically to an existing prefix, using the same commit guard as writeTextAtomic. */
	appendTextAtomic(path: string, suffix: string, options?: WriteTextAtomicOptions): Promise<void>;
	rename(path: string, nextPath: string): Promise<void>;
	unlink(path: string): Promise<void>;
	deleteSessionWithArtifacts(sessionPath: string): Promise<void>;
	openWriter(path: string, options?: { flags?: "a" | "w"; onError?: (err: Error) => void }): SessionStorageWriter;
	/**
	 * Wait for every backing write scheduled by this storage to become durably
	 * visible. Sync backends (file, memory) return immediately because their
	 * writes complete in-body; async backends (Redis/SQL via
	 * {@link IndexedSessionStorage}) await their per-path queues so a caller
	 * driving a graceful shutdown does not exit while a fire-and-forget
	 * `writeTextSync` publish is still on the wire.
	 */
	drain(): Promise<void>;
}

// FinalizationRegistry to clean up leaked file descriptors
const writerRegistry = new FinalizationRegistry<number>(fd => {
	try {
		fs.closeSync(fd);
	} catch {
		// Ignore - fd may already be closed or invalid
	}
});

interface JsonlLocation {
	start: number;
	end: number;
	next?: JsonlLocation;
}
interface JournalLocatorIndex {
	stat: SessionStorageStat;
	locations: Map<string, JsonlLocation>;
	prefixEnd: number;
	terminated?: boolean;
	ready?: Promise<void>;
}

function sameJournalVersion(left: SessionStorageStat, right: SessionStorageStat): boolean {
	return left.dev === right.dev && left.ino === right.ino && left.size === right.size &&
		left.mtimeMs === right.mtimeMs && left.ctimeMs === right.ctimeMs;
}

/** Index only top-level identity tokens; source bodies are parsed only on actual reads. */
function jsonlRecordId(bytes: Uint8Array): string | undefined {
	const line = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	let depth = 0;
	let id: string | undefined;
	for (let position = 0; position < line.length; position++) {
		const byte = line[position];
		if (byte === 0x7b || byte === 0x5b) { depth++; continue; }
		if (byte === 0x7d || byte === 0x5d) { depth--; continue; }
		if (byte !== 0x22) continue;
		let end = line.indexOf(0x22, position + 1);
		while (end !== -1) {
			let slash = end - 1;
			while (slash > position && line[slash] === 0x5c) slash--;
			if ((end - slash) % 2 === 1) break;
			end = line.indexOf(0x22, end + 1);
		}
		if (end === -1) return undefined;
		let next = end + 1;
		while (line[next] === 0x20 || line[next] === 0x09 || line[next] === 0x0d) next++;
		if (depth === 1 && line[next] === 0x3a) {
			let key: string | undefined;
			if (end === position + 3 && line[position + 1] === 0x69 && line[position + 2] === 0x64) key = "id";
			else {
				let escaped = false;
				for (let cursor = position + 1; cursor < end; cursor++) if (line[cursor] === 0x5c) { escaped = true; break; }
				if (escaped) {
					try { key = JSON.parse(line.toString("utf8", position, end + 1)); } catch { return undefined; }
				}
			}
			if (key === "id") {
				next++;
				while (line[next] === 0x20 || line[next] === 0x09 || line[next] === 0x0d) next++;
				id = undefined;
				if (line[next] === 0x22) {
					let finish = line.indexOf(0x22, next + 1);
					while (finish !== -1) {
						let slash = finish - 1;
						while (slash > next && line[slash] === 0x5c) slash--;
						if ((finish - slash) % 2 === 1) break;
						finish = line.indexOf(0x22, finish + 1);
					}
					if (finish === -1) return undefined;
					try { id = JSON.parse(line.toString("utf8", next, finish + 1)); } catch { return undefined; }
					end = finish;
				}
			}
		}
		position = end;
	}
	return id;
}

class FileSessionStorageWriter implements SessionStorageWriter {
	#fd: number;
	#closed = false;
	#error: Error | undefined;
	#onError: ((err: Error) => void) | undefined;

	#onAppend: ((before: fs.Stats, bytes: Buffer) => void) | undefined;
	constructor(fpath: string, options?: { flags?: "a" | "w"; onError?: (err: Error) => void },
		onAppend?: (before: fs.Stats, bytes: Buffer) => void) {
		this.#onError = options?.onError;
		this.#onAppend = onAppend;
		const flags = options?.flags ?? "a";
		// Ensure parent directory exists
		const dir = path.dirname(fpath);
		if (!fs.existsSync(dir)) {
			fs.mkdirSync(dir, { recursive: true });
		}
		// Open file once, keep fd for lifetime
		this.#fd = fs.openSync(fpath, flags === "w" ? "w" : "a");
		// Register for cleanup if abandoned without close()
		writerRegistry.register(this, this.#fd, this);
	}

	#recordError(err: unknown): Error {
		const error = toError(err);
		if (!this.#error) this.#error = error;
		this.#onError?.(error);
		return error;
	}

	#writeNow(line: string): void {
		const before = fs.fstatSync(this.#fd);
		const originalSize = before.size;
		const buf = Buffer.from(line, "utf-8");
		let offset = 0;
		try {
			while (offset < buf.length) {
				const written = fs.writeSync(this.#fd, buf, offset, buf.length - offset);
				if (written === 0) {
					throw new Error("Short write");
				}
				offset += written;
			}
		} catch (writeError) {
			try {
				fs.ftruncateSync(this.#fd, originalSize);
			} catch (rollbackError) {
				throw new AggregateError(
					[toError(writeError), toError(rollbackError)],
					"Session append failed and its partial bytes could not be rolled back",
				);
			}
			throw writeError;
		}
		this.#onAppend?.(before, buf);
	}

	appendSync(line: string): void {
		if (this.#closed) throw new Error("Writer closed");
		if (this.#error) throw this.#error;
		// Write in-body so software crash after the call still sees the entry.
		// Microtask batching used to leave completed transcript lines only in
		// memory until the next event-loop turn; process crash then lost every
		// post-checkpoint event. flush/flushSync remain no-op drains (no fsync).
		try {
			this.#writeNow(line);
		} catch (err) {
			throw this.#recordError(err);
		}
	}

	async append(line: string): Promise<void> {
		this.appendSync(line);
	}

	async flush(): Promise<void> {
		if (this.#error) throw this.#error;
	}

	flushSync(): void {
		if (this.#error) throw this.#error;
	}

	isOpen(): boolean {
		return !this.#closed;
	}

	async close(): Promise<void> {
		if (this.#closed) return;
		this.#closed = true;
		// Unregister from finalization - we're closing properly
		writerRegistry.unregister(this);
		try {
			fs.closeSync(this.#fd);
		} catch {
			// Ignore close errors
		}
		if (this.#error) throw this.#error;
	}

	getError(): Error | undefined {
		return this.#error;
	}
}

export class FileSessionStorage implements SessionStorage {
	#journalIndexes?: Map<string, JournalLocatorIndex>;

	#indexJournalLine(index: JournalLocatorIndex, line: Uint8Array, start: number, end: number): void {
		if (!index.prefixEnd || end < index.prefixEnd) {
			const text = utf8Decoder.decode(line);
			if ((start !== 0 || !parseTitleSlotLine(text)) && parseJsonlLenient(text).length) index.prefixEnd = end;
		}
		const id = jsonlRecordId(line);
		if (id !== undefined) index.locations.set(id, { start, end, next: index.locations.get(id) });
	}

	async #journalIndex(filePath: string): Promise<JournalLocatorIndex> {
		const stat = this.statSync(filePath);
		let index = this.#journalIndexes?.get(filePath);
		if (index && sameJournalVersion(index.stat, stat)) {
			await index.ready;
			return index;
		}
		index = { stat, locations: new Map(), prefixEnd: 0 };
		(this.#journalIndexes ??= new Map()).set(filePath, index);
		const building = index;
		building.ready = (async () => {
			let start = 0, records = 0;
			for await (const line of readLines(Bun.file(filePath).slice(0, stat.size).stream())) {
				this.#indexJournalLine(building, line, start, Math.min(start + line.length + 1, stat.size));
				start += line.length + 1;
				if ((++records & 8191) === 0) await Bun.sleep(0);
			}
			const terminated = start === stat.size;
			// Appending to an unterminated old record changes its physical bounds.
			if ((!terminated && building.stat.size !== stat.size) || !sameJournalVersion(building.stat, this.statSync(filePath))) {
				if (this.#journalIndexes?.get(filePath) === building) this.#journalIndexes.delete(filePath);
			} else if (building.stat.size === stat.size) building.terminated = terminated;
		})();
		try { await building.ready; }
		catch (error) {
			if (this.#journalIndexes?.get(filePath) === building) this.#journalIndexes.delete(filePath);
			throw error;
		} finally { building.ready = undefined; }
		return building;
	}

	#recordJournalAppend(filePath: string, before: fs.Stats, bytes: Buffer): void {
		const index = this.#journalIndexes?.get(filePath);
		if (!index) return;
		try {
			const after = this.statSync(filePath);
			if (!sameJournalVersion(index.stat, before) || index.terminated === false ||
				after.dev !== before.dev || after.ino !== before.ino || after.size !== before.size + bytes.length) {
				this.#journalIndexes!.delete(filePath);
				return;
			}
			let start = 0;
			while (start < bytes.length) {
				const newline = bytes.indexOf(0x0a, start);
				const end = newline === -1 ? bytes.length : newline + 1;
				this.#indexJournalLine(index, bytes.subarray(start, newline === -1 ? end : newline), before.size + start, before.size + end);
				start = end;
			}
			index.stat = after;
			index.terminated = bytes.length === 0 ? index.terminated : bytes[bytes.length - 1] === 0x0a;
		} catch {
			// Locator maintenance cannot turn a completed physical append into a failed append.
			this.#journalIndexes!.delete(filePath);
		}
	}

	/** Journal-owned coordinates only. Every selected whole record and its header are read afresh. */
	async readJsonlLinesById(filePath: string, entryIds: ReadonlySet<string>): Promise<{ prefix: string; lines: string[] }> {
		const index = await this.#journalIndex(filePath);
		const locations: JsonlLocation[] = [];
		for (const id of entryIds) {
			for (let location = index.locations.get(id); location; location = location.next) locations.push(location);
		}
		locations.sort((left, right) => left.start - right.start);
		const file = Bun.file(filePath);
		const prefix = await file.slice(0, index.prefixEnd).text();
		const lines: string[] = [];
		for (const location of locations) {
			let line = await file.slice(Math.max(0, location.start - 1), location.end).text();
			if (location.start > 0) {
				if (!line.startsWith("\n")) continue;
				line = line.slice(1);
			}
			if (!line.endsWith("\n") && location.end !== this.statSync(filePath).size) continue;
			lines.push(line);
		}
		return { prefix, lines };
	}

	ensureDirSync(dir: string): void {
		if (!fs.existsSync(dir)) {
			fs.mkdirSync(dir, { recursive: true });
		}
	}

	existsSync(path: string): boolean {
		return fs.existsSync(path);
	}

	writeTextSync(fpath: string, content: string): void {
		const dir = path.dirname(fpath);
		this.ensureDirSync(dir);
		const tempPath = path.join(dir, `.${path.basename(fpath)}.${Snowflake.next()}.tmp`);
		try {
			fs.writeFileSync(tempPath, content);
		} catch (err) {
			this.#discardTemp(tempPath, fpath);
			throw toError(err);
		}
		try {
			this.renameSync(tempPath, fpath);
		} catch (err) {
			if (!hasFsCode(err, "EPERM")) {
				this.#discardTemp(tempPath, fpath);
				throw toError(err);
			}
			try {
				this.#replaceSessionFileAfterEpermSync(tempPath, fpath, err);
			} catch (fallbackErr) {
				this.#discardTemp(tempPath, fpath);
				throw fallbackErr;
			}
		}
	}

	async updateSessionTitle(fpath: string, update: SessionTitleUpdate): Promise<void> {
		const fd = fs.openSync(fpath, "r+");
		const index = this.#journalIndexes?.get(fpath);
		const before = index ? fs.fstatSync(fd) : undefined;
		try {
			const buf = Buffer.from(serializeTitleSlot(update), "utf-8");
			let offset = 0;
			while (offset < buf.length) {
				const written = fs.writeSync(fd, buf, offset, buf.length - offset, offset);
				if (written === 0) {
					throw new Error("Short write");
				}
				offset += written;
			}
			if (index && before) {
				const after = this.statSync(fpath);
				if (sameJournalVersion(index.stat, before) && after.dev === before.dev && after.ino === before.ino && after.size === before.size) index.stat = after;
				else this.#journalIndexes?.delete(fpath);
			}
		} catch (err) {
			throw toError(err);
		} finally {
			fs.closeSync(fd);
		}
	}

	statSync(path: string): SessionStorageStat {
		const stats = fs.statSync(path);
		return {
			size: stats.size,
			mtimeMs: stats.mtimeMs,
			mtime: stats.mtime,
			dev: stats.dev,
			ino: stats.ino,
			ctimeMs: stats.ctimeMs,
			mode: stats.mode,
		};
	}

	listFilesSync(dir: string, pattern: string): string[] {
		try {
			return Array.from(new Bun.Glob(pattern).scanSync(dir)).map(name => path.join(dir, name));
		} catch {
			return [];
		}
	}

	async exists(path: string): Promise<boolean> {
		try {
			await fs.promises.access(path);
			return true;
		} catch (err) {
			if (isEnoent(err)) return false;
			throw err;
		}
	}

	readText(path: string): Promise<string> {
		return Bun.file(path).text();
	}

	async readTextSlices(path: string, prefixBytes: number, suffixBytes: number): Promise<[string, string]> {
		return peekFileEnds(path, prefixBytes, suffixBytes, (head, tail) => [
			utf8Decoder.decode(head),
			utf8Decoder.decode(tail),
		]);
	}

	async writeText(path: string, content: string): Promise<void> {
		await Bun.write(path, content, { createPath: true });
	}

	async writeTextAtomic(fpath: string, content: string, options?: WriteTextAtomicOptions): Promise<void> {
		const dir = path.resolve(fpath, "..");
		const tempPath = path.join(dir, `.${path.basename(fpath)}.${Snowflake.next()}.tmp`);
		await fs.promises.mkdir(dir, { recursive: true });
		try {
			await fs.promises.writeFile(tempPath, content);
		} catch (err) {
			this.#discardTemp(tempPath, fpath);
			throw toError(err);
		}
		this.#publishAtomic(tempPath, fpath, options);
	}

	async appendTextAtomic(fpath: string, suffix: string, options?: WriteTextAtomicOptions): Promise<void> {
		const dir = path.resolve(fpath, "..");
		const tempPath = path.join(dir, `.${path.basename(fpath)}.${Snowflake.next()}.tmp`);
		try {
			// Clone when supported; copyFile falls back to a native copy without decoding the prefix.
			await fsp.copyFile(fpath, tempPath, fs.constants.COPYFILE_FICLONE);
			await fsp.appendFile(tempPath, suffix);
		} catch (err) {
			this.#discardTemp(tempPath, fpath);
			throw toError(err);
		}
		this.#publishAtomic(tempPath, fpath, options);
	}

	#publishAtomic(tempPath: string, fpath: string, options?: WriteTextAtomicOptions): void {
		// Guard-check + rename MUST NOT be separated by an await. A concurrent
		// synchronous rewrite (flushSync -> #rewriteSynchronously) can otherwise
		// publish a fresh body between the check and the rename, and this stale
		// staged body would overwrite it. Sync rename closes that window.
		if (options?.commitGuard && !options.commitGuard()) {
			this.#discardTemp(tempPath, fpath);
			return;
		}
		try {
			this.renameSync(tempPath, fpath);
			return;
		} catch (err) {
			if (!hasFsCode(err, "EPERM")) {
				this.#discardTemp(tempPath, fpath);
				throw toError(err);
			}
			try {
				this.#replaceSessionFileAfterEpermSync(tempPath, fpath, err, options?.commitGuard);
			} catch (fallbackErr) {
				this.#discardTemp(tempPath, fpath);
				throw fallbackErr;
			}
		}
	}

	/**
	 * Sync rename hook. Split from `rename` so atomic writes and appends can perform their
	 * guard-then-publish step without a yield, and so tests can inject
	 * Windows-style EPERM at the sync layer used by the atomic path.
	 */
	renameSync(source: string, target: string): void {
		fs.renameSync(source, target);
	}

	#discardTemp(tempPath: string, targetPath: string): void {
		try {
			fs.unlinkSync(tempPath);
		} catch (err) {
			if (!isEnoent(err)) {
				logger.warn("Failed to remove session rewrite temp file", {
					sessionFile: targetPath,
					tempPath,
					error: toError(err).message,
				});
			}
		}
	}

	#replaceSessionFileAfterEpermSync(
		tempPath: string,
		targetPath: string,
		renameError: unknown,
		commitGuard?: () => boolean,
	): void {
		const dir = path.resolve(targetPath, "..");
		const backupPath = path.join(dir, `${path.basename(targetPath)}.${Snowflake.next()}.bak`);
		try {
			this.renameSync(targetPath, backupPath);
		} catch (moveAsideError) {
			if (isEnoent(moveAsideError)) {
				if (commitGuard && !commitGuard()) {
					this.#discardTemp(tempPath, targetPath);
					return;
				}
				this.renameSync(tempPath, targetPath);
				return;
			}
			throw toError(renameError);
		}
		if (commitGuard && !commitGuard()) {
			// Restore only if no fresh target took over while the guard rejected us.
			try {
				if (!this.existsSync(targetPath)) this.renameSync(backupPath, targetPath);
			} catch (restoreErr) {
				throw new Error(
					`Failed to restore session file after EPERM commitGuard rejection (original: ${
						toError(renameError).message
					}; rollback: ${toError(restoreErr).message})`,
					{ cause: toError(renameError) },
				);
			}
			this.#discardTemp(tempPath, targetPath);
		} else {
			try {
				this.renameSync(tempPath, targetPath);
			} catch (replaceError) {
				try {
					this.renameSync(backupPath, targetPath);
				} catch (rollbackErr) {
					const rollbackError = toError(rollbackErr);
					throw new Error(
						`Failed to replace session file after EPERM (original: ${toError(renameError).message}; retry: ${
							toError(replaceError).message
						}; rollback: ${rollbackError.message})`,
						{ cause: toError(renameError) },
					);
				}
				throw toError(replaceError);
			}
		}
		try {
			fs.unlinkSync(backupPath);
		} catch (err) {
			if (!isEnoent(err)) {
				logger.warn("Failed to remove session rewrite backup", {
					sessionFile: targetPath,
					backupPath,
					error: toError(err).message,
				});
			}
		}
	}

	async rename(path: string, nextPath: string): Promise<void> {
		try {
			await fs.promises.rename(path, nextPath);
		} catch (err) {
			throw toError(err);
		}
	}

	unlink(path: string): Promise<void> {
		this.#journalIndexes?.delete(path);
		return fs.promises.unlink(path);
	}

	drain(): Promise<void> {
		// File writes complete synchronously in-body via fs.writeFileSync /
		// fs.renameSync, so there is no queued work to await.
		return Promise.resolve();
	}

	openWriter(path: string, options?: { flags?: "a" | "w"; onError?: (err: Error) => void }): SessionStorageWriter {
		return new FileSessionStorageWriter(path, options, (before, bytes) => this.#recordJournalAppend(path, before, bytes));
	}

	/**
	 * Delete a session file and its artifacts directory.
	 * Artifacts are stored in a sibling directory with the same name minus .jsonl extension.
	 */
	async deleteSessionWithArtifacts(sessionPath: string): Promise<void> {
		// Delete the session file itself
		await this.unlink(sessionPath);

		// Compute artifacts directory: /path/to/session.jsonl -> /path/to/session
		const artifactsDir = sessionPath.slice(0, -6);

		// Delete artifacts directory if it exists. Missing directories are fine, but
		// surface real cleanup failures because the session file is already gone.
		try {
			await fsp.rm(artifactsDir, { recursive: true, force: true });
		} catch (err) {
			const error = toError(err);
			throw new Error(
				`Session file deleted but failed to remove artifacts directory ${artifactsDir}: ${error.message}`,
				{
					cause: error,
				},
			);
		}
	}
}

function matchesPattern(name: string, pattern: string): boolean {
	if (pattern === "*") return true;
	if (pattern.startsWith("*.")) {
		return name.endsWith(pattern.slice(1));
	}
	return name === pattern;
}

class MemorySessionStorageWriter implements SessionStorageWriter {
	#storage: MemorySessionStorage;
	#path: string;
	#closed = false;
	#error: Error | undefined;
	#onError: ((err: Error) => void) | undefined;

	constructor(
		storage: MemorySessionStorage,
		path: string,
		options?: { flags?: "a" | "w"; onError?: (err: Error) => void },
	) {
		this.#storage = storage;
		this.#path = path;
		this.#onError = options?.onError;
		if ((options?.flags ?? "a") === "w") {
			this.#storage.writeTextSync(path, "");
		}
	}

	#recordError(err: unknown): Error {
		const error = toError(err);
		if (!this.#error) this.#error = error;
		this.#onError?.(error);
		return error;
	}

	appendSync(line: string): void {
		if (this.#closed) throw new Error("Writer closed");
		if (this.#error) throw this.#error;
		try {
			// O(1) append — push onto the path's indexed in-memory entry.
			this.#storage.appendSync(this.#path, line);
		} catch (err) {
			throw this.#recordError(err);
		}
	}

	async append(line: string): Promise<void> {
		this.appendSync(line);
	}

	async flush(): Promise<void> {
		if (this.#error) throw this.#error;
	}

	flushSync(): void {
		if (this.#error) throw this.#error;
	}

	isOpen(): boolean {
		return !this.#closed;
	}

	async close(): Promise<void> {
		if (this.#closed) return;
		this.#closed = true;
	}

	getError(): Error | undefined {
		return this.#error;
	}
}

interface MemoryFileEntry {
	chunks: string[];
	cumulativeBytes: number[];
	size: number;
	mtimeMs: number;
}

function createMemoryFileEntry(content: string, mtimeMs: number): MemoryFileEntry {
	const size = Buffer.byteLength(content, "utf-8");
	return {
		chunks: size === 0 ? [] : [content],
		cumulativeBytes: size === 0 ? [] : [size],
		size,
		mtimeMs,
	};
}

function appendMemoryChunk(entry: MemoryFileEntry, chunk: string): void {
	const chunkSize = Buffer.byteLength(chunk, "utf-8");
	if (chunkSize === 0) return;
	entry.size += chunkSize;
	entry.chunks.push(chunk);
	entry.cumulativeBytes.push(entry.size);
}

function normalizeByteLimit(maxBytes: number, size: number): number {
	if (!(maxBytes > 0) || size === 0) return 0;
	return Math.min(Math.trunc(maxBytes), size);
}

function lowerBound(values: readonly number[], target: number): number {
	let lo = 0;
	let hi = values.length;
	while (lo < hi) {
		const mid = (lo + hi) >>> 1;
		if (values[mid] < target) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	return lo;
}

function upperBound(values: readonly number[], target: number): number {
	let lo = 0;
	let hi = values.length;
	while (lo < hi) {
		const mid = (lo + hi) >>> 1;
		if (values[mid] <= target) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	return lo;
}

function joinChunkRange(chunks: readonly string[], start: number, end: number): string {
	const count = end - start;
	if (count <= 0) return "";
	if (count === 1) return chunks[start] ?? "";

	let content = "";
	for (let i = start; i < end; i++) {
		content += chunks[i];
	}
	return content;
}

function decodeChunkByteRange(chunk: string, startByte: number, endByte: number, chunkSize: number): string {
	if (startByte >= endByte) return "";
	if (startByte === 0 && endByte === chunkSize) return chunk;
	if (chunk.length === chunkSize) return chunk.slice(startByte, endByte);
	const bytes = Buffer.from(chunk, "utf-8");
	return utf8Decoder.decode(bytes.subarray(startByte, endByte));
}

function materializeMemoryEntry(entry: MemoryFileEntry): string {
	const { chunks } = entry;
	if (chunks.length === 0) return "";
	if (chunks.length === 1) return chunks[0];

	const content = chunks.join("");
	entry.chunks = [content];
	entry.cumulativeBytes = [entry.size];
	return content;
}

function sliceChunksHead(entry: MemoryFileEntry, maxBytes: number): string {
	const limit = normalizeByteLimit(maxBytes, entry.size);
	if (limit === 0) return "";
	if (limit >= entry.size) return materializeMemoryEntry(entry);

	const boundaryIndex = lowerBound(entry.cumulativeBytes, limit);
	const chunkStart = boundaryIndex === 0 ? 0 : entry.cumulativeBytes[boundaryIndex - 1];
	const chunkEnd = entry.cumulativeBytes[boundaryIndex];
	if (chunkEnd === limit) return joinChunkRange(entry.chunks, 0, boundaryIndex + 1);

	const chunk = entry.chunks[boundaryIndex];
	const chunkPrefix = decodeChunkByteRange(chunk, 0, limit - chunkStart, chunkEnd - chunkStart);
	return joinChunkRange(entry.chunks, 0, boundaryIndex) + chunkPrefix;
}

function sliceChunksTail(entry: MemoryFileEntry, maxBytes: number): string {
	const limit = normalizeByteLimit(maxBytes, entry.size);
	if (limit === 0) return "";
	if (limit >= entry.size) return materializeMemoryEntry(entry);

	const startByte = entry.size - limit;
	const boundaryIndex = upperBound(entry.cumulativeBytes, startByte);
	const chunkStart = boundaryIndex === 0 ? 0 : entry.cumulativeBytes[boundaryIndex - 1];
	const chunkEnd = entry.cumulativeBytes[boundaryIndex];
	const chunkOffset = startByte - chunkStart;
	if (chunkOffset === 0) return joinChunkRange(entry.chunks, boundaryIndex, entry.chunks.length);

	const chunk = entry.chunks[boundaryIndex];
	const chunkSuffix = decodeChunkByteRange(chunk, chunkOffset, chunkEnd - chunkStart, chunkEnd - chunkStart);
	return chunkSuffix + joinChunkRange(entry.chunks, boundaryIndex + 1, entry.chunks.length);
}

export class MemorySessionStorage implements SessionStorage {
	// Each path keeps appended string chunks plus cumulative UTF-8 byte offsets.
	// Full reads materialize the chunks into one string chunk, so repeated reads
	// do not re-join stale history. Later appends still stay O(1) by pushing
	// after that materialized chunk. Prefix/suffix reads binary-search byte
	// offsets and join only the requested window.
	#files = new Map<string, MemoryFileEntry>();

	#requireEntry(path: string): MemoryFileEntry {
		const entry = this.#files.get(path);
		if (!entry) throw new Error(`File not found: ${path}`);
		return entry;
	}

	ensureDirSync(_dir: string): void {
		// No-op for in-memory storage.
	}

	existsSync(path: string): boolean {
		return this.#files.has(path);
	}

	writeTextSync(path: string, content: string): void {
		this.#files.set(path, createMemoryFileEntry(content, Date.now()));
	}

	async updateSessionTitle(path: string, update: SessionTitleUpdate): Promise<void> {
		const entry = this.#requireEntry(path);
		this.#files.set(
			path,
			createMemoryFileEntry(overlayTitleSlotContent(materializeMemoryEntry(entry), update), Date.now()),
		);
	}

	/**
	 * Internal O(1) append used by {@link MemorySessionStorageWriter}. Lazily
	 * creates the entry. External callers should go through `openWriter()`
	 * rather than touching the mirror directly.
	 */
	appendSync(path: string, chunk: string): void {
		const mtimeMs = Date.now();
		let entry = this.#files.get(path);
		if (!entry) {
			entry = createMemoryFileEntry("", mtimeMs);
			this.#files.set(path, entry);
		}
		appendMemoryChunk(entry, chunk);
		entry.mtimeMs = mtimeMs;
	}

	statSync(path: string): SessionStorageStat {
		const entry = this.#requireEntry(path);
		return {
			size: entry.size,
			mtimeMs: entry.mtimeMs,
			mtime: new Date(entry.mtimeMs),
		};
	}

	listFilesSync(dir: string, pattern: string): string[] {
		const prefix = dir.endsWith("/") ? dir : `${dir}/`;
		const files: string[] = [];
		for (const path of this.#files.keys()) {
			if (!path.startsWith(prefix)) continue;
			const name = path.slice(prefix.length);
			if (name.includes("/") || name.includes("\\")) continue;
			if (!matchesPattern(name, pattern)) continue;
			files.push(path);
		}
		return files;
	}

	exists(path: string): Promise<boolean> {
		return Promise.resolve(this.existsSync(path));
	}

	readText(path: string): Promise<string> {
		const entry = this.#files.get(path);
		if (!entry) return Promise.reject(new Error(`File not found: ${path}`));
		return Promise.resolve(materializeMemoryEntry(entry));
	}

	readTextSlices(path: string, prefixBytes: number, suffixBytes: number): Promise<[string, string]> {
		const entry = this.#files.get(path);
		if (!entry) return Promise.reject(new Error(`File not found: ${path}`));
		return Promise.resolve([sliceChunksHead(entry, prefixBytes), sliceChunksTail(entry, suffixBytes)]);
	}

	writeText(path: string, content: string): Promise<void> {
		this.writeTextSync(path, content);
		return Promise.resolve();
	}

	writeTextAtomic(path: string, content: string, options?: WriteTextAtomicOptions): Promise<void> {
		if (options?.commitGuard && !options.commitGuard()) return Promise.resolve();
		this.writeTextSync(path, content);
		return Promise.resolve();
	}

	async appendTextAtomic(path: string, suffix: string, options?: WriteTextAtomicOptions): Promise<void> {
		const entry = this.#requireEntry(path);
		if (options?.commitGuard && !options.commitGuard()) return;
		appendMemoryChunk(entry, suffix);
		entry.mtimeMs = Date.now();
	}

	rename(path: string, nextPath: string): Promise<void> {
		const entry = this.#files.get(path);
		if (!entry) return Promise.reject(new Error(`File not found: ${path}`));
		this.#files.set(nextPath, entry);
		this.#files.delete(path);
		return Promise.resolve();
	}

	unlink(path: string): Promise<void> {
		this.#files.delete(path);
		return Promise.resolve();
	}
	deleteSessionWithArtifacts(_sessionPath: string): Promise<void> {
		return Promise.resolve();
	}

	drain(): Promise<void> {
		return Promise.resolve();
	}

	openWriter(path: string, options?: { flags?: "a" | "w"; onError?: (err: Error) => void }): SessionStorageWriter {
		return new MemorySessionStorageWriter(this, path, options);
	}
}
