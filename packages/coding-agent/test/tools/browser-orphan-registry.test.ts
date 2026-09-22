/**
 * Regression tests for issue #10022: the project-shared broker-owned Chromium
 * (`omp.browser.headless`) retains page targets created by omp processes that
 * ended abnormally, because tab ownership was tracked only in per-process
 * memory. `orphan-registry` records ownership durably and reaps targets whose
 * owning process is gone.
 *
 * The contract under test:
 *  - a dead owner's targets are collected for reaping, a live owner's are not;
 *  - the current host incarnation keeps its tabs, while prior same-PID owners
 *    are eligible immediately, including legacy records from before exec;
 *  - a conservative grace window keeps a just-crashed owner's fresh records;
 *  - confirmed closures are removed, while transient failures remain durable
 *    and are retried on the next reap.
 */
import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { daemonRuntimeDir } from "@oh-my-pi/pi-coding-agent/launch/paths";
import {
	collectOrphanTargets,
	forgetSharedTarget,
	reapOrphanSharedTargets,
	recordSharedTarget,
	resetOrphanRegistryForTest,
	type SharedTargetScope,
} from "@oh-my-pi/pi-coding-agent/tools/browser/orphan-registry";
import type { Browser } from "puppeteer-core";

const DAEMON_NAME = "omp.browser.headless";

/** Unique per-test scope so registry dirs never collide across the suite. */
function makeScope(): SharedTargetScope {
	const projectDir = path.join("/tmp", `omp-orphan-test-${crypto.randomUUID()}`);
	return { projectDir, daemonName: DAEMON_NAME };
}

function registryDir(scope: SharedTargetScope): string {
	return path.join(daemonRuntimeDir(scope.projectDir), `${scope.daemonName}.targets`);
}

async function writeOwnershipFile(
	scope: SharedTargetScope,
	record: { pid: number; incarnation?: string; updatedAt: number; targets: string[] },
): Promise<string> {
	const dir = registryDir(scope);
	await fs.mkdir(dir, { recursive: true });
	const filename = record.incarnation ? `${record.pid}.${record.incarnation}.json` : `${record.pid}.json`;
	const file = path.join(dir, filename);
	await Bun.write(file, JSON.stringify(record));
	return file;
}

/** Discover the current record in a fresh scope without depending on its UUID. */
async function currentOwnershipFile(scope: SharedTargetScope): Promise<string> {
	const files = (await fs.readdir(registryDir(scope))).filter(file => file.endsWith(".json"));
	expect(files).toHaveLength(1);
	return path.join(registryDir(scope), files[0]!);
}

/** A pid that has been spawned and reaped, so `kill(pid, 0)` reports ESRCH. */
async function deadPid(): Promise<number> {
	const proc = Bun.spawn(["true"], { stdout: "ignore", stderr: "ignore" });
	await proc.exited;
	return proc.pid;
}

/** Minimal puppeteer Browser stub recording closes and optionally failing selected targets. */
function makeBrowser(
	closed: string[],
	failTargets: ReadonlySet<string> = new Set(),
	beforeClose?: (targetId: string) => Promise<void>,
): Browser {
	const session = {
		send: async (method: string, params?: { targetId: string }) => {
			if (method === "Target.getTargets") {
				return { targetInfos: [...failTargets].map(targetId => ({ targetId })) };
			}
			if (!params) throw new Error(`Missing params for ${method}`);
			await beforeClose?.(params.targetId);
			if (failTargets.has(params.targetId)) throw new Error("transient CDP failure");
			closed.push(params.targetId);
			return { success: true };
		},
		detach: async () => undefined,
	};
	return {
		target: () => ({ createCDPSession: async () => session }),
	} as unknown as Browser;
}

const scopes: SharedTargetScope[] = [];
function trackedScope(): SharedTargetScope {
	const scope = makeScope();
	scopes.push(scope);
	return scope;
}

afterEach(async () => {
	resetOrphanRegistryForTest();
	for (const scope of scopes.splice(0)) {
		await fs.rm(daemonRuntimeDir(scope.projectDir), { recursive: true, force: true }).catch(() => undefined);
	}
});

describe("orphan-registry — ownership scan", () => {
	it("collects a dead owner's targets and leaves a live owner's untouched", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		const live = 424242; // treated as alive by the injected probe below
		await writeOwnershipFile(scope, { pid: dead, updatedAt: 0, targets: ["dead-a", "dead-b"] });
		await writeOwnershipFile(scope, { pid: live, updatedAt: 0, targets: ["live-a"] });

		const scan = await collectOrphanTargets(scope, {
			now: () => 10_000_000,
			isAlive: pid => pid === live,
		});

		expect(scan.owners).toEqual([
			{
				file: path.join(registryDir(scope), `${dead}.json`),
				pid: dead,
				updatedAt: 0,
				targetIds: ["dead-a", "dead-b"],
			},
		]);
	});

	it("excludes the current incarnation and other live PIDs even when their records are old", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "mine-1");
		await recordSharedTarget(scope, "mine-2");
		const ownFile = await currentOwnershipFile(scope);
		const own = await Bun.file(ownFile).json();
		expect(own.targets.sort()).toEqual(["mine-1", "mine-2"]);

		const live = process.pid + 1;
		await writeOwnershipFile(scope, { pid: live, updatedAt: 0, targets: ["live-legacy"] });
		await writeOwnershipFile(scope, {
			pid: live,
			incarnation: crypto.randomUUID(),
			updatedAt: 0,
			targets: ["live-incarnation"],
		});
		// Even a failed probe of this PID cannot make the current incarnation an orphan.
		const scan = await collectOrphanTargets(scope, {
			now: () => own.updatedAt + 60_000,
			isAlive: pid => pid === live,
		});
		expect(scan.owners).toEqual([]);
	});

	it("collects a fresh legacy record of this PID immediately after exec", async () => {
		const scope = trackedScope();
		const file = await writeOwnershipFile(scope, { pid: process.pid, updatedAt: 100_000, targets: ["legacy"] });
		const scan = await collectOrphanTargets(scope, { now: () => 100_000, isAlive: () => true });
		expect(scan.owners).toEqual([{ file, pid: process.pid, updatedAt: 100_000, targetIds: ["legacy"] }]);
	});

	it("collects a fresh prior incarnation of this PID without waiting for the grace window", async () => {
		const scope = trackedScope();
		const incarnation = crypto.randomUUID();
		const file = await writeOwnershipFile(scope, {
			pid: process.pid,
			incarnation,
			updatedAt: 100_000,
			targets: ["prior"],
		});
		const scan = await collectOrphanTargets(scope, { now: () => 100_000, isAlive: () => true });
		expect(scan.owners).toEqual([{ file, pid: process.pid, incarnation, updatedAt: 100_000, targetIds: ["prior"] }]);
	});

	it("forgetSharedTarget drops one id and removes the file once empty", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "a");
		await recordSharedTarget(scope, "b");
		await forgetSharedTarget(scope, "a");
		const ownFile = await currentOwnershipFile(scope);
		const after = (await Bun.file(ownFile).json()) as {
			targets: string[];
		};
		expect(after.targets).toEqual(["b"]);

		await forgetSharedTarget(scope, "b");
		expect(await Bun.file(ownFile).exists()).toBe(false);
	});

	it("keeps a dead owner's records inside the conservative grace window", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		await writeOwnershipFile(scope, { pid: dead, updatedAt: 100_000, targets: ["fresh"] });

		const withinGrace = await collectOrphanTargets(scope, {
			now: () => 105_000,
			isAlive: () => false,
			graceMs: 15_000,
		});
		expect(withinGrace.owners).toEqual([]);

		const pastGrace = await collectOrphanTargets(scope, {
			now: () => 130_000,
			isAlive: () => false,
			graceMs: 15_000,
		});
		expect(pastGrace.owners).toEqual([
			{
				file: path.join(registryDir(scope), `${dead}.json`),
				pid: dead,
				updatedAt: 100_000,
				targetIds: ["fresh"],
			},
		]);
	});
});

describe("orphan-registry — reap", () => {
	it("closes a dead owner's targets via CDP and deletes its ownership file", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		await writeOwnershipFile(scope, {
			pid: dead,
			updatedAt: Date.now() - 60_000,
			targets: ["orphan-1", "orphan-2"],
		});
		const closed: string[] = [];

		const count = await reapOrphanSharedTargets(makeBrowser(closed), scope);

		expect(count).toBe(2);
		expect(closed.sort()).toEqual(["orphan-1", "orphan-2"]);
		expect(await Bun.file(path.join(registryDir(scope), `${dead}.json`)).exists()).toBe(false);
	});

	it("retains transient CDP failures and retries them on the next reap", async () => {
		const scope = trackedScope();
		const dead = await deadPid();
		const updatedAt = Date.now() - 60_000;
		await writeOwnershipFile(scope, {
			pid: dead,
			updatedAt,
			targets: ["closed-now", "retry-later"],
		});
		const firstClosed: string[] = [];

		const firstCount = await reapOrphanSharedTargets(makeBrowser(firstClosed, new Set(["retry-later"])), scope);

		expect(firstCount).toBe(1);
		expect(firstClosed).toEqual(["closed-now"]);
		const retained = (await Bun.file(path.join(registryDir(scope), `${dead}.json`)).json()) as {
			pid: number;
			updatedAt: number;
			targets: string[];
		};
		expect(retained).toEqual({ pid: dead, updatedAt, targets: ["retry-later"] });

		const retryClosed: string[] = [];
		const retryCount = await reapOrphanSharedTargets(makeBrowser(retryClosed), scope);

		expect(retryCount).toBe(1);
		expect(retryClosed).toEqual(["retry-later"]);
		expect(await Bun.file(path.join(registryDir(scope), `${dead}.json`)).exists()).toBe(false);
	});

	it("retains a failed prior incarnation across concurrent current writes and retries only its file", async () => {
		const scope = trackedScope();
		await recordSharedTarget(scope, "current-old");
		const ownFile = await currentOwnershipFile(scope);
		const incarnation = crypto.randomUUID();
		const updatedAt = Date.now();
		const priorFile = await writeOwnershipFile(scope, {
			pid: process.pid,
			incarnation,
			updatedAt,
			targets: ["closed-now", "retry-later"],
		});
		const closing = Promise.withResolvers<void>();
		const resume = Promise.withResolvers<void>();
		const firstClosed: string[] = [];
		const browser = makeBrowser(firstClosed, new Set(["retry-later"]), async targetId => {
			if (targetId !== "retry-later") return;
			closing.resolve();
			await resume.promise;
		});
		const firstReap = reapOrphanSharedTargets(browser, scope);
		try {
			await closing.promise;
			// Interleave normal host ownership writes with the unresolved old close.
			await Promise.all([recordSharedTarget(scope, "current-new"), forgetSharedTarget(scope, "current-old")]);
		} finally {
			resume.resolve();
			await firstReap;
		}
		expect(await firstReap).toBe(1);
		expect(firstClosed).toEqual(["closed-now"]);
		const retained = { pid: process.pid, incarnation, updatedAt, targets: ["retry-later"] };
		expect(await Bun.file(priorFile).json()).toEqual(retained);
		expect((await Bun.file(ownFile).json()).targets).toEqual(["current-new"]);

		// Removing the last current tab must not erase the still-retryable old owner.
		await forgetSharedTarget(scope, "current-new");
		expect(await Bun.file(ownFile).exists()).toBe(false);
		expect(await Bun.file(priorFile).json()).toEqual(retained);
		await recordSharedTarget(scope, "current-active");

		const retryClosed: string[] = [];
		expect(await reapOrphanSharedTargets(makeBrowser(retryClosed), scope)).toBe(1);
		expect(retryClosed).toEqual(["retry-later"]);
		expect(await Bun.file(priorFile).exists()).toBe(false);
		expect((await Bun.file(ownFile).json()).targets).toEqual(["current-active"]);
	});
});
