import { afterEach, describe, expect, it, spyOn, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { daemonRuntimeDir } from "@oh-my-pi/pi-coding-agent/launch/paths";
import * as launch from "@oh-my-pi/pi-coding-agent/tools/browser/launch";
import {
	collectOrphanTargets,
	forgetSharedTarget,
	recordSharedTarget,
	type SharedTargetScope,
} from "@oh-my-pi/pi-coding-agent/tools/browser/orphan-registry";
import {
	acquireBrowser,
	getBrowsersMapForTest,
	releaseBrowser,
} from "@oh-my-pi/pi-coding-agent/tools/browser/registry";
import type { WorkerOutbound } from "@oh-my-pi/pi-coding-agent/tools/browser/tab-protocol";
import {
	cancelIdleCloseForOwner,
	getTabsMapForTest,
	releaseIdleTabsForOwner,
	releaseTab,
	releaseTabsForOwner,
	type WorkerTabSession,
} from "@oh-my-pi/pi-coding-agent/tools/browser/tab-supervisor";
import type { Browser } from "puppeteer-core";

const cleanups: Array<() => Promise<void>> = [];

/** Only fake transports are installed; release, idle sweeping, and durable writes are real. */
async function makeFixture() {
	const id = crypto.randomUUID();
	const scope: SharedTargetScope = {
		projectDir: path.join("/tmp", `omp-close-failure-${id}`),
		daemonName: "omp.browser.headless",
	};
	const owner = `close-owner-${id}`;
	const targets = new Set<string>();
	const blocked = new Set<string>();
	const attempts: string[] = [];
	const tabs: WorkerTabSession[] = [];
	const transport = {
		connected: true,
		disconnect() {
			this.connected = false;
		},
		target: () => ({
			createCDPSession: async () => ({
				send: async (method: string, params?: { targetId: string }) => {
					if (method === "Target.getTargets") {
						return { targetInfos: [...targets].map(targetId => ({ targetId })) };
					}
					if (method !== "Target.closeTarget" || !params) throw new Error(`Unexpected CDP call: ${method}`);
					attempts.push(params.targetId);
					if (blocked.has(params.targetId)) return { success: false };
					return { success: targets.delete(params.targetId) };
				},
				detach: async () => undefined,
			}),
		}),
	};
	const launchSpy = spyOn(launch, "launchHeadlessBrowser").mockResolvedValue({
		browser: transport as unknown as Browser,
	});
	const browser = await acquireBrowser({ kind: "headless", headless: true }, { cwd: scope.projectDir }).finally(() =>
		launchSpy.mockRestore(),
	);
	if ("client" in browser) throw new Error("Expected a Puppeteer browser");
	browser.sharedDaemon = { projectDir: scope.projectDir, name: scope.daemonName };
	const fixture = {
		scope,
		owner,
		targets,
		blocked,
		attempts,
		tabs,
		browser,
		transport,
		ownershipFile: "",
		async addTab(label: string, overrides: Partial<WorkerTabSession> = {}) {
			const targetId = `${id}-${label}`;
			const tab: WorkerTabSession = {
				name: targetId,
				targetId,
				backend: "worker",
				browser,
				state: "alive",
				info: { targetId, url: "about:blank", viewport: { width: 800, height: 600 } },
				pending: new Map(),
				kindTag: "headless",
				ownerSessionId: owner,
				persist: false,
				lastActivityAt: Date.now() - 120_000,
				frozen: false,
				activateForScreenshot: false,
				worker: {
					mode: "worker",
					send() {
						throw new Error("Worker transport unavailable");
					},
					onMessage: () => () => {},
					onError: () => () => {},
					terminate: async () => undefined,
				},
				...overrides,
			};
			targets.add(targetId);
			tabs.push(tab);
			browser.refCount++;
			getTabsMapForTest().set(tab.name, tab);
			await recordSharedTarget(scope, targetId);
			if (!fixture.ownershipFile) {
				const dir = path.join(daemonRuntimeDir(scope.projectDir), `${scope.daemonName}.targets`);
				const files = (await fs.readdir(dir)).filter(file => file.endsWith(".json"));
				expect(files).toHaveLength(1);
				fixture.ownershipFile = path.join(dir, files[0]!);
			}
			return tab;
		},
	};
	cleanups.push(async () => {
		fixture.blocked.clear();
		for (const tab of fixture.tabs) {
			await releaseTab(tab.name);
			await forgetSharedTarget(fixture.scope, tab.targetId);
		}
		cancelIdleCloseForOwner(fixture.owner);
		if (getBrowsersMapForTest().get(browser.key) === browser) await releaseBrowser(browser, { kill: false });
		await fs.rm(daemonRuntimeDir(fixture.scope.projectDir), { recursive: true, force: true });
	});
	return fixture;
}

afterEach(async () => {
	for (const cleanup of cleanups.splice(0)) await cleanup();
});

describe("browser release — confirmed target cleanup", () => {
	it("removes the target and durable ownership after a confirmed CDP close", async () => {
		const fixture = await makeFixture();
		const tab = await fixture.addTab("confirmed");

		expect(await releaseTab(tab.name, { kill: false })).toBe(true);
		expect(fixture.targets.has(tab.targetId)).toBe(false);
		expect(getTabsMapForTest().has(tab.name)).toBe(false);
		expect(await Bun.file(fixture.ownershipFile).exists()).toBe(false);
		expect(getBrowsersMapForTest().has(fixture.browser.key)).toBe(false);
	});

	it("accepts confirmed target absence after the close command cannot find it", async () => {
		const fixture = await makeFixture();
		const tab = await fixture.addTab("already-gone");
		fixture.targets.delete(tab.targetId);

		expect(await releaseTab(tab.name, { kill: false })).toBe(true);
		expect(getTabsMapForTest().has(tab.name)).toBe(false);
		expect(await Bun.file(fixture.ownershipFile).exists()).toBe(false);
	});

	it("rejects an unconfirmed close and preserves ownership until a later release succeeds", async () => {
		const fixture = await makeFixture();
		const listeners = new Set<(message: WorkerOutbound) => void>();
		const tab = await fixture.addTab("retry-release", {
			worker: {
				mode: "worker",
				send(message) {
					if (message.type !== "close") throw new Error(`Unexpected worker message: ${message.type}`);
					queueMicrotask(() => {
						for (const listener of listeners) {
							listener({
								type: "close-failed",
								error: { name: "Error", message: "Page close failed", isToolError: false, isAbort: false },
							});
						}
					});
				},
				onMessage(listener) {
					listeners.add(listener);
					return () => listeners.delete(listener);
				},
				onError: () => () => {},
				terminate: async () => undefined,
			},
		});
		fixture.blocked.add(tab.targetId);

		await expect(releaseTab(tab.name, { kill: false })).rejects.toThrow();
		expect(fixture.targets.has(tab.targetId)).toBe(true);
		expect(getTabsMapForTest().get(tab.name)).toBe(tab);
		expect(getBrowsersMapForTest().get(fixture.browser.key)).toBe(fixture.browser);
		expect(fixture.transport.connected).toBe(true);
		expect(fixture.browser.refCount).toBe(1);
		expect((await Bun.file(fixture.ownershipFile).json()).targets).toEqual([tab.targetId]);

		fixture.blocked.clear();
		expect(await releaseTab(tab.name, { kill: false })).toBe(true);
		expect(fixture.targets.has(tab.targetId)).toBe(false);
		expect(getTabsMapForTest().has(tab.name)).toBe(false);
		expect(await Bun.file(fixture.ownershipFile).exists()).toBe(false);
	});

	it("retries failed cleanup in the live owner's ordinary idle sweep without touching other tabs", async () => {
		const fixture = await makeFixture();
		const failed = await fixture.addTab("retry-sweep");
		const active = await fixture.addTab("active", { lastActivityAt: Date.now() });
		const borrowed = await fixture.addTab("borrowed", { ownerSessionId: `other-${fixture.owner}` });
		fixture.blocked.add(failed.targetId);

		expect(await releaseIdleTabsForOwner(fixture.owner, { idleMs: 60_000 })).toBe(0);
		expect(getTabsMapForTest().get(failed.name)).toBe(failed);
		expect((await Bun.file(fixture.ownershipFile).json()).targets.sort()).toEqual(
			[failed.targetId, active.targetId, borrowed.targetId].sort(),
		);
		// The creator is this still-running process, not a dead-owner orphan.
		expect((await collectOrphanTargets(fixture.scope, { graceMs: 0 })).owners).toEqual([]);

		fixture.blocked.clear();
		expect(await releaseIdleTabsForOwner(fixture.owner, { idleMs: 60_000 })).toBe(1);
		expect(getTabsMapForTest().has(failed.name)).toBe(false);
		expect(fixture.targets.has(failed.targetId)).toBe(false);
		expect((await Bun.file(fixture.ownershipFile).json()).targets.sort()).toEqual(
			[active.targetId, borrowed.targetId].sort(),
		);
		expect(getTabsMapForTest().get(active.name)).toBe(active);
		expect(getTabsMapForTest().get(borrowed.name)).toBe(borrowed);
		expect([...fixture.targets].sort()).toEqual([active.targetId, borrowed.targetId].sort());
		expect(new Set(fixture.attempts)).toEqual(new Set([failed.targetId]));
		expect(fixture.transport.connected).toBe(true);
	});

	it("keeps retrying an explicit close after idle closing is disabled, without closing ordinary idle tabs", async () => {
		const fixture = await makeFixture();
		const failed = await fixture.addTab("cancelled-idle", { persist: true });
		const ordinary = await fixture.addTab("ordinary-idle");
		fixture.blocked.add(failed.targetId);
		await expect(releaseTab(failed.name, { kill: false })).rejects.toThrow();

		vi.useFakeTimers();
		try {
			cancelIdleCloseForOwner(fixture.owner);
			fixture.blocked.clear();
			// Fire the retained retry timer; never manually release or rearm it.
			vi.advanceTimersByTime(30_000);
		} finally {
			vi.useRealTimers();
		}
		// The timer's asynchronous CDP and durable writes finish on real I/O.
		for (let attempt = 0; attempt < 100 && getTabsMapForTest().has(failed.name); attempt++) {
			await Bun.sleep(20);
		}
		expect(getTabsMapForTest().has(failed.name)).toBe(false);
		expect(fixture.targets.has(failed.targetId)).toBe(false);
		expect(getTabsMapForTest().get(ordinary.name)).toBe(ordinary);
		expect([...fixture.targets]).toEqual([ordinary.targetId]);
		expect((await Bun.file(fixture.ownershipFile).json()).targets).toEqual([ordinary.targetId]);
	});

	it("reports a failed bulk release but still closes the owner's releasable sibling", async () => {
		const fixture = await makeFixture();
		const failed = await fixture.addTab("bulk-failed");
		const sibling = await fixture.addTab("bulk-sibling");
		fixture.blocked.add(failed.targetId);

		await expect(releaseTabsForOwner(fixture.owner, { kill: false })).rejects.toThrow();
		expect(getTabsMapForTest().get(failed.name)).toBe(failed);
		expect(getTabsMapForTest().has(sibling.name)).toBe(false);
		expect([...fixture.targets]).toEqual([failed.targetId]);
		expect((await Bun.file(fixture.ownershipFile).json()).targets).toEqual([failed.targetId]);
		expect(fixture.transport.connected).toBe(true);
	});
});
