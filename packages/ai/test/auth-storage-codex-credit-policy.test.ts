import { Database } from "bun:sqlite";
import { afterEach, beforeEach, describe, expect, test, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { AuthStorage, SqliteAuthCredentialStore, type AuthAccountPolicies } from "@oh-my-pi/pi-ai/auth-storage";
import * as oauth from "@oh-my-pi/pi-ai/registry/oauth";
import { openaiCodexUsageProvider } from "@oh-my-pi/pi-ai/usage/openai-codex";
import type { UsageReport } from "@oh-my-pi/pi-ai/usage";
import { loadAuthAccountPolicyConfig } from "@oh-my-pi/pi-ai/auth-broker";

const provider = "openai-codex";
const optedIn: AuthAccountPolicies = [{ provider, account: { accountId: "actual-a" }, useCredits: true, creditAllocation: "100" }];
const optedOut: AuthAccountPolicies = [{ provider, account: { accountId: "actual-a" }, useCredits: false }];
function payload(balance = "80", exhausted = true) {
	return {
		plan_type: "pro",
		rate_limit: { allowed: !exhausted, limit_reached: exhausted,
			primary_window: { used_percent: exhausted ? 100 : 10, limit_window_seconds: 18000, reset_after_seconds: 9000 },
			secondary_window: null },
		credits: { has_credits: true, unlimited: false, balance, overage_limit_reached: false },
		spend_control: { reached: false },
	};
}

describe("explicit Codex credit policy on native generation selection", () => {
	let dir: string;
	let auth: AuthStorage | undefined;
	const reports = new Map<string, UsageReport | null>();
	let issued: string[];

	beforeEach(async () => {
		dir = await fs.mkdtemp(path.join(os.tmpdir(), "codex-credit-policy-"));
		reports.clear();
		issued = [];
		vi.spyOn(oauth, "getOAuthApiKey").mockImplementation(async (_provider, credentials) => {
			const credential = credentials[provider]!;
			issued.push(credential.accountId!);
			return { apiKey: `simulated-${credential.accountId}`, newCredentials: credential };
		});
	});
	afterEach(async () => {
		auth?.close();
		vi.restoreAllMocks();
		await fs.rm(dir, { recursive: true, force: true });
	});

	async function report(accountId: string, body = payload(), policies: AuthAccountPolicies | null = optedIn) {
		return openaiCodexUsageProvider.fetchUsage({ provider,
			credential: { type: "oauth", accessToken: "simulated-only", accountId }, codexCreditPolicies: policies ?? undefined,
		}, { fetch: (async () => Response.json(body)) as typeof fetch });
	}
	async function open(policies: AuthAccountPolicies, ids = ["actual-a"]) {
		auth = await AuthStorage.create(path.join(dir, "agent.db"), { accountPolicies: policies,
			usageProviderResolver: id => id === provider ? { id: provider, fetchUsage: async params => reports.get(params.credential.accountId!) ?? null } : undefined,
		});
		await auth.credentials.set(provider, ids.map(accountId => ({ type: "oauth" as const, accountId,
			access: `simulated-${accountId}`, refresh: "simulated-only", expires: Date.now() + 86_400_000 })));
		return auth;
	}

	async function openFreshnessPool() {
		const blockedIds = ["depleted-a", "depleted-b"];
		const freeIds = ["free-a", "free-b"];
		const ids = [...blockedIds, ...freeIds];
		const policies: AuthAccountPolicies = ids.map(accountId => ({ provider, account: { accountId }, useCredits: false }));
		// SQLite retention uses its own wall clock; keep durable rows ahead of it.
		const state = { now: Date.UTC(2030, 0, 1), failFreeFetch: false, freeFetches: 0 };
		const resetAt = state.now + 9_000_000;
		vi.spyOn(Date, "now").mockImplementation(() => state.now);
		vi.spyOn(Math, "random").mockReturnValue(1);
		const noNetwork = (async () => { throw new Error("Unexpected network access in Codex freshness regression"); }) as typeof fetch;
		vi.spyOn(globalThis, "fetch").mockImplementation(noNetwork);
		const store = new SqliteAuthCredentialStore(new Database(":memory:"));
		function recreate() {
			// Do not close the previous facade: it would close the shared memory database.
			auth = new AuthStorage(store, { accountPolicies: policies, usageFetch: noNetwork,
				usageProviderResolver: id => id === provider ? { id: provider, fetchUsage: async params => {
					const accountId = params.credential.accountId!;
					const free = freeIds.includes(accountId);
					if (free) {
						state.freeFetches += 1;
						if (state.failFreeFetch) return null;
					}
					const body = payload("80", !free);
					Object.assign(body.rate_limit.primary_window, { reset_at: resetAt / 1000,
						reset_after_seconds: Math.ceil((resetAt - state.now) / 1000) });
					return report(accountId, body, policies);
				} } : undefined,
			});
			return auth;
		}
		const storage = recreate();
		await storage.credentials.set(provider, ids.map(accountId => ({ type: "oauth" as const, accountId,
			access: `simulated-${accountId}`, refresh: `simulated-refresh-${accountId}`, expires: state.now + 86_400_000 })));
		const accounts = storage.oauth.accounts(provider);
		for (const account of accounts.filter(account => blockedIds.includes(account.accountId!))) {
			for (const blockScope of ["chat", "shared"]) {
				storage.blocks.upsert({ credentialId: account.credentialId, providerKey: `${provider}:oauth`,
					blockScope, blockedUntilMs: resetAt });
			}
		}
		const credentialIds = accounts.map(account => account.credentialId);
		const originalBlocks = storage.blocks.list(credentialIds);
		expect(originalBlocks).toHaveLength(4);
		async function expectPreserved(current: AuthStorage) {
			expect(current.oauth.accounts(provider).map(account => account.accountId).sort()).toEqual([...ids].sort());
			expect(await store.listDisabledCredentials(provider)).toEqual([]);
			expect(current.blocks.list(credentialIds)).toEqual(originalBlocks);
			expect(issued.every(accountId => freeIds.includes(accountId))).toBe(true);
		}
		return { storage, store, state, ids, policies, resetAt, recreate, expectPreserved,
			freeKeys: freeIds.map(accountId => `simulated-${accountId}`) };
	}

	test("positive usage-cache jitter keeps opted-out free generation available after report freshness expires", async () => {
		const pool = await openFreshnessPool();
		await pool.storage.usage.reports();
		expect(pool.freeKeys).toContain(await pool.storage.keys.get(provider, "free-generation"));
		await pool.expectPreserved(pool.storage);

		pool.state.now += 313_596;
		expect(pool.freeKeys).toContain(await pool.storage.keys.get(provider, "free-generation"));
		await pool.expectPreserved(pool.storage);
	});

	test("legacy persisted positive-jitter reports recover free generation and preserve failure cooldown across recreation", async () => {
		const pool = await openFreshnessPool();
		for (const accountId of pool.ids) {
			const body = payload("80", accountId.startsWith("depleted-"));
			Object.assign(body.rate_limit.primary_window, { reset_at: pool.resetAt / 1000 });
			const value = (await report(accountId, body, pool.policies))!;
			pool.store.setCache(`usage_cache:report:${provider}:default:oauth|account:${accountId}`,
				JSON.stringify({ value, expiresAt: value.fetchedAt + 375_000 }),
				Math.floor((pool.state.now + 86_400_000) / 1000));
		}
		let storage = pool.recreate();
		await storage.credentials.reload();
		pool.state.now += 313_596;
		expect(pool.freeKeys).toContain(await storage.keys.get(provider, "legacy-free-generation"));
		await pool.expectPreserved(storage);

		// Every free report is now stale; unavailable fresh usage must fail closed.
		pool.state.now += 300_001;
		pool.state.failFreeFetch = true;
		const beforeFailure = pool.state.freeFetches;
		expect(await storage.keys.get(provider, "legacy-free-generation")).toBeUndefined();
		expect(pool.state.freeFetches).toBe(beforeFailure + 2);
		const afterFailure = pool.state.freeFetches;
		await pool.expectPreserved(storage);

		// Neither the current process nor a new facade may hammer usage during backoff.
		expect(await storage.keys.get(provider, "legacy-free-generation")).toBeUndefined();
		expect(pool.state.freeFetches).toBe(afterFailure);
		storage = pool.recreate();
		await storage.credentials.reload();
		expect(await storage.keys.get(provider, "legacy-free-generation")).toBeUndefined();
		expect(pool.state.freeFetches).toBe(afterFailure);
		await pool.expectPreserved(storage);

		pool.state.now += 12_501;
		pool.state.failFreeFetch = false;
		expect(pool.freeKeys).toContain(await storage.keys.get(provider, "legacy-free-generation"));
		await pool.expectPreserved(storage);
	});

	test("opt-out blocks exhausted last resort and opaque override without issuing generation auth; control access remains", async () => {
		reports.set("actual-a", await report("actual-a", payload(), optedOut));
		const storage = await open(optedOut);
		storage.keys.setRuntime(provider, "unconsented-opaque-token");
		expect(await storage.keys.get(provider, "exhausted-last-resort")).toBeUndefined();
		expect(issued).toEqual([]);
		storage.keys.removeRuntime(provider);
		const account = storage.oauth.accounts(provider)[0]!;
		expect((await storage.oauth.accessById(provider, account.credentialId))?.ok).toBe(true);
	});

	test("exact reserve equality blocks while the next decimal unit authorizes paid overage", async () => {
		reports.set("actual-a", await report("actual-a", payload("5")));
		const storage = await open(optedIn);
		expect(await storage.keys.get(provider, "at-reserve")).toBeUndefined();
		expect(issued).toEqual([]);
		reports.set("actual-a", await report("actual-a", payload("5.0000000000000000000000000001")));
		await storage.usage.invalidate(provider);
		expect(await storage.keys.get(provider, "above-reserve")).toBe("simulated-actual-a");
	});

	test("stale, unknown and wrong-account reports cannot authorize generation", async () => {
		const storage = await open(optedIn);
		const fresh = (await report("actual-a"))!;
		reports.set("actual-a", { ...fresh, fetchedAt: Date.now() - 300_001 });
		expect(await storage.keys.get(provider, "stale")).toBeUndefined();
		reports.set("actual-a", null);
		await storage.usage.invalidate(provider);
		expect(await storage.keys.get(provider, "unknown")).toBeUndefined();
		reports.set("actual-a", { ...fresh, metadata: { ...fresh.metadata, accountId: "actual-b" } });
		await storage.usage.invalidate(provider);
		expect(await storage.keys.get(provider, "wrong-account")).toBeUndefined();
		expect(issued).toEqual([]);
	});

	test("fallback and rotation never spend from an unlisted actual account; revocation removes prior consent", async () => {
		reports.set("actual-a", await report("actual-a"));
		reports.set("actual-b", await report("actual-b"));
		const storage = await open(optedIn, ["actual-a", "actual-b"]);
		expect(await storage.keys.get(provider, "two-accounts")).toBe("simulated-actual-a");
		await storage.limits.rotate(provider, "two-accounts", { error: new Error("rate_limit_exceeded"), apiKey: "simulated-actual-a" });
		const next = await storage.keys.get(provider, "two-accounts");
		expect(next === undefined || next === "simulated-actual-a").toBe(true);
		expect(issued.includes("actual-b")).toBe(false);
		storage.close();
		auth = undefined;
		const revoked = await open(optedOut, ["actual-a", "actual-b"]);
		issued = [];
		expect(await revoked.keys.get(provider, "two-accounts")).toBeUndefined();
		expect(issued).toEqual([]);
	});

	test("ordinary quota remains available without consent and standalone overage remains unchanged", async () => {
		reports.set("actual-a", await report("actual-a", payload("80", false), optedOut));
		const storage = await open(optedOut);
		expect(await storage.keys.get(provider, "included-quota")).toBe("simulated-actual-a");
		const standalone = await report("actual-a", payload(), null);
		expect(standalone?.metadata?.limitReached).toBe(false);
		expect((await report("actual-a", payload(), optedOut))?.metadata?.limitReached).toBe(true);
	});

	test("native config discovery preserves explicit false and exact reference decimals", async () => {
		const exact: AuthAccountPolicies = [{ provider, account: { accountId: "actual-a" }, useCredits: false, creditAllocation: "100.00000000000000000001" }];
		await Bun.write(path.join(dir, "config.yml"), `auth:\n  accountPolicies: ${JSON.stringify(exact)}\n`);
		expect((await loadAuthAccountPolicyConfig({ agentDir: dir })).accountPolicies).toEqual(exact);
	});
});
