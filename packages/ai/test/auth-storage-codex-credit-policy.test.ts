import { afterEach, beforeEach, describe, expect, test, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { AuthStorage, type AuthAccountPolicies } from "@oh-my-pi/pi-ai/auth-storage";
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
