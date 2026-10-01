import * as AIError from "../error";
import type {
	AuthAccountPolicies,
	AuthAccountPolicy,
	AuthAccountSelector,
	AuthCredential,
	OAuthAccountIdentity,
	OAuthCredential,
} from "./types";
import { DEFAULT_USAGE_RESERVE_PCT } from "./types";

import type { UsageReport } from "../usage";
import { isRecord } from "../utils";
import { USAGE_REPORT_TTL_MS } from "./sqlite-credential-store";

// Provider credits are decimal quantities, not floating-point currency or reset counts.
function creditDecimal(value: unknown): { units: bigint; scale: bigint } | undefined {
	const text = typeof value === "number" && Number.isFinite(value) ? String(value) : value;
	if (typeof text !== "string" || !/^\d+(?:\.\d+)?$/.test(text)) return undefined;
	const [whole, fraction = ""] = text.split(".");
	return { units: BigInt(whole! + fraction), scale: 10n ** BigInt(fraction.length) };
}

/** Consent belongs to the exact provider account and never authorizes an unlimited/unknown balance. */
export function codexCreditConsent(
	policies: AuthAccountPolicies,
	accountId: string | undefined,
	payload: unknown,
): boolean {
	if (!accountId || !isRecord(payload) || !isRecord(payload.credits)) return false;
	const policy = policies.find(entry => entry.provider === "openai-codex" && entry.account.accountId === accountId);
	if (policy?.useCredits !== true) return false;
	const credits = payload.credits;
	if (credits.has_credits !== true || credits.unlimited === true || credits.overage_limit_reached === true) return false;
	if (isRecord(payload.spend_control) && payload.spend_control.reached === true) return false;
	const balance = creditDecimal(credits.balance);
	const allocation = creditDecimal(policy.creditAllocation);
	return balance !== undefined && allocation !== undefined && allocation.units > 0n &&
		balance.units * allocation.scale * 20n > allocation.units * balance.scale;
}
/** Whether every identity field set on `selector` matches `identity`. */
export function matchesAuthAccountSelector(selector: AuthAccountSelector, identity: OAuthAccountIdentity): boolean {
	return (
		(selector.email === undefined || selector.email === identity.email) &&
		(selector.accountId === undefined || selector.accountId === identity.accountId) &&
		(selector.projectId === undefined || selector.projectId === identity.projectId) &&
		(selector.orgId === undefined || selector.orgId === identity.orgId)
	);
}

/** Validated per-account routing policies (priority/reserve) plus the global reserve fallback. */
export class AccountPolicies {
	#accountPolicies: AuthAccountPolicies;
	readonly defaultReservePct: number;

	constructor(policies: AuthAccountPolicies, defaultReservePct: number | undefined) {
		this.#accountPolicies = policies;
		this.defaultReservePct =
			typeof defaultReservePct === "number" && Number.isFinite(defaultReservePct)
				? Math.max(0, Math.min(100, defaultReservePct))
				: DEFAULT_USAGE_RESERVE_PCT;
		this.#validateAccountPolicyConfiguration();
	}

	#validateAccountPolicyConfiguration(): void {
		for (let index = 0; index < this.#accountPolicies.length; index += 1) {
			const policy = this.#accountPolicies[index]!;
			const path = `auth.accountPolicies[${index}]`;
			if (
				typeof policy.provider !== "string" ||
				policy.provider.length === 0 ||
				policy.provider.trim() !== policy.provider
			) {
				throw new AIError.ConfigurationError(
					`${path}.provider must be a non-empty string without surrounding whitespace`,
				);
			}
			if (!policy.account || typeof policy.account !== "object") {
				throw new AIError.ConfigurationError(`${path}.account must be an object`);
			}
			const baseIdentities = [policy.account.email, policy.account.accountId, policy.account.projectId];
			if (!baseIdentities.some(value => typeof value === "string" && value.length > 0)) {
				throw new AIError.ConfigurationError(
					`${path}.account must include at least one of email, accountId, or projectId`,
				);
			}
			for (const field of ["email", "accountId", "projectId", "orgId"] as const) {
				const value = policy.account[field];
				if (value !== undefined && (typeof value !== "string" || value.length === 0)) {
					throw new AIError.ConfigurationError(`${path}.account.${field} must be a non-empty string`);
				}
			}
			if (policy.priority !== undefined && !Number.isFinite(policy.priority)) {
				throw new AIError.ConfigurationError(`${path}.priority must be a finite number`);
			}
			if (
				policy.reservePct !== undefined &&
				(!Number.isFinite(policy.reservePct) || policy.reservePct < 0 || policy.reservePct > 100)
			) {
				throw new AIError.ConfigurationError(`${path}.reservePct must be a finite number between 0 and 100`);
			}
			if (policy.useCredits !== undefined || policy.creditAllocation !== undefined) {
				if (policy.provider !== "openai-codex" || !policy.account.accountId?.trim()) {
					throw new AIError.ConfigurationError(`${path} credit policy requires an exact openai-codex accountId`);
				}
				if (policy.useCredits !== undefined && typeof policy.useCredits !== "boolean") {
					throw new AIError.ConfigurationError(`${path}.useCredits must be a boolean`);
				}
				const allocation = creditDecimal(policy.creditAllocation);
				if ((policy.creditAllocation !== undefined && typeof policy.creditAllocation !== "string") ||
					(policy.creditAllocation !== undefined && !allocation) ||
					(policy.useCredits === true && (!allocation || allocation.units <= 0n))) {
					throw new AIError.ConfigurationError(`${path}.creditAllocation must be an exact positive decimal string when enabled`);
				}
			}
		}
	}

	validateUsageCapability(provider: string, canFetchUsage: boolean): void {
		const policyIndex = this.#accountPolicies.findIndex(
			policy => policy.provider === provider && policy.reservePct !== undefined,
		);
		if (policyIndex !== -1 && !canFetchUsage) {
			throw new AIError.ConfigurationError(
				`auth.accountPolicies[${policyIndex}].reservePct requires a usage provider for ${provider}`,
			);
		}
	}

	validateFor(provider: string, credentials: readonly AuthCredential[]): void {
		const policies = this.#accountPolicies
			.map((policy, index) => ({ policy, index }))
			.filter(({ policy }) => policy.provider === provider);
		if (policies.length === 0) return;
		const oauthCredentials = credentials.filter(
			(credential): credential is OAuthCredential => credential.type === "oauth",
		);
		if (oauthCredentials.length === 0) return;

		const claimedCredentials = new Map<number, number>();
		for (const { policy, index } of policies) {
			const matches: number[] = [];
			for (let credentialIndex = 0; credentialIndex < oauthCredentials.length; credentialIndex += 1) {
				if (matchesAuthAccountSelector(policy.account, oauthCredentials[credentialIndex]!)) {
					matches.push(credentialIndex);
				}
			}
			const path = `auth.accountPolicies[${index}].account`;
			if (matches.length === 0) {
				throw new AIError.ConfigurationError(`${path} matches no stored OAuth account for ${provider}`);
			}
			if (matches.length > 1) {
				throw new AIError.ConfigurationError(
					`${path} matches ${matches.length} stored OAuth accounts for ${provider}; add another identity field`,
				);
			}
			const credentialIndex = matches[0]!;
			const previousPolicyIndex = claimedCredentials.get(credentialIndex);
			if (previousPolicyIndex !== undefined) {
				throw new AIError.ConfigurationError(
					`auth.accountPolicies[${previousPolicyIndex}] and auth.accountPolicies[${index}] match the same stored OAuth account for ${provider}`,
				);
			}
			claimedCredentials.set(credentialIndex, index);
		}
	}

	/**
	 * Return the configured account policy matching an OAuth identity.
	 *
	 * This is a read-only diagnostics surface: it performs the same conjunctive
	 * selector match as routing and never refreshes, ranks, or mutates credentials.
	 */
	find(provider: string, identity: OAuthAccountIdentity): AuthAccountPolicy | undefined {
		return this.#accountPolicies.find(
			policy => policy.provider === provider && matchesAuthAccountSelector(policy.account, identity),
		);
	}

	/** Undefined preserves standalone behavior; an explicit envelope defaults every account to opt-out. */
	codexCreditPolicies(): AuthAccountPolicies | undefined {
		return this.#accountPolicies.some(policy => policy.provider === "openai-codex" &&
			(policy.useCredits !== undefined || policy.creditAllocation !== undefined)) ? this.#accountPolicies : undefined;
	}

	/** Non-bypassable generation gate, including last resorts, unknown usage and post-refresh rotation. */
	allowsCodexRequest(identity: OAuthAccountIdentity, report: UsageReport | null, now = Date.now()): boolean {
		const policies = this.codexCreditPolicies();
		if (!policies) return true;
		if (!report || !identity.accountId || report.metadata?.accountId !== identity.accountId ||
			!Number.isFinite(report.fetchedAt) || report.fetchedAt > now || now - report.fetchedAt > USAGE_REPORT_TTL_MS ||
			!isRecord(report.raw) || !isRecord(report.raw.rate_limit)) return false;
		const plan = report.raw.rate_limit;
		if (plan.limit_reached === true) return codexCreditConsent(policies, identity.accountId, report.raw);
		if (plan.allowed !== true || plan.limit_reached !== false) return false;
		return [plan.primary_window, plan.secondary_window].every(window => window === null || window === undefined ||
			(isRecord(window) && typeof window.used_percent === "number" &&
				Number.isFinite(window.used_percent) && window.used_percent >= 0 && window.used_percent < 100));
	}

	/** Return the configured policy for a stored OAuth credential. */
	forCredential(provider: string, credential: AuthCredential): AuthAccountPolicy | undefined {
		return credential.type === "oauth" ? this.find(provider, credential) : undefined;
	}
}
