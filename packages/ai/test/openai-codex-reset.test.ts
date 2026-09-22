/**
 * Wire-contract regressions for OpenAI Codex "saved rate limit reset"
 * redemption. The endpoints and request shape are reverse-engineered from the
 * Codex desktop app; these tests pin them so the redeem path can't silently
 * drift (and so we never need to spend a real credit to verify it):
 *
 *   GET  /wham/rate-limit-reset-credits
 *   POST /wham/rate-limit-reset-credits/consume  { credit_id, redeem_request_id, account_id? }
 */
import { Database } from "bun:sqlite";
import { describe, expect, it } from "bun:test";
import { AuthStorage, SqliteAuthCredentialStore } from "@oh-my-pi/pi-ai/auth-storage";
import type { FetchImpl } from "@oh-my-pi/pi-ai/types";
import {
	consumeCodexResetCredit,
	listCodexResetCreditHistory,
	listCodexResetCredits,
	pickSoonestExpiringCredit,
} from "@oh-my-pi/pi-ai/usage/openai-codex-reset";

interface Captured {
	url: string;
	method: string;
	headers: Record<string, string>;
	body?: unknown;
}

function recordingFetch(status: number, payload: unknown): { fetch: FetchImpl; calls: Captured[] } {
	const calls: Captured[] = [];
	const fetch = (async (url: string, init?: RequestInit) => {
		calls.push({
			url: String(url),
			method: init?.method ?? "GET",
			headers: (init?.headers as Record<string, string>) ?? {},
			body: init?.body ? JSON.parse(init.body as string) : undefined,
		});
		return new Response(JSON.stringify(payload), {
			status,
			headers: { "content-type": "application/json" },
		});
	}) as unknown as FetchImpl;
	return { fetch, calls };
}

describe("listCodexResetCredits", () => {
	it("lists credits and surfaces available_count from the dedicated route", async () => {
		const { fetch, calls } = recordingFetch(200, {
			credits: [
				{
					id: "RateLimitResetCredit_abc",
					reset_type: "codex_rate_limits",
					status: "available",
					granted_at: "2026-06-12T02:11:50Z",
					expires_at: "2026-07-12T02:11:50Z",
					title: "One free rate limit reset",
					description: "Thanks for using Codex!",
				},
			],
			available_count: 1,
		});
		const list = await listCodexResetCredits({ accessToken: "tok", accountId: "acct-1", fetch });
		expect(list).not.toBeNull();
		expect(list?.availableCount).toBe(1);
		expect(list?.credits[0]?.id).toBe("RateLimitResetCredit_abc");
		expect(list?.credits[0]?.title).toBe("One free rate limit reset");
		expect(calls[0]?.method).toBe("GET");
		expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/wham/rate-limit-reset-credits");
		expect(calls[0]?.headers.Authorization).toBe("Bearer tok");
		expect(calls[0]?.headers["ChatGPT-Account-Id"]).toBe("acct-1");
	});

	it("falls back to counting available credits when available_count is absent", async () => {
		const { fetch } = recordingFetch(200, {
			credits: [
				{ id: "c1", status: "available" },
				{ id: "c2", status: "redeemed" },
			],
		});
		const list = await listCodexResetCredits({ accessToken: "tok", fetch });
		expect(list?.availableCount).toBe(1);
	});

	it("returns null on non-2xx", async () => {
		const { fetch } = recordingFetch(401, { detail: "Unauthorized" });
		expect(await listCodexResetCredits({ accessToken: "tok", fetch })).toBeNull();
	});
});

describe("listCodexResetCreditHistory", () => {
	const window = { window_start: "2026-08-16T00:00:00Z", as_of: "2026-09-16T00:00:00Z" };
	const event = { id: "original-id", kind: "future_backend_kind", occurred_at: "2026-09-01T01:02:03+01:00" };

	it("follows opaque cursors and preserves events and the first snapshot metadata", async () => {
		const fetch = (async (url: string) => {
			const cursor = new URL(url).searchParams.get("cursor");
			if (cursor === null) return Response.json({ ...window, events: [event], next_cursor: "a+/=?&" });
			if (cursor !== "a+/=?&") return new Response(null, { status: 400 });
			return Response.json({ ...window, as_of: "2026-09-16T00:01:00Z", events: [{ ...event, id: "next-id" }] });
		}) as FetchImpl;
		const result = await listCodexResetCreditHistory({ accessToken: "tok", fetch });
		expect(result.history).toEqual([
			{ id: event.id, kind: event.kind, occurredAt: event.occurred_at },
			{ id: "next-id", kind: event.kind, occurredAt: event.occurred_at },
		]);
		expect(result.historyComplete).toBe(true);
		expect(result.historyWindowStart).toBe(window.window_start);
		expect(result.historyAsOf).toBe(window.as_of);
	});

	it("keeps the first page when a later request fails", async () => {
		const fetch = (async (url: string) =>
			new URL(url).searchParams.has("cursor")
				? new Response(null, { status: 503 })
				: Response.json({ ...window, events: [event], next_cursor: "page-2" })) as FetchImpl;
		const result = await listCodexResetCreditHistory({ accessToken: "tok", fetch });
		expect(result.history).toEqual([{ id: event.id, kind: event.kind, occurredAt: event.occurred_at }]);
		expect(result.historyComplete).toBe(false);
		expect(result.historyError).toBeDefined();
	});

	it("stops repeated cursors without claiming the collected history is complete", async () => {
		let calls = 0;
		const fetch = (async () => {
			if (++calls > 2) throw new Error("Unexpected third request");
			return Response.json({ ...window, events: [{ ...event, id: String(calls) }], next_cursor: "same" });
		}) as FetchImpl;
		const result = await listCodexResetCreditHistory({ accessToken: "tok", fetch });
		expect(calls).toBe(2);
		expect(result.history.map(item => item.id)).toEqual(["1", "2"]);
		expect(result.historyComplete).toBe(false);
		expect(result.historyError).toBeDefined();
	});

	it("retains valid events but rejects malformed events as incomplete", async () => {
		const { fetch } = recordingFetch(200, { ...window, events: [{ ...event, occurred_at: "invalid" }, event] });
		const result = await listCodexResetCreditHistory({ accessToken: "tok", fetch });
		expect(result.history).toEqual([{ id: event.id, kind: event.kind, occurredAt: event.occurred_at }]);
		expect(result.historyComplete).toBe(false);
		expect(result.historyError).toBeDefined();
	});

	it("keeps successful list and history results independent of the other request failing", async () => {
		let failHistory = true;
		const fetch = (async (url: string) => {
			const history = url.endsWith("/history");
			if (history === failHistory) return new Response(null, { status: 503 });
			return Response.json(
				history
					? { ...window, events: [event], next_cursor: null }
					: { credits: [{ id: "saved-credit", status: "available" }], available_count: 1 },
			);
		}) as typeof globalThis.fetch;
		const store = new SqliteAuthCredentialStore(new Database(":memory:"));
		const storage = new AuthStorage(store, { usageFetch: fetch });
		try {
			await storage.credentials.set("openai-codex", {
				type: "oauth",
				access: "tok",
				refresh: "unused",
				expires: Date.now() + 3_600_000,
				accountId: "account",
			});
			const [listSuccess] = await storage.resets.list({ includeHistory: true });
			expect(listSuccess?.availableCount).toBe(1);
			expect(listSuccess?.credits[0]?.id).toBe("saved-credit");
			expect(listSuccess?.error).toBeUndefined();
			expect(listSuccess?.historyComplete).toBe(false);
			expect(listSuccess?.historyError).toBeDefined();
			failHistory = false;
			const [historySuccess] = await storage.resets.list({ includeHistory: true });
			expect(historySuccess?.error).toBeDefined();
			expect(historySuccess?.historyComplete).toBe(true);
			expect(historySuccess?.history).toEqual([{ id: event.id, kind: event.kind, occurredAt: event.occurred_at }]);
		} finally {
			store.close();
		}
	});
});

describe("consumeCodexResetCredit", () => {
	it("POSTs credit_id + redeem_request_id and reports ok on code=reset", async () => {
		const { fetch, calls } = recordingFetch(200, { code: "reset" });
		const result = await consumeCodexResetCredit({
			creditId: "RateLimitResetCredit_abc",
			accessToken: "tok",
			accountId: "acct-1",
			redeemRequestId: "req-123",
			fetch,
		});
		expect(result.ok).toBe(true);
		expect(result.code).toBe("reset");
		expect(calls[0]?.method).toBe("POST");
		expect(calls[0]?.url).toBe("https://chatgpt.com/backend-api/wham/rate-limit-reset-credits/consume");
		expect(calls[0]?.body).toEqual({
			credit_id: "RateLimitResetCredit_abc",
			redeem_request_id: "req-123",
			account_id: "acct-1",
		});
		expect(calls[0]?.headers["Content-Type"]).toBe("application/json");
	});

	it("generates a redeem_request_id when none is supplied", async () => {
		const { fetch, calls } = recordingFetch(200, { code: "reset" });
		await consumeCodexResetCredit({ creditId: "c1", accessToken: "tok", fetch });
		const body = calls[0]?.body as { redeem_request_id?: string };
		expect(typeof body.redeem_request_id).toBe("string");
		expect(body.redeem_request_id?.length).toBeGreaterThan(0);
	});

	it("reports not-ok for business outcomes like already_redeemed", async () => {
		const { fetch } = recordingFetch(200, { code: "already_redeemed" });
		const result = await consumeCodexResetCredit({ creditId: "c1", accessToken: "tok", fetch });
		expect(result.ok).toBe(false);
		expect(result.code).toBe("already_redeemed");
	});

	it("synthesizes an http_<status> code on unexpected failures", async () => {
		const { fetch } = recordingFetch(500, {});
		const result = await consumeCodexResetCredit({ creditId: "c1", accessToken: "tok", fetch });
		expect(result.ok).toBe(false);
		expect(result.code).toBe("http_500");
	});
});

describe("pickSoonestExpiringCredit", () => {
	it("spends in expiry order: soonest available credit first", () => {
		const credits = [
			{ id: "late", status: "available", expiresAt: "2026-08-12T00:00:00Z" },
			{ id: "soon", status: "available", expiresAt: "2026-07-31T18:00:00Z" },
			{ id: "mid", status: "available", expiresAt: "2026-08-11T00:00:00Z" },
		];
		expect(pickSoonestExpiringCredit(credits)?.id).toBe("soon");
	});

	it("never picks a non-available credit over an available one", () => {
		const credits = [
			{ id: "spent", status: "redeemed", expiresAt: "2026-07-31T18:00:00Z" },
			{ id: "live", status: "available", expiresAt: "2026-08-12T00:00:00Z" },
		];
		expect(pickSoonestExpiringCredit(credits)?.id).toBe("live");
	});

	it("ranks dated credits before undated ones and treats missing status as available", () => {
		const credits = [{ id: "undated" }, { id: "dated", expiresAt: "2026-08-12T00:00:00Z" }];
		expect(pickSoonestExpiringCredit(credits)?.id).toBe("dated");
		expect(pickSoonestExpiringCredit([{ id: "undated" }])?.id).toBe("undated");
	});

	it("falls back to the first credit when none are available (backend surfaces the outcome)", () => {
		const credits = [
			{ id: "first", status: "redeemed" },
			{ id: "second", status: "redeemed" },
		];
		expect(pickSoonestExpiringCredit(credits)?.id).toBe("first");
		expect(pickSoonestExpiringCredit([])).toBeUndefined();
	});
});
