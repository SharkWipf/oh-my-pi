import { describe, expect, it } from "bun:test";
import {
	PreservedMessageIndex,
	type PolicyDirection,
	type PolicyKind,
	type PolicyLimit,
	type PolicyRange,
	type PolicySlot,
} from "../src/session/preserved-message-index";

function user(raw: number, overrides: Partial<PolicySlot> = {}): PolicySlot {
	return {
		user: true,
		raw,
		candidate: raw,
		eligible: true,
		always: false,
		manual: false,
		nonUserCount: 0,
		...overrides,
	};
}

function price(
	slot: PolicySlot,
	position: number,
	kind: PolicyKind,
	hard: number,
	automatic: boolean,
): number | undefined {
	if (!slot.user) return kind === "always" && slot.nonUserCount > 0 ? slot.raw : undefined;
	if (!automatic) return kind === "always" && slot.manual ? slot.raw : undefined;
	if (kind === "always" && !slot.always && !slot.manual) return undefined;
	if (kind === "raw" || position >= hard || slot.manual) return slot.raw;
	return slot.eligible && !Number.isNaN(slot.candidate) ? slot.candidate : undefined;
}

// Deliberately plain source walk: no blocks, cumulative searches, or production predicates.
function oracle(
	slots: PolicySlot[],
	limit: PolicyLimit,
	direction: PolicyDirection,
	kind: PolicyKind,
	hard: number,
	automatic: boolean,
) {
	const selected: number[] = [];
	let count = 0;
	let tokens = 0;
	let blocker: number | undefined;
	if (limit.mode === "off") return { selected, count, tokens, blocker };
	for (let n = 0; n < slots.length; n++) {
		const position = direction === "first" ? n : slots.length - 1 - n;
		const slot = slots[position];
		const cost = price(slot, position, kind, hard, automatic);
		if (cost === undefined) continue;
		const weight = slot.user ? 1 : slot.nonUserCount;
		if (
			(limit.mode === "messages" && count + weight > limit.value) ||
			(limit.mode === "tokens" && tokens + cost > limit.value)
		) {
			blocker = position;
			break;
		}
		selected.push(position);
		count += weight;
		tokens += cost;
	}
	selected.sort((a, b) => a - b);
	return { selected, count, tokens, blocker };
}

describe("PreservedMessageIndex", () => {
	it("stops asymmetric suffixes at the first nonfit and retains zero-price edge ties", () => {
		const index = new PreservedMessageIndex();
		for (const cost of [6, 6]) index.append(user(cost));
		expect(index.query({ mode: "tokens", value: 10 }, "recent", "raw")).toEqual({
			start: 1,
			end: 2,
			count: 1,
			tokens: 6,
			blocker: 0,
		});
		index.truncate(0);
		for (const cost of [0, 0, 3, 0, 0]) index.append(user(cost, { always: true }));
		for (const kind of ["raw", "eligible", "always"] as const) {
			const first = index.query({ mode: "tokens", value: 0 }, "first", kind);
			const recent = index.query({ mode: "tokens", value: 0 }, "recent", kind);
			expect([...index.iterate(first, kind)]).toEqual([0, 1]);
			expect([...index.iterate(recent, kind)]).toEqual([3, 4]);
			expect(first.blocker).toBe(2);
			expect(recent.blocker).toBe(2);
		}
		index.update(2, user(0));
		expect(index.query({ mode: "tokens", value: 0 }, "recent", "raw").count).toBe(5);
		expect(index.query({ mode: "off" }, "first", "always").count).toBe(0);
	});

	it("crosses full zero-price blocks but never crosses their positive blocker", () => {
		const index = new PreservedMessageIndex();
		for (let i = 0; i < 12289; i++) index.append(user(i === 8192 ? 1 : 0, { manual: true }));
		for (const direction of ["first", "recent"] as const) {
			const selected = index.query({ mode: "tokens", value: 0 }, direction, "always", 0, false);
			expect(selected.count).toBe(direction === "first" ? 8192 : 4096);
			expect(selected.blocker).toBe(8192);
			expect(index.includes(8192, selected, "always", 0, false)).toBe(false);
		}
		index.update(8192, user(0, { manual: true }));
		expect(index.query({ mode: "tokens", value: 0 }, "recent", "always", 0, false).count).toBe(12289);
	});

	it("applies virtual H to raw Exclude/Never users without changing stored policy", () => {
		const index = new PreservedMessageIndex();
		index.append(user(9, { candidate: NaN, always: true }));
		index.append(user(8, { candidate: 1, eligible: false }));
		index.append(user(7, { candidate: NaN, eligible: false, manual: true }));
		const all = { mode: "all" } as const;
		expect(index.query(all, "first", "eligible")).toMatchObject({ count: 1, tokens: 7 });
		expect(index.query(all, "first", "eligible", 0)).toMatchObject({ count: 3, tokens: 24 });
		expect(index.query(all, "first", "always", 0)).toMatchObject({ count: 2, tokens: 16 });
		expect(index.query(all, "first", "always", 1)).toMatchObject({ count: 1, tokens: 7 });
		expect(index.query(all, "first", "eligible")).toMatchObject({ count: 1, tokens: 7 });
		expect(index.query(all, "first", "raw", 0, false).count).toBe(0);
		expect(index.query(all, "first", "always", 0, false)).toMatchObject({ count: 1, tokens: 7 });
	});

	it("charges one complete N atom by distinct source count, never splitting it", () => {
		const index = new PreservedMessageIndex();
		index.append(user(12, { user: false, nonUserCount: 3 }));
		index.append(user(0, { user: false }));
		index.append(user(0, { user: false }));
		index.append(user(1, { manual: true }));
		const denied = index.query({ mode: "messages", value: 2 }, "first", "always");
		expect(denied).toMatchObject({ count: 0, tokens: 0, blocker: 0 });
		const admitted = index.query({ mode: "messages", value: 3 }, "first", "always");
		expect([...index.iterate(admitted, "always")]).toEqual([0]);
		expect(admitted).toMatchObject({ count: 3, tokens: 12, blocker: 3 });
		expect(index.query({ mode: "all" }, "first", "always", 0, false)).toMatchObject({ count: 4, tokens: 13 });
		expect(index.measureUnion([{ range: admitted, kind: "always" }])).toEqual({ count: 0, tokens: 0 });
		expect(index.measureUnion([{ range: admitted, kind: "always" }], index.length, true, false)).toEqual({
			count: 3,
			tokens: 12,
		});
	});

	it("matches an independent oracle across blocks, deltas, rollback, reappend and overlapping unions", () => {
		let seed = 18731;
		const random = (max: number) => {
			seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
			return seed % max;
		};
		const nextSlot = (): PolicySlot => {
			const raw = random(17);
			return user(raw, {
				user: random(3) !== 0,
				candidate: random(5) === 0 ? NaN : random(19),
				eligible: random(4) !== 0,
				always: random(3) === 0,
				manual: random(7) === 0,
				nonUserCount: random(8) === 0 ? random(4) + 1 : 0,
			});
		};
		const index = new PreservedMessageIndex();
		const slots: PolicySlot[] = [];
		const append = (count: number) => {
			for (let i = 0; i < count; i++) {
				const slot = nextSlot();
				slots.push(slot);
				index.append(slot);
			}
		};
		const check = () => {
			expect(index.length).toBe(slots.length);
			for (let trial = 0; trial < 90; trial++) {
				const kind: PolicyKind = (["raw", "eligible", "always"] as const)[trial % 3];
				const direction = trial % 2 === 0 ? "first" : "recent";
				const hard = [0, slots.length, 4096, 4101, random(slots.length + 1)][trial % 5];
				const automatic = trial % 7 !== 0;
				const limit: PolicyLimit =
					trial % 4 === 0
						? { mode: "all" }
						: trial % 4 === 1
							? { mode: "off" }
							: { mode: trial % 4 === 2 ? "tokens" : "messages", value: random(slots.length * 5) };
				const expected = oracle(slots, limit, direction, kind, hard, automatic);
				const actual = index.query(limit, direction, kind, hard, automatic);
				expect({ count: actual.count, tokens: actual.tokens, blocker: actual.blocker }).toEqual({
					count: expected.count,
					tokens: expected.tokens,
					blocker: expected.blocker,
				});
				expect([...index.iterate(actual, kind, hard, automatic)]).toEqual(expected.selected);
				for (const position of [
					0,
					4095,
					4096,
					actual.start - 1,
					actual.start,
					actual.end - 1,
					actual.end,
					slots.length,
				]) {
					expect(index.includes(position, actual, kind, hard, automatic)).toBe(
						expected.selected.includes(position),
					);
				}
			}
			for (const automatic of [false, true]) {
				const hardRange = index.query({ mode: "messages", value: 2000 }, "recent", "raw", index.length, automatic);
				const hard = hardRange.start;
				const selections: { range: PolicyRange; kind: PolicyKind }[] = [
					{ range: hardRange, kind: "raw" },
					{
						range: index.query({ mode: "messages", value: 4000 }, "first", "eligible", hard, automatic),
						kind: "eligible",
					},
					{
						range: index.query({ mode: "messages", value: 3000 }, "recent", "eligible", hard, automatic),
						kind: "eligible",
					},
					{ range: index.query({ mode: "all" }, "first", "always", hard, automatic), kind: "always" },
				];
				for (const userOnly of [false, true]) {
					let count = 0;
					let tokens = 0;
					for (let position = 0; position < slots.length; position++) {
						const slot = slots[position];
						if (userOnly && !slot.user) continue;
						for (const selection of selections) {
							if (position < selection.range.start || position >= selection.range.end) continue;
							const cost = price(slot, position, selection.kind, hard, automatic);
							if (cost === undefined) continue;
							count += slot.user ? 1 : slot.nonUserCount;
							tokens += cost;
							break;
						}
					}
					expect(index.measureUnion(selections, hard, automatic, userOnly)).toEqual({ count, tokens });
				}
			}
		};
		append(12319);
		check();
		for (let i = 0; i < 1500; i++) {
			const position = random(slots.length);
			slots[position] = nextSlot();
			index.update(position, slots[position]);
		}
		check();
		for (const length of [8192, 4111, 4096, 0]) {
			index.truncate(length);
			slots.length = length;
			append(9000);
			check();
		}
	});
});
