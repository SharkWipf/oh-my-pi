import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import { getProjectAgentDir, TempDir } from "@oh-my-pi/pi-utils";
import { YAML } from "bun";
import { Settings } from "../src/config/settings";
import {
	compilePreservedUserMessageRegexRules,
	decodeCompactionMessageOverride,
	decodePreservedUserMessageClassifications,
	migrateLegacyCompactionPin,
	packPreservedUserMessageClassifications,
	parsePreservationLimit,
	readPreservationPolicySettings,
	serializePreservationLimit,
	unpackPreservedUserMessageClassifications,
	validatePreservedUserMessageRegexCondition,
} from "../src/session/preserved-message-settings";
import { beginSettingsTest, restoreSettingsTestState, type SettingsTestState } from "./helpers/settings-test-state";

describe("preservation limits and durable codecs", () => {
	it("distinguishes disabled, unbounded, and finite zero without accepting invalid canonical counts", () => {
		expect(parsePreservationLimit("off")).toEqual({ mode: "off" });
		expect(parsePreservationLimit("all")).toEqual({ mode: "all" });
		expect(parsePreservationLimit("tokens:0")).toEqual({ mode: "tokens", value: 0 });
		expect(parsePreservationLimit("context-percent:0")).toEqual({ mode: "context-percent", value: 0 });
		expect(parsePreservationLimit("messages:0")).toBeUndefined();
		expect(parsePreservationLimit("messages:1.5")).toBeUndefined();
		expect(parsePreservationLimit("tokens:9007199254740992")).toBeUndefined();
		expect(parsePreservationLimit("context-percent:100.1")).toBeUndefined();
		expect(parsePreservationLimit("tokens:NaN")).toBeUndefined();
		expect(serializePreservationLimit({ mode: "tokens", value: Number.MAX_SAFE_INTEGER })).toBe("tokens:9007199254740991");
		expect(serializePreservationLimit({ mode: "context-percent", value: 12.5 })).toBe("context-percent:12.5");
	});

	it("preserves successful all-false and eleven-bit tags but rejects the entire malformed record", () => {
		const tags = [{ id: "zero", mask: 0 }, { id: "all", mask: 2047 }];
		expect(unpackPreservedUserMessageClassifications(packPreservedUserMessageClassifications(tags))).toEqual(tags);
		expect(unpackPreservedUserMessageClassifications({ v: 1, c: ["valid", 1, "invalid", 4096] })).toEqual([]);
		expect(decodePreservedUserMessageClassifications({ v: 2, c: ["valid", 1] }).status).toBe("unsupported");
		expect(decodePreservedUserMessageClassifications({ v: 1, c: ["valid"] }).status).toBe("malformed");
		expect(decodePreservedUserMessageClassifications({ version: 1, preservedIds: ["old"], classifiedIds: ["old"] }).status).toBe("malformed");
		expect(() => packPreservedUserMessageClassifications([{ id: "", mask: 0 }])).toThrow();
	});

	it("decodes manual array and singular records independently of binary classifier history", () => {
		expect(decodeCompactionMessageOverride({ messageId: "source", state: "exclude" })).toEqual({ messageIds: ["source"], state: "exclude" });
		expect(decodeCompactionMessageOverride({ messageIds: ["source", "source"], state: "auto" })).toEqual({ messageIds: ["source"], state: "auto" });
		expect(migrateLegacyCompactionPin({ messageId: "source", pinned: false })).toEqual({ messageIds: ["source"], state: "auto" });
		expect(migrateLegacyCompactionPin({ messageId: "source" })).toEqual({ messageIds: ["source"], state: "keep" });
		expect(decodeCompactionMessageOverride({ messageIds: ["source", 2], state: "keep" })).toBeUndefined();
		expect(migrateLegacyCompactionPin({ preservedIds: ["source"], classifiedIds: ["source"] })).toBeUndefined();
	});

	it("retains legacy case-insensitive regex actions and Final while rejecting unsupported RE2 syntax", () => {
		const rules = compilePreservedUserMessageRegexRules({ "hello": "keep", "bye": { state: "exclude", caseInsensitive: false, final: true }, "(?<=x)y": "keep", "broken": { state: "keep", final: "true" } });
		expect(rules.map(rule => [rule.pattern.test(rule.condition.toUpperCase()), rule.action, rule.final])).toEqual([[true, "keep", false], [false, "exclude", true]]);
		expect(() => validatePreservedUserMessageRegexCondition("(?<=x)y")).toThrow();
	});
});

describe("effective layered preservation migration", () => {
	let state: SettingsTestState;
	let temp: TempDir;
	let agentDir: string;
	let cwd: string;
	beforeEach(() => {
		state = beginSettingsTest();
		temp = TempDir.createSync("@pi-preservation-settings-");
		agentDir = temp.join("agent");
		cwd = temp.join("project");
		fs.mkdirSync(agentDir, { recursive: true });
		fs.mkdirSync(getProjectAgentDir(cwd), { recursive: true });
	});
	afterEach(async () => {
		restoreSettingsTestState(state);
		await temp.remove();
	});

	it("normalizes legacy zero pairs only after global/project/overlay/runtime precedence", async () => {
		const globalPath = `${agentDir}/config.yml`;
		const projectPath = `${getProjectAgentDir(cwd)}/config.yml`;
		const globalText = YAML.stringify({ compaction: { keepFirstNMessages: 0, keepLastNMessages: 0 } });
		await Bun.write(globalPath, globalText);
		await Bun.write(projectPath, YAML.stringify({ compaction: { keepLastNMessages: 1 } }));
		const settings = await Settings.loadReadOnly({ cwd, agentDir });
		expect(settings.get("compaction.keepFirstLimit")).toBe("off");
		expect(settings.get("compaction.keepLastLimit")).toBe("messages:1");
		settings.override("compaction.keepLastLimit", "messages:0");
		expect(settings.get("compaction.keepFirstLimit")).toBe("all");
		expect(settings.get("compaction.keepLastLimit")).toBe("all");
		settings.clearOverride("compaction.keepLastLimit");
		expect(settings.get("compaction.keepFirstLimit")).toBe("off");
		expect(await Bun.file(globalPath).text()).toBe(globalText);
	});

	it("composes legacy percentage intent across layers but canonical fields win within their layer", async () => {
		await Bun.write(`${agentDir}/config.yml`, YAML.stringify({ compaction: { keepFirstNMessages: 8, keepFirstMessagesPercent: 25 } }));
		await Bun.write(`${getProjectAgentDir(cwd)}/config.yml`, YAML.stringify({ compaction: { keepFirstNMessages: 2 } }));
		const overlay = temp.join("overlay.yml");
		await Bun.write(overlay, YAML.stringify({ compaction: { keepFirstLimit: "tokens:0", keepFirstMessagesPercent: 90, keepRecentUserMessages: 0 } }));
		const legacy = await Settings.loadReadOnly({ cwd, agentDir });
		expect(legacy.get("compaction.keepFirstLimit")).toBe("context-percent:25");
		const canonical = await Settings.loadReadOnly({ cwd, agentDir, configFiles: [overlay] });
		expect(canonical.get("compaction.keepFirstLimit")).toBe("tokens:0");
		expect(canonical.get("compaction.keepRecentUserMessagesLimit")).toBe("off");
	});

	it("keeps percentage zero finite, negative percentage falls through, and invalid legacy numbers become tokens zero", async () => {
		const settings = Settings.isolated({
			"compaction.keepFirstNMessages": 9,
			"compaction.keepFirstMessagesPercent": 0,
			"compaction.keepLastNMessages": 3,
			"compaction.keepLastMessagesPercent": -1,
			"compaction.keepRecentUserMessages": Infinity,
		} as Parameters<typeof Settings.isolated>[0]);
		expect(settings.get("compaction.keepFirstLimit")).toBe("context-percent:0");
		expect(settings.get("compaction.keepLastLimit")).toBe("messages:3");
		expect(settings.get("compaction.keepRecentUserMessagesLimit")).toBe("tokens:0");
		const clone = await settings.cloneForCwd(cwd);
		expect(readPreservationPolicySettings(clone).first).toEqual({ mode: "context-percent", value: 0 });
	});

	it("keeps new canonical independent defaults and stored policy active with automatic selection off", () => {
		const settings = Settings.isolated({ "compaction.keepFirstLimit": "messages:3", "compaction.keepUserMessages": false });
		const policy = readPreservationPolicySettings(settings);
		expect(policy.recent).toEqual({ mode: "all" });
		expect(policy.enabled).toBe(false);
		expect(policy.classifier).toBe(true);
		expect(policy.categoryActions.longTermRule).toBe("keep");
	});
});
