import { afterEach, beforeEach, expect, it, spyOn } from "bun:test";
import type { Component } from "@oh-my-pi/pi-tui";
import { Settings, resetSettingsForTest } from "../src/config/settings";
import { RewindSelectorComponent } from "../src/modes/components/rewind-selector";
import { SelectorController } from "../src/modes/controllers/selector-controller";
import { initTheme, theme } from "../src/modes/theme/theme";
import type { InteractiveModeContext } from "../src/modes/types";
import { SessionManager } from "../src/session/session-manager";

beforeEach(async () => {
	await Settings.init({ inMemory: true, cwd: process.cwd() });
	await initTheme();
});
afterEach(() => resetSettingsForTest());

it("reopens at the current source tail after navigation, append, branch and same-ID rewrite", async () => {
	const manager = SessionManager.inMemory(process.cwd());
	const rootId = manager.appendMessage({ role: "user", content: "Root request", timestamp: 1 });
	manager.appendMessage({ role: "user", content: "Original tail", timestamp: 2 });
	let mounted: RewindSelectorComponent | undefined;
	let painted = false;
	const getEntry = manager.getEntry.bind(manager);
	const readSource = spyOn(manager, "getEntry").mockImplementation(id => {
		if (!painted) throw new Error("Rewind read source before its first frame");
		return getEntry(id);
	});
	const eagerBranch = spyOn(manager, "getBranch").mockImplementation(() => {
		throw new Error("Rewind eagerly materialized the branch");
	});
	const displayed = new Set<RewindSelectorComponent>();
	const session = { getToolByName: () => undefined, hasBuiltInTool: () => true };
	const controller = new SelectorController({
		sessionManager: manager,
		session,
		viewSession: session,
		effectiveHideThinkingBlock: false,
		proseOnlyThinking: true,
		ui: {
			terminal: { rows: 24, columns: 100 },
			showOverlay(component: Component) {
				if (!(component instanceof RewindSelectorComponent)) throw new Error("Expected actual rewind selector");
				mounted = component;
				displayed.add(component);
				return { hide() { mounted = undefined; } };
			},
			setFocus() {},
			requestRender() {},
			renderNow() {
				mounted?.render(100);
				painted = true;
			},
			requestComponentRender() {},
		},
		editor: {},
		editorContainer: { children: [] },
		showStatus(message: string) { throw new Error(message); },
	} as unknown as InteractiveModeContext);
	const open = async () => {
		controller.showUserMessageSelector();
		if (!mounted) throw new Error("Rewind overlay did not mount");
		const selector = mounted;
		await selector.ready;
		return selector;
	};
	const outlined = (selector: RewindSelectorComponent) => selector.render(100)
		.map(row => Bun.stripANSI(row))
		.filter(row => row.includes(theme.boxDotted.vertical))
		.join("\n");
	try {
		controller.showUserMessageSelector();
		if (!mounted) throw new Error("Rewind overlay did not mount");
		const cancelled = mounted;
		expect(cancelled.isLoading).toBe(true);
		expect(readSource).not.toHaveBeenCalled();
		cancelled.handleInput("\u001b");
		expect(mounted).toBeUndefined();
		await cancelled.ready;
		expect(readSource).not.toHaveBeenCalled();
		painted = false;

		const first = await open();
		expect(outlined(first)).toContain("Original tail");
		first.handleInput("\u001b[A");
		expect(outlined(first)).toContain("Root request");
		first.handleInput("\u001b");

		const reopened = await open();
		expect(outlined(reopened)).toContain("Original tail");
		reopened.handleInput("\u001b");
		manager.appendMessage({ role: "user", content: "Appended request", timestamp: 3 });
		const appended = await open();
		expect(outlined(appended)).toContain("Appended request");
		expect(outlined(appended)).not.toContain("Original tail");

		// A source mutation while open retires the old overlay, not just its next-open cache.
		manager.branch(rootId);
		expect(mounted).toBeUndefined();
		const branched = await open();
		expect(outlined(branched)).toContain("Root request");
		expect(outlined(branched)).not.toContain("Appended request");
		branched.handleInput("\u001b");

		// Same entry ID and branch length: source identity alone cannot detect this rewrite.
		const root = manager.getEntry(rootId);
		if (root?.type !== "message" || root.message.role !== "user") throw new Error("Missing source user entry");
		root.message = { ...root.message, content: "Rewritten request" };
		await manager.rewriteEntries();
		const rewritten = await open();
		expect(outlined(rewritten)).toContain("Rewritten request");
		expect(outlined(rewritten)).not.toContain("Root request");
		rewritten.handleInput("\u001b");
	} finally {
		readSource.mockRestore();
		eagerBranch.mockRestore();
		for (const selector of displayed) selector.dispose();
		manager.releaseRetainedEntries();
	}
});
