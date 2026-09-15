import type { CompactionDiagnostics } from "@oh-my-pi/pi-agent-core/compaction/diagnostics";
import { type Component, Ellipsis, matchesKey, ScrollView, Text } from "@oh-my-pi/pi-tui";
import type { KeybindingsManager } from "../../config/keybindings";
import { theme } from "../theme/theme";
import { renderCompactionDiagnosticsDetails } from "../utils/context-usage";
import { OverlayPanel, PanelDivider } from "./overlay-box";

interface ContextDetailsHost {
	readonly terminal: { readonly rows: number };
	requestRender(): void;
}

/** Read-only inventory: switching snapshots never replays hooks or rerenders archive frames. */
export class ContextDetailsOverlay implements Component {
	readonly #panel = new OverlayPanel("Context Details");
	readonly #body: Text;
	readonly #footer = new Text("", 0, 0);
	readonly #scroll: ScrollView;
	readonly #snapshots: CompactionDiagnostics[];
	#index = 0;
	#width: number | undefined;
	#height: number | undefined;

	constructor(
		private readonly host: ContextDetailsHost,
		private readonly keybindings: KeybindingsManager,
		current: CompactionDiagnostics | undefined,
		recorded: CompactionDiagnostics | undefined,
		private readonly onClose: () => void,
		prepared?: CompactionDiagnostics,
	) {
		this.#snapshots = [current, recorded, prepared].filter((value): value is CompactionDiagnostics => value !== undefined);
		this.#body = new Text(this.#snapshotText(), 0, 0);
		this.#scroll = new ScrollView([], {
			height: 1,
			scrollbar: "auto",
			ellipsis: Ellipsis.Omit,
			theme: { track: text => theme.fg("dim", text), thumb: text => theme.fg("accent", text) },
		});
		this.#footer.setStyleFn(text => theme.fg("dim", text));
		this.#panel.addChild(this.#scroll);
		this.#panel.addChild(new PanelDivider());
		this.#panel.addChild(this.#footer);
		this.#updateFooter();
	}

	#snapshotText(): string {
		const snapshot = this.#snapshots[this.#index];
		return snapshot
			? renderCompactionDiagnosticsDetails(snapshot)
			: "Context diagnostics are unavailable. Legacy compactions without recorded facts cannot provide historical settings or source attribution.";
	}

	handleInput(data: string): void {
		const keys = this.keybindings;
		if (keys.matches(data, "tui.select.cancel")) {
			this.onClose();
			return;
		}
		if (keys.matches(data, "tui.select.confirm") && this.#snapshots.length > 1) {
			this.#index = (this.#index + 1) % this.#snapshots.length;
			this.#body.setText(this.#snapshotText());
			this.#updateFooter();
			this.#width = undefined;
			this.#scroll.scrollToTop();
		} else if (keys.matches(data, "tui.select.up")) this.#scroll.scroll(-1);
		else if (keys.matches(data, "tui.select.down")) this.#scroll.scroll(1);
		else if (keys.matches(data, "tui.select.pageUp")) this.#scroll.page(-1);
		else if (keys.matches(data, "tui.select.pageDown")) this.#scroll.page(1);
		else if (matchesKey(data, "home")) this.#scroll.scrollToTop();
		else if (matchesKey(data, "end")) this.#scroll.scrollToBottom();
		this.host.requestRender();
	}

	invalidate(): void {
		this.#width = undefined;
		this.#height = undefined;
		this.#panel.invalidate();
		this.#updateFooter();
		this.#body.invalidate();
	}

	dispose(): void {
		this.#panel.dispose();
	}

	#updateFooter(): void {
		const keys = this.keybindings;
		const hints = [
			keys.getDisplayString("tui.select.up") + "/" + keys.getDisplayString("tui.select.down") + " scroll",
			keys.getDisplayString("tui.select.pageUp") + "/" + keys.getDisplayString("tui.select.pageDown") + " page",
		];
		if (this.#snapshots.length > 1) hints.push(keys.getDisplayString("tui.select.confirm") + " next snapshot (" + (this.#index + 1) + "/" + this.#snapshots.length + ")");
		hints.push(keys.getDisplayString("tui.select.cancel") + " close");
		this.#footer.setText(hints.join(" · "));
	}

	render(width: number): readonly string[] {
		const innerWidth = Math.max(1, width - 4);
		const height = Math.max(1, this.host.terminal.rows - 3 - this.#footer.render(innerWidth).length);
		if (this.#width !== width) {
			// Reserve the scrollbar column so exact-width source labels are never clipped.
			this.#scroll.setLines(this.#body.render(Math.max(1, innerWidth - 1)));
			this.#width = width;
		}
		if (this.#height !== height) {
			this.#scroll.setHeight(height);
			this.#height = height;
		}
		return this.#panel.render(width);
	}
}
