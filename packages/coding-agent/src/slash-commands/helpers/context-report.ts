import { computeContextBreakdown, renderCompactionDiagnosticsDetails, renderCompactionDiagnosticsSummary } from "../../modes/utils/context-usage";
import type { SlashCommandRuntime } from "../types";
import { renderAsciiBar } from "./format";

/**
 * Build the `/context` ACP-mode text. Tries the rich breakdown first
 * (categories + auto-compact buffer + free slack) and falls back to the
 * minimal "window/used" lines when the breakdown helper throws.
 */
export function buildContextReportText(runtime: Pick<SlashCommandRuntime, "session">, action: "usage" | "details" = "usage"): string {
	if (action === "details") {
		const current = runtime.session.getCompactionDiagnostics("current");
		const recorded = runtime.session.getCompactionDiagnostics("recorded");
		const prepared = runtime.session.getPreparedCompactionDiagnostics();
		const sections: string[] = [];
		if (current) sections.push(renderCompactionDiagnosticsDetails(current));
		if (recorded) sections.push(renderCompactionDiagnosticsDetails(recorded));
		if (prepared) sections.push(renderCompactionDiagnosticsDetails(prepared));
		return sections.length ? sections.join("\n\n---\n\n") : "Context diagnostics are unavailable. Legacy compactions without recorded facts cannot provide historical settings or source attribution.";
	}
	try {
		const breakdown = computeContextBreakdown(runtime.session, { snapcompactSavings: true });
		if (breakdown.contextWindow <= 0) {
			return "Context usage is unavailable: no model is selected for this session.";
		}
		const usedPct = Math.round((breakdown.usedTokens / breakdown.contextWindow) * 100);
		const lines = [`Context window: ${breakdown.contextWindow} tokens (${usedPct}% used)`];
		for (const category of breakdown.categories) {
			if (category.tokens === 0) continue;
			const fraction = category.tokens / breakdown.contextWindow;
			lines.push(`  ${category.label.padEnd(16)} ${renderAsciiBar(fraction)}  ${category.tokens} tokens`);
		}
		if (breakdown.autoCompactBufferTokens > 0) {
			const fraction = breakdown.autoCompactBufferTokens / breakdown.contextWindow;
			lines.push(
				`  ${"Auto-compact buf".padEnd(16)} ${renderAsciiBar(fraction)}  ${breakdown.autoCompactBufferTokens} tokens`,
			);
		}
		if (breakdown.freeTokens > 0) {
			const fraction = breakdown.freeTokens / breakdown.contextWindow;
			lines.push(`  ${"Free".padEnd(16)} ${renderAsciiBar(fraction)}  ${breakdown.freeTokens} tokens`);
		}
		const snap = breakdown.snapcompact;
		if (snap) {
			if (!snap.visionCapable) {
				lines.push("Snapcompact: inactive (model has no image input)");
			} else {
				lines.push("Snapcompact (estimated wire savings):");
				if (snap.systemPrompt) {
					const sp = snap.systemPrompt;
					lines.push(
						sp.applied
							? `  System prompt: ${sp.textTokens} text tokens → ${sp.frames} frame${sp.frames === 1 ? "" : "s"} ≈ ${sp.imageTokens} tokens (saves ~${sp.savedTokens})`
							: "  System prompt: stays text (no net savings)",
					);
				}
				if (snap.toolResults) {
					const tr = snap.toolResults;
					lines.push(
						tr.swapped > 0
							? `  Tool results: ${tr.swapped} of ${tr.total} imaged, ${tr.textTokens} text tokens → ${tr.frames} frames ≈ ${tr.imageTokens} tokens (saves ~${tr.savedTokens})`
							: `  Tool results: none imaged (${tr.total} in history)`,
					);
				}
				if (snap.savedTokens > 0) {
					lines.push(`  Estimated next request: ~${breakdown.usedTokens - snap.savedTokens} tokens on the wire`);
				}
			}
		}
		if (breakdown.recordedCompaction) lines.push("", "Last compaction (recorded)", renderCompactionDiagnosticsSummary(breakdown.recordedCompaction));
		lines.push("", "Use /context details for the ordered inventory, settings and measurement basis.");
		return lines.join("\n");
	} catch {
		const fallback = runtime.session.getContextUsage();
		if (!fallback) return "Context usage is unavailable.";
		return ["Context", `Window: ${fallback.contextWindow}`, `Used: ${fallback.tokens ?? 0}`].join("\n");
	}
}
