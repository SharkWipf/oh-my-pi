/**
 * Show provider usage limits for every authenticated account.
 */

import { Args, Command, Flags } from "@oh-my-pi/pi-utils/cli";
import { usageHelp as commandHelp } from "../cli/command-help";
import { runUsageCommand } from "../cli/usage-cli";

export default class Usage extends Command {
	static description = commandHelp.description;
	static args = {
		action: Args.string({
			description: "Optional subcommand to execute",
			required: false,
			options: ["invalidate", "clients", "policy-capability", "reset-capability", "redeem-reset"],
		}),
	};

	static flags = {
		json: Flags.boolean({ char: "j", description: "Output usage reports as JSON", default: false }),
		provider: Flags.string({ char: "p", description: "Only show usage for this provider id (e.g. anthropic)" }),
		"account-id": Flags.string({ description: "Exact Codex account id for explicit redeem-reset" }),
		"credit-id": Flags.string({ description: "Explicitly confirmed saved-reset grant id for redeem-reset" }),
		"request-id": Flags.string({ description: "Durable Codex reset consume idempotency key" }),
		redact: Flags.boolean({
			char: "r",
			description: "Redact account emails/ids (shortest unique prefix) for sharing screenshots",
			default: false,
		}),
		history: Flags.boolean({
			description: "Show recorded usage-limit history (hourly snapshots) instead of a live snapshot",
			default: false,
		}),
		"reset-credits": Flags.boolean({
			description: "Fetch live Codex saved resets and full reset history (normal usage only)",
			default: false,
		}),
		days: Flags.integer({ char: "d", description: "History window in days (with --history or clients)", default: 7 }),
	};

	static examples = [
		"# Detailed per-account usage breakdown across all providers\n  omp usage",
		"# Only Anthropic accounts\n  omp usage --provider anthropic",
		"# Redact account identifiers for screenshots\n  omp usage --redact",
		"# Machine-readable output\n  omp usage --json",
		"# Live Codex saved resets and complete reset history as JSON\n  omp usage --provider openai-codex --reset-credits --json",
		"# Offline headless reset capability (does not load credentials or redeem)\n  omp usage reset-capability --json",
		"# Explicitly redeem one confirmed saved reset for one exact Codex account\n  omp usage redeem-reset --provider openai-codex --account-id ACTUAL_ACCOUNT_ID --credit-id CONFIRMED_GRANT_ID --request-id ACTION_ID",
		"# Usage-limit trend over the last 30 days\n  omp usage --history --days 30",
		"# Per-client token burn (which machine/app spent what) over the last 30 days\n  omp usage clients --days 30",
		"# Invalidate cached usage reports for all providers\n  omp usage invalidate",
		"# Invalidate cached usage reports for a specific provider\n  omp usage invalidate --provider anthropic",
	];

	async run(): Promise<void> {
		const { args, flags } = await this.parse(Usage);
		await runUsageCommand({
			action: args.action,
			json: flags.json,
			provider: flags.provider,
			accountId: flags["account-id"],
			creditId: flags["credit-id"],
			requestId: flags["request-id"],
			redact: flags.redact,
			history: flags.history,
			resetCredits: flags["reset-credits"],
			days: flags.days,
		});
	}
}
