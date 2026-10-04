# Prewalk

Prewalk hands off from a planning model to a faster or cheaper execution model after planning reaches implementation. By default it runs once; an optional setting restarts it for each new user message. The planning model inspects the repository, creates a todo list, and begins the change before the target model continues.

Prewalk is off by default. Its default target is the model assigned to the `@smol` role.

## Enable prewalk

Enable prewalk persistently in the global config:

```bash
omp config set prewalk.enabled true
```

The equivalent YAML in `~/.omp/agent/config.yml` or a project `.omp/config.yml` is:

```yaml
prewalk:
  enabled: true
```

Session flags override the configured value:

| Flag | Effect |
| --- | --- |
| `--prewalk` | Arm prewalk for the new session. |
| `--no-prewalk` | Leave prewalk disabled for the session, even when `prewalk.enabled` is `true`. |
| `--prewalk-into <model-or-role>` | Arm prewalk and use the supplied model pattern or role instead of `@smol`. |

For example:

```bash
omp --prewalk
omp --prewalk-into @smol
omp --prewalk-into openai/gpt-5-mini
```

At startup, OMP resolves the target with the normal model-role and model-matching rules. If the target cannot be resolved or has no configured credentials, OMP prints a warning and starts with prewalk unarmed.

## Handoff trigger

An armed prewalk injects a planning nudge. When the `todo` tool is active, any successful `todo` call—including the read-only `view` operation—opens the handoff gate. OMP then switches models after the first completed `edit` or `write` call.

Calls to other tools do not trigger the handoff. A read-only `xd://` device request routed through `write`, such as LSP navigation, also does not count; only device operations classified as workspace writes or execution count.

After each handoff, the current prewalk disarms itself. Without automatic restarts, it stays disarmed until explicitly armed again. A target already matching the model and configured thinking level needs no handoff.

## Restart for every user message

In `/settings`, open **Model → Prewalk** and enable **Prewalk Every User Message**:

```yaml
prewalk:
  enabled: true
  afterEveryUserMessage: true
```

Each newly delivered user prompt, steering injection, follow-up, aside or user-authored custom message starts a fresh planning cycle. The original planning model and configured thinking level are restored before the next safe model request. Internal notices and restored history do not start cycles. An Eval cell still running in the background defers restoration until a later safe request boundary; it is never switched mid-cell.

On a resumed session without an existing cycle, the next new user message resolves the planning model from `@default` and the execution target from `@smol`. `--no-prewalk` disables automatic restarts. Disabling **Enable Prewalk** in settings cancels the armed handoff and clears its status without switching the running model.

The status line shows a walking person while a handoff is armed and a standing person after a successful handoff while its execution model remains active. Unicode uses 🚶 and 🧍; Nerd Font and ASCII presets have corresponding alternatives. Other modes remain visible alongside it. Custom status layouts need the `mode` segment to display these icons.

## Arm from an active session

Run either slash command without restarting OMP:

```text
/prewalk
/prewalk restart
```

`/prewalk` arms a one-shot handoff from the active model to the current `@smol` assignment.

After a handoff, `/prewalk restart` immediately returns the session to the current `@default` assignment and re-arms the handoff to `@smol`. Both roles are resolved when the command runs, so the cycle is independent of concrete model names and does not alter either role's persisted configuration.

If prewalk is already armed, the command leaves the existing target in place. To choose a different target at startup, use `--prewalk-into`.

## Subagent prewalk

Task subagents have separate controls: agent frontmatter, `task.prewalk`, per-agent `task.agentPrewalk` overrides, and **Tasks → Subagents → Prewalk Unpinned Subagents** (`task.prewalkWithoutModelOverride`). The explicit per-agent setting overrides frontmatter; `task.prewalk` retains its existing behavior for the bundled task agent.

The unpinned-subagent setting defaults off. When enabled, eligible launches start on `@slow` with its configured thinking level, then hand off to the existing prewalk target or otherwise the model and thinking level the child would normally use. Explicit agent models, configured role choices, per-agent model overrides, launch overrides and extension-selected models are exempt. An unconfigured bundled `@task` default and ordinary parent-model inheritance are not overrides. Explicit per-agent prewalk off stays off; plan-mode launches do not receive this automatic fallback. An unavailable planning model is reported and skipped rather than failing the child launch.
