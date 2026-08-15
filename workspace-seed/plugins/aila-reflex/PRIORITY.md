# Reflex P2 session preemption (opt-in)

P2 preemption makes the **reflex platform AILA's home**, so the MAIN cron wake
runs *on* the reflex adapter and a P2 event can suspend MAIN, run a PRIORITY
session, and let MAIN resume on the next cron cadence. Because this makes reflex
**load-bearing for the main wake loop**, it is **disabled by default** and must be
turned on explicitly.

The shipped default (P4/P5 reflex digest via `pre_llm_call` + `reflex_expand`)
does **not** require any of this.

## Enable

1. **Opt in** — set the plugin env flag so the reflex gateway platform registers:

   ```
   AILA_REFLEX_P2=1
   ```

2. **Home channel** — pick a stable id the reflex platform owns:

   ```
   REFLEX_HOME_CHANNEL=aila-main
   ```

3. **Enable the platform** in the gateway config (connected platform list) so the
   daemon starts the reflex adapter.

4. **Point the wake at reflex** — set the cron `wake` job's delivery target to the
   `reflex` platform in `~/.hermes/cron/jobs.json` (`deliver: reflex`), so MAIN
   runs under the reflex-owned, deterministic `SessionSource`.

## Behavior once enabled

- The reflex adapter watches `EventStore.unseen({P2})`.
- On a P2 event: interrupt MAIN at a turn boundary (summary persisted), spawn a
  single-level PRIORITY session seeded with the event.
- PRIORITY output is routed by modality: mic/audio -> speaker, camera/video ->
  display, filesystem/health -> tooling/logs.
- PRIORITY terminates on normal session end, a termination event (e.g. audio
  keyword), or idle timeout. MAIN resumes on the next cron cadence.

See `aila_v2_reflex_design.md` §11-12 for the full design.
