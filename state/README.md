Auto-managed by the workflows: `history.json` (every video + metrics), `strategy.json` (learned bandit posteriors),
`analytics_summary.json` (rolling stats), and `runway_usage.json` (Runway Dev UTC-day reserved-credit ledger when
that optional provider is enabled). Committed back to the repo after each run. Don't edit by hand.

The Runway ledger stores conservative reservations before any paid task is submitted, so a timeout or failed task
still consumes the local daily cap. Per-video sidecars hold the generation output's task/model/prompt/hash/cost and
rights-confirmation provenance. The ledger is an additional guard, not a substitute for setting a spend limit in the
Runway project/account.
