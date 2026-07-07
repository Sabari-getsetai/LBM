# Changelog

## [Unreleased]
### Added
- Project scaffolding: `pyproject.toml`, package layout under `src/lbm`,
  default config, README/TASK/CHANGELOG.
- Vocab module with padding/unknown-token handling and time-gap bucketing.
- YOOCHOOSE parsers: `load_clicks()` and `load_buys()` for reading raw
  `.dat` files into DataFrames with proper schema (session_id, timestamp,
  item_id, category for clicks; adds price, quantity for buys).
- Session event streams: `build_session_events()` combines clicks+buys into
  chronological per-session events with action_type (view/buy).
- Session filtering and outcome labels: `filter_sessions()` drops short
  sessions and truncates long ones; `session_outcomes()` extracts sessions
  with purchases.
- Tokenize sessions: `SessionRecord` dataclass and `tokenize_sessions()`
  function to encode event DataFrames into integer-id SessionRecord instances
  ready for embedding lookups in downstream model/dataset/baseline.
