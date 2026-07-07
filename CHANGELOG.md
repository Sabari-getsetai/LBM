# Changelog

## [Unreleased]
### Added
- Project scaffolding: `pyproject.toml`, package layout under `src/lbm`,
  default config, README/TASK/CHANGELOG.
- Vocab module with padding/unknown-token handling and time-gap bucketing.
- YOOCHOOSE parsers: `load_clicks()` and `load_buys()` for reading raw
  `.dat` files into DataFrames with proper schema (session_id, timestamp,
  item_id, category for clicks; adds price, quantity for buys).
