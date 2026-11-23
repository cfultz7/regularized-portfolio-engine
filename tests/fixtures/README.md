# Test fixtures (SYNTHETIC)

Every file here is a small, hand-written **synthetic** fixture. None contains provider data.

- `french_industries_fixture.csv` mimics the layout of the Kenneth French daily industry file:
  header notes, a value-weighted section, an equal-weighted section (values of 50% that must never
  be selected), a `-99.99` missing sentinel, CRLF line endings, and a 2026 row that must be excluded.
- `french_factors_fixture.csv` mimics the Fama/French daily factors layout (RF in percent).
