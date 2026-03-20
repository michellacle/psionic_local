# Tassadar Fast-Route Exactness Audit

`TAS-174` is the tranche where the chosen fast article path stops being a
partially direct integration and becomes a bounded no-fallback exact route on
the declared canonical article workload families.

Before this change, the trained
`tassadar-article-transformer-trace-bound-trained-v0` model already owned the
selected `HullCache` fast path at the descriptor and served route boundary, but
the runtime still fell back on the long-loop, Sudoku, and Hungarian article
families. That meant fast-path ownership was real while fast-path exactness and
zero-fallback posture were still not.

After this change:

- the runtime keeps the committed article-class corpus direct on `HullCache`
  while continuing to reject arbitrary unsupported backward-branch programs
- the committed HullCache closure report now records LongLoopKernel,
  SudokuClass, and HungarianMatching as exact with zero fallback rows
- the dedicated fast-route exactness session and hybrid workflow artifacts now
  expose representative long-loop, Sudoku, and Hungarian cases as direct
  `HullCache` executions on the trained Transformer-backed model
- the repo now ships one explicit machine-readable closure artifact at
  `fixtures/tassadar/reports/tassadar_article_fast_route_exactness_report.json`
  plus the paired operator summary at
  `fixtures/tassadar/reports/tassadar_article_fast_route_exactness_summary.json`

This closes only fast-route exactness and no-fallback posture inside the
current canonical article profile. It does not close the remaining
`TAS-175` throughput-floor tranche, and it does not widen the final
article-equivalence claim beyond that bounded statement.
