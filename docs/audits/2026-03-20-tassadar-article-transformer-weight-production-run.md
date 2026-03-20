# Tassadar Article Transformer Weight Production Run

`TAS-169` closes the first real trained-weight tranche for the owned
article-Transformer route.

The repo now has one committed trained trace-bound article model under
`fixtures/tassadar/models/`, produced through `psionic-train` rather than
declared only as a structural descriptor. The current production lane is
intentionally bounded:

- it trains only the trace-bound article wrapper already frozen by `TAS-168`
- it uses one explicit `32`-token trace-prefix window from the canonical
  Hungarian article demo as the trained case
- it keeps the kernel-family article cases visible as held-out evidence
- it records checkpoint restore parity and artifact reload parity

That is enough to close "real weights exist" for the owned route.

It is not enough to close:

- full article-class exactness
- reference-linear direct-proof closure
- fast-route closure
- benchmark parity
- contamination or anti-memorization audits
- final article-equivalence green status

The public claim boundary stays narrow on purpose: this audit freezes one
bounded trained artifact and its receipts, not the later stronger claims.
