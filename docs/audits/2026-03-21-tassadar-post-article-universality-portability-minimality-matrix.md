# TAS-193 Audit

`TAS-193` closes the post-article universality portability/minimality matrix
tranche for the rebased canonical route.

The canonical artifact is now
`fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_report.json`.
It extends the post-`TAS-186` rebased universality lane across one declared CPU
machine matrix, one explicit route-carrier classification, and one
machine-level minimality contract instead of inheriting bounded universality
from adjacent green artifacts by implication.

The report keeps three machine cells explicit: two declared CPU classes that
carry the rebased universality lane and one out-of-envelope CPU class that
stays explicitly suppressed. It also classifies the selected direct HullCache
route as the only route inside the universality carrier while keeping the
linear recurrent and research fast routes outside that carrier.

The operator-readable summary now lives at
`fixtures/tassadar/reports/tassadar_post_article_universality_portability_minimality_matrix_summary.json`,
the provider receipt now lives at
`crates/psionic-provider/src/tassadar_post_article_universality_portability_minimality_matrix.rs`,
the served conformance envelope now lives at
`fixtures/tassadar/reports/tassadar_post_article_universality_served_conformance_envelope.json`,
and the dedicated checker now lives at
`scripts/check-tassadar-post-article-universality-portability-minimality-matrix.sh`.

This tranche is still narrower than the later rebased verdict split and plugin
boundary work. It does not yet publish the rebased theory/operator/served
verdict split, admit served/public universality, admit weighted plugin
control, or admit arbitrary software capability.
