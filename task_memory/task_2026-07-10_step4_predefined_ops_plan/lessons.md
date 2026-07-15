## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Added vetted lessons from full-contract sharding, strict resume, and final semantic validation |
| 2026-07-10 | Created lessons placeholder for Step4 predefined ops planning |
| 2026-07-12 | Added vetted source-precedence and precision-realization lessons |
| 2026-07-13 | Added verified lesson separating runtime acceptance from simulator fidelity |
| 2026-07-13 | Added independently reviewed lesson for cap-saturated search results |
| 2026-07-13 | Added vetted lessons on materialized search contracts and threshold-sensitive empirical corrections |

# Lessons: Step4 Predefined Ops Planning

1. A concrete sharding YAML and an architecture calculator can both be authoritative for different concerns: record an explicit source-of-truth matrix rather than selecting one file globally.
2. An omitted YAML field plus a code default proves the current simulator realization, but not necessarily the intended model architecture. Numeric architecture formulas can outweigh omission-derived defaults and require explicit adjudication.
3. Precision declarations and simulator metadata realization are separate facts. A plumbing gap must not be reinterpreted as a model-precision decision; effective dtype/weight bytes should be tested at the roofline query boundary.
4. For hybrid attention/FFN models, validating marginal totals is insufficient. Tests should lock the cross-composition counts so an incorrect pairing cannot pass: here `4 dense+SWA`, `23 MoE+full`, and `65 MoE+SWA`.
5. DS-V4-specific measured modules should not be reused for Step4 roofline-only modeling because they encode DS-V4 architecture and measured-data assumptions.
6. A parallel configuration being accepted by an enumerator or runtime does not prove simulator fidelity. Uneven expert placement must be represented in the roofline model; floor-dividing `num_experts / EP` can hide the latency-critical heavier ranks even when the serving runtime supports them.
7. A search result that reaches its configured batch cap is a censored bound, not a proven optimum. Only the saturated cap should be expanded, expansion must repeat until a non-saturated or terminal result is observed, and cap-saturated rows must stay out of final ranking.
8. Candidate lists and total counts do not fully specify a deployment search. A reproducible contract must lock materialized rows, cross-field equalities, role-prefixed fields, experiment partitioning, worker/replica semantics, memory adjudication, and topology tiers; otherwise a valid-looking Cartesian product can encode invalid runtime meanings.
9. Equal empirical correction factors do not guarantee a fair model comparison when SLA thresholds or mode selection are involved. For a strict analytic/roofline study, neutralize every downstream empirical factor that changes latency, throughput, feasibility, or ranking, and isolate any calibrated-serving experiment under a separate name.
10. A filtered execution contract and a composable shard contract answer different identity questions. When strict merge validates one full input manifest, every shard must be executed against that same full contract while retaining its own exact mode-spec subset; never rewrite or normalize incompatible shard headers during merge.
11. Formula-only provenance must distinguish an executed operation from an explicit no-op. An exact zero-latency `not_executed` placeholder is valid only when its operation name and control-flow condition prove no kernel ran; every executed operation must still carry direct `sol` evidence, and validators must reject any profiling or guessed source.
12. A strict resume claim requires more than a successful return value. Use fail-on-call loader/executor hooks, verify the exact expected record count, and compare every durable checkpoint/final-artifact hash before and after resume.
13. Cross-model summaries must keep unpaired feasibility outcomes rather than imputing missing values. Aggregate numeric means only over paired keys, report one-model-only counts separately, preserve metric polarity, and exclude undefined TPOT zero-baseline relatives from relative means.
14. Artifact validators should derive expectations from declared dataclasses and serialization schemas rather than guessing field names or normalized empty representations. A validator defect must be diagnosed and rerun across the complete contract; it must never be treated as evidence that the immutable artifact is wrong without a field-level mismatch.
