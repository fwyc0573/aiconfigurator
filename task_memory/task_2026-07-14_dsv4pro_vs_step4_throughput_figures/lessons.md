## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Added verified reusable lessons from the four-system throughput-ratio workflow |

# Lessons: DeepSeek-V4-Pro vs Step4 Throughput Figures

## Strict disaggregated TTFT semantics must be explicit

For this workflow, a rank-one candidate is eligible only when corrected disaggregated `TTFT < 5000 ms`. The numeric target alone is insufficient because nearby code paths use both strict and non-strict comparisons. Persist the comparison semantics in the ranking contract and validate the stored `ttft_pass` flag against the raw TTFT value.

## Preserve a single best-configuration identity across derived metrics

The configuration is ranked by `output_token_throughput`. Prefill and decode ratios must then be extracted from that same rank-one row. Independently maximizing prefill throughput would answer a different question and break the approved experiment contract.

## Infeasibility is part of the experiment result

A broad model/system/ISL matrix can contain memory- or SLA-infeasible points. The correct representation is an explicit terminal-status count and a missing-point inventory. Plotting should leave gaps; it must not interpolate or synthesize ratios. Direct complete-ratio validation should remain fail-fast unless the caller explicitly selects the plotting path that records unpaired points.

## Checkpoint contracts protect long-running shards

The GB300 shard required more than two hours of CPU-bound SOL processing. Its SQLite checkpoint and execution-contract hash made durable progress observable and prevented accidental resume under modified runner semantics. Do not edit a hashed runner or replace its checkpoint while a shard is active.

## High CPU with active object processing is not a deadlock

For the GB300 SOL path, near-100% main-thread CPU, increasing CPU time, and short read-only stack samples in pyarrow/Python object processing demonstrated continued computation. This evidence distinguished a slow data-processing path from a deadlock and avoided an unnecessary restart.

## System-subset menus and final menus should be separate

When one shard is still running, derive plotted systems from the supplied completed shards and publish an interim menu separately. After all shards finish, generate a new final menu from all sources. This keeps both views reproducible and prevents an empty line from implying completed data.

## H800 provenance must remain visible

H800 results in this workflow are simulated SOL results, not silicon measurements. Preserve that provenance in the menu, test report, summary, and any downstream interpretation.
