# Why `merge_insert` OOMs on large datasets — root-cause analysis and fix plan

Analysis of out-of-memory failures when running `merge_insert` between two large datasets
(hundreds of GB – TB scale, ~50k rows, fat rows of 0.1–2+ MB each: nested lidar/image binary
columns), joining on two key columns with a partial-schema (subschema) source.

All file/line references are at commit `f24e42c` (9.0.0-beta.15).

## TL;DR

* The OOM is **not caused by using multiple join keys**. Key count only influences *which
  execution path* the planner picks. Two of the three paths fully **materialize either the
  source or large parts of the target in memory**; the third streams and is memory-safe.
* Your single-key merges most likely didn't OOM because they took the memory-safe path
  (full-schema source, no scalar index on the key → streaming v2 plan). A single-key merge
  **can absolutely OOM the same way** if the key has a scalar index (v1 indexed path) or if
  the source is a partial schema (v2 partial-schema plan).
* Two independent root causes, both triggered by your workload shape:
  1. **v2 "fast path" with a partial-schema source** collects the *target table's missing
     columns for every row* into the hash-join build side, using an **unbounded memory
     pool** (`TaskContext::default()`). For a table whose payload columns are hundreds of
     GB, this is a guaranteed kernel OOM regardless of how few rows you update.
  2. **v1 indexed-scan path** (taken when *every* join column has a scalar index — enabled
     for composite keys by PR #6878) materializes the **entire source twice**: once in an
     unbounded, unaccounted `ReplayExec` buffer and once in the `CollectLeft` hash-join
     build side. Downstream, the partial-update writer buffers **all updates of a fragment**
     in memory at once (fragments default to up to 1M rows / 90 GB).
* Short-term: chunk the source into bounded pieces, or use a full-schema source with
  `use_index(false)`. Long-term fix plan is at the bottom (P0: stop routing payload columns
  through join build sides; give v2 a real memory pool; make `ReplayExec` spillable).

---

## 1. How a merge_insert is routed

`execute_uncommitted_impl` (`rust/lance/src/dataset/write/merge_insert.rs:1921`) picks one
of three main paths:

```
can_use_create_plan()?  (merge_insert.rs:1815)
│
├─ YES → v2 planned path (create_plan → FullSchemaMergeInsertExec / DeleteOnlyMergeInsertExec)
│        Chosen when the source schema is the full schema OR a subset of it, AND the join
│        would NOT use a scalar index (merge_insert.rs:1872-1885: only when every `on`
│        column has an exact-equality scalar index does it route away from v2).
│
└─ NO  → v1 legacy path (create_joined_stream, merge_insert.rs:1052)
         ├─ every join column indexed & WhenNotMatchedBySource::Keep
         │    → create_indexed_scan_joined_stream (merge_insert.rs:786)   [v1 indexed]
         └─ otherwise
              → create_full_table_joined_stream (merge_insert.rs:991)     [v1 full-scan join]
         then:
         ├─ full-schema source → Merger → rewrite matched rows as new fragments
         └─ partial-schema source → update_fragments (merge_insert.rs:1086) column patch
```

Key consequence: **before PR #6878**, a two-key join could never satisfy "every join column
indexed", so it always went to the v2 path. **After #6878**, if both key columns have BTree
indices, the same call is routed to the v1 indexed path. The two paths fail differently
(see §2 and §3), but both are memory-unsafe for fat-row datasets.

## 2. Root cause A — v2 path with a partial-schema source collects the target in RAM

The v2 plan is built in `create_plan` (`merge_insert.rs:1628`):

* `scan_aliased.join(source_df_aliased, …)` puts the **target scan on the left**, and the
  physical plan uses `HashJoinExec: mode=CollectLeft` — i.e. the **left/target side is
  fully collected into an in-memory hash table** before the source is streamed through.
  (See plan snapshot in `test_plan_upsert`, `merge_insert.rs:6522-6533`.)
* For a **full-schema** source this is fine: projection pushdown reduces the target scan to
  `[join keys, _rowid, _rowaddr]` (the snapshot shows `LanceRead … projection=[key]`).
  50k keys ≈ a few MB. This is why your ordinary single-key upserts never OOMed.
* For a **partial-schema** source, `create_plan` fills every dataset column missing from the
  source with `col("target.<col>")` (`merge_insert.rs:1693-1700`). Those columns must flow
  out of the join, so the target-side `LanceRead` now projects **all missing payload
  columns** — confirmed by `test_merge_insert_subcols_v2_explain_plan`
  (`merge_insert.rs:5611-5619`: "the target side of the join reads the `other` column").
  With `CollectLeft`, that means **every heavy column not present in your source (images,
  other lidar fields, …) for all 50k rows is loaded into the hash-join build side** —
  essentially the whole physical dataset, hundreds of GB.
* The plan is executed with `TaskContext::default()` (`merge_insert.rs:1752`), i.e.
  DataFusion's **unbounded memory pool, no spilling**. Nothing stops the build side from
  growing until the kernel kills the process. This matches "easily get OOM even when
  updating only 30-40%" — the collected build side is *the whole target*, independent of
  how many rows match.

Additional cost of this path: updated rows are rewritten as **full rows** into new fragments
(`FullSchemaMergeInsertExec` → `write_fragments_internal`, RewriteRows), so a partial update
also re-reads and re-writes all untouched heavy columns of every matched row.

## 3. Root cause B — v1 indexed path materializes the source twice (and more)

`create_indexed_scan_joined_stream` (`merge_insert.rs:786`) — the path your composite-key
merge takes when both key columns are indexed (post-#6878; single indexed keys have always
taken it):

1. **`ReplayExec::new(Capacity::Unbounded, input)`** (`merge_insert.rs:809`). The comment is
   explicit: *"this needs to have unbounded capacity, and so we need to fully read the new
   data into memory"*. The source is forked to (a) the index-probe side and (b) the join;
   because a hash join drains one side completely before the other, the replay buffer ends
   up holding **the entire source** — all columns, not just keys — in an ordinary
   `VecDeque` that is **not registered with any DataFusion memory pool** (unaccounted RSS;
   `rust/lance/src/io/exec/utils.rs:142`, `rust/lance-core/src/utils/futures.rs`).
2. **`HashJoinExec … PartitionMode::CollectLeft` with the source as the left side**
   (`merge_insert.rs:935-948`). The **whole source is collected a second time** into the
   join build hash table. This side *is* pool-accounted, and the pool here is
   `FairSpillPool(100 MB)` by default (`execute_plan` with `use_spilling: true`;
   `LanceExecutionOptions::mem_pool_size`, `rust/lance-datafusion/src/exec.rs:313` —
   100 MB/partition, overridable via `LANCE_MEM_POOL_SIZE`). Hash joins cannot spill, so:
   * with defaults, any source larger than ~100 MB should abort with a
     `Resources exhausted` error rather than OOM;
   * if `LANCE_MEM_POOL_SIZE` was raised (the natural reaction to that error), you get
     source×2 in RAM (replay + hash table) plus everything below → kernel OOM.
3. **`TakeExec`** materializes matched target rows (at your row sizes, a single 8192-row
   batch is ~8-16 GB; batches are only byte-capped later, in the sort before
   `update_fragments`).
4. **`update_fragments`** (`merge_insert.rs:1086`) — the partial-schema writer:
   * sorts by `_rowaddr` (spillable, batches capped at 25 MB — fine), but then
   * `BatchStreamGrouper` (`rust/lance-datafusion/src/dataframe.rs:68`) collects **all
     update batches of one fragment into a `Vec<RecordBatch>`** before the write task
     starts. Default write params are 1M rows / 90 GB per file (`write.rs:417-421`), so a
     bulk-written 50 GB dataset can be a *single fragment* — the whole update set for it is
     buffered at once.
   * The memory reservation for the group is taken *after* the group is already in memory,
     and is **silently bypassed** when no other task is running
     (`merge_insert.rs:1430-1440`: "If there are no tasks running, we can bypass the pool
     limits"). `interleave_batches` then makes another copy while merging old and new rows.

## 4. Answers to your specific questions

**Why did this happen?** Your job hits one of the two materializing paths above. Which one
depends on whether both join columns have scalar indices:

| Your setup | Path taken | Failure mode |
|---|---|---|
| Both keys indexed (post-#6878) | v1 indexed | `Resources exhausted` error at ~100 MB source (default pool), or kernel OOM from source×2 + per-fragment buffering if the pool was raised |
| Not all keys indexed | v2 partial-schema | Kernel OOM: entire target's missing columns collected into the join build with an unbounded pool |

A quick way to tell which one you hit: run `job.explain_plan()` / check the log line
"Executing plan:" — the v1 indexed plan contains `Replay`, `MapIndexExec`, `Take`; the v2
plan starts with `MergeInsert: on=[…]`. Also, a Rust/Python *error* mentioning
"Resources exhausted" = v1; a killed process = v2 (or v1 with a raised pool).

**Is this specific to multi-key joins?** No. Key count is only a routing input:

* Single key **with** a scalar index → v1 indexed path → same double-source
  materialization. This predates #6878; you'd hit it with one key and a big source.
* Single key, partial-schema source, no index → v2 partial-schema → same target-collect
  OOM as the two-key case.
* Single key, **full-schema** source, no index → v2 full-schema → build side is keys+ids
  only, source streams; memory-safe. This is almost certainly the configuration in which
  you "didn't see merge_insert cause OOM".

What #6878 changed for you is that a two-key join with both columns indexed now lands on
the v1 indexed path (previously impossible), and that path is the most memory-hungry per
byte of *source*. The v2 partial-schema hazard (per byte of *target*) existed before and
after.

**Why fat rows make everything worse.** Most operators size batches by row count (e.g. 8192
rows). At 1 MB+/row a "normal" batch is 8+ GB. Only the sort in `update_fragments` byte-caps
batches (25 MB). Every un-capped stage (source scanning, take, join output) can allocate
multi-GB batches, so even the "streaming" parts have very high peaks with this data shape.

## 5. Immediate workarounds (no code changes)

1. **Chunk the source.** Run N merge_inserts, each with a bounded slice of the source
   (e.g. 1–2 GB per run, filtered by key range). Bounds both the replay/build materialization
   (v1) and the row-rewrite working set. Cost: N commits/versions and N target key scans.
   With indexed keys (v1 path) the per-run target work is only the index probe + take, so
   this is cheap and is the most practical mitigation today.
2. **Avoid the partial-schema v2 path for TB-scale tables.** If you cannot chunk, provide a
   *full-schema* source (include all columns, even unchanged ones, e.g. by scanning the
   target and attaching the new column) and set `use_index(false)`: that forces the
   streaming v2 full-schema plan whose build side is keys-only. Cost: full row rewrite of
   matched rows (heavy write amplification, but bounded memory).
3. **If you stay on the v1 indexed path**, size for it: RAM ≥ ~2.5× source bytes and
   `LANCE_MEM_POOL_SIZE` ≥ source size (plus per-fragment update sets for the write stage).
   Only viable when the source slice is much smaller than RAM — which is really workaround 1.
4. For this migration pattern specifically (derive a new sub-field for an existing column
   across most rows), a `add_columns`/column-patch style flow (schema evolution + rewriting
   only the affected column fragment-by-fragment) is fundamentally cheaper than a join —
   your manual delete+append workaround approximated this. `merge_insert` should eventually
   handle it, per the plan below.

## 6. Fix plan

### P0-1: v2 partial-schema — stop routing target payload columns through the join build
`create_plan` (`merge_insert.rs:1693-1700`). Join only on
`[keys, _rowid, _rowaddr, sentinel]`; fetch the missing columns *after* the join,
streaming, via a take-by-`_rowaddr` (the `TakeExec` operator already exists and the custom
`MergeInsertPlanner` can place it), or fetch them inside the write exec per output batch.
Build side returns to keys+ids regardless of source schema. This alone fixes the guaranteed
OOM for wide/heavy targets.
*Even better long-term:* route v2 partial-schema updates to a column-patch write (like v1
`update_fragments`) instead of full-row rewrite — removes both the memory issue and the
read/write amplification of untouched heavy columns.

### P0-2: v2 — execute with a real memory pool
`execute_uncommitted_v2` (`merge_insert.rs:1752`) uses `TaskContext::default()` (unbounded
pool, no disk manager). Use the shared Lance session context
(`get_session_context(LanceExecutionOptions { use_spilling: true, … })`) so operators are
accounted and fail gracefully (or spill where supported) instead of taking the process down.

### P0-3: v1 indexed path — eliminate double source materialization
`create_indexed_scan_joined_stream` (`merge_insert.rs:804-948`):
* Make the replay buffer **spill-backed**: the machinery already exists
  (`create_replay_spill` used by `new_source_iter` for retries, `write.rs:1782-1809`) —
  cap in-memory replay at N MB, spill the rest to temp disk.
* Feed the index probe from a **keys-only projection** so whichever copy must be buffered
  is narrow, and/or restructure the join so the build side is the keys+rowaddr side rather
  than the full-width source.
* Register whatever remains buffered with the memory pool so the failure mode is a clear
  error, not RSS growth.

### P1-4: `update_fragments` — bound per-fragment buffering
`merge_insert.rs:1132-1459` + `BatchStreamGrouper` (`dataframe.rs:68`). The input is already
sorted by `_rowaddr`, so each fragment's updates can be **streamed into the fragment
updater incrementally** instead of collected into a `Vec<RecordBatch>`. Remove (or at least
account/log) the silent pool-limit bypass at `merge_insert.rs:1435-1439`; reserve memory
*before* buffering, not after.

### P1-5: byte-capped batching for fat rows
`HardCapBatchSizeExec` exists (`merge_insert.rs:1105-1130`) but is only applied before the
sort. Apply a byte cap at the merge-insert source boundary and after take/join so no stage
sees multi-GB batches. (A reasonable default: 32–64 MB.)

### P2-6: statistics + join strategy
`LanceScanExec::partition_statistics` (`rust/lance/src/io/exec/scan.rs:730`) reports row
counts but no meaningful byte sizes for heavy binary columns, so DataFusion's join-selection
cannot know the build side is enormous. Report (approximate) byte-size statistics from
fragment/data-file metadata, and consider partitioned or spillable join strategies for
large builds.

### Test plan for the fixes
* Extend the existing plan-snapshot tests (`test_plan_upsert`,
  `test_merge_insert_subcols_v2_explain_plan`) to assert the target-side `LanceRead`
  projection stays keys+ids for partial-schema sources (P0-1) and that a memory-limited
  context is used (P0-2).
* Add a memory-bounded integration test: dataset with a wide binary column (e.g. 2k rows ×
  1 MB), partial-schema merge_insert under a small `LANCE_MEM_POOL_SIZE`, assert success
  with bounded peak (v1 indexed and v2 variants; single and composite keys, per the
  multi-fragment testing standard).
* Benchmark in `rust/lance/benches/merge_insert.rs` with a fat-row config to catch
  regressions in both time and (via `MemoryPool` metrics) peak reservation.
