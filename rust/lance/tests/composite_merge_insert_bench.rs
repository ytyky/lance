// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One-shot wall-clock comparison for indexed composite-key merge_insert.
//! Run with:
//!   cargo test --release -p lance --test composite_merge_insert_bench \
//!       -- --ignored --nocapture

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance::dataset::write::merge_insert::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;

const TARGET_ROWS: i64 = 2_000_000;
const SOURCE_ROWS: i64 = 100;
const ROWS_PER_FRAG: usize = 20_000;

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn make_batch(start: i64, n: i64) -> RecordBatch {
    let a = Int64Array::from_iter_values(start..start + n);
    let b = Int64Array::from_iter_values((start..start + n).map(|i| i * 7 + 3));
    let value = Int64Array::from_iter_values(start..start + n);
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(a), Arc::new(b), Arc::new(value)],
    )
    .unwrap()
}

async fn build_dataset(path: &str) -> Dataset {
    let mut batches = Vec::new();
    let mut start = 0i64;
    while start < TARGET_ROWS {
        let n = (TARGET_ROWS - start).min(ROWS_PER_FRAG as i64);
        batches.push(make_batch(start, n));
        start += n;
    }
    let params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAG,
        max_rows_per_group: ROWS_PER_FRAG,
        mode: WriteMode::Create,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema());
    Dataset::write(reader, path, Some(params)).await.unwrap();

    let mut ds = Dataset::open(path).await.unwrap();
    ds.create_index(
        &["a"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        true,
    )
    .await
    .unwrap();
    ds.create_index(
        &["b"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        true,
    )
    .await
    .unwrap();
    ds
}

fn make_source() -> RecordBatch {
    // Source rows: half are updates (existing IDs), half are inserts (IDs past the end).
    let half = SOURCE_ROWS / 2;
    let updates: Vec<i64> = (0..half).map(|i| i * (TARGET_ROWS / half)).collect();
    let inserts: Vec<i64> = (0..half).map(|i| TARGET_ROWS + i).collect();
    let a_values: Vec<i64> = updates.iter().chain(inserts.iter()).copied().collect();
    let b_values: Vec<i64> = a_values.iter().map(|i| i * 7 + 3).collect();
    let value_values: Vec<i64> = (0..SOURCE_ROWS).map(|i| -i - 1).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(a_values)),
            Arc::new(Int64Array::from(b_values)),
            Arc::new(Int64Array::from(value_values)),
        ],
    )
    .unwrap()
}

#[ignore]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_indexed_composite_merge_insert() {
    let dir = TempStrDir::default();
    let path = dir.as_str().to_string();
    let ds = build_dataset(&path).await;
    let base_version = ds.version().version;

    // Warm the OS page cache + JIT-y bits with one untimed run.
    {
        let mut warmup_ds = Dataset::open(&path).await.unwrap();
        warmup_ds = warmup_ds.checkout_version(base_version).await.unwrap();
        warmup_ds.restore().await.unwrap();
        let src = make_source();
        let reader = RecordBatchIterator::new(std::iter::once(Ok(src.clone())), src.schema());
        MergeInsertBuilder::try_new(
            Arc::new(warmup_ds),
            vec!["a".to_string(), "b".to_string()],
        )
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(reader)
        .await
        .unwrap();
    }

    const ITERS: u32 = 5;
    let mut times = Vec::with_capacity(ITERS as usize);
    for _ in 0..ITERS {
        let mut bench_ds = Dataset::open(&path).await.unwrap();
        bench_ds = bench_ds.checkout_version(base_version).await.unwrap();
        bench_ds.restore().await.unwrap();
        let src = make_source();
        let reader = RecordBatchIterator::new(std::iter::once(Ok(src.clone())), src.schema());

        let t0 = Instant::now();
        MergeInsertBuilder::try_new(
            Arc::new(bench_ds),
            vec!["a".to_string(), "b".to_string()],
        )
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(reader)
        .await
        .unwrap();
        times.push(t0.elapsed());
    }

    times.sort();
    let min = times.first().unwrap();
    let median = times[times.len() / 2];
    let max = times.last().unwrap();
    println!(
        "\n=== composite_merge_insert: target={} rows, source={} rows, iters={} ===",
        TARGET_ROWS, SOURCE_ROWS, ITERS
    );
    println!("  min    : {:?}", min);
    println!("  median : {:?}", median);
    println!("  max    : {:?}", max);
    println!("  all    : {:?}", times);
}
