// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, RecordBatch, UInt8Array, UInt32Array, UInt64Array};
use arrow_schema::Schema;
use arrow_select;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::{
    execution::{SendableRecordBatchStream, TaskContext},
    physical_plan::{
        DisplayAs, ExecutionPlan, PlanProperties,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::{StreamExt, stream};
use lance_core::{Error, ROW_ADDR, ROW_ID};
use lance_datafusion::projection::ProjectionPlan;
use lance_table::format::RowIdMeta;
use roaring::RoaringTreemap;

use crate::dataset::transaction::UpdateMode::RewriteRows;
use crate::dataset::utils::CapturedRowIds;
use crate::dataset::write::merge_insert::inserted_rows::{
    KeyExistenceFilter, KeyExistenceFilterBuilder, extract_key_value_from_batch,
};
use crate::dataset::write::merge_insert::{
    MERGE_SOURCE_SENTINEL, SourceDedupeBehavior, create_duplicate_row_error,
    format_key_values_on_columns, resolve_target_bases,
};
use crate::dataset::{ProjectionRequest, TakeBuilder};
use crate::{
    Dataset,
    dataset::{
        transaction::{Operation, Transaction},
        write::{
            WriteParams, cleanup_data_fragments,
            merge_insert::{
                MERGE_ACTION_COLUMN, MergeInsertParams, MergeStats, assign_action::Action,
                exec::MergeInsertMetrics,
            },
            write_fragments_internal,
        },
    },
};

use super::apply_deletions;

/// Shared state for merge insert operations to simplify lock management
struct MergeState {
    /// Row addresses that need to be deleted, due to a row update or delete action
    delete_row_addrs: RoaringTreemap,
    /// Shared collection to capture row ids that need to be updated
    updating_row_ids: Arc<Mutex<CapturedRowIds>>,
    /// Track keys of newly inserted rows (not updates).
    inserted_rows_filter: KeyExistenceFilterBuilder,
    /// Merge operation metrics
    metrics: MergeInsertMetrics,
    /// Whether the dataset uses stable row ids.
    stable_row_ids: bool,
    /// Set to track processed row IDs to detect duplicates
    processed_row_ids: HashSet<u64>,
    /// The "on" column names for merge operation
    on_columns: Vec<String>,
    /// How to handle duplicate source rows
    source_dedupe_behavior: SourceDedupeBehavior,
}

impl MergeState {
    fn new(
        metrics: MergeInsertMetrics,
        stable_row_ids: bool,
        on_columns: Vec<String>,
        field_ids: Vec<i32>,
        source_dedupe_behavior: SourceDedupeBehavior,
    ) -> Self {
        Self {
            delete_row_addrs: RoaringTreemap::new(),
            updating_row_ids: Arc::new(Mutex::new(CapturedRowIds::new(stable_row_ids))),
            inserted_rows_filter: KeyExistenceFilterBuilder::new(field_ids),
            metrics,
            stable_row_ids,
            processed_row_ids: HashSet::new(),
            on_columns,
            source_dedupe_behavior,
        }
    }

    /// Process a single row based on its action, updating internal state
    fn process_row_action(
        &mut self,
        action: Action,
        row_idx: usize,
        row_addr_array: &UInt64Array,
        row_id_array: &UInt64Array,
        batch: &RecordBatch,
    ) -> DFResult<Option<usize>> {
        match action {
            Action::Delete => {
                // Delete action - only delete, don't write back
                if !row_addr_array.is_null(row_idx) {
                    let row_addr = row_addr_array.value(row_idx);
                    let row_id = row_id_array.value(row_idx);

                    // A source with duplicate keys matches the same target row
                    // more than once; apply the same dedupe policy as updates.
                    // (Target-only deletes from `delete_not_matched_by_source`
                    // also reach here but never duplicate, so they never trip
                    // `Fail`.)
                    if !self.processed_row_ids.insert(row_id) {
                        match self.source_dedupe_behavior {
                            SourceDedupeBehavior::Fail => {
                                return Err(create_duplicate_row_error(
                                    batch,
                                    row_idx,
                                    &self.on_columns,
                                ));
                            }
                            SourceDedupeBehavior::FirstSeen => {
                                self.metrics.num_skipped_duplicates.add(1);
                                return Ok(None); // Skip this duplicate row
                            }
                        }
                    }

                    self.delete_row_addrs.insert(row_addr);
                    self.metrics.num_deleted_rows.add(1);
                }
                Ok(None) // Don't keep this row
            }
            Action::UpdateAll => {
                // Update action - delete old row AND insert new data
                if !row_addr_array.is_null(row_idx) {
                    let row_addr = row_addr_array.value(row_idx);
                    let row_id = row_id_array.value(row_idx);

                    // Check for duplicate _rowid in the current merge operation
                    if !self.processed_row_ids.insert(row_id) {
                        match self.source_dedupe_behavior {
                            SourceDedupeBehavior::Fail => {
                                return Err(create_duplicate_row_error(
                                    batch,
                                    row_idx,
                                    &self.on_columns,
                                ));
                            }
                            SourceDedupeBehavior::FirstSeen => {
                                self.metrics.num_skipped_duplicates.add(1);
                                return Ok(None); // Skip this duplicate row
                            }
                        }
                    }

                    self.delete_row_addrs.insert(row_addr);

                    if self.stable_row_ids {
                        self.updating_row_ids.lock().unwrap().capture(&[row_id])?;
                    }
                    // Don't count as actual delete - this is an update
                }

                self.metrics.num_updated_rows.add(1);
                Ok(Some(row_idx)) // Keep this row for writing
            }
            Action::Insert => {
                // Insert action - just insert new data
                // Capture the key value for conflict detection (only for inserts, not updates)
                if let Some(key_value) =
                    extract_key_value_from_batch(batch, row_idx, &self.on_columns)
                {
                    self.inserted_rows_filter
                        .insert(key_value)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                self.metrics.num_inserted_rows.add(1);
                Ok(Some(row_idx)) // Keep this row for writing
            }
            Action::Nothing => {
                // Do nothing action - keep the row but don't count it
                Ok(None)
            }
            Action::Fail => {
                // Fail action - return an error to fail the operation
                Err(datafusion::error::DataFusionError::Execution(format!(
                    "Merge insert failed: found matching row with key values: {}",
                    format_key_values_on_columns(batch, row_idx, &self.on_columns)
                )))
            }
        }
    }
}

/// Where each output column of the write stream comes from, in dataset
/// schema order.
#[derive(Debug, Clone, Copy)]
enum WriteColumnSource {
    /// The column is present in the exec's input (provided by the source).
    Input(usize),
    /// The column is missing from the source (partial-schema upsert) and is
    /// fetched from the target by row address; index into
    /// [`TargetColumnFiller::fields`].
    FillFromTarget(usize),
}

/// Fetches target-side values for dataset columns that a partial-schema
/// source does not provide.
///
/// The values are fetched one batch at a time, by the `_rowaddr` of the
/// matched target rows, so peak memory is bounded by the batch size. This
/// deliberately replaces the earlier approach of copying these columns
/// through the join in the logical plan: there the target scan fed the
/// build side of a `CollectLeft` hash join, so every payload column of
/// every target row was collected into memory at once, which caused OOM on
/// wide tables.
#[derive(Debug)]
struct TargetColumnFiller {
    dataset: Arc<Dataset>,
    projection: Arc<ProjectionPlan>,
    /// The filled fields, matching the projection. Nullable because
    /// inserted rows (no matched target row) receive NULL.
    fields: Vec<arrow_schema::FieldRef>,
}

impl TargetColumnFiller {
    fn try_new(dataset: Arc<Dataset>, fields: Vec<arrow_schema::FieldRef>) -> DFResult<Self> {
        let names = fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>();
        let projection = ProjectionRequest::from_columns(names, dataset.schema())
            .into_projection_plan(dataset.clone())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Self {
            dataset,
            projection: Arc::new(projection),
            fields,
        })
    }

    /// Returns one array per filled field, aligned with `row_addrs`: rows
    /// with a valid row address (updates) get the target's current value,
    /// rows with a NULL row address (inserts) get NULL.
    async fn fetch(&self, row_addrs: &UInt64Array) -> DFResult<Vec<ArrayRef>> {
        let valid_addrs: Vec<u64> = (0..row_addrs.len())
            .filter(|&i| row_addrs.is_valid(i))
            .map(|i| row_addrs.value(i))
            .collect();

        if valid_addrs.is_empty() {
            return Ok(self
                .fields
                .iter()
                .map(|field| arrow_array::new_null_array(field.data_type(), row_addrs.len()))
                .collect());
        }

        let mut sorted_addrs = valid_addrs;
        sorted_addrs.sort_unstable();
        sorted_addrs.dedup();

        let taken = TakeBuilder::try_new_from_addresses(
            self.dataset.clone(),
            sorted_addrs,
            self.projection.clone(),
        )
        .map_err(|e| DataFusionError::External(Box::new(e)))?
        .with_row_address(true)
        .execute()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Map each returned row address back to its position in the taken
        // batch so we do not depend on the take preserving request order.
        let taken_addrs = taken
            .column_by_name(ROW_ADDR)
            .ok_or_else(|| {
                DataFusionError::Internal("take result is missing the _rowaddr column".to_string())
            })?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal("Expected UInt64Array for _rowaddr column".to_string())
            })?;
        let addr_to_taken_idx: HashMap<u64, u32> = taken_addrs
            .values()
            .iter()
            .enumerate()
            .map(|(idx, addr)| (*addr, idx as u32))
            .collect();

        let indices =
            (0..row_addrs.len())
                .map(|i| {
                    if row_addrs.is_valid(i) {
                        let addr = row_addrs.value(i);
                        addr_to_taken_idx.get(&addr).copied().map(Some).ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "Row address {} matched by the merge join was not returned by take",
                            addr
                        ))
                    })
                    } else {
                        Ok(None)
                    }
                })
                .collect::<DFResult<UInt32Array>>()?;

        self.fields
            .iter()
            .map(|field| {
                let column = taken.column_by_name(field.name()).ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "take result is missing the {:?} column",
                        field.name()
                    ))
                })?;
                arrow_select::take::take(column, &indices, None).map_err(DataFusionError::from)
            })
            .collect()
    }
}

/// Precomputed layout of the write stream: where the control columns live
/// in the input, how to assemble output columns in dataset schema order,
/// and (for partial-schema sources) how to fill the missing columns.
#[derive(Debug)]
struct WriteStreamContext {
    rowaddr_idx: usize,
    rowid_idx: usize,
    action_idx: usize,
    column_sources: Vec<WriteColumnSource>,
    filler: Option<TargetColumnFiller>,
    output_schema: Arc<Schema>,
}

/// Inserts new rows and updates existing rows in the target table.
///
/// This does the actual write.
///
/// This is implemented by moving updated rows to new fragments. This mode
/// is most optimal when updating the full schema.
///
#[derive(Debug)]
pub struct FullSchemaMergeInsertExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    params: MergeInsertParams,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    merge_stats: Arc<Mutex<Option<MergeStats>>>,
    transaction: Arc<Mutex<Option<Transaction>>>,
    affected_rows: Arc<Mutex<Option<RoaringTreemap>>>,
    inserted_rows_filter: Arc<Mutex<Option<KeyExistenceFilter>>>,
    /// Whether the ON columns match the schema's unenforced primary key.
    /// If true, inserted_rows_filter will be included in the transaction for conflict detection.
    is_primary_key: bool,
}

impl FullSchemaMergeInsertExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        params: MergeInsertParams,
    ) -> DFResult<Self> {
        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(empty_schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));

        // Check if ON columns match the schema's unenforced primary key
        let field_ids: Vec<i32> = params
            .on
            .iter()
            .filter_map(|name| dataset.schema().field(name).map(|f| f.id))
            .collect();
        let pk_field_ids: Vec<i32> = dataset
            .schema()
            .unenforced_primary_key()
            .iter()
            .map(|f| f.id)
            .collect();
        let is_primary_key = !pk_field_ids.is_empty() && field_ids == pk_field_ids;

        Ok(Self {
            input,
            dataset,
            params,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            merge_stats: Arc::new(Mutex::new(None)),
            transaction: Arc::new(Mutex::new(None)),
            affected_rows: Arc::new(Mutex::new(None)),
            inserted_rows_filter: Arc::new(Mutex::new(None)),
            is_primary_key,
        })
    }

    /// Takes the merge statistics if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn merge_stats(&self) -> Option<MergeStats> {
        self.merge_stats
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Takes the transaction if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn transaction(&self) -> Option<Transaction> {
        self.transaction
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Returns the filter for inserted row keys if the execution has completed.
    /// This contains keys of newly inserted rows (not updates) for conflict detection.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn inserted_rows_filter(&self) -> Option<KeyExistenceFilter> {
        self.inserted_rows_filter
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    /// Takes the affected rows (deleted/updated row addresses) if the execution has completed.
    /// Returns `None` if the execution is still in progress or hasn't started.
    pub fn affected_rows(&self) -> Option<RoaringTreemap> {
        self.affected_rows
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Creates a filtered stream that captures row addresses for deletion and returns
    /// a stream with only the source data columns (no _rowaddr or __action columns)
    fn create_filtered_write_stream(
        &self,
        input_stream: SendableRecordBatchStream,
        merge_state: Arc<Mutex<MergeState>>,
    ) -> DFResult<SendableRecordBatchStream> {
        let enable_stable_row_ids = {
            let state = merge_state.lock().map_err(|e| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Failed to lock merge state: {}",
                    e
                ))
            })?;
            state.stable_row_ids
        };

        if enable_stable_row_ids {
            self.create_ordered_update_insert_stream(input_stream, merge_state)
        } else {
            self.create_streaming_write_stream(input_stream, merge_state)
        }
    }

    /// High-performance streaming implementation for non-stable row ID scenarios
    ///
    /// It processes batches one at a time as they arrive from the input stream,
    /// immediately filtering and transforming each batch without buffering.
    fn create_streaming_write_stream(
        &self,
        input_stream: SendableRecordBatchStream,
        merge_state: Arc<Mutex<MergeState>>,
    ) -> DFResult<SendableRecordBatchStream> {
        let ctx = self.prepare_stream_schema(input_stream.schema())?;

        let output_schema = ctx.output_schema.clone();
        let stream = input_stream.then(move |batch_result| {
            let ctx = ctx.clone();
            let merge_state = merge_state.clone();
            async move {
                let batch = batch_result?;
                let (row_addr_array, row_id_array, action_array) = Self::extract_control_arrays(
                    &batch,
                    ctx.rowaddr_idx,
                    ctx.rowid_idx,
                    ctx.action_idx,
                )?;

                // Process each row using the shared state
                let mut keep_rows: Vec<u32> = Vec::with_capacity(batch.num_rows());

                {
                    let mut merge_state = merge_state.lock().map_err(|e| {
                        datafusion::error::DataFusionError::Internal(format!(
                            "Failed to lock merge state: {}",
                            e
                        ))
                    })?;

                    for row_idx in 0..batch.num_rows() {
                        let action_code = action_array.value(row_idx);
                        let action = Action::try_from(action_code).map_err(|e| {
                            datafusion::error::DataFusionError::Internal(format!(
                                "Invalid action code {}: {}",
                                action_code, e
                            ))
                        })?;

                        if merge_state
                            .process_row_action(
                                action,
                                row_idx,
                                row_addr_array,
                                row_id_array,
                                &batch,
                            )?
                            .is_some()
                        {
                            keep_rows.push(row_idx as u32);
                        }
                    }
                }

                Self::create_write_batch(&batch, keep_rows, &ctx).await
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }

    /// Creates an ordered update-insert stream ensuring updated data before inserted data.
    ///
    /// 1. Separating the input stream into update and insert streams
    /// 2. Using chain operations to guarantee all update batches are processed before any insert batches
    /// 3. Returning the combined ordered stream
    fn create_ordered_update_insert_stream(
        &self,
        input_stream: SendableRecordBatchStream,
        merge_state: Arc<Mutex<MergeState>>,
    ) -> DFResult<SendableRecordBatchStream> {
        let (update_stream, insert_stream) =
            self.split_updates_and_inserts(input_stream, merge_state)?;

        let output_schema = update_stream.schema();

        // Chain the update and insert streams to ensure order
        let combined_stream = update_stream.chain(insert_stream);

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            combined_stream,
        )))
    }

    /// Common schema preparation logic
    fn prepare_stream_schema(
        &self,
        input_schema: arrow_schema::SchemaRef,
    ) -> DFResult<Arc<WriteStreamContext>> {
        // Find column indices
        let (rowaddr_idx, _) = input_schema.column_with_name(ROW_ADDR).ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "Expected _rowaddr column in merge insert input".to_string(),
            )
        })?;

        let (rowid_idx, _) = input_schema.column_with_name(ROW_ID).ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "Expected _rowid column in merge insert input".to_string(),
            )
        })?;

        let (action_idx, _) = input_schema
            .column_with_name(MERGE_ACTION_COLUMN)
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Expected {} column in merge insert input",
                    MERGE_ACTION_COLUMN
                ))
            })?;

        // Emit data columns in dataset-schema order, keyed by name. Columns
        // present in the input come from the source; dataset columns that a
        // partial-schema source omits are fetched from the target per batch
        // (see [`TargetColumnFiller`]) rather than routed through the join.
        // Name-based lookup is also a strictly-safer choice for the
        // full-schema path: it turns an implicit positional assumption into
        // an explicit name-based invariant.
        let mut name_to_idx: HashMap<&str, usize> =
            HashMap::with_capacity(input_schema.fields().len());
        for (idx, field) in input_schema.fields().iter().enumerate() {
            let name = field.name();
            // Skip special columns: _rowaddr, _rowid, __action, and the
            // source-presence sentinel.
            if idx == rowaddr_idx
                || idx == rowid_idx
                || idx == action_idx
                || name == ROW_ADDR
                || name == ROW_ID
                || name == MERGE_ACTION_COLUMN
                || name == MERGE_SOURCE_SENTINEL
            {
                continue;
            }
            name_to_idx.insert(name.as_str(), idx);
        }

        let dataset_arrow_schema: arrow_schema::Schema = self.dataset.schema().into();
        let dataset_fields = dataset_arrow_schema.fields();
        let mut column_sources: Vec<WriteColumnSource> = Vec::with_capacity(dataset_fields.len());
        let mut fill_fields: Vec<arrow_schema::FieldRef> = Vec::new();
        let mut output_fields: Vec<Arc<arrow_schema::Field>> =
            Vec::with_capacity(dataset_fields.len());
        let mut num_input_columns = 0;
        for dataset_field in dataset_fields {
            match name_to_idx.get(dataset_field.name().as_str()) {
                Some(&idx) => {
                    column_sources.push(WriteColumnSource::Input(idx));
                    output_fields.push(Arc::new(input_schema.field(idx).clone()));
                    num_input_columns += 1;
                }
                None => {
                    // Nullable regardless of the dataset field: inserted rows
                    // have no matched target row and receive NULL (the writer
                    // schema is what the commit validates against).
                    let field = Arc::new(dataset_field.as_ref().clone().with_nullable(true));
                    column_sources.push(WriteColumnSource::FillFromTarget(fill_fields.len()));
                    fill_fields.push(field.clone());
                    output_fields.push(field);
                }
            }
        }

        if num_input_columns == 0 {
            return Err(datafusion::error::DataFusionError::Internal(
                "No data columns found in merge insert input".to_string(),
            ));
        }

        let filler = if fill_fields.is_empty() {
            None
        } else {
            Some(TargetColumnFiller::try_new(
                self.dataset.clone(),
                fill_fields,
            )?)
        };

        let output_schema = Arc::new(Schema::new(output_fields));

        Ok(Arc::new(WriteStreamContext {
            rowaddr_idx,
            rowid_idx,
            action_idx,
            column_sources,
            filler,
            output_schema,
        }))
    }

    /// Extract control arrays from batch
    fn extract_control_arrays(
        batch: &RecordBatch,
        rowaddr_idx: usize,
        rowid_idx: usize,
        action_idx: usize,
    ) -> DFResult<(&UInt64Array, &UInt64Array, &UInt8Array)> {
        // Get row address, row id and __action arrays
        let row_addr_array = batch
            .column(rowaddr_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(
                    "Expected UInt64Array for _rowaddr column".to_string(),
                )
            })?;

        let row_id_array = batch
            .column(rowid_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(
                    "Expected UInt64Array for _rowid column".to_string(),
                )
            })?;

        let action_array = batch
            .column(action_idx)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Expected UInt8Array for {} column",
                    MERGE_ACTION_COLUMN
                ))
            })?;

        Ok((row_addr_array, row_id_array, action_array))
    }

    /// Create the batch to write from the selected rows, assembling columns
    /// in dataset schema order. Columns a partial-schema source does not
    /// provide are fetched from the target for this batch only, keeping
    /// memory bounded by the batch size.
    async fn create_write_batch(
        batch: &RecordBatch,
        keep_rows: Vec<u32>,
        ctx: &WriteStreamContext,
    ) -> DFResult<RecordBatch> {
        // If no rows to keep, return empty batch
        if keep_rows.is_empty() {
            let empty_columns: Vec<_> = ctx
                .output_schema
                .fields()
                .iter()
                .map(|field| arrow_array::new_empty_array(field.data_type()))
                .collect();
            return RecordBatch::try_new(ctx.output_schema.clone(), empty_columns)
                .map_err(datafusion::error::DataFusionError::from);
        }

        // Create indices for rows to keep
        let indices = arrow_array::UInt32Array::from(keep_rows);

        // Take only the rows we want to keep
        let filtered_batch = arrow_select::take::take_record_batch(batch, &indices)?;

        let fill_arrays = if let Some(filler) = &ctx.filler {
            let row_addrs = filtered_batch
                .column(ctx.rowaddr_idx)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(
                        "Expected UInt64Array for _rowaddr column".to_string(),
                    )
                })?;
            filler.fetch(row_addrs).await?
        } else {
            Vec::new()
        };

        let output_columns: Vec<_> = ctx
            .column_sources
            .iter()
            .map(|source| match source {
                WriteColumnSource::Input(idx) => filtered_batch.column(*idx).clone(),
                WriteColumnSource::FillFromTarget(idx) => fill_arrays[*idx].clone(),
            })
            .collect();

        RecordBatch::try_new(ctx.output_schema.clone(), output_columns)
            .map_err(datafusion::error::DataFusionError::from)
    }

    /// Calculate write metrics from new fragments
    fn calculate_write_metrics(new_fragments: &[lance_table::format::Fragment]) -> (usize, usize) {
        let mut total_bytes = 0u64;
        let mut total_files = 0usize;

        for fragment in new_fragments {
            for data_file in &fragment.files {
                if let Some(size) = data_file.file_size_bytes.get() {
                    total_bytes += u64::from(size);
                }
                total_files += 1;
            }
        }

        (total_bytes as usize, total_files)
    }

    fn split_updates_and_inserts(
        &self,
        input_stream: SendableRecordBatchStream,
        merge_state: Arc<Mutex<MergeState>>,
    ) -> DFResult<(SendableRecordBatchStream, SendableRecordBatchStream)> {
        let ctx = self.prepare_stream_schema(input_stream.schema())?;
        let output_schema = ctx.output_schema.clone();

        let (update_tx, update_rx) = tokio::sync::mpsc::unbounded_channel();
        let (insert_tx, insert_rx) = tokio::sync::mpsc::unbounded_channel();

        let merge_state_clone = merge_state;

        tokio::spawn(async move {
            let mut input_stream = input_stream;

            while let Some(batch_result) = input_stream.next().await {
                match batch_result {
                    Ok(batch) => {
                        match Self::process_and_split_batch(&batch, &ctx, merge_state_clone.clone())
                            .await
                        {
                            Ok((update_batch_opt, insert_batch_opt)) => {
                                if let Some(update_batch) = update_batch_opt
                                    && update_tx.send(Ok(update_batch)).is_err()
                                {
                                    break;
                                }

                                if let Some(insert_batch) = insert_batch_opt
                                    && insert_tx.send(Ok(insert_batch)).is_err()
                                {
                                    break;
                                }
                            }
                            Err(e) => {
                                Self::handle_stream_processing_error(e, &update_tx, &insert_tx);
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        Self::handle_stream_processing_error(e, &update_tx, &insert_tx);
                        break;
                    }
                }
            }
        });

        let update_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(update_rx);
        let update_stream = Box::pin(RecordBatchStreamAdapter::new(
            output_schema.clone(),
            update_stream,
        ));

        let insert_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(insert_rx);
        let insert_stream = Box::pin(RecordBatchStreamAdapter::new(output_schema, insert_stream));

        Ok((update_stream, insert_stream))
    }

    async fn process_and_split_batch(
        batch: &RecordBatch,
        ctx: &WriteStreamContext,
        merge_state: Arc<Mutex<MergeState>>,
    ) -> DFResult<(Option<RecordBatch>, Option<RecordBatch>)> {
        let (row_addr_array, row_id_array, action_array) =
            Self::extract_control_arrays(batch, ctx.rowaddr_idx, ctx.rowid_idx, ctx.action_idx)?;

        let mut update_indices: Vec<u32> = Vec::new();
        let mut insert_indices: Vec<u32> = Vec::new();

        {
            let mut merge_state = merge_state.lock().map_err(|e| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Failed to lock merge state: {}",
                    e
                ))
            })?;

            for row_idx in 0..batch.num_rows() {
                let action_code = action_array.value(row_idx);
                let action = Action::try_from(action_code).map_err(|e| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "Invalid action code {}: {}",
                        action_code, e
                    ))
                })?;

                if merge_state
                    .process_row_action(action, row_idx, row_addr_array, row_id_array, batch)?
                    .is_some()
                {
                    match action {
                        Action::UpdateAll => update_indices.push(row_idx as u32),
                        Action::Insert => insert_indices.push(row_idx as u32),
                        _ => {}
                    }
                }
            }
        }

        let update_batch = if !update_indices.is_empty() {
            Some(Self::create_write_batch(batch, update_indices, ctx).await?)
        } else {
            None
        };

        let insert_batch = if !insert_indices.is_empty() {
            Some(Self::create_write_batch(batch, insert_indices, ctx).await?)
        } else {
            None
        };

        Ok((update_batch, insert_batch))
    }

    fn handle_stream_processing_error(
        error: datafusion::error::DataFusionError,
        update_tx: &tokio::sync::mpsc::UnboundedSender<DFResult<RecordBatch>>,
        insert_tx: &tokio::sync::mpsc::UnboundedSender<DFResult<RecordBatch>>,
    ) {
        // Send to first open one. It doesn't matter which one receives it as
        // long as the user gets the error in the end.
        if let Err(tokio::sync::mpsc::error::SendError(error)) = update_tx.send(Err(error)) {
            let _ = insert_tx.send(error);
        }
    }
}

impl DisplayAs for FullSchemaMergeInsertExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default
            | datafusion::physical_plan::DisplayFormatType::Verbose => {
                let on_keys = self.params.on.join(", ");
                let when_matched = match &self.params.when_matched {
                    crate::dataset::WhenMatched::DoNothing => "DoNothing".to_string(),
                    crate::dataset::WhenMatched::UpdateAll => "UpdateAll".to_string(),
                    crate::dataset::WhenMatched::UpdateIf(condition) => {
                        format!("UpdateIf({})", condition)
                    }
                    crate::dataset::WhenMatched::UpdateIfExpr(expr) => {
                        format!("UpdateIf({})", expr.human_display())
                    }
                    crate::dataset::WhenMatched::Fail => "Fail".to_string(),
                    crate::dataset::WhenMatched::Delete => "Delete".to_string(),
                };
                let when_not_matched = if self.params.insert_not_matched {
                    "InsertAll"
                } else {
                    "DoNothing"
                };
                let when_not_matched_by_source = match &self.params.delete_not_matched_by_source {
                    crate::dataset::WhenNotMatchedBySource::Keep => "Keep",
                    crate::dataset::WhenNotMatchedBySource::Delete => "Delete",
                    crate::dataset::WhenNotMatchedBySource::DeleteIf(_) => "DeleteIf",
                };

                write!(
                    f,
                    "MergeInsert: on=[{}], when_matched={}, when_not_matched={}, when_not_matched_by_source={}",
                    on_keys, when_matched, when_not_matched, when_not_matched_by_source
                )
            }
            datafusion::physical_plan::DisplayFormatType::TreeRender => {
                write!(f, "MergeInsert[{}]", self.dataset.uri())
            }
        }
    }
}

impl ExecutionPlan for FullSchemaMergeInsertExec {
    fn name(&self) -> &str {
        "FullSchemaMergeInsertExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(arrow_schema::Schema::empty())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "FullSchemaMergeInsertExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            dataset: self.dataset.clone(),
            params: self.params.clone(),
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
            merge_stats: self.merge_stats.clone(),
            transaction: self.transaction.clone(),
            affected_rows: self.affected_rows.clone(),
            inserted_rows_filter: self.inserted_rows_filter.clone(),
            is_primary_key: self.is_primary_key,
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }

    fn required_input_distribution(&self) -> Vec<datafusion_physical_expr::Distribution> {
        // We require a single partition for the merge operation to ensure all data is processed
        vec![datafusion_physical_expr::Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // We just want one stream.
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let _baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        // Input schema structure based on our logical plan:
        // - target._rowaddr: Address of existing rows to update/delete
        // - source.*: Source data columns (variable schema)
        // - __action: Merge action (1=update, 2=insert, 0=delete, etc.)

        // Execute the input plan to get the merge data stream
        let input_stream = self.input.execute(partition, context)?;

        // Step 1: Create shared state and streaming processor for row addresses and write data
        // Get field IDs for the ON columns from the dataset schema
        let field_ids: Vec<i32> = self
            .params
            .on
            .iter()
            .filter_map(|name| self.dataset.schema().field(name).map(|f| f.id))
            .collect();
        let merge_state = Arc::new(Mutex::new(MergeState::new(
            MergeInsertMetrics::new(&self.metrics, partition),
            self.dataset.manifest.uses_stable_row_ids(),
            self.params.on.clone(),
            field_ids,
            self.params.source_dedupe_behavior,
        )));
        let write_data_stream =
            self.create_filtered_write_stream(input_stream, merge_state.clone())?;

        // Use flat_map to handle the async write operation
        let dataset = self.dataset.clone();
        let params = self.params.clone();
        let merge_stats_holder = self.merge_stats.clone();
        let transaction_holder = self.transaction.clone();
        let affected_rows_holder = self.affected_rows.clone();
        let inserted_rows_filter_holder = self.inserted_rows_filter.clone();
        let merged_generations = self.params.merged_generations.clone();
        let is_primary_key = self.is_primary_key;
        let updating_row_ids = {
            let state = merge_state.lock().unwrap();
            state.updating_row_ids.clone()
        };

        let result_stream = stream::once(async move {
            // Step 2: Write new fragments using the filtered data (inserts + updates)
            let target_bases_info = resolve_target_bases(&dataset, &params).await?;
            // Keep a copy so failures after the write can clean up routed files.
            let cleanup_bases = target_bases_info.clone();
            let (mut new_fragments, _) = write_fragments_internal(
                Some(&dataset),
                dataset.object_store.clone(),
                &dataset.base,
                dataset.schema().clone(),
                write_data_stream,
                WriteParams::default(),
                target_bases_info,
            )
            .await?;

            let row_id_result: lance_core::Result<()> = (|| {
                if let Some(row_id_sequence) = updating_row_ids.lock().unwrap().row_id_sequence() {
                    let fragment_sizes = new_fragments
                        .iter()
                        .map(|f| f.physical_rows.unwrap() as u64);

                    let sequences = lance_table::rowids::rechunk_sequences(
                        [row_id_sequence.clone()],
                        fragment_sizes,
                        true,
                    )
                    .map_err(|e| {
                        Error::internal(format!(
                            "Captured row ids not equal to number of rows written: {}",
                            e
                        ))
                    })?;

                    for (fragment, sequence) in new_fragments.iter_mut().zip(sequences) {
                        let serialized = lance_table::rowids::write_row_ids(&sequence);
                        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
                    }
                }
                Ok(())
            })();
            if let Err(e) = row_id_result {
                cleanup_data_fragments(
                    &dataset.object_store,
                    &dataset.base,
                    cleanup_bases.as_deref(),
                    &new_fragments,
                )
                .await;
                return Err(e.into());
            }

            // Step 2.5: Calculate write metrics from new fragments
            let (total_bytes_written, total_files_written) =
                Self::calculate_write_metrics(&new_fragments);

            // Step 3: Apply deletions to existing fragments
            let merge_state =
                Arc::into_inner(merge_state).expect("MergeState should only have 1 reference now");
            let merge_state =
                Mutex::into_inner(merge_state).expect("MergeState lock should be available");
            let delete_row_addrs_clone = merge_state.delete_row_addrs;
            let inserted_rows_filter = if is_primary_key {
                Some(KeyExistenceFilter::from_bloom_filter(
                    &merge_state.inserted_rows_filter,
                ))
            } else {
                None
            };

            let (updated_fragments, removed_fragment_ids) =
                match apply_deletions(&dataset, &delete_row_addrs_clone).await {
                    Ok(result) => result,
                    Err(e) => {
                        // The new data files are not committed; remove them (including
                        // ones routed to target bases) before surfacing the error.
                        cleanup_data_fragments(
                            &dataset.object_store,
                            &dataset.base,
                            cleanup_bases.as_deref(),
                            &new_fragments,
                        )
                        .await;
                        return Err(e.into());
                    }
                };

            // Step 4: Create the transaction operation
            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified: vec![], // No fields are modified in schema for upsert
                merged_generations,
                // Use the full pre-order field list (not just top-level `fields`) so
                // that nested leaf field ids are included. A merge_insert rewrites whole
                // rows, so every field is potentially modified; omitting nested ids would
                // let `register_pure_rewrite_rows_update_frags_in_indices` wrongly extend a
                // nested-field index over the rewritten fragment, silently dropping rows.
                fields_for_preserving_frag_bitmap: dataset
                    .schema()
                    .fields_pre_order()
                    .map(|f| f.id as u32)
                    .collect(),
                update_mode: Some(RewriteRows),
                inserted_rows_filter: inserted_rows_filter.clone(),
                updated_fragment_offsets: None,
            };

            // Step 5: Create and store the transaction
            let transaction = Transaction::new(dataset.manifest.version, operation, None);

            // Step 6: Store transaction, merge stats, and affected rows for later retrieval
            {
                // Update write metrics before converting to stats
                merge_state.metrics.bytes_written.add(total_bytes_written);
                merge_state
                    .metrics
                    .num_files_written
                    .add(total_files_written);

                // Get the final stats from the shared state
                let stats = MergeStats::from(&merge_state.metrics);

                if let Ok(mut transaction_guard) = transaction_holder.lock() {
                    transaction_guard.replace(transaction);
                }
                if let Ok(mut merge_stats_guard) = merge_stats_holder.lock() {
                    merge_stats_guard.replace(stats);
                }
                if let Ok(mut affected_rows_guard) = affected_rows_holder.lock() {
                    affected_rows_guard.replace(delete_row_addrs_clone);
                }
                if let Ok(mut filter_guard) = inserted_rows_filter_holder.lock() {
                    *filter_guard = inserted_rows_filter;
                }
            };

            // Step 7: Return empty result (write operations don't return data)
            let empty_schema = Arc::new(arrow_schema::Schema::empty());
            let empty_batch = RecordBatch::new_empty(empty_schema);
            Ok(empty_batch)
        });

        let empty_schema = Arc::new(arrow_schema::Schema::empty());
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            empty_schema,
            result_stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::UInt64Array;

    #[test]
    fn test_merge_state_duplicate_rowid_detection_fail() {
        let metrics = MergeInsertMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut merge_state = MergeState::new(
            metrics,
            false,
            Vec::new(),
            Vec::new(),
            SourceDedupeBehavior::Fail,
        );

        let row_addr_array = UInt64Array::from(vec![1000, 2000, 3000]);
        let row_id_array = UInt64Array::from(vec![100, 100, 300]); // Duplicate row_id 100

        let result1 = merge_state.process_row_action(
            Action::UpdateAll,
            0,
            &row_addr_array,
            &row_id_array,
            &RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())),
        );
        assert!(result1.is_ok(), "First call should succeed");

        let result2 = merge_state.process_row_action(
            Action::UpdateAll,
            1,
            &row_addr_array,
            &row_id_array,
            &RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())),
        );
        assert!(
            result2.is_err(),
            "Second call with duplicate _rowid should fail"
        );

        let error_msg = result2.unwrap_err().to_string();
        assert!(
            error_msg.contains("Ambiguous merge insert")
                && error_msg.contains("multiple source rows"),
            "Error message should mention ambiguous merge insert and multiple source rows, got: {}",
            error_msg
        );

        let result3 = merge_state.process_row_action(
            Action::UpdateAll,
            2,
            &row_addr_array,
            &row_id_array,
            &RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())),
        );
        assert!(
            result3.is_ok(),
            "Third call with different _rowid should succeed"
        );
    }

    #[test]
    fn test_merge_state_duplicate_rowid_first_seen() {
        let metrics = MergeInsertMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut merge_state = MergeState::new(
            metrics,
            false,
            Vec::new(),
            Vec::new(),
            SourceDedupeBehavior::FirstSeen,
        );

        let row_addr_array = UInt64Array::from(vec![1000, 2000, 3000]);
        let row_id_array = UInt64Array::from(vec![100, 100, 300]); // Duplicate row_id 100

        let result1 = merge_state.process_row_action(
            Action::UpdateAll,
            0,
            &row_addr_array,
            &row_id_array,
            &RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())),
        );
        assert!(result1.is_ok(), "First call should succeed");
        assert_eq!(result1.unwrap(), Some(0), "First row should be kept");

        let result2 = merge_state.process_row_action(
            Action::UpdateAll,
            1,
            &row_addr_array,
            &row_id_array,
            &RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())),
        );
        assert!(
            result2.is_ok(),
            "Second call with duplicate _rowid should succeed with FirstSeen"
        );
        assert_eq!(
            result2.unwrap(),
            None,
            "Duplicate row should be skipped (return None)"
        );

        // Verify the metric was incremented
        assert_eq!(
            merge_state.metrics.num_skipped_duplicates.value(),
            1,
            "num_skipped_duplicates should be 1"
        );

        let result3 = merge_state.process_row_action(
            Action::UpdateAll,
            2,
            &row_addr_array,
            &row_id_array,
            &RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())),
        );
        assert!(
            result3.is_ok(),
            "Third call with different _rowid should succeed"
        );
        assert_eq!(result3.unwrap(), Some(2), "Third row should be kept");
    }
}
