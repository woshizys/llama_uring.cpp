//! Rust FFI surface connecting llama.cpp MoE expert cache calls to io-scheduler.

use io_scheduler::expert_manager::{
    ExpertHandle as SchedulerExpertHandle, ExpertKey, ExpertTicket as SchedulerExpertTicket,
    LlamaExpertManager, MoePart, Slice, TensorLayout, TensorLayoutKind, TensorMeta,
};
use io_scheduler::predictor::{
    ExpertPredictor, LowRankMlpPredictor, ParsedExpertKey, PredictionRequest, TraceBasedPredictor,
};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::{Arc, Mutex};
use tokio::runtime::{Builder, Handle, Runtime};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct llama_expert_slice_ffi {
    pub part: i32,
    pub file_id: i32,
    pub file_offset: u64,
    pub file_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct llama_expert_prediction_ffi {
    pub layer: i32,
    pub expert: i32,
    pub probability: f32,
    pub deadline_ns: u64,
}

#[derive(Debug)]
pub enum ExpertCacheError {
    InvalidPart(i32),
    InvalidLayoutKind(i32),
    InvalidId(&'static str, i32),
    InvalidSize(&'static str),
    IntegerOverflow(&'static str, u64),
    NullManager,
    NullHandle,
    NullTicket,
    NullSlice,
    NullLayoutParts,
    NullPath,
    NullData,
    NullWrittenOut,
    NullExperts,
    NullExpertOut,
    NullPredictions,
    NullLayerOut,
    NullHandles,
    Backend(String),
    Panic,
}

impl std::fmt::Display for ExpertCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPart(part) => write!(f, "invalid MoE part: {part}"),
            Self::InvalidLayoutKind(kind) => write!(f, "invalid tensor layout kind: {kind}"),
            Self::InvalidId(name, value) => write!(f, "invalid {name}: {value}"),
            Self::InvalidSize(name) => write!(f, "{name} must be > 0"),
            Self::IntegerOverflow(name, value) => {
                write!(f, "{name} does not fit usize: {value}")
            }
            Self::NullManager => write!(f, "null expert manager pointer"),
            Self::NullHandle => write!(f, "null expert handle pointer"),
            Self::NullTicket => write!(f, "null expert ticket pointer"),
            Self::NullSlice => write!(f, "null expert slice pointer"),
            Self::NullLayoutParts => write!(f, "null tensor layout parts pointer"),
            Self::NullPath => write!(f, "null expert file path pointer"),
            Self::NullData => write!(f, "null KV state data pointer"),
            Self::NullWrittenOut => write!(f, "null KV written output pointer"),
            Self::NullExperts => write!(f, "null expert ids pointer"),
            Self::NullExpertOut => write!(f, "null output expert pointer"),
            Self::NullHandles => write!(f, "null output handles pointer"),
            Self::Backend(msg) => write!(f, "{msg}"),
            Self::Panic => write!(f, "panic crossed FFI boundary"),
            Self::NullPredictions => write!(f, "null output predictions pointer"),
            Self::NullLayerOut => write!(f, "null output layer pointer"),
        }
    }
}

impl std::error::Error for ExpertCacheError {}
fn predictor_from_env() -> Result<Box<dyn ExpertPredictor>, ExpertCacheError> {
    let top_k = std::env::var("PDCAT_PREDICTOR_TOP_K")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256);
    if let Some(path) = std::env::var_os("PDCAT_PREDICTOR_MANIFEST") {
        let predictor = LowRankMlpPredictor::from_manifest(&path, top_k)
            .map_err(|error| ExpertCacheError::Backend(error.to_string()))?;
        let expected_model_id = std::env::var("PDCAT_MODEL_ID").map_err(|_| {
            ExpertCacheError::Backend(
                "PDCAT_MODEL_ID is required with PDCAT_PREDICTOR_MANIFEST".to_string(),
            )
        })?;
        let expected_model_sha256 = std::env::var("PDCAT_MODEL_SHA256").map_err(|_| {
            ExpertCacheError::Backend(
                "PDCAT_MODEL_SHA256 is required with PDCAT_PREDICTOR_MANIFEST".to_string(),
            )
        })?;
        let expected_num_experts = std::env::var("PDCAT_MODEL_EXPERT_COUNT")
            .map_err(|_| {
                ExpertCacheError::Backend(
                    "PDCAT_MODEL_EXPERT_COUNT is required with PDCAT_PREDICTOR_MANIFEST"
                        .to_string(),
                )
            })?
            .parse::<usize>()
            .map_err(|_| {
                ExpertCacheError::Backend(
                    "PDCAT_MODEL_EXPERT_COUNT must be a positive integer".to_string(),
                )
            })?;
        let manifest = predictor.manifest();
        if manifest.model_id != expected_model_id {
            return Err(ExpertCacheError::Backend(format!(
                "predictor model_id {} does not match current model {}",
                manifest.model_id, expected_model_id
            )));
        }
        if manifest
            .model_sha256
            .as_deref()
            .is_none_or(|value| !value.eq_ignore_ascii_case(&expected_model_sha256))
        {
            return Err(ExpertCacheError::Backend(
                "predictor model SHA-256 does not match the current model".to_string(),
            ));
        }
        if expected_num_experts == 0 || manifest.num_experts != expected_num_experts {
            return Err(ExpertCacheError::Backend(format!(
                "predictor expert count {} does not match current model {}",
                manifest.num_experts, expected_num_experts
            )));
        }
        eprintln!(
            "[PDCAT][predictor-load] backend=safetensors-mlp model={} manifest={} layers={} experts={}",
            predictor.manifest().model_id,
            predictor.manifest_path().display(),
            predictor.manifest().layers.len(),
            predictor.manifest().num_experts,
        );
        return Ok(Box::new(predictor));
    }
    Ok(Box::new(TraceBasedPredictor::new(top_k)))
}

#[allow(non_camel_case_types)]
pub struct llama_expert_manager_ffi {
    manager: LlamaExpertManager,
    runtime: Runtime,
    predictor: Mutex<Box<dyn ExpertPredictor>>,
}

impl llama_expert_manager_ffi {
    pub fn new(
        capacity: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        precision_bits: usize,
    ) -> Result<Self, ExpertCacheError> {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| {
                ExpertCacheError::Backend(format!("failed to build runtime: {error}"))
            })?;
        let manager = {
            let _guard = runtime.enter();
            LlamaExpertManager::new(capacity, hidden_dim, intermediate_dim, precision_bits)
        };

        Ok(Self {
            manager,
            runtime,
            predictor: Mutex::new(predictor_from_env()?),
        })
    }

    pub fn new_with_slot_size(capacity: usize, slot_size: usize) -> Result<Self, ExpertCacheError> {
        Self::new_with_layout(capacity, slot_size, TensorLayout::default_non_contiguous())
    }

    pub fn new_with_layout(
        capacity: usize,
        slot_size: usize,
        layout: TensorLayout,
    ) -> Result<Self, ExpertCacheError> {
        if capacity == 0 {
            return Err(ExpertCacheError::InvalidSize("capacity"));
        }
        if slot_size == 0 {
            return Err(ExpertCacheError::InvalidSize("slot_size"));
        }
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| {
                ExpertCacheError::Backend(format!("failed to build runtime: {error}"))
            })?;
        let manager = {
            let _guard = runtime.enter();
            LlamaExpertManager::new_with_slot_size_and_layout(capacity, slot_size, layout)
        };

        Ok(Self {
            manager,
            runtime,
            predictor: Mutex::new(predictor_from_env()?),
        })
    }
}

#[allow(non_camel_case_types)]
pub struct llama_expert_handle_ffi {
    handle: ExpertHandleStorage,
}

enum ExpertHandleStorage {
    Owned(SchedulerExpertHandle),
    Shared(Arc<SchedulerExpertHandle>),
}

#[allow(non_camel_case_types)]
pub struct llama_expert_ticket_ffi {
    ticket: SchedulerExpertTicket,
    runtime: Handle,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(error: ExpertCacheError) {
    let msg = error.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(msg).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn manager_mut<'a>(
    manager: *mut llama_expert_manager_ffi,
) -> Result<&'a mut llama_expert_manager_ffi, ExpertCacheError> {
    unsafe { manager.as_mut() }.ok_or(ExpertCacheError::NullManager)
}

fn handle_from_scheduler(handle: SchedulerExpertHandle) -> *mut llama_expert_handle_ffi {
    Box::into_raw(Box::new(llama_expert_handle_ffi {
        handle: ExpertHandleStorage::Owned(handle),
    }))
}

fn handle_from_shared(handle: Arc<SchedulerExpertHandle>) -> *mut llama_expert_handle_ffi {
    Box::into_raw(Box::new(llama_expert_handle_ffi {
        handle: ExpertHandleStorage::Shared(handle),
    }))
}

fn handle_ref<'a>(
    handle: *const llama_expert_handle_ffi,
) -> Result<&'a llama_expert_handle_ffi, ExpertCacheError> {
    unsafe { handle.as_ref() }.ok_or(ExpertCacheError::NullHandle)
}

fn scheduler_handle_ref<'a>(handle: &'a llama_expert_handle_ffi) -> &'a SchedulerExpertHandle {
    match &handle.handle {
        ExpertHandleStorage::Owned(handle) => handle,
        ExpertHandleStorage::Shared(handle) => handle.as_ref(),
    }
}

fn ticket_ref<'a>(
    ticket: *const llama_expert_ticket_ffi,
) -> Result<&'a llama_expert_ticket_ffi, ExpertCacheError> {
    unsafe { ticket.as_ref() }.ok_or(ExpertCacheError::NullTicket)
}

fn ticket_mut<'a>(
    ticket: *mut llama_expert_ticket_ffi,
) -> Result<&'a mut llama_expert_ticket_ffi, ExpertCacheError> {
    unsafe { ticket.as_mut() }.ok_or(ExpertCacheError::NullTicket)
}

fn path_from_ptr<'a>(path: *const c_char) -> Result<&'a str, ExpertCacheError> {
    if path.is_null() {
        return Err(ExpertCacheError::NullPath);
    }

    unsafe { CStr::from_ptr(path) }.to_str().map_err(|error| {
        ExpertCacheError::Backend(format!("invalid expert file path UTF-8: {error}"))
    })
}

fn part_from_i32(part: i32) -> Result<MoePart, ExpertCacheError> {
    MoePart::try_from(part).map_err(ExpertCacheError::InvalidPart)
}

fn layout_kind_from_i32(kind: i32) -> Result<TensorLayoutKind, ExpertCacheError> {
    TensorLayoutKind::try_from(kind).map_err(ExpertCacheError::InvalidLayoutKind)
}

fn layout_from_ffi(
    kind: i32,
    parts: *const i32,
    part_count: usize,
) -> Result<TensorLayout, ExpertCacheError> {
    if part_count > 0 && parts.is_null() {
        return Err(ExpertCacheError::NullLayoutParts);
    }

    let raw_parts = if part_count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(parts, part_count) }
    };
    let layout_parts = raw_parts
        .iter()
        .copied()
        .map(part_from_i32)
        .collect::<Result<Vec<_>, ExpertCacheError>>()?;

    Ok(TensorLayout::new(layout_kind_from_i32(kind)?, layout_parts))
}

fn id_to_usize(name: &'static str, value: i32) -> Result<usize, ExpertCacheError> {
    usize::try_from(value).map_err(|_| ExpertCacheError::InvalidId(name, value))
}

fn u64_to_usize(name: &'static str, value: u64) -> Result<usize, ExpertCacheError> {
    usize::try_from(value).map_err(|_| ExpertCacheError::IntegerOverflow(name, value))
}

fn tensor_meta_from_ffi(
    layer: i32,
    expert: i32,
    slice: llama_expert_slice_ffi,
) -> Result<TensorMeta, ExpertCacheError> {
    Ok(TensorMeta {
        part: part_from_i32(slice.part)?,
        layer: id_to_usize("layer", layer)?,
        expert: id_to_usize("expert", expert)?,
        slice: Slice {
            file_id: slice.file_id,
            file_offset: u64_to_usize("file_offset", slice.file_offset)?,
            file_size: u64_to_usize("file_size", slice.file_size)?,
        },
    })
}

fn scheduler_error(error: impl std::fmt::Display) -> ExpertCacheError {
    ExpertCacheError::Backend(error.to_string())
}

/// Creates an io-scheduler backed expert manager.
#[no_mangle]
pub extern "C" fn llama_expert_manager_new(
    capacity: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    precision_bits: usize,
) -> *mut llama_expert_manager_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        llama_expert_manager_ffi::new(capacity, hidden_dim, intermediate_dim, precision_bits)
    }));

    match result {
        Ok(Ok(manager)) => Box::into_raw(Box::new(manager)),
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Creates an io-scheduler backed expert manager with an exact per-expert slot size.
#[no_mangle]
pub extern "C" fn llama_expert_manager_new_with_slot_size(
    capacity: usize,
    slot_size: usize,
) -> *mut llama_expert_manager_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        llama_expert_manager_ffi::new_with_slot_size(capacity, slot_size)
    }));

    match result {
        Ok(Ok(manager)) => Box::into_raw(Box::new(manager)),
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Creates an io-scheduler backed expert manager with an exact per-expert slot
/// size and explicit in-slot tensor layout.
#[no_mangle]
pub extern "C" fn llama_expert_manager_new_with_layout(
    capacity: usize,
    slot_size: usize,
    layout_kind: i32,
    layout_parts: *const i32,
    part_count: usize,
) -> *mut llama_expert_manager_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let layout = layout_from_ffi(layout_kind, layout_parts, part_count)?;
        llama_expert_manager_ffi::new_with_layout(capacity, slot_size, layout)
    }));

    match result {
        Ok(Ok(manager)) => Box::into_raw(Box::new(manager)),
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Returns the last FFI error message for the current thread.
///
/// The pointer remains valid until the next FFI call on the same thread.
#[no_mangle]
pub extern "C" fn llama_expert_manager_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(msg) => msg.as_ptr(),
        None => ptr::null(),
    })
}

/// Registers one tensor slice for `(layer, expert, part)`.
///
/// Return code:
/// - `0`: success
/// - non-zero: failure; call `llama_expert_manager_last_error_message()`
#[no_mangle]
pub extern "C" fn llama_expert_manager_register_slice(
    manager: *mut llama_expert_manager_ffi,
    layer: i32,
    expert: i32,
    slice: *const llama_expert_slice_ffi,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        let slice = unsafe { slice.as_ref() }.ok_or(ExpertCacheError::NullSlice)?;
        let tensor_meta = tensor_meta_from_ffi(layer, expert, *slice)?;
        manager
            .runtime
            .block_on(manager.manager.register_tensor_core(tensor_meta));
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

/// Registers the GGUF file path for a llama.cpp file id.
///
/// Return code:
/// - `0`: success
/// - non-zero: failure; call `llama_expert_manager_last_error_message()`
#[no_mangle]
pub extern "C" fn llama_expert_manager_register_file(
    manager: *mut llama_expert_manager_ffi,
    file_id: i32,
    path: *const c_char,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        let path = path_from_ptr(path)?;
        manager
            .runtime
            .block_on(
                manager
                    .manager
                    .register_file_core(file_id, path.to_string()),
            )
            .map_err(scheduler_error)
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

/// Writes one complete llama slot-state image through the shared P4 scheduler.
///
/// The input bytes are borrowed only for the duration of this call. On success,
/// `written_out` receives exactly `data_size`.
#[no_mangle]
pub extern "C" fn llama_expert_manager_write_background_file(
    manager: *mut llama_expert_manager_ffi,
    path: *const c_char,
    data: *const u8,
    data_size: usize,
    kv_slot_id: i32,
    written_out: *mut usize,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        let path = path_from_ptr(path)?;
        let _ = id_to_usize("kv_slot_id", kv_slot_id)?;
        if data_size > 0 && data.is_null() {
            return Err(ExpertCacheError::NullData);
        }
        if written_out.is_null() {
            return Err(ExpertCacheError::NullWrittenOut);
        }
        let bytes = if data_size == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(data, data_size) }
        };
        let written = manager
            .runtime
            .block_on(manager.manager.write_background_file(
                std::path::Path::new(path),
                bytes,
                kv_slot_id,
            ))
            .map_err(scheduler_error)?;
        unsafe {
            *written_out = written;
        }
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

/// Ensures that an expert is resident and pinned.
///
/// Returns null on failure; call `llama_expert_manager_last_error_message()`.
/// The returned handle must be released exactly once with
/// `llama_expert_manager_release`.
#[no_mangle]
pub extern "C" fn llama_expert_manager_ensure(
    manager: *mut llama_expert_manager_ffi,
    layer: i32,
    expert: i32,
) -> *mut llama_expert_handle_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        let key = ExpertKey::new(id_to_usize("layer", layer)?, id_to_usize("expert", expert)?);
        manager
            .runtime
            .block_on(manager.manager.ensure(key))
            .map_err(scheduler_error)
    }));

    match result {
        Ok(Ok(handle)) => handle_from_scheduler(handle),
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Ensures multiple experts from the same layer and returns one handle per input expert.
///
/// Return code:
/// - `0`: success
/// - non-zero: failure; call `llama_expert_manager_last_error_message()`
#[no_mangle]
pub extern "C" fn llama_expert_manager_ensure_many(
    manager: *mut llama_expert_manager_ffi,
    layer: i32,
    experts: *const i32,
    count: usize,
    handles_out: *mut *mut llama_expert_handle_ffi,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        if count > 0 && experts.is_null() {
            return Err(ExpertCacheError::NullExperts);
        }
        if count > 0 && handles_out.is_null() {
            return Err(ExpertCacheError::NullHandles);
        }

        let layer = id_to_usize("layer", layer)?;
        let expert_ids = if count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(experts, count) }
        };
        let keys = expert_ids
            .iter()
            .copied()
            .map(|expert| Ok(ExpertKey::new(layer, id_to_usize("expert", expert)?)))
            .collect::<Result<Vec<_>, ExpertCacheError>>()?;

        let handles = manager
            .runtime
            .block_on(manager.manager.ensure_many(keys))
            .map_err(scheduler_error)?;

        let out = unsafe { std::slice::from_raw_parts_mut(handles_out, count) };
        for (slot, handle) in out.iter_mut().zip(handles.into_iter()) {
            *slot = handle_from_scheduler(handle);
        }

        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

/// Submits speculative expert prefetches without waiting for I/O completion.
///
/// A later demand for the same expert promotes a queued prefetch to the demand
/// queue. Return code is zero on successful admission.
#[no_mangle]
pub extern "C" fn llama_expert_manager_prefetch_many(
    manager: *mut llama_expert_manager_ffi,
    layer: i32,
    experts: *const i32,
    count: usize,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        if count > 0 && experts.is_null() {
            return Err(ExpertCacheError::NullExperts);
        }

        let layer = id_to_usize("layer", layer)?;
        let expert_ids = if count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(experts, count) }
        };
        let keys = expert_ids
            .iter()
            .copied()
            .map(|expert| Ok(ExpertKey::new(layer, id_to_usize("expert", expert)?)))
            .collect::<Result<Vec<_>, ExpertCacheError>>()?;

        manager
            .runtime
            .block_on(manager.manager.prefetch_many(keys))
            .map_err(scheduler_error)
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

/// Schedules every expert from the next registered MoE layer as P1 mandatory
/// staging. The next layer is discovered from model registration metadata, so
/// non-consecutive MoE layouts do not require architecture-specific constants.
#[no_mangle]
pub extern "C" fn llama_expert_manager_prefetch_next_layer_mandatory(
    manager: *mut llama_expert_manager_ffi,
    current_layer: i32,
    target_layer_out: *mut i32,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        if target_layer_out.is_null() {
            return Err(ExpertCacheError::NullLayerOut);
        }
        let current_layer = id_to_usize("current layer", current_layer)?;
        let scheduled = manager
            .runtime
            .block_on(manager.manager.prefetch_next_layer_mandatory(current_layer))
            .map_err(scheduler_error)?;
        let Some((target_layer, count)) = scheduled else {
            unsafe { *target_layer_out = -1 };
            return Ok(0);
        };
        unsafe {
            *target_layer_out = i32::try_from(target_layer).map_err(|_| {
                ExpertCacheError::Backend("target layer does not fit i32".to_string())
            })?;
        }
        i32::try_from(count)
            .map_err(|_| ExpertCacheError::Backend("expert count does not fit i32".to_string()))
    }));

    match result {
        Ok(Ok(count)) => count,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

/// Observes the native router output, predicts the next registered MoE layer,
/// and admits every returned candidate as speculative prefetch. Prediction never
/// changes the native router selection.
#[no_mangle]
pub extern "C" fn llama_expert_manager_predict_prefetch_next(
    manager: *mut llama_expert_manager_ffi,
    layer: i32,
    experts: *const i32,
    count: usize,
    router_scores: *const f32,
    max_predictions: usize,
    current_token_id: u64,
    deadline_ns: u64,
    predictions_out: *mut llama_expert_prediction_ffi,
    predictions_capacity: usize,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        if count > 0 && experts.is_null() {
            return Err(ExpertCacheError::NullExperts);
        }
        if predictions_capacity > 0 && predictions_out.is_null() {
            return Err(ExpertCacheError::NullPredictions);
        }
        if max_predictions == 0 || predictions_capacity == 0 {
            return Ok(0);
        }

        let layer = id_to_usize("layer", layer)?;
        manager.manager.set_metric_token_id(current_token_id);
        let expert_ids = if count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(experts, count) }
        };
        let current_router_scores = if router_scores.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(router_scores, count) }.to_vec()
        };
        let current_experts = expert_ids
            .iter()
            .copied()
            .map(|expert| {
                let expert = id_to_usize("expert", expert)?;
                Ok(format!("runtime/l{layer}-e{expert}"))
            })
            .collect::<Result<Vec<_>, ExpertCacheError>>()?;
        if current_experts.is_empty() {
            return Ok(0);
        }

        let Some(next_layer) = manager
            .runtime
            .block_on(manager.manager.next_registered_layer_cyclic(layer))
        else {
            return Ok(0);
        };
        let registered = manager
            .runtime
            .block_on(manager.manager.registered_experts_for_layer(next_layer));
        if registered.is_empty() {
            return Ok(0);
        }
        let registered = registered
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let target_token_id = if next_layer <= layer {
            current_token_id.checked_add(1).ok_or_else(|| {
                ExpertCacheError::Backend("predictor token generation overflow".to_string())
            })?
        } else {
            current_token_id
        };

        let predictor_started = std::time::Instant::now();
        let raw_predictions = {
            let mut predictor = manager.predictor.lock().map_err(|_| {
                ExpertCacheError::Backend("online predictor mutex is poisoned".to_string())
            })?;
            predictor
                .observe(Some("online"), layer, &current_experts)
                .map_err(scheduler_error)?;
            predictor
                .predict_next(&PredictionRequest {
                    request_id: Some("online".to_string()),
                    current_layer: layer,
                    target_layer: Some(next_layer),
                    current_experts,
                    current_router_scores,
                    max_predictions: max_predictions.min(predictions_capacity),
                })
                .map_err(scheduler_error)?
        };
        let predictor_us = predictor_started.elapsed().as_secs_f64() * 1.0e6;

        let mut seen = std::collections::HashSet::new();
        let mut admitted = Vec::new();
        for prediction in raw_predictions {
            if !prediction.score.is_finite() {
                continue;
            }
            let parsed = ParsedExpertKey::parse(&prediction.expert_id).map_err(scheduler_error)?;
            if parsed.layer != next_layer {
                continue;
            }
            let key = ExpertKey::new(parsed.layer, parsed.expert);
            if !registered.contains(&key) || !seen.insert(key) {
                continue;
            }
            admitted.push((key, prediction.score));
            if admitted.len() >= max_predictions.min(predictions_capacity) {
                break;
            }
        }

        if admitted.is_empty() {
            return Ok(0);
        }
        manager
            .runtime
            .block_on(manager.manager.prefetch_predictions(
                admitted.clone(),
                deadline_ns,
                predictor_us,
                target_token_id,
            ))
            .map_err(scheduler_error)?;

        let output =
            unsafe { std::slice::from_raw_parts_mut(predictions_out, predictions_capacity) };
        for (slot, (key, probability)) in output.iter_mut().zip(admitted.iter()) {
            slot.layer = i32::try_from(key.layer()).map_err(|_| {
                ExpertCacheError::Backend("predicted layer id does not fit i32".to_string())
            })?;
            slot.expert = i32::try_from(key.expert()).map_err(|_| {
                ExpertCacheError::Backend("predicted expert id does not fit i32".to_string())
            })?;
            slot.probability = *probability;
            slot.deadline_ns = deadline_ns;
        }

        i32::try_from(admitted.len())
            .map_err(|_| ExpertCacheError::Backend("prediction count does not fit i32".to_string()))
    }));

    match result {
        Ok(Ok(count)) => count,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

/// Submits a batch of experts from the same layer for asynchronous loading.
///
/// Returns null on failure; call `llama_expert_manager_last_error_message()`.
/// The returned ticket must be released/freed with `llama_expert_ticket_release`
/// and `llama_expert_ticket_free`.
#[no_mangle]
pub extern "C" fn llama_expert_manager_submit_batch_async(
    manager: *mut llama_expert_manager_ffi,
    layer: i32,
    experts: *const i32,
    count: usize,
    token_id: u64,
) -> *mut llama_expert_ticket_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        if count > 0 && experts.is_null() {
            return Err(ExpertCacheError::NullExperts);
        }

        let layer = id_to_usize("layer", layer)?;
        let expert_ids = if count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(experts, count) }
        };
        let keys = expert_ids
            .iter()
            .copied()
            .map(|expert| Ok(ExpertKey::new(layer, id_to_usize("expert", expert)?)))
            .collect::<Result<Vec<_>, ExpertCacheError>>()?;

        let ticket = manager
            .runtime
            .block_on(
                manager
                    .manager
                    .submit_batch_async(keys, (token_id != u64::MAX).then_some(token_id)),
            )
            .map_err(scheduler_error)?;

        Ok(llama_expert_ticket_ffi {
            ticket,
            runtime: manager.runtime.handle().clone(),
        })
    }));

    match result {
        Ok(Ok(ticket)) => Box::into_raw(Box::new(ticket)),
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Gets one pinned expert handle from a ticket, blocking until it is loaded.
///
/// Returns null on failure; call `llama_expert_manager_last_error_message()`.
/// The returned handle must be released exactly once with
/// `llama_expert_manager_release`.
#[no_mangle]
pub extern "C" fn llama_expert_ticket_get_handle(
    ticket: *const llama_expert_ticket_ffi,
    layer: i32,
    expert: i32,
) -> *mut llama_expert_handle_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let ticket = ticket_ref(ticket)?;
        let key = ExpertKey::new(id_to_usize("layer", layer)?, id_to_usize("expert", expert)?);
        ticket
            .runtime
            .block_on(ticket.ticket.get_handle(&key))
            .map(handle_from_shared)
            .map_err(scheduler_error)
    }));

    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Returns one pinned handle if the expert is already ready in the ticket.
///
/// Returns null when the expert is not ready or on failure. On failure,
/// `llama_expert_manager_last_error_message()` is set.
#[no_mangle]
pub extern "C" fn llama_expert_ticket_try_get_handle(
    ticket: *const llama_expert_ticket_ffi,
    layer: i32,
    expert: i32,
) -> *mut llama_expert_handle_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let ticket = ticket_ref(ticket)?;
        let key = ExpertKey::new(id_to_usize("layer", layer)?, id_to_usize("expert", expert)?);
        Ok(ticket.runtime.block_on(ticket.ticket.try_get_handle(&key)))
    }));

    match result {
        Ok(Ok(Some(handle))) => handle_from_shared(handle),
        Ok(Ok(None)) => ptr::null_mut(),
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Waits until any expert in `experts` is ready and returns its pinned handle.
///
/// The chosen expert id is written to `expert_out`. Returns null on failure;
/// call `llama_expert_manager_last_error_message()`.
#[no_mangle]
pub extern "C" fn llama_expert_ticket_wait_any_ready(
    ticket: *const llama_expert_ticket_ffi,
    layer: i32,
    experts: *const i32,
    count: usize,
    expert_out: *mut i32,
) -> *mut llama_expert_handle_ffi {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let ticket = ticket_ref(ticket)?;
        if count > 0 && experts.is_null() {
            return Err(ExpertCacheError::NullExperts);
        }
        if expert_out.is_null() {
            return Err(ExpertCacheError::NullExpertOut);
        }

        let layer = id_to_usize("layer", layer)?;
        let expert_ids = if count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(experts, count) }
        };
        let keys = expert_ids
            .iter()
            .copied()
            .map(|expert| Ok(ExpertKey::new(layer, id_to_usize("expert", expert)?)))
            .collect::<Result<Vec<_>, ExpertCacheError>>()?;

        let (key, handle) = ticket
            .runtime
            .block_on(ticket.ticket.wait_any_ready(&keys))
            .map_err(scheduler_error)?;
        let expert = i32::try_from(key.expert()).map_err(|_| {
            ExpertCacheError::Backend("ready expert id does not fit i32".to_string())
        })?;
        unsafe {
            *expert_out = expert;
        }
        Ok(handle_from_shared(handle))
    }));

    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

/// Releases all handles pinned by a ticket. Handles already returned to C remain
/// pinned until individually released.
#[no_mangle]
pub extern "C" fn llama_expert_ticket_release(ticket: *mut llama_expert_ticket_ffi) {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), ExpertCacheError> {
        let ticket = ticket_mut(ticket)?;
        ticket.runtime.block_on(ticket.ticket.release());
        Ok(())
    }));

    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => set_last_error(error),
        Err(_) => set_last_error(ExpertCacheError::Panic),
    }
}

/// Frees an expert ticket created by Rust.
#[no_mangle]
pub extern "C" fn llama_expert_ticket_free(ticket: *mut llama_expert_ticket_ffi) {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), ExpertCacheError> {
        if ticket.is_null() {
            return Ok(());
        }
        let _drop_ticket = unsafe { Box::from_raw(ticket) };
        Ok(())
    }));

    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => set_last_error(error),
        Err(_) => set_last_error(ExpertCacheError::Panic),
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_manager_stats(
    manager: *mut llama_expert_manager_ffi,
    hit_out: *mut usize,
    miss_out: *mut usize,
) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let manager = manager_mut(manager)?;
        let (hit, miss) = manager.manager.stats();
        if !hit_out.is_null() {
            unsafe { *hit_out = hit };
        }
        if !miss_out.is_null() {
            unsafe { *miss_out = miss };
        }
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_handle_host_base_ptr(
    handle: *const llama_expert_handle_ffi,
) -> *mut u8 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        Ok(scheduler_handle_ref(handle).host_ptr())
    }));

    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_handle_device_base_ptr(
    handle: *const llama_expert_handle_ffi,
) -> *const u8 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        Ok(scheduler_handle_ref(handle).device_ptr())
    }));

    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_handle_host_ptr(
    handle: *const llama_expert_handle_ffi,
    part: i32,
) -> *mut u8 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        let part = part_from_i32(part)?;
        Ok(scheduler_handle_ref(handle)
            .get_part(part)
            .map_err(scheduler_error)?
            .host_ptr as *mut u8)
    }));

    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_handle_device_ptr(
    handle: *const llama_expert_handle_ffi,
    part: i32,
) -> *const u8 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        let part = part_from_i32(part)?;
        Ok(scheduler_handle_ref(handle)
            .get_part(part)
            .map_err(scheduler_error)?
            .device_ptr)
    }));

    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(error)) => {
            set_last_error(error);
            ptr::null()
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_handle_part_size(
    handle: *const llama_expert_handle_ffi,
    part: i32,
) -> usize {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        let part = part_from_i32(part)?;
        Ok(scheduler_handle_ref(handle)
            .get_part(part)
            .map_err(scheduler_error)?
            .data_size)
    }));

    match result {
        Ok(Ok(size)) => size,
        Ok(Err(error)) => {
            set_last_error(error);
            0
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_handle_slot_id(handle: *const llama_expert_handle_ffi) -> i32 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        i32::try_from(scheduler_handle_ref(handle).slot_id())
            .map_err(|_| ExpertCacheError::Backend("expert slot id does not fit i32".to_string()))
    }));

    match result {
        Ok(Ok(slot)) => slot,
        Ok(Err(error)) => {
            set_last_error(error);
            -1
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn llama_expert_handle_generation(handle: *const llama_expert_handle_ffi) -> u64 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        Ok(scheduler_handle_ref(handle).generation())
    }));

    match result {
        Ok(Ok(generation)) => generation,
        Ok(Err(error)) => {
            set_last_error(error);
            0
        }
        Err(_) => {
            set_last_error(ExpertCacheError::Panic);
            0
        }
    }
}

/// Releases one pinned expert handle.
#[no_mangle]
pub extern "C" fn llama_expert_manager_release(
    manager: *mut llama_expert_manager_ffi,
    handle: *mut llama_expert_handle_ffi,
) {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), ExpertCacheError> {
        let _ = manager_mut(manager)?;
        let handle = unsafe { handle.as_mut() }.ok_or(ExpertCacheError::NullHandle)?;
        let _drop_handle = unsafe { Box::from_raw(handle) };
        Ok(())
    }));

    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => set_last_error(error),
        Err(_) => set_last_error(ExpertCacheError::Panic),
    }
}

/// Frees an expert manager created by Rust.
#[no_mangle]
pub extern "C" fn llama_expert_manager_free(manager: *mut llama_expert_manager_ffi) {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), ExpertCacheError> {
        if manager.is_null() {
            return Ok(());
        }
        let _drop_manager = unsafe { Box::from_raw(manager) };
        Ok(())
    }));

    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => set_last_error(error),
        Err(_) => set_last_error(ExpertCacheError::Panic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_conversion_rejects_invalid_part() {
        let raw = llama_expert_slice_ffi {
            part: 99,
            file_id: 0,
            file_offset: 0,
            file_size: 0,
        };

        assert!(tensor_meta_from_ffi(0, 0, raw).is_err());
    }
}
