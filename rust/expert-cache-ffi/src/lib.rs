//! Rust FFI surface connecting llama.cpp MoE expert cache calls to io-scheduler.

use io_scheduler::expert_manager::{
    ExpertHandle as SchedulerExpertHandle, ExpertKey, LlamaExpertManager, MoePart, Slice,
    TensorMeta,
};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use tokio::runtime::{Builder, Runtime};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct llama_expert_slice_ffi {
    pub part: i32,
    pub file_id: i32,
    pub file_offset: u64,
    pub file_size: u64,
}

#[derive(Debug)]
pub enum ExpertCacheError {
    InvalidPart(i32),
    InvalidId(&'static str, i32),
    InvalidSize(&'static str),
    IntegerOverflow(&'static str, u64),
    NullManager,
    NullHandle,
    NullSlice,
    NullPath,
    NullExperts,
    NullHandles,
    Backend(String),
    Panic,
}

impl std::fmt::Display for ExpertCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPart(part) => write!(f, "invalid MoE part: {part}"),
            Self::InvalidId(name, value) => write!(f, "invalid {name}: {value}"),
            Self::InvalidSize(name) => write!(f, "{name} must be > 0"),
            Self::IntegerOverflow(name, value) => {
                write!(f, "{name} does not fit usize: {value}")
            }
            Self::NullManager => write!(f, "null expert manager pointer"),
            Self::NullHandle => write!(f, "null expert handle pointer"),
            Self::NullSlice => write!(f, "null expert slice pointer"),
            Self::NullPath => write!(f, "null expert file path pointer"),
            Self::NullExperts => write!(f, "null expert ids pointer"),
            Self::NullHandles => write!(f, "null output handles pointer"),
            Self::Backend(msg) => write!(f, "{msg}"),
            Self::Panic => write!(f, "panic crossed FFI boundary"),
        }
    }
}

impl std::error::Error for ExpertCacheError {}

#[allow(non_camel_case_types)]
pub struct llama_expert_manager_ffi {
    manager: LlamaExpertManager,
    runtime: Runtime,
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

        Ok(Self { manager, runtime })
    }

    pub fn new_with_slot_size(capacity: usize, slot_size: usize) -> Result<Self, ExpertCacheError> {
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
            LlamaExpertManager::new_with_slot_size(capacity, slot_size)
        };

        Ok(Self { manager, runtime })
    }
}

#[allow(non_camel_case_types)]
pub struct llama_expert_handle_ffi {
    handle: SchedulerExpertHandle,
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
    Box::into_raw(Box::new(llama_expert_handle_ffi { handle }))
}

fn handle_ref<'a>(
    handle: *const llama_expert_handle_ffi,
) -> Result<&'a llama_expert_handle_ffi, ExpertCacheError> {
    unsafe { handle.as_ref() }.ok_or(ExpertCacheError::NullHandle)
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
        let expert_ids = unsafe { std::slice::from_raw_parts(experts, count) };
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
pub extern "C" fn llama_expert_handle_host_ptr(
    handle: *const llama_expert_handle_ffi,
    part: i32,
) -> *mut u8 {
    clear_last_error();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle_ref(handle)?;
        let part = part_from_i32(part)?;
        Ok(handle
            .handle
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
        Ok(handle
            .handle
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
        Ok(handle
            .handle
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
        i32::try_from(handle.handle.slot_id())
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
