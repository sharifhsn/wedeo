// Persistent thread pool for frame-level decode parallelism.
//
// Replaces per-frame `std::thread::spawn` with a fixed pool of N worker
// threads that receive work via `std::sync::mpsc` channels. Workers are
// stateless — each `FrameWork` is self-contained.
//
// Work distribution: `Arc<Mutex<mpsc::Receiver<Option<FrameWork>>>>` shared
// across workers. Each worker locks, calls `recv()`, unlocks. Automatic
// load balancing. `None` = shutdown sentinel.
//
// Result delivery: `mpsc::Sender<FrameResult>` per worker → single
// `mpsc::Receiver<FrameResult>` on main thread. Results may arrive
// out of order; callers use `sequence_id` for FIFO ordering.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use wedeo_core::{ColorDesc, CropRect, Error, Rational, Timestamp};

use crate::deblock;
use crate::decoder::H264Decoder;
use crate::mb::FrameDecodeContext;
use crate::shared_picture::SharedPicture;
use crate::slice::SliceHeader;

/// Per-slice snapshot of DPB-derived state + RBSP data for worker thread.
/// Mirrors `SliceWorkUnit` in decoder.rs — moved here for cross-module access.
pub(crate) struct SliceWorkUnit {
    pub hdr: SliceHeader,
    pub rbsp: Vec<u8>,
    pub sps: crate::sps::Sps,
    pub pps: crate::pps::Pps,
    pub ref_pics_l0: Vec<Arc<SharedPicture>>,
    pub ref_pics_l1: Vec<Arc<SharedPicture>>,
    pub slice_idx: u16,
    pub cur_l0_ref_poc: Vec<i32>,
    pub cur_l0_ref_dpb: Vec<usize>,
    pub cur_l1_ref_poc: Vec<i32>,
    pub cur_l1_ref_dpb: Vec<usize>,
    pub col_mv: Vec<[i16; 2]>,
    pub col_ref: Vec<i8>,
    pub col_mv_l1: Vec<[i16; 2]>,
    pub col_ref_l1: Vec<i8>,
    pub col_mb_intra: Vec<bool>,
    pub col_poc: i32,
    pub col_l1_is_long_term: bool,
    pub col_ref_poc_l0: Vec<i32>,
    pub col_ref_poc_l1: Vec<i32>,
    pub implicit_weight: Vec<Vec<i32>>,
    pub cur_poc: i32,
    /// MB address where the next slice starts (or total_mbs for the last slice).
    /// Used to terminate the decode loop at slice boundaries for parallel decode.
    pub next_slice_first_mb: u32,
}

/// Self-contained work unit for one frame decode.
pub(crate) struct FrameWork {
    pub fdc: FrameDecodeContext,
    pub slices: Vec<SliceWorkUnit>,
    pub deblock_enabled: bool,
    pub sequence_id: u64,
    // InFlightDecode metadata
    pub poc: i32,
    pub frame_num_h264: u32,
    pub nal_ref_idc: u8,
    pub is_idr: bool,
    pub is_key: bool,
    pub last_hdr: SliceHeader,
    pub ref_list_l0: Vec<usize>,
    pub ref_list_l1: Vec<usize>,
    /// VFR timing propagated from the input packet.
    pub pts: Timestamp,
    pub dts: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
    /// Active stream metadata captured at frame dispatch time.
    pub coded_width: u32,
    pub coded_height: u32,
    pub color: ColorDesc,
    pub sar: Rational,
    pub crop: CropRect,
}

/// Result of a completed frame decode.
pub(crate) struct FrameResult {
    pub decode: Box<InFlightDecode>,
    pub sequence_id: u64,
}

/// Metadata for a completed frame decode (same as before, now public(crate)).
pub(crate) struct InFlightDecode {
    pub fdc: FrameDecodeContext,
    pub poc: i32,
    pub frame_num_h264: u32,
    pub nal_ref_idc: u8,
    pub is_idr: bool,
    pub is_key: bool,
    pub last_hdr: SliceHeader,
    pub ref_list_l0: Vec<usize>,
    pub ref_list_l1: Vec<usize>,
    /// VFR timing propagated from the input packet.
    pub pts: Timestamp,
    pub dts: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
    /// Active stream metadata captured at frame dispatch time.
    pub coded_width: u32,
    pub coded_height: u32,
    pub color: ColorDesc,
    pub sar: Rational,
    pub crop: CropRect,
    /// First slice decode error encountered, if any.
    pub decode_error: Option<Error>,
}

/// Persistent thread pool for frame decode workers (or inline mode when `num_workers <= 1`).
pub(crate) struct DecodeThreadPool {
    workers: Vec<std::thread::JoinHandle<()>>,
    work_tx: Option<mpsc::Sender<Option<Box<FrameWork>>>>,
    result_rx: Option<mpsc::Receiver<FrameResult>>,
    inline_results: VecDeque<FrameResult>,
}

impl DecodeThreadPool {
    /// Create a new pool with `num_workers` threads (`<= 1` runs inline with zero threads spawned).
    pub fn new(num_workers: usize) -> Self {
        if num_workers <= 1 {
            return Self {
                workers: Vec::new(),
                work_tx: None,
                result_rx: None,
                inline_results: VecDeque::new(),
            };
        }

        let (work_tx, work_rx) = mpsc::channel::<Option<Box<FrameWork>>>();
        let (result_tx, result_rx) = mpsc::channel::<FrameResult>();
        let work_rx = Arc::new(Mutex::new(work_rx));

        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let rx = Arc::clone(&work_rx);
            let tx = result_tx.clone();
            let handle = std::thread::spawn(move || {
                worker_loop(rx, tx);
            });
            workers.push(handle);
        }

        Self {
            workers,
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            inline_results: VecDeque::new(),
        }
    }

    /// Submit a frame for decode. Executes inline when no worker threads are running.
    pub fn submit(&mut self, work: Box<FrameWork>) {
        if let Some(tx) = &self.work_tx {
            tx.send(Some(work)).expect("pool work channel closed");
        } else {
            let sequence_id = work.sequence_id;
            let decode = decode_frame(*work);
            self.inline_results.push_back(FrameResult {
                decode,
                sequence_id,
            });
        }
    }

    /// Receive the next completed frame result. Blocks until one is available.
    pub fn recv(&mut self) -> Option<FrameResult> {
        if let Some(rx) = &self.result_rx {
            rx.recv().ok()
        } else {
            self.inline_results.pop_front()
        }
    }

    /// Shut down the pool: send N shutdown sentinels and join all workers.
    pub fn shutdown(&mut self) {
        if let Some(tx) = &self.work_tx {
            for _ in &self.workers {
                let _ = tx.send(None);
            }
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for DecodeThreadPool {
    fn drop(&mut self) {
        if !self.workers.is_empty() {
            self.shutdown();
        }
    }
}

/// Worker loop: receive work items, decode frames, send results.
fn worker_loop(
    work_rx: Arc<Mutex<mpsc::Receiver<Option<Box<FrameWork>>>>>,
    result_tx: mpsc::Sender<FrameResult>,
) {
    loop {
        // Lock → recv → unlock. Minimal contention.
        let work = {
            let rx = work_rx.lock().unwrap();
            rx.recv()
        };

        match work {
            Ok(Some(fw)) => {
                let sequence_id = fw.sequence_id;
                let decode = decode_frame(*fw);
                if result_tx
                    .send(FrameResult {
                        decode,
                        sequence_id,
                    })
                    .is_err()
                {
                    // Main thread dropped the receiver — exit.
                    break;
                }
            }
            Ok(None) | Err(_) => {
                // Shutdown sentinel or channel closed.
                break;
            }
        }
    }
}

/// Decode a single frame from a FrameWork.
///
/// Multi-slice parallel decode is disabled: the `decode_frame_multi_slice`
/// path has unsound `unsafe impl Sync` (shared bi_scratch/qp/mc_scratch
/// without synchronization) and does not enforce slice boundaries
/// (`next_slice_first_mb` is computed but never checked in the decode loop).
/// All frames use the sequential path until these issues are resolved.
fn decode_frame(fw: FrameWork) -> Box<InFlightDecode> {
    decode_frame_sequential(fw)
}

/// Fast path: single-slice frame decode. Identical to the original code.
/// Inline deblocking runs with 1-row delay during decode.
fn decode_frame_sequential(fw: FrameWork) -> Box<InFlightDecode> {
    let mut fdc = fw.fdc;

    let mut decode_error: Option<Error> = None;

    // Decode the single slice (or all slices sequentially for safety)
    for slice in fw.slices.into_iter() {
        // Sequential path: move data instead of cloning (original behavior)
        fdc.current_slice = slice.slice_idx;
        fdc.cur_l0_ref_poc = slice.cur_l0_ref_poc;
        fdc.cur_l0_ref_dpb = slice.cur_l0_ref_dpb;
        fdc.cur_l1_ref_poc = slice.cur_l1_ref_poc;
        fdc.cur_l1_ref_dpb = slice.cur_l1_ref_dpb;
        fdc.col_mv = slice.col_mv;
        fdc.col_ref = slice.col_ref;
        fdc.col_mv_l1 = slice.col_mv_l1;
        fdc.col_ref_l1 = slice.col_ref_l1;
        fdc.col_mb_intra = slice.col_mb_intra;
        fdc.col_poc = slice.col_poc;
        fdc.col_l1_is_long_term = slice.col_l1_is_long_term;
        fdc.col_ref_poc_l0 = slice.col_ref_poc_l0;
        fdc.col_ref_poc_l1 = slice.col_ref_poc_l1;
        fdc.implicit_weight = slice.implicit_weight;
        fdc.cur_poc = slice.cur_poc;

        if let Err(err) = H264Decoder::decode_slice_into(
            &slice.rbsp,
            &slice.hdr,
            &slice.sps,
            &slice.pps,
            &mut fdc,
            &slice.ref_pics_l0,
            &slice.ref_pics_l1,
        ) && decode_error.is_none()
        {
            decode_error = Some(err);
        }
    }

    // Last-row deblock (inline deblock defers by 1 row)
    deblock_last_row(&mut fdc, fw.deblock_enabled);
    fdc.pic.mark_complete();

    Box::new(InFlightDecode {
        fdc,
        poc: fw.poc,
        frame_num_h264: fw.frame_num_h264,
        nal_ref_idc: fw.nal_ref_idc,
        is_idr: fw.is_idr,
        is_key: fw.is_key,
        last_hdr: fw.last_hdr,
        ref_list_l0: fw.ref_list_l0,
        ref_list_l1: fw.ref_list_l1,
        pts: fw.pts,
        dts: fw.dts,
        duration: fw.duration,
        time_base: fw.time_base,
        coded_width: fw.coded_width,
        coded_height: fw.coded_height,
        color: fw.color,
        sar: fw.sar,
        crop: fw.crop,
        decode_error,
    })
}

/// Deblock the last row (deferred by inline deblock's 1-row delay).
fn deblock_last_row(fdc: &mut FrameDecodeContext, deblock_enabled: bool) {
    if deblock_enabled {
        let mb_height = fdc.mb_height;
        let mb_width = fdc.mb_width;
        if fdc.is_mbaff {
            let pair_rows = mb_height / 2;
            if pair_rows > 0 {
                deblock::deblock_row_mbaff(
                    &mut fdc.pic,
                    &fdc.mb_info,
                    &fdc.slice_table,
                    &fdc.slice_deblock_params,
                    pair_rows - 1,
                    mb_width,
                );
            }
        } else if mb_height > 0 {
            deblock::deblock_row(
                &mut fdc.pic,
                &fdc.mb_info,
                &fdc.slice_table,
                &fdc.slice_deblock_params,
                mb_height - 1,
                mb_width,
            );
        }
    }
}
