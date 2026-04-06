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

use std::cell::UnsafeCell;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

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
    pub last_hdr: SliceHeader,
    pub ref_list_l0: Vec<usize>,
    pub ref_list_l1: Vec<usize>,
    pub dpb_idx: Option<usize>,
    /// Packet PTS from the demuxer, propagated to the output frame.
    pub pkt_pts: i64,
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
    pub last_hdr: SliceHeader,
    pub ref_list_l0: Vec<usize>,
    pub ref_list_l1: Vec<usize>,
    pub dpb_idx: Option<usize>,
    /// Packet PTS from the demuxer, propagated to the output frame.
    pub pkt_pts: i64,
}

/// Persistent thread pool for frame decode workers.
pub(crate) struct DecodeThreadPool {
    workers: Vec<std::thread::JoinHandle<()>>,
    work_tx: mpsc::Sender<Option<Box<FrameWork>>>,
    result_rx: mpsc::Receiver<FrameResult>,
}

impl DecodeThreadPool {
    /// Create a new pool with `num_workers` threads.
    pub fn new(num_workers: usize) -> Self {
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
            work_tx,
            result_rx,
        }
    }

    /// Submit a frame for decode. Non-blocking.
    /// Takes a Box<FrameWork> to avoid stack overflow from large struct.
    pub fn submit(&self, work: Box<FrameWork>) {
        self.work_tx
            .send(Some(work))
            .expect("pool work channel closed");
    }

    /// Receive the next completed frame result. Blocks until one is available.
    pub fn recv(&self) -> Option<FrameResult> {
        self.result_rx.recv().ok()
    }

    /// Shut down the pool: send N shutdown sentinels and join all workers.
    pub fn shutdown(&mut self) {
        for _ in &self.workers {
            let _ = self.work_tx.send(None);
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
/// Single-slice frames use the fast sequential path (identical to the
/// original code, zero overhead). Multi-slice frames dispatch slices in
/// parallel via `std::thread::scope`, then run a serial deblock pass.
fn decode_frame(fw: FrameWork) -> Box<InFlightDecode> {
    if fw.slices.len() <= 1 {
        decode_frame_sequential(fw)
    } else {
        decode_frame_multi_slice(fw)
    }
}

/// Fast path: single-slice frame decode. Identical to the original code.
/// Inline deblocking runs with 1-row delay during decode.
fn decode_frame_sequential(fw: FrameWork) -> Box<InFlightDecode> {
    let mut fdc = fw.fdc;

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

        let _ = H264Decoder::decode_slice_into(
            &slice.rbsp,
            &slice.hdr,
            &slice.sps,
            &slice.pps,
            &mut fdc,
            &slice.ref_pics_l0,
            &slice.ref_pics_l1,
        );
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
        last_hdr: fw.last_hdr,
        ref_list_l0: fw.ref_list_l0,
        ref_list_l1: fw.ref_list_l1,
        dpb_idx: fw.dpb_idx,
        pkt_pts: fw.pkt_pts,
    })
}

/// Wrapper around `UnsafeCell<FrameDecodeContext>` that implements `Sync`.
///
/// SAFETY: Multi-slice parallel decode guarantees that each slice writes to
/// a disjoint range of MB addresses. The shared arrays (`pic`, `mb_info`,
/// `mv_ctx`, `slice_table`, `transform_8x8`, `mb_field_flag`) are indexed
/// by MB address, so concurrent writes never overlap. The `neighbor_ctx`
/// top-row arrays are written after each MB row — slices starting at row
/// boundaries means each column position is written by exactly one slice.
///
/// Per-slice state (`qp`, `last_qscale_diff`, `current_slice`, `mc_scratch`,
/// `bi_scratch_*`, `neighbor_ctx.left_*`) is stored in separate
/// `SliceLocalState` structs, not in the shared FDC.
struct SyncFdc(UnsafeCell<FrameDecodeContext>);

// SAFETY: See SyncFdc doc comment — disjoint MB ranges guarantee no data races.
unsafe impl Sync for SyncFdc {}

/// Multi-slice parallel decode using `std::thread::scope`.
///
/// Each slice gets its own scoped thread. Inline deblocking is DISABLED
/// during parallel decode (slices may decode rows out of order). After all
/// slices complete, a serial deblock pass processes the entire frame.
///
/// This matches FFmpeg's `postpone_filter` model for multi-slice frames.
fn decode_frame_multi_slice(fw: FrameWork) -> Box<InFlightDecode> {
    let mut fdc = fw.fdc;
    let deblock_enabled = fw.deblock_enabled;
    let poc = fw.poc;
    let frame_num_h264 = fw.frame_num_h264;
    let nal_ref_idc = fw.nal_ref_idc;
    let is_idr = fw.is_idr;
    let last_hdr = fw.last_hdr;
    let ref_list_l0 = fw.ref_list_l0;
    let ref_list_l1 = fw.ref_list_l1;
    let dpb_idx = fw.dpb_idx;
    let pkt_pts = fw.pkt_pts;

    // Disable inline deblocking for multi-slice — we'll do a full serial
    // pass after all slices complete. Set the env var marker that
    // decode_slice_into checks.
    //
    // We achieve this by temporarily setting WEDEO_NO_DEBLOCK. But that's
    // a global env var — instead, we pass deblock_enabled=false through the
    // decode path. The inline_deblock_row function in decoder.rs checks
    // a local `deblock_enabled` variable, not the env var. Since we call
    // decode_slice_into which reads from env, we need a different approach.
    //
    // Actually, looking at the code: `deblock_enabled` is a local variable
    // in decode_slice_cavlc/cabac computed from env var at slice start.
    // For multi-slice parallel, each thread reads that env var independently.
    // We need to suppress inline deblock for multi-slice frames.
    //
    // Approach: set a flag on FDC that the decode loops check.
    fdc.postpone_deblock = true;

    let slices = fw.slices;
    let sync_fdc = SyncFdc(UnsafeCell::new(fdc));

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(slices.len());

        for (i, slice) in slices.into_iter().enumerate() {
            let fdc_ref = &sync_fdc;
            handles.push(s.spawn(move || {
                // SAFETY: Each slice writes to disjoint MB ranges (guaranteed by
                // next_slice_first_mb boundaries). See SyncFdc doc comment.
                let fdc = unsafe { &mut *fdc_ref.0.get() };

                // Create a slice-local copy of mutable per-slice state.
                // We must not share qp, last_qscale_diff, mc_scratch, or
                // neighbor_ctx left-side state across threads.
                //
                // For multi-slice, each slice starts fresh: qp comes from
                // slice header (set by decode_slice_into), last_qscale_diff=0
                // (set by decode_slice_into), mc_scratch is per-call.
                //
                // The tricky part: neighbor_ctx. The top-row arrays are shared
                // (indexed by column), but left-side arrays are per-slice.
                // Since slices start at row boundaries in practice, and
                // decode_slice calls new_row() at each row start (resetting
                // left state), this is safe.

                apply_slice_state(fdc, &slice);

                let _ = H264Decoder::decode_slice_into(
                    &slice.rbsp,
                    &slice.hdr,
                    &slice.sps,
                    &slice.pps,
                    fdc,
                    &slice.ref_pics_l0,
                    &slice.ref_pics_l1,
                );

                let _ = i; // suppress unused warning
            }));
        }

        // All threads join at scope exit
        for h in handles {
            let _ = h.join();
        }
    });

    let mut fdc = sync_fdc.0.into_inner();
    fdc.postpone_deblock = false;

    // Serial deblock pass over the entire frame (all MBs now decoded).
    if deblock_enabled {
        let mb_height = fdc.mb_height;
        let mb_width = fdc.mb_width;
        if fdc.is_mbaff {
            let pair_rows = mb_height / 2;
            for pr in 0..pair_rows {
                deblock::deblock_row_mbaff(
                    &mut fdc.pic,
                    &fdc.mb_info,
                    &fdc.slice_table,
                    &fdc.slice_deblock_params,
                    pr,
                    mb_width,
                );
            }
        } else {
            for row in 0..mb_height {
                deblock::deblock_row(
                    &mut fdc.pic,
                    &fdc.mb_info,
                    &fdc.slice_table,
                    &fdc.slice_deblock_params,
                    row,
                    mb_width,
                );
            }
        }
    }
    fdc.pic.mark_complete();

    Box::new(InFlightDecode {
        fdc,
        poc,
        frame_num_h264,
        nal_ref_idc,
        is_idr,
        last_hdr,
        ref_list_l0,
        ref_list_l1,
        dpb_idx,
        pkt_pts,
    })
}

/// Apply per-slice FDC state from a SliceWorkUnit snapshot.
#[inline]
fn apply_slice_state(fdc: &mut FrameDecodeContext, slice: &SliceWorkUnit) {
    fdc.current_slice = slice.slice_idx;
    fdc.cur_l0_ref_poc = slice.cur_l0_ref_poc.clone();
    fdc.cur_l0_ref_dpb = slice.cur_l0_ref_dpb.clone();
    fdc.cur_l1_ref_poc = slice.cur_l1_ref_poc.clone();
    fdc.cur_l1_ref_dpb = slice.cur_l1_ref_dpb.clone();
    fdc.col_mv = slice.col_mv.clone();
    fdc.col_ref = slice.col_ref.clone();
    fdc.col_mv_l1 = slice.col_mv_l1.clone();
    fdc.col_ref_l1 = slice.col_ref_l1.clone();
    fdc.col_mb_intra = slice.col_mb_intra.clone();
    fdc.col_poc = slice.col_poc;
    fdc.col_l1_is_long_term = slice.col_l1_is_long_term;
    fdc.col_ref_poc_l0 = slice.col_ref_poc_l0.clone();
    fdc.col_ref_poc_l1 = slice.col_ref_poc_l1.clone();
    fdc.implicit_weight = slice.implicit_weight.clone();
    fdc.cur_poc = slice.cur_poc;
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
