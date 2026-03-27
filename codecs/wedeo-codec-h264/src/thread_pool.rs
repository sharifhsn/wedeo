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

/// Decode a single frame from a FrameWork. Extracted from the inline
/// closure in `finish_current_frame` (Phase 6).
fn decode_frame(fw: FrameWork) -> Box<InFlightDecode> {
    let mut fdc = fw.fdc;

    // Decode all buffered slices
    for slice in fw.slices.into_iter() {
        // Apply per-slice FDC state from snapshot
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
    if fw.deblock_enabled {
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
    })
}
