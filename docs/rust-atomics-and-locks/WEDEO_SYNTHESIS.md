# Wedeo Threading Synthesis: Rust Atomics and Locks Applied

Practical analysis of the wedeo H.264 decoder's threading code against
the patterns and principles from *Rust Atomics and Locks* by Mara Bos.

---

## 1. Memory Ordering for Row Wavefront

### Current code (deblock.rs)

```rust
// Producer side — after completing row my_row:
row_progress[my_row as usize].store(1, Ordering::Release);

// Consumer side — before starting row my_row, wait for row my_row-1:
while row_progress[(my_row - 1) as usize].load(Ordering::Acquire) == 0 {
    std::hint::spin_loop();
}
```

### Verdict: Acquire/Release is correct and sufficient

Ch 3 ("Release and Acquire Ordering") defines the contract precisely:

> A happens-before relationship is formed when an acquire-load operation
> observes the result of a release-store operation. In this case, the store
> and everything before it, happened before the load and everything after it.

Our wavefront has exactly this topology:

1. Thread A finishes deblocking row N (pixel writes to `pic.y`, `pic.u`,
   `pic.v` are all non-atomic).
2. Thread A does `row_progress[N].store(1, Release)`.
3. Thread B does `row_progress[N].load(Acquire)` and sees `1`.
4. Thread B starts deblocking row N+1, reading pixels that overlap with
   row N's write region (the 3-pixel overlap from the vertical filter).

The Release store in step 2 guarantees that all pixel writes from step 1
are visible to Thread B after the Acquire load in step 3. This is the
textbook "release data, acquire data" pattern described in the locking
example in Ch 3.

### When would SeqCst be needed?

Ch 3 ("Sequentially Consistent Ordering") states:

> While it might seem like the easiest memory ordering to reason about,
> SeqCst ordering is almost never necessary in practice. In nearly all
> cases, regular acquire and release ordering suffice.

SeqCst is needed only when you need a **globally consistent total order**
across multiple independent atomic variables observed by multiple threads.
The canonical example from Ch 3:

```rust
static A: AtomicBool = AtomicBool::new(false);
static B: AtomicBool = AtomicBool::new(false);

// Thread 1: A.store(true, SeqCst); if !B.load(SeqCst) { ... }
// Thread 2: B.store(true, SeqCst); if !A.load(SeqCst) { ... }
```

Both threads must agree on whether A or B was set first. This pattern does
not appear in our wavefront — each row only depends on one predecessor
row, and the dependency is a simple single-variable release/acquire pair.

**Conclusion: Acquire/Release is the right choice. Upgrading to SeqCst
would add unnecessary overhead (a full memory barrier on ARM64) with no
correctness benefit.**

### Optimization note: fence batching

Ch 3 ("Fences") shows that a single `fence(Acquire)` after a relaxed load
can replace per-variable acquire loads, and vice versa for release:

```rust
// Instead of:
a.store(1, Release);
// Could be:
fence(Release);
a.store(1, Relaxed);
```

This is not useful for our single-variable wavefront pattern, but becomes
relevant if the future `SharedPicture` design has multiple atomics
(e.g. row progress + ready flag) — a single acquire fence after checking
all of them avoids multiple acquire loads.

---

## 2. SyncPic Safety Proof

### Current code (deblock.rs)

```rust
struct SyncPic(*mut PictureBuffer);
unsafe impl Send for SyncPic {}
unsafe impl Sync for SyncPic {}
```

### What the book says about Send/Sync (Ch 1)

Ch 1 ("Thread Safety: Send and Sync"):

> Raw pointers (`*const T` and `*mut T`) are neither Send nor Sync,
> since the compiler doesn't know much about what they represent.

> Implementing these traits requires the `unsafe` keyword, since the
> compiler cannot check for you if it's correct. It's a promise you make
> to the compiler, which it will just have to trust.

The `unsafe impl Send + Sync` is needed because `*mut PictureBuffer`
opts out of both traits. The question is whether the safety invariants
actually hold.

### Safety argument

The `SyncPic` is used exclusively within `std::thread::scope`, which
guarantees all spawned threads join before `scope` returns. The invariants:

1. **Lifetime**: The `PictureBuffer` outlives the scope (it's `&mut
   PictureBuffer` from the caller). `scope` joins all threads, so the
   pointer is never dangling.
2. **Exclusive access**: The wavefront ensures disjoint row access. Row N
   writes to pixels `[N*16-3, N*16+15]` in luma (and corresponding chroma).
   No two rows with both in-flight can touch the same pixels, because a
   thread waits for `row_progress[N-1]` before starting row N. So two
   `&mut` aliases to different regions never overlap.
3. **The Release/Acquire pair** ensures pixel writes from the previous row
   are visible before the next row reads them.

### Is there a better pattern?

Ch 4 ("Building Our Own Spin Lock") shows the idiomatic pattern for
wrapping data in `UnsafeCell` with restricted `Sync`:

```rust
pub struct SpinLock<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}
unsafe impl<T> Sync for SpinLock<T> where T: Send {}
```

Ch 6 ("Building Our Own Arc") shows the `NonNull<T>` + manual
Send/Sync pattern:

```rust
unsafe impl<T: Send + Sync> Send for Arc<T> {}
unsafe impl<T: Send + Sync> Sync for Arc<T> {}
```

**Recommended improvement:** Replace the raw pointer wrapper with a type
that encodes the row-disjointness invariant more clearly and avoids the
`unsafe { &mut *sync_pic.0 }` in the hot loop. Two options:

**Option A: Row-sliced borrows (no unsafe in worker)**

Split the picture into per-row slices *before* spawning threads, so each
thread receives `&mut [u8]` for its luma/chroma rows. This eliminates the
`SyncPic` entirely. The challenge is that deblocking row N reads the bottom
pixels of row N-1, so the split must account for the 3-pixel overlap.
This can be done with a custom `RowSlice` type that borrows the full
picture immutably for reads and has exclusive slices for writes.

**Option B: UnsafeCell wrapper (safer than raw pointer)**

```rust
struct SharedPicture {
    inner: UnsafeCell<PictureBuffer>,
}
unsafe impl Sync for SharedPicture {}
// Not Send — stays on the creating thread's stack.

impl SharedPicture {
    /// SAFETY: Caller must ensure no other thread is writing to
    /// the same row region concurrently.
    unsafe fn get_mut(&self) -> &mut PictureBuffer {
        &mut *self.inner.get()
    }
}
```

This follows the Ch 4 pattern (UnsafeCell + limited Sync impl) and is
marginally safer than a raw `*mut` because:
- The `UnsafeCell` is the canonical Rust primitive for interior
  mutability (Ch 1, "UnsafeCell").
- It communicates intent: "this is shared mutable state with a custom
  synchronization protocol."
- The `Sync` impl is on a named type with documentation, not a one-off
  wrapper around a raw pointer.

**Verdict: The current code is sound, but Option B is a low-cost
improvement that better communicates the safety invariant.**

---

## 3. Arc in DPB

### Current code (dpb.rs)

```rust
pub struct DpbEntry {
    pub pic: Arc<PictureBuffer>,
    // ...
}
```

### Book's Arc correctness requirements (Ch 6)

Ch 6 ("Building Our Own Arc") lays out the key properties:

1. **Clone increments ref count (Relaxed is fine):**

   > We can use Relaxed memory ordering to increment the reference counter,
   > since there are no operations on other variables that need to strictly
   > happen before or after this atomic operation.

2. **Drop decrements with Release; final drop needs Acquire fence:**

   > We can't use Relaxed ordering, since we need to make sure that nothing
   > is still accessing the data when we drop it. [...] We'll use only
   > Release for the fetch_sub operation and a separate Acquire fence only
   > when necessary.

   ```rust
   if self.data().ref_count.fetch_sub(1, Release) == 1 {
       fence(Acquire);
       unsafe { drop(Box::from_raw(self.ptr.as_ptr())); }
   }
   ```

3. **Send + Sync require T: Send + Sync:**

   > Sending an Arc<T> across threads results in a T object being shared,
   > requiring T to be Sync. Similarly, sending an Arc<T> across threads
   > could result in another thread dropping that T, effectively
   > transferring it to the other thread, requiring T: Send.

4. **Overflow protection:**

   > We'll keep the original fetch_add and simply abort the whole process
   > if we get uncomfortably close to overflowing.

### Does our usage satisfy these requirements?

**Yes, with caveats.** We use the standard library's `Arc`, which
implements all the above correctly. Our usage pattern:

- `PictureBuffer` is `Clone` and contains `Vec<u8>` — it is `Send + Sync`.
- `Arc<PictureBuffer>` is created once per decoded frame.
- Clones are shared into reference lists and DPB entries on the **same
  thread** (single-threaded decode path today).
- The `Arc` is read-only after DPB insertion (reference pictures are never
  mutated).

**Current single-threaded safety:** Since all DPB operations happen on one
thread today, the `Arc` overhead (atomic ref count operations) is pure
overhead. A `Rc` would suffice and be faster. However, the `Arc` is
forward-looking: when frame-level threading is added, reference pictures
will be shared across decode threads, and `Arc` is the correct choice.

**Future multi-threaded concern:** When multiple threads decode frames
concurrently, they will need to share `Arc<PictureBuffer>` across threads
for reference frame access. The `Arc` handles this correctly as long as:
- No thread mutates the `PictureBuffer` after DPB insertion. (Currently
  enforced by the single-threaded decode path; will need enforcement in
  the multi-threaded design.)
- The `PictureBuffer` is fully written before being wrapped in `Arc`.
  This is guaranteed today because deblocking completes before DPB
  insertion.

### get_mut pattern (Ch 6)

The book shows `Arc::get_mut` for conditional exclusive access:

```rust
pub fn get_mut(arc: &mut Self) -> Option<&mut T> {
    if arc.data().ref_count.load(Relaxed) == 1 {
        fence(Acquire);
        unsafe { Some(&mut arc.ptr.as_mut().data) }
    } else {
        None
    }
}
```

This pattern could be useful for recycling `PictureBuffer` allocations: if
a picture's `Arc` strong count is 1 (only the DPB holds it), we can reuse
the allocation via `Arc::get_mut` instead of allocating a new one. This
avoids the clone cost of `PictureBuffer` (which contains large `Vec<u8>`
plane allocations).

---

## 4. Spin Loop vs Condvar

### Current code (deblock.rs)

```rust
while row_progress[(my_row - 1) as usize].load(Ordering::Acquire) == 0 {
    std::hint::spin_loop();
}
```

### What the book says

**Ch 4 ("Building Our Own Spin Lock"):**

> A spin lock is a mutex that does exactly that. Attempting to lock an
> already locked mutex will result in busy-looping or spinning: repeatedly
> trying over and over again until it finally succeeds. This can waste
> processor cycles, but can sometimes result in lower latency when locking.

> Within the while loop, we use a spin loop hint, which tells the
> processor that we're spinning while waiting for something to change.
> [...] it might temporarily slow down or prioritize other useful things
> it can do.

**Ch 9 ("Building Our Own Locks" — Mutex, Optimizing Further):**

> The only way to avoid the wait and wake operations is to go back to our
> spin lock implementation. While spinning is usually very inefficient,
> it at least does avoid the potential overhead of a syscall. The only
> situation where spinning can be more efficient is when waiting for only
> a very short time.

> We can try to combine the best of both approaches by spinning for a very
> short amount of time before resorting to calling wait(). That way, if
> the lock is released very quickly, we don't need to call wait() at all.

The book's optimized mutex (Ch 9) spins for ~100 iterations before falling
back to `wait()`:

```rust
fn lock_contended(state: &AtomicU32) {
    let mut spin_count = 0;
    while state.load(Relaxed) == 1 && spin_count < 100 {
        spin_count += 1;
        std::hint::spin_loop();
    }
    if state.compare_exchange(0, 1, Acquire, Relaxed).is_ok() {
        return;
    }
    while state.swap(2, Acquire) != 0 {
        wait(state, 2);
    }
}
```

**Ch 1 ("Condition Variables"):**

> Condition variables are a more commonly used option for waiting for
> something to happen to data protected by a mutex. [...] To avoid the
> issue of missing notifications in the brief moment between unlocking a
> mutex and waiting for a condition variable, condition variables provide
> a way to atomically unlock the mutex and start waiting.

### Analysis for our deblocking wavefront

The wavefront spin wait is correct for our use case because:

1. **Wait time is short and bounded.** Thread B waits for thread A to
   finish one MB row of deblocking. With ~30 MBs per row at 1080p, this is
   microseconds of work — well within the "very short time" where spinning
   beats syscalls.

2. **Compute-bound pipeline.** During deblocking, all cores should be
   doing useful work. Spinning threads do not block useful work because the
   wavefront is designed so that all threads except possibly one are
   deblocking a row. A context switch from a `wait()` syscall would be far
   more expensive than the brief spin.

3. **No oversubscription.** The code caps `num_threads` at
   `available_parallelism()` and `mb_height / 2`, so we never have more
   spinning threads than physical cores.

### When to switch to Condvar

A Condvar (or futex-style wait) should be used when:

- The wait time is **unpredictable or potentially long** (milliseconds+).
- The system is **oversubscribed** (more threads than cores).
- The waiting thread would be **wasting a core** that another thread could
  use productively.

For the future **frame-level threading** (where a thread decodes frame N
while another decodes frame N+1, waiting for reference rows from frame N),
wait times are longer (an entire MB row of decode + reconstruct + deblock),
and spinning would waste a core. **Use a Condvar or futex-style wait for
frame-level row dependencies.**

### Recommended hybrid approach for frame-level threading

Follow the Ch 9 pattern — spin briefly, then fall back:

```rust
fn wait_for_row(row_progress: &AtomicI32, target_row: i32) {
    // Phase 1: spin briefly (good for short waits)
    let mut spin_count = 0;
    while row_progress.load(Relaxed) < target_row && spin_count < 100 {
        spin_count += 1;
        std::hint::spin_loop();
    }
    // Phase 2: check with Acquire before proceeding or sleeping
    if row_progress.load(Acquire) >= target_row {
        return;
    }
    // Phase 3: fall back to OS wait
    // (use atomic-wait crate or std::thread::park/Condvar)
    while row_progress.load(Acquire) < target_row {
        atomic_wait::wait(row_progress, row_progress.load(Relaxed));
    }
}
```

Note the use of `Relaxed` during the spin phase (cheaper on ARM64) with a
final `Acquire` load to establish the happens-before relationship, matching
the fence optimization pattern from Ch 3.

---

## 5. Future SharedPicture Design

### Planned design

A `SharedPicture` type for frame-level threading:
- Multiple decode threads share a picture (one writer, N readers).
- The writer thread decodes and deblocks rows top-to-bottom.
- Reader threads (decoding later frames) read reference pixels from rows
  that the writer has completed.
- Row progress is tracked atomically.

### Applicable patterns from the book

#### Pattern 1: UnsafeCell + AtomicI32 progress (Ch 4 SpinLock pattern)

```rust
pub struct SharedPicture {
    /// The picture data. Writer has exclusive access to unfinished rows;
    /// readers may access completed rows.
    data: UnsafeCell<PictureBuffer>,
    /// Row index of the last fully deblocked row. -1 = no rows ready.
    /// Writer stores with Release; readers load with Acquire.
    row_progress: AtomicI32,
}

// SAFETY: Access is governed by row_progress — readers only access
// rows <= row_progress.load(Acquire), and the writer only writes
// rows > row_progress (not yet published).
unsafe impl Sync for SharedPicture {}
```

This mirrors Ch 4's `SpinLock<T>` pattern: UnsafeCell for the data,
atomic for the synchronization state, manual Sync impl with a documented
safety invariant.

The Release/Acquire contract from Ch 3 ensures that when a reader sees
`row_progress >= N`, all pixel writes for rows 0..=N are visible.

#### Pattern 2: Arc for ownership (Ch 6)

```rust
pub type SharedPictureRef = Arc<SharedPicture>;
```

Using `Arc` for the `SharedPicture` follows Ch 6's model:
- The decode thread creates the `SharedPicture` and holds one `Arc`.
- Each reference frame slot in other decode threads holds a cloned `Arc`.
- When all references are dropped, the picture and its allocation are
  freed.

Key Ch 6 requirements:
- `SharedPicture` must be `Send + Sync` (we impl `Sync` manually above;
  `Send` is automatic since `UnsafeCell<PictureBuffer>` is `Send`).
- The `Arc` ref count uses `Relaxed` for increment, `Release` for
  decrement, `Acquire` fence on final drop — standard library handles this.

#### Pattern 3: Condvar for row-ready notification (Ch 1 + Ch 9)

Ch 1 ("Condition Variables") explains:

> Condition variables are a more commonly used option for waiting for
> something to happen to data protected by a mutex. They have two basic
> operations: wait and notify.

For frame-level threading, the reader thread may need to wait for a
specific row to become available. A Condvar is appropriate here because
the wait times can be long (the writer might be decoding complex MBs).

However, Ch 9 ("Condition Variable") shows a lighter alternative using
futex-style `wait`/`wake` operations on the `AtomicI32` directly,
avoiding the need for a `Mutex`:

```rust
impl SharedPicture {
    /// Called by the writer after completing row `row`.
    pub fn publish_row(&self, row: i32) {
        self.row_progress.store(row, Release);
        atomic_wait::wake_all(&self.row_progress);
    }

    /// Called by readers to wait until row `row` is available.
    pub fn wait_for_row(&self, row: i32) {
        loop {
            let current = self.row_progress.load(Acquire);
            if current >= row {
                return;
            }
            atomic_wait::wait(&self.row_progress, current);
        }
    }
}
```

This follows the Ch 9 mutex pattern where `wait()` only blocks if the
value hasn't changed, avoiding missed wakeups. As Ch 9 notes:

> In general, the atomic wait and wake functions never play a factor in
> correctness, from a memory safety perspective. They are only a (very
> serious) optimization to avoid busy-looping.

This means our Release/Acquire on `row_progress` provides the correctness
guarantee, and `wait`/`wake` are purely an efficiency optimization.

#### Pattern 4: Weak pointers for output queue (Ch 6)

Ch 6 introduces `Weak<T>` for breaking reference cycles. In our design,
the output queue could hold `Weak<SharedPicture>` references:
- If the DPB still holds a strong `Arc` (picture is a reference frame),
  the picture stays alive.
- If the DPB has evicted it, the `Weak` can detect this and skip output.

This is probably unnecessary for our current design (the DPB and output
queue have clear ownership semantics), but is worth knowing about.

### Putting it together

```rust
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

pub struct SharedPicture {
    data: UnsafeCell<PictureBuffer>,
    row_progress: AtomicI32,  // last completed row, -1 = none
    mb_height: u32,           // immutable after creation
}

// SAFETY: Concurrent access is mediated by row_progress.
// Writer (single thread) writes to rows > row_progress.
// Readers access rows <= row_progress.load(Acquire).
unsafe impl Sync for SharedPicture {}

impl SharedPicture {
    pub fn new(pic: PictureBuffer) -> Arc<Self> {
        let mb_height = pic.mb_height;
        Arc::new(Self {
            data: UnsafeCell::new(pic),
            row_progress: AtomicI32::new(-1),
            mb_height,
        })
    }

    /// Get shared read access to the picture data.
    /// SAFETY: Caller must only access rows <= self.row_progress.
    pub unsafe fn data(&self) -> &PictureBuffer {
        &*self.data.get()
    }

    /// Get exclusive write access to the picture data.
    /// SAFETY: Caller must be the sole writer and only write to
    /// rows > self.row_progress.
    pub unsafe fn data_mut(&self) -> &mut PictureBuffer {
        &mut *self.data.get()
    }

    pub fn publish_row(&self, row: i32) {
        self.row_progress.store(row, Ordering::Release);
        // Wake all waiting reader threads.
        atomic_wait::wake_all(
            // atomic-wait requires &AtomicU32; transmute or use
            // a u32 progress counter instead of i32.
            todo!("wire up wake")
        );
    }

    pub fn wait_for_row(&self, row: i32) {
        let mut spin = 0;
        // Phase 1: brief spin (good for in-flight rows)
        while self.row_progress.load(Ordering::Relaxed) < row && spin < 100 {
            spin += 1;
            std::hint::spin_loop();
        }
        // Phase 2: Acquire load for happens-before
        if self.row_progress.load(Ordering::Acquire) >= row {
            return;
        }
        // Phase 3: OS-level wait
        loop {
            let current = self.row_progress.load(Ordering::Acquire);
            if current >= row {
                return;
            }
            // wait on the atomic until it changes
            todo!("wire up wait");
        }
    }

    pub fn is_complete(&self) -> bool {
        self.row_progress.load(Ordering::Relaxed) >= self.mb_height as i32 - 1
    }
}
```

**Note:** `atomic-wait` only supports `AtomicU32`. For production, use
`AtomicU32` for `row_progress` (with u32::MAX as the "no rows ready"
sentinel), or use `std::sync::Condvar` + `Mutex<i32>` as the wait
mechanism while keeping the `AtomicI32` for the fast-path relaxed check.

---

## 6. Specific Code Recommendations

### 6.1. Replace SyncPic with UnsafeCell wrapper

**File:** `deblock.rs:16-23`

**Current:**
```rust
struct SyncPic(*mut PictureBuffer);
unsafe impl Send for SyncPic {}
unsafe impl Sync for SyncPic {}
```

**Recommended:**
```rust
/// Wrapper for shared mutable access to a PictureBuffer during
/// parallel deblocking. Access safety is guaranteed by the row
/// wavefront protocol: each thread writes exclusively to its
/// assigned row, and waits for the previous row to complete
/// (Acquire/Release on row_progress) before starting.
struct SharedPic(UnsafeCell<PictureBuffer>);

// SAFETY: Concurrent access is mediated by row_progress atomics.
// The wavefront ensures no two threads write the same pixel region.
unsafe impl Sync for SharedPic {}
```

This follows the Ch 4 UnsafeCell pattern and avoids raw pointer
manipulation. The `Send` impl is not needed since the `SharedPic` is
created on the stack and shared by reference into the scoped threads.

### 6.2. Use Relaxed for next_row counter

**File:** `deblock.rs:1738`

**Current:**
```rust
let my_row = next_row.fetch_add(1, Ordering::Relaxed);
```

**Correct as-is.** The `next_row` counter is purely a work-distribution
mechanism. No data depends on which specific value a thread reads — each
thread gets a unique row number from the atomic increment. As Ch 3
("Relaxed Ordering") states:

> Relaxed still guarantees consistency on a single atomic variable, but
> does not promise anything about the relative order of operations between
> different variables.

The per-row data dependencies are handled by the `row_progress` atomics,
not by `next_row`. Relaxed is correct.

### 6.3. Consider Relaxed load in spin loop, Acquire on exit

**File:** `deblock.rs:1745`

**Current:**
```rust
while row_progress[(my_row - 1) as usize].load(Ordering::Acquire) == 0 {
    std::hint::spin_loop();
}
```

**Optional optimization:** Use Relaxed loads during the spin loop and a
single Acquire fence when exiting, following the Ch 3 fence pattern:

```rust
while row_progress[(my_row - 1) as usize].load(Ordering::Relaxed) == 0 {
    std::hint::spin_loop();
}
std::sync::atomic::fence(Ordering::Acquire);
```

On x86, this makes no difference (acquire loads are free). On ARM64,
this replaces N `ldar` instructions (one per spin iteration) with N `ldr`
instructions plus one `dmb ish` fence. For a tight spin loop, the
difference is negligible, but it is technically more efficient and
follows the pattern from Ch 3 ("Fences") where multiple relaxed loads
are guarded by a single acquire fence.

**Recommendation: Keep the current Acquire load.** The clarity benefit
of having the ordering on the load itself (rather than a separate fence)
outweighs the negligible performance difference. This matches the book's
own advice — the fence optimization is primarily useful when guarding
*multiple* atomic variables, not a single one in a tight loop.

### 6.4. Buffer recycling via Arc::get_mut

**File:** `dpb.rs` — when allocating PictureBuffers for new frames.

The standard library's `Arc::get_mut` checks if the strong count is 1 and
returns `&mut T` if so, following the Ch 6 pattern. This can avoid
allocating new `Vec<u8>` buffers for each frame:

```rust
// When a DPB slot is freed and has ref_count == 1, recycle its buffer:
if let Some(entry) = dpb.entries[idx].take() {
    if let Some(pic) = Arc::into_inner(entry.pic) {
        // Reuse pic's Vec allocations for the next frame
        recycled_buffers.push(pic);
    }
}
```

`Arc::into_inner` (stable since Rust 1.70) returns `Some(T)` if this is
the last `Arc`, consuming it without the overhead of `try_unwrap`.

### 6.5. Add debug assertions for wavefront safety

The book emphasizes that `unsafe` code should document and check its
invariants. Add debug assertions to the wavefront:

```rust
// Before writing pixels in row my_row:
debug_assert!(
    my_row == 0 || row_progress[(my_row - 1) as usize].load(Ordering::Relaxed) != 0,
    "row wavefront violated: row {} started before row {} completed",
    my_row, my_row - 1
);
```

This catches wavefront bugs in debug builds without runtime cost in
release builds.

### 6.6. Document the memory ordering contract

Add a module-level comment in `deblock.rs` explaining the memory ordering
contract, referencing the specific pattern from Ch 3:

```rust
// Memory ordering contract for parallel deblocking:
//
// The row wavefront uses Acquire/Release on row_progress to create
// happens-before relationships between consecutive rows. This is the
// "release data, acquire data" pattern (Bos, Ch 3):
//
//   Thread A: [write pixels for row N] → row_progress[N].store(1, Release)
//   Thread B: row_progress[N].load(Acquire) → [read/write pixels for row N+1]
//
// The Release ensures all pixel writes from row N are visible to
// any thread that subsequently observes the Acquire load returning 1.
// No SeqCst is needed because each row depends on exactly one
// predecessor — there is no cross-variable ordering requirement.
```

---

## Summary Table

| Topic | Current State | Book Reference | Action |
|-------|--------------|----------------|--------|
| Wavefront memory ordering | Acquire/Release | Ch 3: Release and Acquire Ordering | Correct, no change needed |
| SyncPic raw pointer | `*mut` + Send + Sync | Ch 1: Send/Sync, Ch 4: UnsafeCell pattern | Replace with UnsafeCell wrapper |
| Arc in DPB | `Arc<PictureBuffer>` | Ch 6: Building Our Own Arc | Correct; add buffer recycling via Arc::into_inner |
| Spin loop in wavefront | spin_loop() hint | Ch 4: SpinLock, Ch 9: Optimizing Further | Correct for short waits; use hybrid for frame-level |
| Future SharedPicture | Not yet implemented | Ch 4, 5, 6, 9 patterns | UnsafeCell + AtomicI32 + Arc + hybrid wait |
| next_row Relaxed | Relaxed fetch_add | Ch 3: Relaxed Ordering | Correct, no change needed |
