# Contributing to wedeo

Contributions are welcome from both humans and AI agents.

## Before you start

1. Read [CLAUDE.md](CLAUDE.md) for architecture, conventions, and technical requirements
2. Read [DIVERGENCES.md](DIVERGENCES.md) for known behavioral differences vs FFmpeg
3. Read the FFmpeg C source for whatever you're implementing (`./FFmpeg/` submodule)
4. Build and run tests to confirm your environment works:
   ```bash
   cargo build && cargo nextest run && cargo clippy
   ```
   If nextest is not installed, `cargo test` also works.

## What we accept

- New codec/format implementations following the patterns in CLAUDE.md
- Bug fixes with test cases
- FATE coverage improvements
- Performance improvements with benchmarks

## What we do not accept

These rules exist to prevent low-value churn. If a PR triggers any of these,
it will be closed without review.

1. **No cosmetic-only PRs.** No reformatting, rewording comments, reorganizing
   imports, or adding documentation to undocumented FFmpeg code. If you touch a
   file, you must be fixing a bug or adding functionality.

2. **No speculative abstractions.** No traits, generics, or builders for
   hypothetical future needs. Add abstraction when there are two concrete users,
   not before.

3. **No unnecessary documentation.** Where FFmpeg has no comment, add none.
   Copy FFmpeg comments verbatim where they exist. `// Increment the counter`
   above `counter += 1` will get your PR rejected.

4. **No dependency churn.** Don't replace working code with new crates unless
   they have >100K downloads and you can demonstrate a measurable improvement.

5. **No "improvements" without measurement.** Performance claims need benchmarks.
   Correctness claims need a test that fails before and passes after.

6. **No incomplete codec implementations.** Must include decoder/encoder + FATE
   tests + bitexact or SNR comparison vs FFmpeg. No skeletons, no stubs.

7. **No generated boilerplate.** AI-generated trait impls, error types, or module
   scaffolding that doesn't serve a concrete need.

8. **No drive-by annotations.** Don't add type annotations, docstrings, or
   comments to code you didn't change.

9. **No CLAUDE.md changes without maintainer approval.** `CLAUDE.md` is the
   project's source of truth for architecture, conventions, and AI agent
   instructions. Changes require explicit sign-off from a maintainer.

## Verification requirements

Every PR must pass:

- `cargo clippy` — zero warnings
- `cargo fmt --check` — pass
- `cargo nextest run` (or `cargo test`) — all existing tests pass
- FATE tests pass for any codec/format changes
- Bitexact framecrc or SNR measurement vs FFmpeg for codec work
- New divergences documented in [DIVERGENCES.md](DIVERGENCES.md)

## Commit message format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

<body: why, not what>
```

**Types:** `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `ci`, `build`, `chore`

**Scopes** (optional): `core`, `codec/pcm`, `codec/h264`, `format/wav`,
`format/h264`, `play`, `cli`, `fate`, `symphonia`

## PR body

Every PR must include these sections:

1. **What** — one sentence
2. **Why** — problem solved or capability added
3. **Verification** — test output, FATE comparison, or benchmark
4. **FFmpeg reference** (codec/format work) — which C source files were read

## For AI agents

- Read existing code BEFORE writing anything
- Match existing code style exactly — look at neighboring files
- No `#[allow(clippy::...)]` without a comment explaining why
- No `unsafe` without a `// SAFETY:` comment explaining the invariants
- Use `wrapping_add`/`wrapping_mul`/`wrapping_neg` for arithmetic that must
  match C overflow behavior
- 64-byte SIMD padding on all buffer allocations (`INPUT_BUFFER_PADDING_SIZE`)
- If CLAUDE.md says to do it one way, do it that way
- Run the full test suite before submitting

## Adding a new codec

1. Create `codecs/wedeo-codec-<name>/` with `Cargo.toml` depending on
   `wedeo-core` + `wedeo-codec` + `inventory`
2. Implement the `Decoder` trait (`send_packet`/`receive_frame`/`flush`)
3. Create a `DecoderFactory` impl and register with `inventory::submit!`
4. Add the crate to `workspace.members` in root `Cargo.toml`
5. Add `use wedeo_codec_<name> as _;` in `wedeo-cli` and `wedeo-fate` to
   ensure linking
6. Add FATE tests comparing framecrc output against FFmpeg

## Adding a new format

1. Create `formats/wedeo-format-<name>/` with `Cargo.toml` depending on
   `wedeo-core` + `wedeo-format` + `inventory`
2. Implement the `Demuxer` trait (`read_header`/`read_packet`/`seek`) and
   `DemuxerFactory` with `probe()`
3. Register with `inventory::submit!`
4. Add FATE tests

## License

By contributing, you agree that your contributions will be licensed under
LGPL-2.1-or-later, consistent with the project license.
