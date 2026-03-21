# Parallel Claude Code Workflow

## How to run multiple Claude Code sessions without conflicts

### The dependency rule

```
SAFE IN PARALLEL (leaf crates — never import each other):
  codecs/wedeo-codec-h264/    ← H.264 debugging
  adapters/wedeo-rav1d/       ← AV1 adapter (NEW)
  formats/wedeo-format-mp4/   ← MP4 muxer (NEW)
  bins/wedeo-play/            ← Video player

SHARED (coordinate changes):
  Cargo.toml                  ← workspace.members list
  crates/wedeo-core/          ← base types
  crates/wedeo-codec/         ← decoder/encoder traits
  crates/wedeo-format/        ← demuxer/muxer traits
  bins/wedeo-cli/src/main.rs  ← use ... as _; imports
  tests/fate/Cargo.toml       ← test dependencies
```

### Setup: use git worktrees

Each parallel session works in its own worktree so they don't see each
other's uncommitted changes:

```bash
# From the main repo:
git worktree add ../wedeo-rav1d   -b feat/rav1d-av1-adapter
git worktree add ../wedeo-player  -b feat/video-player-audio
git worktree add ../wedeo-mp4     -b feat/mp4-muxer
# H.264 work stays on main (it's the primary workstream)
```

### Session launch commands

**Session 1: rav1d/AV1 adapter** (in ../wedeo-rav1d/)
```
cd ../wedeo-rav1d
claude "Implement the plan in plans/rav1d-av1-adapter.md. Read CONTRIBUTING.md
and CLAUDE.md first. Create the adapter crate, register it, add FATE tests.
Work in this worktree on the feat/rav1d-av1-adapter branch."
```

**Session 2: Video player with audio** (in ../wedeo-player/)
```
cd ../wedeo-player
claude "Implement the plan in plans/video-player-audio.md. Read the existing
wedeo-play code first, then add audio playback with A/V sync. Work in this
worktree on the feat/video-player-audio branch."
```

**Session 3: H.264 remaining** (in main repo)
```
cd /path/to/wedeo
claude "Implement the plan in plans/h264-remaining.md. Start with CVWP2/CVWP3
reorder investigation (likely quick win), then CVWP5 empirical MV extraction."
```

**Session 4: MP4 muxer** (in ../wedeo-mp4/)
```
cd ../wedeo-mp4
claude "Implement the plan in plans/mp4-muxer.md. Read CONTRIBUTING.md and
the existing WAV muxer first. Write the MP4 box serialization from scratch.
Work in this worktree on the feat/mp4-muxer branch."
```

### Merge order

When sessions complete, merge in this order to minimize conflicts:

1. **H.264 fixes** (on main, no merge needed)
2. **rav1d adapter** — touches Cargo.toml + cli + fate
3. **MP4 muxer** — touches Cargo.toml + cli (resolve with rav1d's changes)
4. **Video player** — only touches bins/wedeo-play/ (no conflicts)

```bash
# After each feature branch is done:
git checkout main
git merge feat/rav1d-av1-adapter
git merge feat/mp4-muxer           # may need to resolve Cargo.toml conflicts
git merge feat/video-player-audio   # clean merge
# Clean up worktrees:
git worktree remove ../wedeo-rav1d
git worktree remove ../wedeo-mp4
git worktree remove ../wedeo-player
```

### Conflict resolution protocol

If two branches both modify `Cargo.toml` workspace members:
- The second merge will have a conflict in the `members = [...]` list
- Resolution: include BOTH new entries, keep alphabetical order
- Same for `wedeo-cli/src/main.rs` `use` imports

### What each session MUST NOT touch

| Session | Do NOT modify |
|---------|--------------|
| rav1d | Any existing codec/format crate, wedeo-core types |
| MP4 muxer | Any existing codec/format crate, Muxer trait |
| Video player | Any crate outside bins/wedeo-play/ |
| H.264 | Any crate outside codecs/wedeo-codec-h264/ and scripts/ |

### Verification before merge

Each session must pass before merging:
```bash
cargo check --workspace
cargo clippy --workspace  # 0 warnings
cargo fmt --check --all
cargo nextest run         # all existing tests still pass
```
