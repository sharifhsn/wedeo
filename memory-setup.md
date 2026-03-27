
# Claude Code Memory & Review System — Full Setup Plan

**Target machine**: New MacBook  
**Date**: March 2026  
**Architecture**: recall-stack Layers 1–3 (CLAUDE.md + primer.md + git context) + Mnemon (replaces Hindsight) + adversarial review protocol + retrospective system

---

## Prerequisites

```bash
# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Node.js (required for Claude Code)
brew install node

# Claude Code
npm install -g @anthropic-ai/claude-code

# Mnemon (prebuilt binary via Homebrew — Go not needed)
brew install mnemon-dev/tap/mnemon

# Verify
claude --version
mnemon --version

# Authenticate Claude Code
claude login
```

---

## Step 1: Directory Structure

```bash
mkdir -p ~/.claude/hooks
mkdir -p ~/.claude/commands
mkdir -p ~/.claude/scripts
```

**Important ordering note**: Steps 2–4 create CLAUDE.md, primer.md docs,
and the git hook *script file*. Do **not** create or edit
`~/.claude/settings.json` yet — Step 5 (`mnemon setup --global`) will
create it. Step 6 then merges your git hook entry into Mnemon's
settings.json. If you create settings.json before Mnemon runs, Mnemon
may overwrite it.

---

## Step 2: Global CLAUDE.md (Layer 1)

This is the master instruction file. It includes the adversarial review protocol, retrospective protocol, PR checklist, and all your coding preferences.

Create `~/.claude/CLAUDE.md`:

```markdown
# PREFERENCES
- Use `uv` for all Python commands, with script dependency syntax for ad hoc scripts
- For larger Python projects, use `uv` project management
- For Python libraries, use Pydantic for robustness and Polars instead of Pandas
- Always recommend Rust-written alternatives for software (unless the working language is not Rust)
- Ask clarifying questions if a prompt is unclear
- One clear next action per response, not a list
- Flag anything uncertain with [UNCLEAR]
- Always ask for explicit verification prompt before making changes to a database

# AGENT RULES
- At session start, read .claude/primer.md in the project root directory
  if it exists. If it doesn't exist or is empty, ask what we're working on.
- Also read tasks/lessons.md in the project root if it exists, and apply
  every rule before touching any code.
- Keep primer.md under 100 lines
- Never ask for context that exists in imported files
- After completing a meaningful task (implemented a feature, fixed a bug,
  finished a refactor — not after answering a question or making a trivial
  edit), silently overwrite .claude/primer.md in the project root (create
  the directory if missing) with: active project, what's been completed,
  exact next step, open blockers. Keep under 100 lines. This ensures
  primer.md survives abrupt exits and is scoped to the project, not shared
  across unrelated repos.
- Before closing, check for uncommitted changes and remind me to commit.
- When the conversation is getting long and you're concerned about context
  pressure, proactively rewrite .claude/primer.md with the current state
  and suggest I run /compact. Do not wait for a specific percentage — the
  PreCompact hook handles the hard cutoff.

## primer.md vs Mnemon: what goes where
- **primer.md** is ephemeral session state: what's in-flight right now,
  what to do next, current blockers. Think of it as a sticky note. If it
  won't matter in 3 days, it goes here.
- **Mnemon** is durable knowledge: patterns, architectural decisions,
  debugging insights, lessons learned. Think of it as institutional memory.
  If it will matter across sessions or projects, it goes here.
- Do not store the same information in both places.

# SELF-LEARNING
- After any correction from me, immediately add an entry to
  tasks/lessons.md in the project root (create the file if it doesn't exist)
- Format: [date] | what went wrong | rule to follow next time
- For lessons that apply across projects (not project-specific), also store
  them via `mnemon remember` so they survive outside this repo

# ADVERSARIAL REVIEW PROTOCOL
After generating or significantly modifying code (>50 lines, structural
changes, or executing a multi-step plan), automatically run an adversarial
review loop.

## Severity Scale
- **CRITICAL**: Will cause incorrect behavior, data loss, security
  vulnerability, or crash in normal use. Must fix before any code ships.
- **MAJOR**: Significant logic error, unhandled edge case likely to be
  hit, API contract violation, resource leak, or race condition. Won't
  crash immediately but will cause real problems. Must fix before the
  review can converge.
- **NIT**: Style, naming, minor readability, non-blocking suggestions,
  or edge cases that are extremely unlikely. Noted but do not block
  convergence. Fix at your discretion.

## The Two Round Types

### Fix Round
A fix round finds issues and resolves them. It consists of:

1. **Review**: Examine the code from a specific analytical angle (rotate):
   - control flow & logic
   - data flow & invariants
   - failure modes & edge cases
   - API boundaries & type contracts
   - (on round 5+: revisit earlier angles, prioritizing areas touched
     by recent fixes)
   Check for: logic errors, edge cases, off-by-ones, error handling gaps,
   type mismatches, race conditions, security issues, API contract
   violations, resource leaks.
   Classify each finding using the severity scale above.
   State the round number, the analytical angle, and list all findings.

2. **Fix**: Apply fixes for all CRITICAL and MAJOR findings.

3. **Regression check**: After fixing, explicitly verify that the fixes
   did not break anything that was previously working, especially in
   areas adjacent to the changed code.

A fix round that finds CRITICAL or MAJOR issues MUST be followed by
another fix round. Keep running fix rounds until a round's review step
finds zero CRITICAL and zero MAJOR issues.

### Verification Round
A verification round confirms the fixes are clean. It ONLY runs after
a fix round whose review step found zero CRITICAL and zero MAJOR issues.

1. **Review from a different angle** than the last fix round. Explicitly
   check for regressions introduced by ALL previous fix rounds, not just
   the most recent one.

2. **If CRITICAL or MAJOR issues are found**: The verification round
   fails. Return to fix rounds. Run fix rounds until clean again, then
   attempt another verification round.

3. **If only NITs or nothing found**: Convergence is reached.

## Convergence
- State: "Adversarial review converged after N rounds (M fix rounds +
  V verification rounds)."
- After convergence, list any known gaps that remain unfixed: missing
  test coverage, unhandled edge cases deferred by design, hardcoded
  values, TODOs, performance concerns, or assumptions about the
  environment. For each gap, note why it's being left (out of scope,
  acceptable risk, blocked on upstream, etc.). Record these as a
  `## Known Gaps` section in the relevant file (inline comment block,
  PROGRESS.md, or tasks/lessons.md — whichever fits). If a gap is
  significant enough to affect future sessions, also store it via
  `mnemon remember`.

## Skip Conditions
- Do NOT run for trivial changes (renaming, comments, formatting, <20 lines).
- If I say "skip review" in my prompt, skip it.

# SESSION RETROSPECTIVE PROTOCOL
When I say "retro" or "retrospective", perform:

1. **Session summary**: What was accomplished? Current state?
2. **Dead ends & loops**: Approaches tried and abandoned. Why they failed.
   What signal should have triggered abandoning sooner.
3. **High-value actions**: Moves with disproportionate impact. Go-to
   approaches for next time.
4. **Automatable work**: Anything done 2+ times that could be scripted.
5. **Debugging patterns**: Bug types encountered. Most effective diagnostics.
6. **Knowledge gaps**: Things looked up mid-session that should be known upfront.

## Script Generation (after step 4)
For each item identified in "Automatable work", up to a maximum of 3 per retro:
- Actually write the script (shell, Python with uv script syntax, etc.)
- Place it in the project's `scripts/` directory (create if missing)
- Run the Adversarial Review Protocol on each generated script
- Add a brief entry to PROGRESS.md noting the new script and what it automates
- If the script is general-purpose (not project-specific), also place a copy
  in `~/.claude/scripts/` for cross-project reuse
- If more than 3 items were identified, list the remainder in PROGRESS.md
  under a "## Deferred Automation" section to pick up in the next session

## Memory Triage (after generating retrospective and scripts)
Classify each lesson into one of:
- **CLAUDE_MD**: Fundamental project-level rules (rare). Ask me before writing.
- **MNEMON**: Durable cross-session knowledge. Use `mnemon remember` to store.
- **PROGRESS_DOC**: State updates. Update PROGRESS.md in project root.
- **LESSONS**: Correction-style learning. Append to tasks/lessons.md.

# WORKFLOW
- Enter plan mode for any non-trivial task (3+ steps)
- If something goes wrong mid-task, stop and re-plan
- Never mark a task complete without proving it works
- When given a bug: just fix it, no hand-holding
- Commit at logical checkpoints, not just at the end

# PR PREPARATION CHECKLIST
When I say "prep PR", "PR review", or "ready for review", run through
every section below before marking the PR ready. This is separate from
the Adversarial Review Protocol (which covers code correctness during
development). This checklist covers everything a human reviewer will
scrutinize.

## Code Quality
- Run all linters the project uses (ruff, codespell, rstcheck, etc.)
- Search for debug artifacts: TODO, FIXME, print statements, commented-out code
- Verify no new imports are added unnecessarily, and all existing imports are still used
- Check for consistent naming conventions within the PR

## Commit Hygiene
- Each commit message should be accurate (don't claim "10x speedup" if benchmark shows 1.0x)
- Separate concerns: bugfixes, cleanups, docs, and tests in their own commits
- Commit prefixes should match project conventions (PERF:, FIX:, DOC:, TST:, STY:, etc.)
- Run `codespell` on commit messages for spelling errors
- Verify bisectability: key commits should pass tests independently

## Changelog & Documentation
- Match project conventions exactly: line endings (LF not CRLF), character encoding
  (ASCII x not Unicode ×), RST roles
- Use `uvx rstcheck` to validate RST files
- Changelog entries should accurately describe what changed
- Separate changelog entries for separate concerns (newfeature vs bugfix)

## PR Body & Comments
- Keep PR body in sync with actual commit count and structure
- Update stale comments when restructuring (add blockquote notes pointing to updates)
- Include reproducible benchmark script (collapsible details section)
- PR body should match the PR template structure if the project has one
- Include "AI was used" disclosure if applicable

## Test Coverage
- Verify test fixtures actually test what they claim (e.g., a "NumPy fallback"
  fixture should truly disable the fast path)
- Check that external callers of modified internal functions still work
- Run label/related tests when modifying shared internal APIs

## Reviewer Experience
- Order commits by "bang for buck" (easy wins first, complex changes last)
- Each commit description should explain WHY, not just WHAT
- Note which commits can be accepted/rejected independently
- Include benchmark tables with per-commit cumulative speedup
- Offer alternatives for controversial design choices

## Diff Hygiene
- Run `codespell` on the diff itself (`git diff main... | codespell -`), not just the files
- Verify no conflicting open PRs touch the same files
- Check that the PR title fits in ~72 chars (GitHub truncates in list views)
- Verify changelog RST role names exist in the project
- Ensure trailing newlines in all new files (POSIX requirement)

## API Compatibility
- Check that deleted functions/classes aren't called from other modules (grep the entire codebase)
- When changing return types of internal functions, verify ALL callers are updated
- New kwargs with defaults keep backward compatibility — but verify callers don't pass positional args
- If a function is imported by other modules, its signature change is higher risk

## Cross-Platform & Edge Cases
- Check uninitialized arrays are only accessed at set indices
- Verify division-by-zero protection
- Check that fallback code paths work when optional dependencies are missing
- Verify parallel execution safety (no shared mutable state across workers)

## Maintainer Relationship
- Answer EVERY question a maintainer asks — don't leave any unanswered
- If you say you'll do something, actually do it or explain why you didn't
- When CI fails, proactively verify if it's your fault (check other PRs) and report findings
- Post a concise follow-up comment when the PR is ready, closing the loop on all open questions

## PR Metadata
- Mark as draft until truly ready for review
- Request reviewers relevant to the module
- Keep track of draft/ready status
```

---

## Step 3: primer.md (Layer 2 — project-scoped)

Primers are stored per-project, not globally. The global CLAUDE.md
instructs Claude to read `<project-root>/.claude/primer.md` at session
start (as a normal file, not via `@import` — the `@` directive resolves
relative to the file it's in, which would be `~/.claude/`, not the project).

**For each project you work on**, the primer auto-creates on first session
when Claude completes a task. You don't need to seed them manually.

**Add `.claude/primer.md` to each project's `.gitignore`** — the primer
contains ephemeral session state, not repo knowledge:

```bash
echo ".claude/primer.md" >> .gitignore
```

---

## Step 4: Git Context Hook (Layer 3)

Create `~/.claude/hooks/session-start-git.sh`:

```bash
#!/bin/bash
# Layer 3: Inject git context at session start

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "[GIT CONTEXT] Not inside a git repository."
    exit 0
fi

BRANCH=$(git branch --show-current 2>/dev/null)
RECENT_COMMITS=$(git log --oneline -10 2>/dev/null)
MODIFIED=$(git diff --name-only 2>/dev/null)
STAGED=$(git diff --cached --name-only 2>/dev/null)

cat <<EOF
[GIT CONTEXT]
Branch: ${BRANCH}
Recent commits:
${RECENT_COMMITS}
Modified files: ${MODIFIED}
Staged files: ${STAGED}
EOF
```

```bash
chmod +x ~/.claude/hooks/session-start-git.sh
```

---

## Step 5: Mnemon Setup (Layer 4 — replaces Hindsight)

Mnemon handles all cross-session memory. It installs its own hooks (Prime,
Remind, Nudge, Compact) that cover SessionStart, UserPromptSubmit, Stop,
and PreCompact lifecycle events.

### Install globally (applies to all projects):

```bash
mnemon setup --global
```

This interactively deploys:
- **Skill file** (`SKILL.md`) — teaches Claude the `mnemon remember` and `mnemon recall` command syntax
- **Behavioral guide** (`~/.mnemon/prompt/guide.md`) — controls when Claude recalls and what it considers worth remembering
- **Four hooks**:
  - `prime.sh` (SessionStart) — loads the behavioral guide
  - `user_prompt.sh` (UserPromptSubmit) — reminds agent to recall & remember
  - `stop.sh` (Stop) — nudges agent to remember after completing work
  - `compact.sh` (PreCompact) — extracts critical insights before context compression

### Verify:

```bash
mnemon --version
ls ~/.mnemon/
# Should see: prompt/, data/ (or similar structure)
```

### Optional: Ollama for enhanced retrieval

Mnemon works fully without this, but adding local embeddings enables
vector+keyword hybrid search:

```bash
brew install ollama
ollama pull nomic-embed-text
# Mnemon auto-detects Ollama at localhost:11434
```

### Key Mnemon commands (the agent runs these, not you):

```bash
mnemon remember "lesson text here"   # Store a memory
mnemon recall "query"                # Search memories
mnemon list                          # Show recent memories
mnemon forget <id>                   # Remove a memory
mnemon store create <name>           # Create isolated store
mnemon store set <name>              # Switch active store
```

### Per-project isolation (optional):

If you want separate memory per project:

```bash
mnemon store create ourfirm
mnemon store set ourfirm
# Or per-session: MNEMON_STORE=ourfirm claude
```

---

## Step 6: Merge Hook Configurations

Mnemon's `setup --global` will have written hooks into `~/.claude/settings.json`.
You need to also add the git context hook from Layer 3. The final
`~/.claude/settings.json` should look like this (merge carefully — Mnemon's
hooks are the authority, just add the git context hook to the SessionStart
array):

Check what Mnemon wrote:

```bash
cat ~/.claude/settings.json
```

Then add the git context hook to the `SessionStart` array. The result should
have entries for both `session-start-git.sh` AND Mnemon's `prime.sh` under
SessionStart. Example (your Mnemon paths may differ):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          {
            "type": "command",
            "command": "bash \"$HOME/.claude/hooks/session-start-git.sh\"",
            "timeout": 10
          }
        ]
      },
      ... (Mnemon's prime.sh entry — already present from mnemon setup)
    ],
    "UserPromptSubmit": [
      ... (Mnemon's user_prompt.sh — already present)
    ],
    "Stop": [
      ... (Mnemon's stop.sh — already present)
    ],
    "PreCompact": [
      ... (Mnemon's compact.sh — already present)
    ]
  }
}
```

**Do not overwrite Mnemon's entries.** Just add the git context hook alongside them.

---

## Step 7: /retro Slash Command

Create `~/.claude/commands/retro.md`:

```markdown
Perform a session retrospective now. Follow the Session Retrospective Protocol
from CLAUDE.md:

1. Session summary
2. Dead ends & loops
3. High-value actions
4. Automatable work
5. Debugging patterns
6. Knowledge gaps

Then for each item in "Automatable work" (max 3 — defer the rest to PROGRESS.md):
- Actually write the script. Do not just describe what could be scripted.
- Place it in the project's scripts/ directory (create if missing).
- Run the Adversarial Review Protocol on each script you generate.
- If the script is general-purpose, also copy it to ~/.claude/scripts/.

Then perform Memory Triage:
- Classify each lesson as CLAUDE_MD / MNEMON / PROGRESS_DOC / LESSONS
- Ask me to confirm any CLAUDE.md changes
- For MNEMON items, use `mnemon remember` to store them
- For PROGRESS_DOC, update PROGRESS.md
- For LESSONS, append to tasks/lessons.md
```

---

## Step 8: Post-Commit Hook (Probably unnecessary)

The git context SessionStart hook (Step 4) already injects the last 10
commits via `git log --oneline -10` at the start of every session. A
post-commit hook that appends to a file is redundant with this and grows
unboundedly.

**Skip this step unless** you have a specific reason to want a persistent
commit log outside of git itself (e.g., you want Claude to see commits
from before the last 10). If you do use it, cap the file:

```bash
# Run inside the repo
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
MSG=$(git log -1 --pretty=format:"%h %s")
DATE=$(date +%Y-%m-%d)
echo "[$DATE] $MSG" >> .claude-memory.md
# Cap at 50 entries
tail -50 .claude-memory.md > .claude-memory.md.tmp && mv .claude-memory.md.tmp .claude-memory.md
EOF
chmod +x .git/hooks/post-commit
```

---

## Final File Tree

```
~/.claude/
├── CLAUDE.md                        # Layer 1: Rules + review + retro + PR checklist
├── settings.json                    # All hooks (git context + Mnemon's 4 hooks)
├── hooks/
│   └── session-start-git.sh         # Layer 3: Git context injection
│   └── (Mnemon hooks also here)     # Layer 4: prime.sh, user_prompt.sh, stop.sh, compact.sh
├── commands/
│   └── retro.md                     # /retro slash command
└── scripts/                         # General-purpose scripts from retros

~/.mnemon/
├── prompt/
│   └── guide.md                     # Behavioral guide (customize memory behavior)
├── data/
│   └── default/                     # Default memory store (SQLite)
└── (skill files deployed by setup)

<project-root>/
├── .claude/
│   └── primer.md                    # Layer 2: THIS project's session state (gitignored)
├── CLAUDE.md                        # Optional: project-specific rules (layers on global)
├── PROGRESS.md                      # Project state, milestones, what's next
├── tasks/
│   └── lessons.md                   # Correction log (project-specific)
└── scripts/                         # Project-specific automation from retros
```

---

## What Fires When

| Trigger | What happens |
|---------|-------------|
| Session starts | Git context injected + Mnemon primes with behavioral guide + recalls relevant memories |
| You send a message | Mnemon reminds agent to evaluate recall/remember before working |
| Code generation >50 lines, structural changes, or multi-step plan execution | Adversarial review loop fires (from CLAUDE.md rules) |
| Meaningful task completed | primer.md auto-rewrites with current state + Mnemon nudges agent to remember |
| You type `/retro` | Manual retrospective → writes scripts for automatable work → adversarial review on each script → memory triage → stores lessons via Mnemon |
| Context getting long | Claude proactively rewrites primer.md, suggests `/compact` (soft heuristic — PreCompact hook handles the hard cutoff) |
| Auto-compaction fires | Mnemon's compact hook extracts critical insights before compression |
| Terminal killed | primer.md already saved from last meaningful task — nothing major lost |

---

## Customization

### Tune Mnemon's behavior:

Edit `~/.mnemon/prompt/guide.md` to control:
- What the agent considers worth remembering
- When it should recall vs. skip
- How aggressively it stores debugging patterns

### Tune adversarial review:

Adjust the line threshold (currently >50 lines) or analytical angles in
`~/.claude/CLAUDE.md` based on your experience.

### Per-project overrides:

Any project can have its own `CLAUDE.md` at the repo root that layers on
top of the global one. For example:

```markdown
# Project: rust-h264-decoder
- Always run `cargo clippy` after changes
- Test with: `cargo test -- --nocapture`
- FFmpeg test files are in tests/fixtures/
```

---

## Verification Checklist

After setup, run through these to confirm everything works:

1. [ ] `claude --version` returns current version
2. [ ] `mnemon --version` returns v0.1.2+
3. [ ] Start a Claude Code session in a git repo — ask "what branch am I on and what are my recent commits?" to verify git context was injected (hook output goes to the model, not your terminal)
4. [ ] Ask Claude to write a >50 line function — should trigger adversarial review
5. [ ] Complete a task — check that `<project-root>/.claude/primer.md` was created/updated
6. [ ] Type `/retro` — should produce structured retrospective with script generation
7. [ ] Start a new session — Mnemon should recall relevant memories from previous session
8. [ ] `mnemon list` shows stored memories
