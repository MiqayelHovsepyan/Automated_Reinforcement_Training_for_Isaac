# Auto-Train Setup Guide

Automated RL training loop powered by Claude Code. Claude trains your policy, visually inspects gait quality from a side-view camera, tunes hyperparameters with convergence analysis, and repeats — all unattended.

## Prerequisites

- NVIDIA GPU with Isaac Sim support (RTX 3060+)
- Isaac Lab installed in a Python venv (`.venv/`)
- RSL-RL training scripts (`scripts/rsl_rl/train.py`, `play.py`, `cli_args.py`)
- Claude Code (`npm install -g @anthropic-ai/claude-code`)

---

## Part 1: Setup

### Step 1 — Clone the auto-train repo

```bash
git clone https://github.com/MiqayelHovsepyan/Automated_Reinforcement_Learning_Training_for_Isaac_Lab.git
```

This repo contains:

```
Automated_Reinforcement_Learning_Training_for_Isaac_Lab/
  auto_train/
    SKILL.md                     # Claude Code skill definition
    AUTO_TRAIN_SETUP.md          # This setup guide
    README.md                    # Project documentation
    resources/                   # Python scripts
      __init__.py
      run_phase.py               # orchestrator: train -> metrics -> play -> frames -> report
      train_with_overrides.py    # train.py + JSON override support
      analyze_metrics.py         # TensorBoard -> JSON + convergence + suspicious patterns
      play_for_inspection.py     # side-view camera play wrapper for gait inspection
      extract_frames.py          # MP4 -> PNG frames
      wait_for_phase.py          # blocks until training completes (replaces sleep-poll loop)
```

### Step 2 — Copy the auto_train folder into your project

```bash
# Single copy — everything goes into .claude/skills/
cp -r Automated_Reinforcement_Learning_Training_for_Isaac_Lab/auto_train/ your_project/.claude/skills/auto_train/
```

Your project structure should look like:

```
your_project/
├── scripts/
│   └── rsl_rl/
│       ├── train.py          # (your existing train script)
│       ├── play.py           # (your existing play script)
│       └── cli_args.py       # (your existing CLI args)
├── .claude/
│   └── skills/
│       └── auto_train/
│           ├── SKILL.md
│           ├── resources/
│           │   ├── __init__.py
│           │   ├── run_phase.py
│           │   ├── train_with_overrides.py
│           │   ├── analyze_metrics.py
│           │   ├── play_for_inspection.py
│           │   ├── extract_frames.py
│           │   └── wait_for_phase.py
│           └── experiments/        # (created automatically)
│               └── .scratch/
└── logs/                           # (created automatically)
```

**Why:** Everything is self-contained in one folder. The skill, scripts, and experiments all live together.

### Step 3 — Adapt imports to your project

In `.claude/skills/auto_train/resources/train_with_overrides.py` **and** `play_for_inspection.py`, change:

```python
import cf_lab.tasks  # noqa: F401
# -> replace with your package:
import your_package.tasks  # noqa: F401
```

**Why:** This registers your environments with Isaac Lab so training and play can find them.

### Step 4 — Adapt SKILL.md to your project

Update these in `.claude/skills/auto_train/SKILL.md`:

1. **Working directory:** `cd cf_lab` -> `cd your_project_dir`
2. **Venv path:** Update `.venv/bin/python` if your venv is elsewhere
3. **Task import check:** Update the Level 2 safety check `ast.parse(...)` path to your edited files
4. **Device scaling:** Adjust `num_envs` for your robot/environment
5. **Robot body names:** The Body Coverage Audit section uses generic body names — ensure your robot's URDF body names are discoverable from the asset config
6. **Velocity tracking thresholds:** The Production Readiness Checklist uses default thresholds (exp-kernel reward > 0.6 for xy, > 0.3 for yaw). Adjust if your task uses different reward formulations

**Why:** Claude follows these paths and thresholds literally. Wrong values = failed training or premature production runs.

### Step 5 — Install dependencies

```bash
source .venv/bin/activate
uv pip install tbparse opencv-python-headless
```

- `tbparse` — converts TensorBoard metrics to JSON with convergence analysis
- `opencv-python-headless` — extracts video frames so Claude can visually inspect the robot's gait

### Step 6 — Verify Play task IDs exist

Auto-train needs a `-Play-v0` variant of your task (auto-derived: `Task-v0` -> `Task-Play-v0`):

```
Isaac-Velocity-Flat-Ayg-v0        # training
Isaac-Velocity-Flat-Ayg-Play-v0   # play (fewer envs, for video recording)
```

If you don't have `-Play-v0` variants, register them (inherit training config, reduce `num_envs`).

**Note:** The play wrapper `play_for_inspection.py` overrides `num_envs` to 2–4 and sets a side-view camera regardless of the Play task's default config. But the Play task must still be registered.

---

## Part 2: Running

### Step 1 — Disable machine sleep

**Why:** Training iterations take 30 min to 2+ hours. If the machine sleeps, GPU stops, session is wasted.

### Step 2 — Launch Claude Code

```bash
cd your_project
claude --dangerously-skip-permissions
```

**Why:** Auto-train executes dozens of actions autonomously (write files, launch processes, read images). This flag auto-approves all tool calls — required when no human is watching. The skill is hardened to never block on any operation.

### Step 3 — Start auto-train

```
/auto_train <task_id> level <1|2> on <gpu> <vram> [optional notes]
```

Examples:
```
/auto_train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 3060 12GB
/auto_train Isaac-WTW-Flat-Ayg-v0 level 2 on RTX 4090 24GB focus on foot clearance
```

| Argument | Meaning |
|----------|---------|
| `level 1` | JSON overrides only (reward weights, PPO hyperparams). No source edits. |
| `level 2` | Full autonomy — can edit configs, add/remove rewards, write new reward functions. |
| `on <gpu> <vram>` | Claude uses this to pick `num_envs`. Wrong values = OOM. |
| `[notes]` | Domain hints Claude uses to guide strategy. |

### Step 4 — Walk away

Claude will:
1. Audit body coverage and analyze rewards before any training
2. Run **5–15 short tuning iterations** (300–500 iters each) testing one hypothesis per run
3. Pass a formal **Production Readiness Checklist** (velocity tracking, visual gait quality, no reward hacking, etc.)
4. Run a final production training with the best config
5. Bake winning parameters into source config

### Step 5 — Check results

```bash
cat .claude/skills/auto_train/experiments/*/journal.md    # what Claude tried and learned
ls -lt logs/rsl_rl/*/                                      # training checkpoints
```

The journal contains:
- Body coverage audit table
- Per-iteration detailed metrics with convergence analysis
- Cross-iteration comparison table (after iter 3+)
- Visual assessment with CAN/CANNOT verification framework
- Production readiness assessment (PASS/FAIL table)
- Post-production summary with recommended next steps

---

## Controls, Resume, and Safety

**User controls** (type into terminal if watching):
`stop` | `level 1` | `level 2` | `focus on X`

**Resume after disconnect:** The current training survives (detached via `nohup`). Start a new session:
```
Continue auto-train, read journal at .claude/skills/auto_train/experiments/<name>/journal.md
```

**Safety guarantees:**

| Concern | Handled by |
|---------|-----------|
| Training runs for hours | Detached via `nohup setsid` — survives any timeout |
| Claude blocks on a question | Forbidden — `AskUserQuestion` not in allowed-tools, and SKILL.md forbids all blocking operations including file creation confirmations |
| Infinite loop | Hard cap at 15 tuning iterations |
| Premature production run | Production Readiness Checklist requires 6 PASS criteria (velocity tracking, visual gait, body coverage, no reward hacking, sufficient iterations, convergence) |
| Repeated failures | Aborts after 3 consecutive failures |
| Reward hacking (hip-walking, etc.) | Body coverage audit + suspicious pattern detection + side-view camera + velocity tracking gate |
| Misleading metrics | Convergence analysis detects early plateau; suspicious patterns flag when gait reward games velocity tracking |
| Bad visual assessment | Honest assessment rules: CAN/CANNOT framework, never claim "good gait" without evidence |
| Disk fills up | Stops if < 10 GB free |
| Wrong Python (system vs venv) | Uses `.venv/bin/python` explicitly |
| Training crashes | Crash handler writes report so Claude detects it |
| OOM kill | PID tracked — Claude detects dead process |
| Bad override JSON | Pre-flight validation catches format errors before Isaac Sim boots |

---

## What's New (v2)

If you're upgrading from an earlier version, here's what changed:

| Feature | Before | After |
|---------|--------|-------|
| Visual inspection | Overhead camera, 50 robots — useless for gait assessment | Side-view camera at robot height, 2–4 robots via `play_for_inspection.py` |
| Metrics analysis | Final values + crude "stable/improving/degrading" trend | Convergence detection (where each metric plateaued), curve shape classification, suspicious pattern detection |
| Tuning iterations | 2–3 runs at 2000 iters | 5–15 runs at 300–500 iters (faster hypothesis testing) |
| Production decision | Subjective ("looks good") | Formal Production Readiness Checklist with 6 PASS/FAIL gates |
| Body coverage | Not checked — hip-walking went undetected | Mandatory audit before iteration 1 |
| Velocity tracking | Noted as "poor" but training continued | Velocity Tracking Gate — blocking issue if below threshold |
| Journal format | Basic: hypothesis, result, metrics | Detailed: metrics tables with convergence, CAN/CANNOT visual assessment, cross-iteration comparison, reasoning |
| Blocking risk | Could block on AskUserQuestion | All blocking operations forbidden — `AskUserQuestion` removed from allowed-tools, file writes pre-authorized |
| Frames extracted | 8 frames from overhead video | 12 frames from side-view video |
| Phase report | Metrics + frame paths | + convergence_summary + suspicious_patterns + health summary |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: isaaclab` | Wrong Python. Check `.venv/bin/python -c "import isaaclab"`. Update venv path in SKILL.md. |
| No play video found | `-Play-v0` task not registered, or play crashed. Check `.claude/skills/auto_train/experiments/.scratch/current_phase.log`. |
| OOM crashes | Reduce `num_envs` in SKILL.md device scaling section. |
| Claude stopped after 1 iteration | Forgot `--dangerously-skip-permissions`. |
| Report stuck on "running" | Process OOM-killed. Check `dmesg \| grep -i oom`. Reduce `num_envs`. |
| `validation_error` in report | Override JSON has wrong format (nested dict instead of flat dot-paths). See SKILL.md Override JSON Format section. |
| `play_for_inspection.py` crashes | Check that `ViewerCfg` is importable: `.venv/bin/python -c "from isaaclab.envs import ViewerCfg; print('OK')"`. If not, your Isaac Lab version may be too old — `ViewerCfg` was added in Isaac Lab 2023+. |
| Camera shows black frames | Isaac Sim needs a few steps to initialize rendering. The frame extractor skips the first 10% of frames automatically. If all frames are black, increase `--video-length`. |
| `suspicious_patterns` false positives | The detection uses heuristics (e.g., tracking < 15% of total reward = suspicious). If your task legitimately has low tracking contribution, adjust thresholds in the SKILL.md Production Readiness Checklist or ignore the warning with documented reasoning in the journal. |
| `converged_early` on all terms | The convergence detector uses a 2% relative change threshold with 50-point windows. For very short runs (< 100 iters), most metrics will show insufficient data. Use at least 300 iterations for meaningful convergence analysis. |
