# Auto-Train Setup Guide

Automated RL training loop powered by Claude Code. Claude trains your policy, visually inspects gait quality, tunes hyperparameters, and repeats — all unattended.

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
  auto_train (scripts)/
    __init__.py
    run_phase.py               # orchestrator: train -> metrics -> play -> frames -> report
    train_with_overrides.py    # train.py + JSON override support
    analyze_metrics.py         # TensorBoard -> JSON
    extract_frames.py          # MP4 -> PNG frames
  SKILL.md                     # Claude Code skill definition
```

### Step 2 — Copy scripts and skill into your project

```bash
# Copy auto-train scripts
cp -r "Automated_Reinforcement_Learning_Training_for_Isaac_Lab/auto_train (scripts)/" your_project/scripts/auto_train/

# Copy the skill definition
mkdir -p your_project/.claude/skills/auto-train/
cp Automated_Reinforcement_Learning_Training_for_Isaac_Lab/SKILL.md your_project/.claude/skills/auto-train/SKILL.md
```

**Why:** `run_phase.py` runs the full pipeline per iteration. Claude reads the output report + frames to decide what to change next. `SKILL.md` is loaded when you invoke `/auto-train` and contains Claude's complete instructions for the training loop.

### Step 3 — Adapt imports to your project

In `scripts/auto_train/train_with_overrides.py`, change:

```python
import cf_lab.tasks  # noqa: F401
# -> replace with your package:
import your_package.tasks  # noqa: F401
```

**Why:** This registers your environments with Isaac Lab so training can find them.

### Step 4 — Adapt SKILL.md to your project

Update these in `.claude/skills/auto-train/SKILL.md`:

1. **Working directory:** `cd cf_lab` -> `cd your_project_dir`
2. **Venv path:** Update `.venv/bin/python` if your venv is elsewhere
3. **Task import check:** Update the Level 2 safety check `ast.parse(...)` path to your edited files
4. **Device scaling:** Adjust `num_envs` for your robot/environment

**Why:** Claude follows these paths literally. Wrong paths = failed training.

### Step 5 — Install dependencies

```bash
source .venv/bin/activate
uv pip install tbparse opencv-python-headless
```

- `tbparse` — converts TensorBoard metrics to JSON for Claude to analyze
- `opencv-python-headless` — extracts video frames so Claude can visually inspect the robot's gait

### Step 6 — Verify Play task IDs exist

Auto-train needs a `-Play-v0` variant of your task (auto-derived: `Task-v0` -> `Task-Play-v0`):

```
Isaac-Velocity-Flat-Ayg-v0        # training
Isaac-Velocity-Flat-Ayg-Play-v0   # play (fewer envs, for video recording)
```

If you don't have `-Play-v0` variants, register them (inherit training config, reduce `num_envs`).

### Step 7 — Create experiments directory

```bash
mkdir -p experiments/.scratch
```

**Why:** Stores override files, report files, and session journals.

---

## Part 2: Running

### Step 1 — Disable machine sleep

**Why:** Training iterations take 30 min to 2+ hours. If the machine sleeps, GPU stops, session is wasted.

### Step 2 — Launch Claude Code

```bash
cd your_project
claude --dangerously-skip-permissions
```

**Why:** Auto-train executes dozens of actions autonomously (write files, launch processes, read images). This flag auto-approves all tool calls — required when no human is watching.

### Step 3 — Start auto-train

```
/auto-train <task_id> level <1|2> on <gpu> <vram> [optional notes]
```

Examples:
```
/auto-train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 3060 12GB
/auto-train Isaac-WTW-Flat-Ayg-v0 level 2 on RTX 4090 24GB focus on foot clearance
```

| Argument | Meaning |
|----------|---------|
| `level 1` | JSON overrides only (reward weights, PPO hyperparams). No source edits. |
| `level 2` | Full autonomy — can edit configs, add/remove rewards, write new reward functions. |
| `on <gpu> <vram>` | Claude uses this to pick `num_envs`. Wrong values = OOM. |
| `[notes]` | Domain hints Claude uses to guide strategy. |

### Step 4 — Walk away

Claude will iterate up to 15 times, then run a final production training with the best config.

### Step 5 — Check results

```bash
cat experiments/*/journal.md    # what Claude tried and learned
ls -lt logs/rsl_rl/*/           # training checkpoints
```

---

## Controls, Resume, and Safety

**User controls** (type into terminal if watching):
`stop` | `level 1` | `level 2` | `focus on X`

**Resume after disconnect:** The current training survives (detached via `nohup`). Start a new session:
```
Continue auto-train, read journal at experiments/<name>/journal.md
```

**Safety guarantees:**

| Concern | Handled by |
|---------|-----------|
| Training runs for hours | Detached via `nohup setsid` — survives any timeout |
| Claude blocks on a question | Forbidden — makes a decision and logs reasoning |
| Infinite loop | Hard cap at 15 iterations |
| Repeated failures | Aborts after 3 consecutive failures |
| Disk fills up | Stops if < 10 GB free |
| Wrong Python (system vs venv) | Uses `.venv/bin/python` explicitly |
| Training crashes | Crash handler writes report so Claude detects it |
| OOM kill | PID tracked — Claude detects dead process |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: isaaclab` | Wrong Python. Check `.venv/bin/python -c "import isaaclab"`. Update venv path in SKILL.md. |
| No play video found | `-Play-v0` task not registered, or play crashed. Check `experiments/.scratch/current_phase.log`. |
| OOM crashes | Reduce `num_envs` in SKILL.md device scaling section. |
| Claude stopped after 1 iteration | Forgot `--dangerously-skip-permissions`. |
| Report stuck on "running" | Process OOM-killed. Check `dmesg \| grep -i oom`. Reduce `num_envs`. |
