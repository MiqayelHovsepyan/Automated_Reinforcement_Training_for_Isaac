# Automated Reinforcement Learning Training for Isaac Lab with Claude Code

An automated RL training loop powered by [Claude Code](https://claude.ai/claude-code) that trains, evaluates visually, tunes hyperparameters, and iterates — designed to run unattended (e.g., overnight).

Built for quadruped robots trained in [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab), but adaptable to any Isaac Lab RL task.

## How It Works

Claude Code acts as an expert RL engineer in a loop:

1. **Audits** body coverage (every non-foot body must be penalized/terminated) and analyzes reward terms mathematically
2. **Writes** parameter overrides as JSON (no source file edits in Level 1)
3. **Launches** short tuning runs (300–500 iters) via `run_phase.py` as a detached process
4. **Waits** for completion using `wait_for_phase.py` (blocks until done, shows progress)
5. **Analyzes** TensorBoard metrics (convergence detection, suspicious pattern analysis) + side-view video frames (visual gait inspection)
6. **Logs** everything to a journal with detailed metrics tables, CAN/CANNOT verification framework, and cross-iteration comparison
7. **Repeats** with tuned parameters (one variable per iteration, minimum 5 tuning iterations) until the Production Readiness Checklist passes, then runs a final production training

```
┌─────────────────────────────────────────────────────────┐
│                  Claude Code (AI Agent)                   │
│                                                           │
│   Audit → Analyze → Override → Train → Inspect → Decide  │
│       ↑                                        │          │
│       └──────────── Repeat (5-15x) ────────────┘          │
│                                                           │
│   Production Readiness Checklist → Production Run → Bake  │
└─────────────────────────────────────────────────────────┘
        │                              ↑
        ▼                              │
┌──────────────┐  metrics + frames  ┌──────────────┐
│ run_phase.py │ ─────────────────→ │ phase_report │
│  (detached)  │                    │    .json     │
└──────────────┘                    └──────────────┘
    │
    ├── train_with_overrides.py      (Isaac Lab training with JSON overrides)
    ├── analyze_metrics.py           (TensorBoard → JSON + convergence + suspicious patterns)
    ├── play_for_inspection.py       (side-view camera, 2-4 robots, gait inspection)
    └── extract_frames.py            (MP4 → PNG for visual inspection)
```

## What's in This Repo

| File | Description |
|------|-------------|
| `auto_train/resources/run_phase.py` | Orchestrator: train → metrics → play → frames → report. Includes convergence summary and suspicious pattern detection in output. |
| `auto_train/resources/train_with_overrides.py` | Modified Isaac Lab `train.py` with `--overrides-file` JSON support |
| `auto_train/resources/analyze_metrics.py` | Extracts TensorBoard events into structured JSON with convergence detection, curve shape analysis, and suspicious pattern detection |
| `auto_train/resources/play_for_inspection.py` | Custom play wrapper with side-view camera at robot height, following a single robot (2–4 envs) for meaningful visual gait inspection |
| `auto_train/resources/extract_frames.py` | Extracts evenly-spaced PNG frames from rollout videos (uses OpenCV) |
| `auto_train/resources/wait_for_phase.py` | Blocks until training completes, prints final report (replaces sleep-poll loop) |
| `auto_train/resources/__init__.py` | Python package marker |
| `auto_train/SKILL.md` | Claude Code skill definition that drives the entire auto-train loop |

## Setup

### Prerequisites

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) installed and working
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (`rsl-rl-lib >= 3.0.1`)
- [Claude Code](https://claude.ai/claude-code) CLI installed
- An Isaac Lab project with registered RL tasks (e.g., `Isaac-Velocity-Flat-Ayg-v0`)

### Installation

**Single-folder copy** — everything goes into `.claude/skills/`:

```bash
# From your Isaac Lab project root
cp -r /path/to/Automated_Reinforcement_Learning_Training_for_Isaac_Lab/auto_train/ .claude/skills/auto_train/
```

Your project structure should look like:

```
your_isaac_lab_project/
├── scripts/
│   └── rsl_rl/
│       ├── train.py          # (your existing train script)
│       └── play.py           # (your existing play script)
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
│           └── experiments/          # (created automatically)
│               └── .scratch/
└── logs/                             # (created automatically)
```

Then install Python dependencies (inside your Isaac Lab venv):

```bash
source .venv/bin/activate
uv pip install tbparse opencv-python-headless
```

See [AUTO_TRAIN_SETUP.md](AUTO_TRAIN_SETUP.md) for detailed step-by-step instructions.

## Usage

### Start an Auto-Train Session

In Claude Code, invoke the skill:

```
/auto_train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 4090 24GB
```

**Arguments format:** `<task_name> level <1|2> on <device_info> [optional notes]`

- **Task name**: Your registered Isaac Lab task ID
- **Level 1** (Reward Tuning): Only JSON overrides — reward weights and PPO hyperparams. No source files edited. Safe and reversible.
- **Level 2** (Full Tuning): Full autonomy — can edit source configs, add/remove rewards, write new reward functions.
- **Device info**: GPU model and VRAM, used to scale `num_envs` appropriately
- **Optional notes**: Domain knowledge, known issues, focus areas (e.g., "robot tends to spider-walk", "focus on foot clearance")

### Examples

```
# Basic flat terrain training
/auto_train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 3060 12GB

# Full tuning with domain hints
/auto_train Isaac-Velocity-Rough-Ayg-v0 level 2 on A100 80GB robot tends to stumble on stairs, previous best was 15.0

# Overnight unattended run
claude --dangerously-skip-permissions "/auto_train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 4090 24GB"
```

### Controls During Training

While auto-train is running, you can type:

| Command | Effect |
|---------|--------|
| `stop` | Stop after the current phase completes |
| `level 1` / `level 2` | Switch privilege level mid-session |
| `focus on X` | Redirect Claude's tuning priority (e.g., "focus on foot clearance") |

### Resuming After Disconnect

If a conversation disconnects, start a new one and say:

```
Continue auto-train, read journal at .claude/skills/auto_train/experiments/<name>/journal.md
```

Claude will read the journal, pick up from the last iteration, and continue.

## Device Scaling

The skill automatically scales `num_envs` and iteration counts based on your device:

| Device | VRAM | Recommended `num_envs` |
|--------|------|----------------------|
| Laptop GPU (RTX 3060) | 12 GB | ~2048 |
| Desktop GPU (RTX 4090) | 24 GB | ~4096 |
| Cloud GPU (A100) | 80 GB | 4096–8192 |

## Tuning Strategy

Auto-train uses **short tuning runs** (300–500 iterations) to test hypotheses quickly, with a minimum of **5 tuning iterations** before considering a production run. This replaces the old approach of 2–3 long runs at 2000 iterations.

A formal **Production Readiness Checklist** must pass before the final production run:
- Body coverage audit (all non-foot bodies penalized/terminated)
- Velocity tracking above threshold (the primary task objective)
- Visual gait quality confirmed via side-view camera
- No suspicious reward hacking patterns detected
- Sufficient tuning iterations completed
- Key metrics showing convergence

## How the Override System Works

Level 1 tuning uses JSON override files to modify parameters without touching source code:

```json
{
  "rewards.track_lin_vel_xy_exp.weight": 2.0,
  "rewards.lin_vel_z_l2.weight": -1.5,
  "agent.learning_rate": 0.0003,
  "agent.entropy_coef": 0.008
}
```

- **Flat dot-path keys** map to `@configclass` attribute paths (e.g., `rewards.gait.weight`)
- `agent.*` prefixed keys apply to the RSL-RL agent config
- All other keys apply to the environment config
- Type casting is automatic (matches existing config types)
- Original source files remain untouched
- **Pre-flight validation** catches format errors (nested dicts, invalid JSON) before Isaac Sim boots

## Output Structure

Each auto-train session produces:

```
.claude/skills/auto_train/experiments/<experiment_name>/
└── journal.md                    # Full log of every iteration with detailed metrics

logs/rsl_rl/<task>/
└── <timestamp>/
    ├── model_*.pt                # Checkpoints
    ├── params/
    │   ├── env.yaml              # Full env config (with overrides applied)
    │   ├── agent.yaml            # Full agent config
    │   └── overrides.json        # Raw override file used
    ├── metrics.json              # Extracted TensorBoard metrics with convergence analysis
    ├── phase_report.json         # Full phase report with convergence summary + suspicious patterns
    ├── frames/
    │   ├── frame_001.png–012.png # Visual inspection frames (side-view)
    │   └── frames_info.json      # Frame manifest
    └── videos/
        └── play/*.mp4            # Side-view policy rollout video
```

## Key Design Decisions

- **Pre-training body coverage audit**: Before training, every robot body is verified to be covered by either termination or penalty — prevents reward exploitation through uncovered body parts (e.g., hip-walking).
- **Pre-training reward analysis**: Reward terms are mathematically analyzed for feasibility — prevents wasting iterations on misconfigured thresholds.
- **Short tuning, long production**: Tuning runs use 300–500 iterations for quick hypothesis testing. Production runs use convergence-appropriate iteration counts. Minimum 5 tuning iterations before production.
- **Convergence detection**: `analyze_metrics.py` detects where each metric plateaued, whether it converged too early, and classifies curve shapes — enabling informed decisions about iteration counts.
- **Suspicious pattern detection**: Automatically flags reward hacking indicators (gait gaming velocity tracking, high total reward with low tracking contribution, early flatlined tracking terms).
- **Side-view visual inspection**: `play_for_inspection.py` provides a camera at robot height following a single robot (2–4 envs), replacing the useless overhead view of 50+ robots.
- **Honest visual assessment**: The skill requires explicit CAN/CANNOT verification framework — Claude must state what the camera angle can and cannot show, never claiming "good gait" without evidence.
- **Production Readiness Checklist**: A formal gate with 6 pass/fail criteria that must all pass before production training.
- **Velocity tracking gate**: Velocity tracking is treated as the primary objective — poor tracking is a blocking issue that must be fixed before other tuning.
- **Hardened unattended operation**: The skill NEVER blocks on user input — no AskUserQuestion, no file creation confirmations, no option selection. All decisions are autonomous with reasoning logged in the journal.
- **Progress-aware waiting**: `run_phase.py` writes progress (iteration, reward, ETA) to the report file during training. `wait_for_phase.py` blocks until done instead of polling.
- **Pre-flight validation**: Override JSON is validated for format errors before Isaac Sim boots, catching mistakes in milliseconds.
- **Detached processes**: Training runs via `nohup setsid` to survive Claude Code timeouts and conversation disconnects.
- **Journal-based resumability**: Everything is logged to `journal.md`, so training can resume from any point after a disconnect.
- **Self-contained folder**: The entire auto-train system lives in `.claude/skills/auto_train/` — single copy to set up on any project.

## License

BSD-3-Clause (following Isaac Lab's license)
