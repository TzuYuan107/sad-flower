# SAD-Flower

## Environment Setup with Docker

We recommend running all code in Docker for consistent environment setup. We provide a `Dockerfile` and helper scripts to simplify the build and run process.
### Prerequisites

Before you begin, ensure you have the following installed:
- **Docker** ([installation guide](https://docs.docker.com/engine/install/))
- **NVIDIA Docker runtime** (required for GPU support) ([installation guide](https://github.com/NVIDIA/nvidia-docker))
- A **GPU** with NVIDIA drivers installed (optional but recommended for faster training)

### Step 1: Build the Docker Image

The `BuildDocker.sh` script builds the Docker image from the provided Dockerfile.

**Default usage:**
```bash
bash BuildDocker.sh
```

This builds the image with the default name `sad_flower`.

**Custom image name:**

If you want to use a different image name, edit `BuildDocker.sh` and change your desired name, e.g.:
```bash
docker build -t "project_name" .
```

### Step 2: Run the Docker Container

The `runDocker.sh` script starts the container with the necessary configuration including GPU support and X11 display forwarding.

**Before running, you must modify two things:**

1. **Set the source directory mount path** (line 8)
   
   Point to your local project directory. For example:
   ```bash
   --mount type=bind,source=$HOME/path/to/sad_flower,target=/workspace
   ```
   
   This mounts your local code directory to `/workspace` inside the container.

2. **Use the correct image name** (line 10)
   
   Make sure the image name matches what you built in Step 1:
   ```bash
   sad_flower  # or your custom name
   ```

**Run the container:**
```bash
bash runDocker.sh
```

The script automatically handles:
- ✅ GPU access via NVIDIA Docker runtime
- ✅ X11 display forwarding (for visualization)
- ✅ Privilege escalation for MuJoCo rendering
- ✅ Conda environment activation (gflower)

Once inside the container, you'll be in the `/workspace` directory with all code mounted from your local system. Inside the container, do 
```bash
pip install -e .
```

## Training

The SAD-Flower project trains three separate components in sequence: a flow matching model for policy learning, a value function model, and a system dynamics model. Each component has its own training script located in `run_scripts/`.

### 1. Train Flow Model

The flow model learns the policy using flow matching on diffusion trajectories.

**Script:** `run_scripts/train_maze2d.sh`

**Usage:**
```bash
bash run_scripts/train_maze2d.sh
```

**What it does:**
- Trains a flow matching policy on three maze environments: `maze2d-open-dense-v0`, `maze2d-umaze-v1`, and `maze2d-large-v1`
- Uses Conditional Flow Matching (CFM) or Optimal Transport CFM (OT_CFM)
- Adapts the prediction horizon based on environment complexity (48 steps for open, 128 for umaze, 384 for large)

**Key parameters:**
- `--device`: GPU device 
- `--env`: Target environment
- `--horizon`: Prediction horizon (adapted per environment)
- `--n_train_steps`: Total training steps 
- `--batch_size`: 
- `--learning_rate`: 
- `--flow_matching_type`: CFM or OT_CFM

---

### 2. Train Value Model

The value function model learns to estimate the expected return for state-action pairs.

**Script:** `run_scripts/train_value_maze2d.sh`

**Usage:**
```bash
bash run_scripts/train_value_maze2d.sh
```

**What it does:**
- Trains a value function on maze2d environments
- Uses infinite horizon formulation with environment-specific horizons
- Normalizes inputs using `LimitsNormalizer`

**Key parameters:**
- `--device`: GPU device 
- `--env`: Target environment 
- `--horizon`: Prediction horizon (adapted per environment)
- `--n_train_steps`: Total training steps 
- `--batch_size`: 
- `--learning_rate`: 

---

### 3. Train System Dynamics Model

The dynamics model learns to predict the next state given current state and action, essential for planning.

**Script:** `run_scripts/train_forward_dynamics.sh`

**Usage:**
```bash
bash run_scripts/train_forward_dynamics.sh
```

**What it does:**
- Trains a neural network dynamics model with Jacobian regularization
- Supports high-precision (float64) training for improved numerical stability (default is off)
- Uses different batch sizes optimized for each environment
- Validates periodically during training
- predict delta_state instead of next_state for stable training

**Key parameters:**
- `--device`: GPU device 
- `--env`: Target environment
- `--horizon`: Prediction horizon 
- `--n_train_steps`: Total training steps 
- `--batch_size`: 
- `--val_freq`: Validation frequency 
- `--save_freq`: Checkpoint save frequency 
- `--is_high_precision`: Enable float64 training

---

### Training from scratch

For a complete training pipeline on all components:

1. **Start with flow model training** (longest):
   ```bash
   bash run_scripts/train_maze2d.sh
   ```

2. **While flow model trains, start value model** (in a separate terminal/GPU):
   ```bash
   bash run_scripts/train_value_maze2d.sh
   ```

3. **Start dynamics model training** (in another terminal/GPU):
   ```bash
   bash run_scripts/train_forward_dynamics.sh
   ```

All training outputs are saved to `./logs` with organized folder structure by environment and component type.

## Evaluation Scripts

The `eval_*.sh` scripts are for running trained models and measuring how they behave in the maze environments.

Use them when you want to evaluate policy on maze. Some common parameters:
- Unified params
   - horizon: plan length of flow model
   - ode_t_steps: total ODE steps for flow matching
   - ode_solver: choose the ODE solver; Euler is usually enough, while RK4 is much slower
- obs_Center, obs_radius: defines the two constraints center and radius
   - guidance_method: use "ss" to sample a batch and pick the best; `ss_batch` defaults to 32
- SAD-Flower params
   - start_time: activate time of control, do CBF/CLF after this time
   - constraint_strategy: 
      - QP_PT_ACSCDC: SAD-Flower
      - No: FM baseline
      - Reject: reject sample baseline
   - ACSCDCFlag: 0 uses NN forward dynamics; 1 uses explicit Jacobians and a smaller action correction

We provide an example set of parameters in the script. Pleae note that performance may vary depending on the task, environment, optimizer settings, hardware, etc.

Run them with large maze:
```bash
bash run_scripts/eval_large_maze_FM.sh # evaluate flow without control as baseline
bash run_scripts/eval_large_maze_Reject.sh # evaluate reject sample (sample and verification) as baseline
bash run_scripts/eval_large_maze_SADFlower.sh # evaluate SAD-Flower
```

Run them with umaze:
```bash
bash run_scripts/eval_umaze_FM.sh # evaluate flow without control as baseline
bash run_scripts/eval_umaze_Reject.sh # evaluate reject sample (sample and verification) as baseline
bash run_scripts/eval_umaze_SADFlower.sh # evaluate SAD-Flower
```

## Debug Scripts

Run them with:
```bash
bash run_scripts/debug_exp_mainresult.sh # summarize evaluation records and main results
```


## Reference
```
@misc{huang2026sadflowerflowmatchingsafe,
      title={SAD-Flower: Flow Matching for Safe, Admissible, and Dynamically Consistent Planning}, 
      author={Tzu-Yuan Huang and Armin Lederer and Dai-Jie Wu and Xiaobing Dai and Sihua Zhang and Hsiu-Chin Lin and Shao-Hua Sun and Stefan Sosnowski and Sandra Hirche},
      year={2026},
      eprint={2511.05355},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.05355}, 
}
```

## Acknowledgments

This codebase builds upon and is inspired by:

- **[Diffuser](https://github.com/jannerm/diffuser)** 
- **[Flow Guidance](https://github.com/AI4Science-WestlakeU/flow_guidance)** 

We thank the authors of these projects for their open-source contributions.

