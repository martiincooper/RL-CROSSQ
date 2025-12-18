
<p align="center">
  <img src="http://adityab.github.io/CrossQ/static/images/crossq-fancy.png" align="center" width="300px"/>
</p>

# Super Mario Bros RL Benchmark: CrossQ vs DroQ vs RedQ

This repository contains a benchmark comparing three advanced Reinforcement Learning algorithms—**CrossQ**, **DroQ**, and **RedQ**—on the **Super Mario Bros** environment (`gym-super-mario-bros`).

The goal is to visually and empirically compare which algorithm learns to play the game faster and more efficiently.

## What is CrossQ?

**CrossQ** (Batch Normalization in Deep Reinforcement Learning) is a novel approach to Off-Policy RL that introduces Batch Normalization (BN) into the Critic networks without using Target Networks.

### Why is it Efficient?
- **No Target Networks**: Traditional algorithms like SAC or TD3 use "target networks" (slowly updating copies of the main network) to stabilize training. CrossQ removes these, simplifying the architecture.
- **Batch Normalization**: By correctly applying BN, CrossQ allows for much higher Update-To-Data (UTD) ratios (training the network many times per environment step) without divergence.
- **Sample Efficiency**: It achieves state-of-the-art sample efficiency, often matching or beating Model-Based RL methods while being purely Model-Free.

In this benchmark, we pit it against:
- **RedQ**: Randomized Ensembled Double Q-Learning (High UTD, many critics).
- **DroQ**: Dropout Q-Learning (High UTD, dropout for regularization).

## Setup Instructions

### Prerequisites
- Python >= 3.9
- Jupyter Notebook
- Basic build tools (for compiling some dependencies)

### Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd CrossQ
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment (Conda or venv).
   ```bash
   conda create -n crossq_mario python=3.9
   conda activate crossq_mario
   
   # Install project dependencies
   pip install -e .
   
   # Install Mario env and other tools
   pip install gym-super-mario-bros nes-py gymnasium shimmy opencv-python matplotlib jupyter ipywidgets
   ```

   *Note: If you are on Mac with Apple Silicon (M1/M2/M3), ensure you have compatible `jax` and `tensorflow-probability` versions installed.*

3. **Verify Setup**
   You can run the provided verification script to ensure the environment and models initialize correctly:
   ```bash
   python verify_setup.py
   ```

## Running the Benchmark

The entire benchmark is contained within a Jupyter Notebook.

1. **Launch Jupyter**
   ```bash
   jupyter notebook benchmark_mario.ipynb
   ```

2. **Run All Cells**
   - The notebook will initialize three agents: CrossQ, DroQ, and RedQ.
   - It will train them sequentially on `SuperMarioBros-v0`.
   - **Training**: Rewards are logged to `./mario_benchmark_logs/`.
   - **Plotting**: After training, the notebook generates a learning curve comparing the episode rewards of the three models.
   - **Visual Race**: Finally, a popup window will appear showing the three agents playing the level side-by-side in real-time.

## Project Structure
- `benchmark_mario.ipynb`: Main experiment notebook.
- `mario_wrapper.py`: Custom wrapper to make `gym-super-mario-bros` compatible with modern `gymnasium` and `sbx`.
- `sbx/`: The core RL algorithms library (based on Stable Baselines JAX), modified to support CNNs (`NatureCNN`) for pixel-based Mario environment.
- `train.py`: Original training script from the CrossQ paper (mostly for MuJoCo/continuous control tasks).

## Credits
- Based on the **CrossQ** paper: *Bhatt A., Palenicek D., Belousov B., Argus M., Amiranashvili A., Brox T., Peters J.*
- Original Repository: [adityab/CrossQ](https://github.com/adityab/CrossQ)
- Benchmark implementation by Antigravity (Google DeepMind).
