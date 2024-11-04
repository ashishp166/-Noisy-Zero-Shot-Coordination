# Noisy Zero-Shot Coordination (NZSC)

This repository implements the SelfPlay, Noisy Zero-Shot Coordination (NZSC), and meta-learning methods for training agents to effectively coordinate in such settings.

## Available Environments

- **OS-NLG** (One Shot Noisy Lever Game)
- **I-NLG** (Iterated Lever Game)
- **CEE** (Coordinated Exploration Environment)
- **SSE** (SyncSight Environment)

## Environment Setup

```bash
conda create -n NZSC python=3.10
conda activate NZSC
pip install -r requirements.txt
```

## Training Procedures

### Training via SelfPlay

```bash
python SelfPlay_{env_name}.py
```

Make sure to adjust the environment parameters listed in the `main()` function as needed. Additional environment parameters are available in the config files along with training hyperparameters.

### Training via NZSC

```bash
python NZSC_{env_name}.py
```

**Note**: Make sure to utilize the population of seeds that were trained in SelfPlay.

### Training via MetaNZSC

```bash
python Meta_NZSC_{env_name}.py
```

## Acknowledgements

We would like to thank the authors of JaxMARL for their MultiAgent JAX implementations that inspired the creation of our environments.