[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Zero-Shot Instruction Following in RL via Structured LTL Representations

This repository contains the official implementation of StructLTL ([arxiv.org/2602.14344](https://arxiv.org/pdf/2602.14344)), as well as the environments
*ZoneEnv-NM* and *Warehouse*.

Also included are baselines [DeepLTL](https://arxiv.org/pdf/2410.04631),
[LTL2Action](https://arxiv.org/pdf/2102.06858), and [GenZ-LTL](https://arxiv.org/abs/2508.01561) for comparison. We adapted these baselines to fit within our JAX-based implementation of StructLTL and the environments.

## Installation

We recommend using [pixi](https://pixi.sh/latest/) to install the required dependencies
in a virtual environment. Installing on GPU is highly recommended:
```bash
pixi install -e gpu
pixi run copy-templates
```

### Rabinizer 4

We use [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html) for the
conversion of LTL formulae into LDBAs. Download the program using [this
link](https://www7.in.tum.de/~kretinsk/rabinizer4.zip) and unzip it into the
`dependencies` subfolder. Rabinizer requires Java 17 to be installed on your system and
`$JAVA_HOME` to be set accordingly. To test the installation, run the following:
```bash
./dependencies/rabinizer4/bin/ltl2ldba -h
```
which should print a help message.
We tested the implementation with OpenJDK 17.0.18.

### Docker
We alternatively provide a Dockerfile to build an image with all required dependencies:
```bash
docker build -t structltl:gpu .
```

Note that this is a GPU-enabled image and requires a working [Docker](https://www.docker.com/) and [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installation.

We recommend mounting `runs` and `data` directories for preservation:
```bash
mkdir data runs
docker run --rm -it --gpus all \
  --shm-size=2g \
  --mount type=bind,src="$PWD/data",dst=/workspace/data \
  --mount type=bind,src="$PWD/runs",dst=/workspace/runs \
  structltl:gpu
```

## Experiments

We use [Hydra](https://hydra.cc/docs/intro/) to configure experiments. The below
commands assume you want to train and evaluate StructLTL for the Warehouse
experiment. You can set the experiment Hydra configuration via command line; see the
`conf` subfolder for the pre-made experiment configuration files.

### Precomputing Resets

For efficiency, we precompute the environment resets for both training and evaluation:
```bash
pixi run -e gpu python scripts/precompute_resets.py experiment=struct_ltl/warehouse train=true
pixi run -e gpu python scripts/precompute_resets.py experiment=struct_ltl/warehouse train=false
```

For LTL2Action, we also recommend precomputing the training curriculum:
```bash
pixi run -e gpu python scripts/precompute_curriculum.py experiment=ltl2action/warehouse
```

### Training

To train a policy:
```bash
pixi run -e gpu python scripts/train.py experiment=struct_ltl/warehouse run=tmp
```

To plot training performance:
```bash
pixi run -e gpu python scripts/plotting/plot_training_curves.py
```
**NOTE**: you will need to edit which runs to plot inside `scripts/plotting/plot_training_curves.py`

> [!NOTE]
> If you run into OOM errors, reduce the `num_seeds` that are trained in parallel. You can combine trained models from different runs with [combine_models.py](scripts/combine_models.py).

### Evaluation

To evaluate the trained policy on a set of LTL formulae:
```bash
pixi run -e gpu python scripts/eval/eval.py experiment=struct_ltl/warehouse run=tmp formulas=warehouse/finite eval.finite=True
```

> [!NOTE]
> If you run into OOM errors, the eval script supports `models_per_batch` and `formulas_per_batch` to control the evaluation parallelism. Lowering these will reduce memory requirements, at the expense of longer runtimes.

> [!TIP]
> All formulae used in the paper are provided in `conf/formulas`.

To visualize trajectories for the trained policy on an LTL formula (both for drawing trajectories and rendering them in real-time):
```bash
pixi run -e gpu python scripts/eval/visualize_trajectories.py experiment=struct_ltl/warehouse run=tmp eval.formula="F (vase & region_a & X(!vase & region_a))"
```

To compute evaluation curves:
```bash
pixi run -e gpu python scripts/eval/compute_eval_curves.py experiment=struct_ltl/warehouse run=tmp +formulas=warehouse/finite
```

To plot evaluation curves:
```bash
pixi run -e gpu python scripts/plotting/plot_eval_curves.py
```
**NOTE**: you will need to edit which runs to plot inside `scripts/plotting/plot_eval_curves.py`

### Ablation Studies

This repository includes code for reproducing the ablation studies D.2.1 and D.2.2. See `conf/experiment/tokenized_ltl/warehouse.yaml` for a configuration of StructLTL with a flat sequence model, and `conf/experiment/gcn_ltl/warehouse.yaml` for the GNN configuration. To train StructLTL with a GRU encoder instead of the attention mechanism, specify `model/sequence=gru`. 

## License

This project is licensed under the terms of the [MIT License](/LICENSE).

## Citation

If you find this code useful in your research, please consider citing our paper:
```bibtex
@inproceedings{structltl,
    title     = {Zero-Shot Instruction Following in {RL} via Structured {LTL} Representations},
    author    = {Mathias Jackermeier and Mattia Giuri and Jacques Cloete and Alessandro Abate},
    booktitle = {arXiv},
    year      = {2026}
}
```