This code is built on top of [CrossQ](https://github.com/adityabhatt/crossq).

---

## Setup

Create a conda environment and install dependencies:

```bash
conda create -n crossq python=3.11.5
conda activate crossq
conda install -c nvidia cuda-nvcc=12.3.52

pip install -e .
pip install "jax[cuda12_pip]==0.4.19" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Running Experiments

Example: training with MA-BN (without WandB). 
The following command trains an agent on `Humanoid-v4`, where the actor uses TT mode and the critic uses ETT mode:

```bash
python train.py -algo crossq -env Humanoid-v4 -seed 0 -wandb_mode 'disabled' -update_mode TTETT -eval_qbias 1
```

---

## Citation

If you use this code in your research, please cite the original CrossQ paper:

```bibtex
@inproceedings{
  bhatt2024crossq,
  title={CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity},
  author={Aditya Bhatt and Daniel Palenicek and Boris Belousov and Max Argus and Artemij Amiranashvili and Thomas Brox and Jan Peters},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=PczQtTsTIX}
}
```

---

## Acknowledgements

This implementation is based on [Stable Baselines JAX](https://github.com/araffin/sbx) and [CrossQ](https://github.com/adityabhatt/crossq).