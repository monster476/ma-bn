# MA-BN Evaluation of DRQv2 in DMC Environments

This code is built on top of [DRQv2](https://github.com/denisyarats/drqv2) and provides evaluation of the Mode-Aware Batch Normalization (MA-BN) method in DeepMind Control (DMC) environments.

---

## Instructions

### Install MuJoCo

If not already installed:

1. Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).  
2. Download MuJoCo binaries [here](https://www.roboti.us/index.html).  
3. Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.  
4. Use the environment variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.  
5. Append the MuJoCo subdirectory `bin` path into the environment variable `LD_LIBRARY_PATH`.

### Install system libraries

```bash
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

### Install Python dependencies

```bash
conda env create -f conda_env.yml
conda activate drqv2
```

---

## Running MA-BN Experiments

To train with MA-BN:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py agent.critic_norm=2 agent.actor_norm=1 +task=hopper_hop
```

Notes:

* The mode is specified in the configuration file `cfgs/task/config_oni.yaml` under the `update_mode` parameter.  
* For MA-BN, set `update_mode = TTETT`.

---

## Citation

If you use this repo in your research, please consider citing the DRQv2 paper:

```bibtex
@article{yarats2021drqv2,
  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},
  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
  journal={arXiv preprint arXiv:2107.09645},
  year={2021}
}
```

Please also cite our original paper:

```bibtex
@inproceedings{yarats2021image,
  title={Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels},
  author={Denis Yarats and Ilya Kostrikov and Rob Fergus},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=GY6-6sTvGaf}
}
```
