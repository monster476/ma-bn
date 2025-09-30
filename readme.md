<h2 align="center">An Investigation of Batch Normalization in Off-Policy Actor-Critic Algorithms
</h2>


<h5 align="center">
<div align="center">

[Li Wang](https://scholar.google.com/citations?user=Qtt1QBMAAAAJ&hl=zh-CN)<sup>1</sup>,
[Sudun](https://github.com/Sudun)<sup>1</sup>,
[Xingjian Zhang](https://scholar.google.com/citations?user=H34fwioAAAAJ&hl=zh-CN)<sup>1</sup>,
[Wenjun Wu](https://iai.buaa.edu.cn/info/1013/1093.htm)<sup>1,2,3,✉</sup>, 
[Lei Huang](https://huangleibuaa.github.io/)<sup>1,2,3,✉</sup>

<sup>1</sup>School of Artificial Intelligence, Beihang University, Beijing, China<br>
<sup>2</sup>Hangzhou International Innovation Institute, Beihang University, Hangzhou, China<br>
<sup>3</sup>Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, Beihang University

</div>

## 📰 News

- [2025-9] 🎉 Our arXiv paper [MA-BN](https://arxiv.org/abs/2509.23750) is released!

## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/2435/2435606.png"> About

Batch Normalization (BN) has played a pivotal role in the success of deep learning by improving training stability, mitigating overfitting, and enabling more effective optimization. However, its adoption in deep reinforcement learning (DRL) has been limited due to the inherent non-i.i.d. nature of data and the dynamically shifting distributions induced by the agent’s learning process. In this paper, we argue that, despite these challenges, BN retains unique advantages in DRL settings, particularly through its stochasticity and its ability to ease training. When applied appropriately, BN can adapt to evolving data distributions and enhance both convergence speed and final performance. To this end, we conduct a comprehensive empirical study on the use of BN in off-policy actor-critic algorithms, systematically analyzing how different training and evaluation modes impact performance. We further identify failure modes that lead to instability or divergence, analyze their underlying causes, and propose the Mode-Aware Batch Normalization (MA-BN) method with practical actionable recommendations for robust BN integration in DRL pipelines. We also empirically validate that, in RL settings, MA-BN accelerates and stabilizes training, broadens the effective learning rate range, enhances exploration, and reduces overall optimization difficulty.



## 📌 Usage

For code execution and reproducibility, please refer to the CrossQ and drqv2 directories in the repository.



## 📝 Citation

If you would like to cite our work, please use the following format:
```bibtex
@misc{wang2025investigationbatchnormalizationoffpolicy,
      title={An Investigation of Batch Normalization in Off-Policy Actor-Critic Algorithms}, 
      author={Li Wang and Sudun and Xingjian Zhang and Wenjun Wu and Lei Huang},
      year={2025},
      eprint={2509.23750},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.23750}, 
}
```
