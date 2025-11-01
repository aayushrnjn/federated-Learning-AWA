
[![arXiv](https://img.shields.io/badge/arXiv-2503.15842-b31b1b.svg)](https://arxiv.org/abs/2503.15842)

**This is the official implementation of the CVPR 2025 paper "[FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors](https://arxiv.org/abs/2503.15842)".**

## FedAWA
Our code is based on Python version 3.7 and PyTorch version 1.13.1. You can run FedAWA with the following command:

```
python main.py --dataset cifar100 --local_model ResNet20 --server_method fedawa --client_method local_train #FedAWA on CIFAR-100 dataset with ResNet20 model
```



## Citing This Repository

Please cite our paper if you find this repo useful in your work:

```
@misc{shi2025fedawaadaptiveoptimizationaggregation,
      title={FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors}, 
      author={Changlong Shi and He Zhao and Bingjie Zhang and Mingyuan Zhou and Dandan Guo and Yi Chang},
      year={2025},
      eprint={2503.15842},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.15842}, 
}
```


## Acknowledgement

We would like to thank the authors for releasing the public repository: [ICML-2023-FedLAW](https://github.com/ZexiLee/ICML-2023-FedLAW/tree/main)

## Contact

Please feel free to contact via email (changlong.shi@connect.polyu.hk) if you have any questions.
