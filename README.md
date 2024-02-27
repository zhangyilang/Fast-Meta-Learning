# Fast-Meta-Learning

Welcome to Fast-Meta-Learning! This repository is dedicated to accelerating the implementation of meta-learning algorithms through **parallel** adaptation for a batch of tasks. Unlike to existing repos (e.g., [maml](https://github.com/cbfinn/maml), [MAML-PyTorch](https://github.com/dragen1860/MAML-Pytorch), …) that conduct task-adaptation **sequentially**, this implementation is able to achieve `batch_size` times speedup leveraging parallel processing, albeit at the cost of consuming `batch_size` times more GPU memory. 

## 🌟Features🌟

* **Parallel implementation:** Achieves `batch_size` times faster adaptation compared to sequential methods.
* **Easy setup:** Meta-learning algorithm implementation relies solely on PyTorch (≥2.0) built-in functions, free of external packages. 
* **Flexibility:** Provides a generic framework that is easy to utilize and modify. 

## Environment setup

The parallel implementation of meta-learning algorithms relies merely on PyTorch (≥2.0), while the datasets and models are built on top of [learn2learn](https://github.com/learnables/learn2learn/). 

To set up the environment using Anaconda and pip, execute the following shell commands:

```shell
$ bash env_setup
```

Alternatively, you can use only pip:

```shell
$ pip install torch torchvision torchaudio learn2learn
```

The codes have been tested under the following environment:

---

* Python 3.10

* PyTorch 2.2.1
* TorchVision 0.17.1
* CUDA 12.1
* cuDNN 8.9.2
* learn2learn 0.2.0

---

**⚠️Note:** [learn2learn](https://github.com/learnables/learn2learn/) may fail to build under python 3.11

## How to use the codes

All the parameters including dataset, algorithm, and hyperparameters are centralized in `main.py`, while algorithm implementations can be found in `src/`. As illustrative examples, implementations of [MAML](https://proceedings.mlr.press/v70/finn17a.html), [MetaSGD](https://arxiv.org/pdf/1707.09835.pdf), and [MetaCurvature](https://proceedings.neurips.cc/paper_files/paper/2019/hash/57c0531e13f40b91b3b0f1a30b529a1d-Abstract.html) are provided. 

To carry out the numerical test, use the shell command

```shell
$ python main.py "--arguments" "values"
```

where `arguments` and `values` are the algorithm parameters that you want to alter.

For instance, the following command runs MAML on 5way-1shot *mini*-ImageNet dataset (which will be downloaded to `./datasets/`): 

```shell
$ mkdir datasets
$ python main.py --algorithm MAML --dataset mini-ImageNet --data-dir ./datasets/ --num-cls 5 --num-trn-data 1
```

To carry out your own algorithm, you will need to define a subclass inheriting the abstract base class `MetaLearningAlgBase` in `meta_alg_base.py`. Then, you can implement the abstract methods `_get_meta_model` and `adapt`, which respectively define the meta-parameter and task-adaptation process. 

## Citation

If you find this repo useful, please consider citing the following paper based on the [repo](https://github.com/zhangyilang/MetaProxNet) of which this codebase is developed:

```latex
@inproceedings{MetaProxNet, 
  author={Zhang, Yilang and Giannakis, Georgios B.}, 
  title={Meta-Learning Priors Using Unrolled Proximal Networks}, 
  booktitle={International Conference on Learning Representations}, 
  year={2024}, 
  url={https://openreview.net/forum?id=b3Cu426njo},
}
```