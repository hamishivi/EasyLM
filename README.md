# EasyLM (Forked)
Large language models (LLMs) made easy, EasyLM is a one stop solution for
pre-training, finetuning, evaluating and serving LLMs in JAX/Flax. EasyLM can
scale up LLM training to hundreds of TPU/GPU accelerators by leveraging
JAX's pjit functionality.

Building on top of Hugginface's [transformers](https://huggingface.co/docs/transformers/main/en/index)
and [datasets](https://huggingface.co/docs/datasets/index), this repo provides
an easy to use and easy to customize codebase for training large language models
without the complexity in many other frameworks.

EasyLM is built with JAX/Flax. By leveraging JAX's pjit utility, EasyLM is able
to train large models that don't fit on a single accelerator by sharding
the model weights and training data across multiple accelerators. Currently,
EasyLM supports multiple TPU/GPU training in a single host as well as multi-host
training on Google Cloud TPU Pods.

The **fork** adds support for some RLHF methods, such as Direct Preference Optimization (DPO) and 
Proximal Policy Optimization (PPO) for *some* models. It has been used for models, such as [Tulu 2](https://huggingface.co/allenai/tulu-2-dpo-70b), with more coming soon.

The original `EasyLM` is no longer supported. This is largely supported for research use and doesn't
come with standard maitenance rules and guidelines.

Currently, the following models are supported to some capacity.
* [LLaMA(-2)](https://arxiv.org/abs/2302.13971), with DPO and PPO
* [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)
* [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)
* [OLMo] (coming soon!)

For the core models used, for now just Llama, the directory `EasyLM/models/llama/` contains scripts such as `convert_hf_to_easylm.py` and `convert_easylm_to_hf.py` for easy integration with other libraries. 

Models trained here can also be evaluated with [AllenAI's Open Instruct](https://github.com/allenai/open-instruct) repository via `scripts/submit_open_instruct_eval.py`.

## Directory Structure

The directory is *organized* as follows:
```
├── README.md                   <- The top-level README for researchers using this project
├── beaker_configs/             <- [AI2 only] example config and automatically generated experiment configs
├── conversion_scripts/         <- Scripts for creating .json datasets from HuggingFace format (see `create_preference_data.sh`)
├── docs/                       <- New and existing documentation
|   ├── ai2.md                      ├── In depth tutorial on how to use EasyLM on AI2 infra
|   └── *.md                        └── Preexisting docs
├── EasyLM/                     <- Core utils and modeling files
|   ├── models/                     ├── Packages and scripts specific to each model's architecture
|   ├── scripts/                    ├── Benchmarking and evaluation scripts
|   └── *.py                        └── Utilities and tools
├── examples/                   <- Bash scripts for running EasyLM training
├── scripts/                    <- Misc. extra scripts for benchmarking and evaluation.
└── LICENSE
```

---

## Discord Server
The original authors are running an unofficial Discord community (unaffiliated with Google) for discussion related to training LLMs in JAX. [Follow this link to join the Discord server](https://discord.gg/Rf4drG3Bhp). They have dedicated channels for several JAX based LLM frameworks, include EasyLM, [JaxSeq](https://github.com/Sea-Snell/JAXSeq), [Alpa](https://github.com/alpa-projects/alpa) and [Levanter](https://github.com/stanford-crfm/levanter).


## Models Trained with EasyLM
### OpenLLaMA
OpenLLaMA is our permissively licensed reproduction of LLaMA which can be used
for commercial purposes. Check out the [project main page here](https://github.com/openlm-research/open_llama).
The OpenLLaMA can serve as drop in replacement for the LLaMA weights in EasyLM.
Please refer to the [LLaMA documentation](docs/llama.md) for more details.


### Koala
Koala is our new chatbot fine-tuned on top of LLaMA. If you are interested in
our Koala chatbot, you can check out the [blogpost](https://bair.berkeley.edu/blog/2023/04/03/koala/)
and [documentation for running it locally](docs/koala.md).


### Tulu 2
[Tulu 2 is a suite](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101) of DPO aligned models built on top of the Llama 2 suite.

## Installation
The installation method differs between GPU hosts and Cloud TPU hosts. The first
step is to pull from GitHub.

``` shell
git clone https://github.com/hamishivi/EasyLM.git
cd EasyLM
export PYTHONPATH="${PWD}:$PYTHONPATH"
```

#### Installing on GPU Host
The GPU environment can be installed via [Anaconda](https://www.anaconda.com/products/distribution).

``` shell
conda env create -f scripts/gpu_environment.yml
conda activate EasyLM
```

#### Installing on Cloud TPU Host
The TPU host VM comes with Python and PIP pre-installed. Simply run the following
script to set up the TPU host.

``` shell
./scripts/tpu_vm_setup.sh
```


## [Documentations](docs/README.md)
The EasyLM documentations can be found in the [docs](docs/) directory.


## Reference
If you found EasyLM useful in your research or applications, please cite using the following BibTeX:
```
@software{geng2023easylm,
  author = {Geng, Xinyang},
  title = {EasyLM: A Simple And Scalable Training Framework for Large Language Models},
  month = March,
  year = 2023,
  url = {https://github.com/young-geng/EasyLM}
}
```
And the citation for this fork specifically, if you wish:
```
@software{hamishivi2023easylmfork,
  author = {Ivison, Hamish and Wang, Yizhong, and Pyatkin, Valentina and Liu, Jiacheng and Lu, Jiasen and Wu, Zeqiu},
  title = {EasyLM-Fork: A Simple And Scalable Training Framework for Large Language Models},
  month = October,
  year = 2023,
  url = {https://github.com/hamishivi/EasyLM}
}
```


## Credits
* The LLaMA implementation is from [JAX_llama](https://github.com/Sea-Snell/JAX_llama)
* The JAX/Flax GPT-J and RoBERTa implementation are from [transformers](https://huggingface.co/docs/transformers/main/en/index)
* Most of the JAX utilities are from [mlxu](https://github.com/young-geng/mlxu)
* The codebase is heavily inspired by [JAXSeq](https://github.com/Sea-Snell/JAXSeq)
