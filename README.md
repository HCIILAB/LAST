<div align="center">    

# Read Ten Lines at One Glance: Line-Aware Semi-Autoregressive Transformer for Multi-Line Handwritten Mathematical Expression Recognition
[![Python 3.7 Badge](https://img.shields.io/badge/Python-3.7-blue?link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-370%2F)](https://www.python.org/downloads/release/python-370/) [![PyTorch 1.8.1 Badge](https://img.shields.io/badge/PyTorch-1.8.1-brightgreen?link=https%3A%2F%2Fpytorch.org%2F)](https://pytorch.org/) [![PyTorch Lightning Badge](https://img.shields.io/badge/PyTorch%20Lightning-1.4.9-orange?link=https%3A%2F%2Fwww.pytorchlightning.ai%2F)](https://www.pytorchlightning.ai/)
</div>



## Project structure
```bash
├── README.md
├── last               # model definition folder
├── config.yaml         # config for hyperparameter
├── lightning_logs      # training logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch.ckpt
│       ├── config.yaml
│       └── hparams.yaml
├── requirements.txt
├── setup.cfg
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd LAST
# install project   
conda create -y -n LAST python=3.7
conda activate LAST
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
```

## Evaluation

```bash
python test.py  
```
### Dataset

M²E Dataset is available at: https://www.modelscope.cn/datasets/Wente47/M2E

## Related Work

Some awesome open-sourced HMER works are listed below:

[WAP](https://github.com/JianshuZhang/WAP) [![PR Badge](https://img.shields.io/badge/PR-2017-brightgreen)](http://XXX)

[DWAP-TD](https://github.com/JianshuZhang/TreeDecoder) [![ICML Badge](https://img.shields.io/badge/ICML-2020-green)](http://XXX)

[BTTR](https://github.com/Green-Wood/BTTR) [![ICDAR Badge](https://img.shields.io/badge/ICDAR-2021-yellowgreen)](http://XXX)

[TSDNet](https://github.com/zshhans/TSDNet) [![ACM Badge](https://img.shields.io/badge/ACM_MM-2022-yellow)](http://XXX)

[ABM](https://github.com/XH-B/ABM) [![AAAI Badge](https://img.shields.io/badge/AAAI-2022-yellow)](http://XXX)

[SAN](https://github.com/tal-tech/SAN) [![CVPR Badge](https://img.shields.io/badge/CVPR-2022-orange)](http://XXX)

[CoMER](https://github.com/Green-Wood/CoMER) [![ECCV Badge](https://img.shields.io/badge/ECCV-2022-red)](http://XXX)

[CAN](https://github.com/LBH1024/CAN) [![ECCV Badge](https://img.shields.io/badge/ECCV-2022-blue)](http://XXX)
