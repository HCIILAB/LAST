<div align="center">    

# Read Ten Lines at One Glance: Line-Aware Semi-Autoregressive Transformer for Multi-Line Handwritten Mathematical Expression Recognition
[![GitHub Badge](https://img.shields.io/badge/GitHub-WGeong-blueviolet?logo=github)](https://github.com/W-Geong) [![GitHub Badge](https://img.shields.io/badge/GitHub-DLVC-success?logo=github)](https://github.com/HCIILAB)

[![Python 3.7 Badge](https://img.shields.io/badge/Python-3.7-blue?link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-370%2F)](https://www.python.org/downloads/release/python-370/) [![PyTorch 1.8.1 Badge](https://img.shields.io/badge/PyTorch-1.8.1-yellowgreen?link=https%3A%2F%2Fpytorch.org%2F)](https://pytorch.org/) [![PyTorch Lightning Badge](https://img.shields.io/badge/PyTorch%20Lightning-1.4.9-orange?link=https%3A%2F%2Fwww.pytorchlightning.ai%2F)](https://www.pytorchlightning.ai/)
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

[WAP](https://github.com/JianshuZhang/WAP) [![PR Badge](https://img.shields.io/badge/PR-2017-brightgreen)](https://www.sciencedirect.com/science/article/abs/pii/S0031320317302376)

[DWAP-TD](https://github.com/JianshuZhang/TreeDecoder) [![ICML Badge](https://img.shields.io/badge/ICML-2020-green)](https://proceedings.mlr.press/v119/zhang20g.html)

[BTTR](https://github.com/Green-Wood/BTTR) [![ICDAR Badge](https://img.shields.io/badge/ICDAR-2021-yellowgreen)](https://link.springer.com/chapter/10.1007/978-3-030-86331-9_37)

[TSDNet](https://github.com/zshhans/TSDNet) [![ACM Badge](https://img.shields.io/badge/ACM_MM-2022-yellow)](https://dl.acm.org/doi/10.1145/3503161.3548424)

[ABM](https://github.com/XH-B/ABM) [![AAAI Badge](https://img.shields.io/badge/AAAI-2022-yellow)](https://ojs.aaai.org/index.php/AAAI/article/view/19885)

[SAN](https://github.com/tal-tech/SAN) [![CVPR Badge](https://img.shields.io/badge/CVPR-2022-orange)](https://openaccess.thecvf.com/content/CVPR2022/html/Yuan_Syntax-Aware_Network_for_Handwritten_Mathematical_Expression_Recognition_CVPR_2022_paper.html)

[CoMER](https://github.com/Green-Wood/CoMER) [![ECCV Badge](https://img.shields.io/badge/ECCV-2022-red)](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_23)

[CAN](https://github.com/LBH1024/CAN) [![ECCV Badge](https://img.shields.io/badge/ECCV-2022-blue)](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_12)

