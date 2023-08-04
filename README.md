<div align="center">    

# Read Ten Lines at One Glance: Line-Aware Semi-Autoregressive Transformer for Multi-Line Handwritten Mathematical Expression Recognition

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

M2E
```
Download the dataset from the official website: 
```

## Recommendation

Some other excellent open-sourced HMER algorithms can be found here:

[WAP](https://github.com/JianshuZhang/WAP)[PR'2017]
[DWAP-TD](https://github.com/JianshuZhang/TreeDecoder)[ICML'2020]
[BTTR](https://github.com/Green-Wood/BTTR)[ICDAR'2021]
[ABM](https://github.com/XH-B/ABM)[AAAI'2022]
[SAN](https://github.com/tal-tech/SAN)[CVPR'2022]
[CoMER](https://github.com/Green-Wood/CoMER)[ECCV'2022]
[CoMER](https://github.com/Green-Wood/CoMER)[ECCV'2022]
[CAN](https://github.com/LBH1024/CAN)[ECCV'2022]
