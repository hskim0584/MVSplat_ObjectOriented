# MVSplat

Official implementation of **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images**

Authors: [Yuedong Chen](https://donydchen.github.io/), [Haofei Xu](https://haofeixu.github.io/), [Chuanxia Zheng](https://chuanxiaz.com/), [Bohan Zhuang](https://bohanzhuang.github.io/), [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/), [Andreas Geiger](https://www.cvlibs.net/), [Tat-Jen Cham](https://personal.ntu.edu.sg/astjcham/) and [Jianfei Cai](https://jianfei-cai.github.io/).

### [Project Page](https://donydchen.github.io/mvsplat/) | [arXiv](https://arxiv.org/abs/2403.14627) | [Pretrained Models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU) 

https://github.com/donydchen/mvsplat/assets/5866866/c5dc5de1-819e-462f-85a2-815e239d8ff2

## Installation

To get started, create a conda virtual environment using Python 3.10+ and install the requirements:

```bash
conda create -n mvsplat python=3.10
conda activate mvsplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Acquiring Datasets

### RealEstate10K and ACID

Our MVSplat uses the same training datasets as pixelSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

### DTU (For Testing Only)

* Download the preprocessed DTU data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).
* Convert DTU to chunks by running `python src/scripts/convert_dtu.py --input_dir PATH_TO_DTU --output_dir datasets/dtu`
* [Optional] Generate the evaluation index by running `python src/scripts/generate_dtu_evaluation_index.py --n_contexts=N`, where N is the number of context views. (For N=2 and N=3, we have already provided our tested version under `/assets`.)

## Running the Code

### Evaluation

To render novel views and compute evaluation metrics from a pretrained model,

* get the [pretrained models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU), and save them to `/checkpoints`

* run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# acid
python -m src.main +experiment=acid \
checkpointing.load=checkpoints/acid.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true
```

* the rendered novel views will be stored under `outputs/test`

To render videos from a pretrained model, run the following

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false
```

### Training

Run the following:

```bash
# download the backbone pretrained weight from unimath and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
# train mvsplat
python -m src.main +experiment=re10k data_loader.train.batch_size=14
```

Our models are trained with a single A100 (80GB) GPU. They can also be trained on multiple GPUs with smaller RAM by setting a smaller `data_loader.train.batch_size` per GPU.

### Ablations

We also provide a collection of our [ablation models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU) (under folder 'ablations'). To evaluate them, *e.g.*, the 'base' model, run the following command

```bash
# Table 3: base
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_base \
model.encoder.wo_depth_refine=true 
```

### Cross-Dataset Generalization

We use the default model trained on RealEstate10K to conduct cross-dataset evalutions. To evaluate them, *e.g.*, on DTU, run the following command

```bash
# Table 2: RealEstate10K -> DTU
python -m src.main +experiment=dtu \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
test.compute_scores=true
```

**More running commands can be found at [more_commands.sh](more_commands.sh).**

## BibTeX

```bibtex
@article{chen2024mvsplat,
    title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
    author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
    journal = {arXiv preprint arXiv:2403.14627},
    year    = {2024},
}
```

## Acknowledgements

The project is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat) and has incorporated numerous code snippets from [UniMatch](https://github.com/autonomousvision/unimatch). Many thanks to these two projects for their excellent contributions!
