#!/usr/bin/env bash

# In this file, we provide commands to get the quantitative results presented in the MVSplat paper.
# Commands are provided by following the order of Tables appearing in the paper.



#---------------------HS Train Code---------------------
#python -m src.main +experiment=re10k data_loader.train.batch_size=14 :  Original Train Code
# For this Server, train baatch size must be <= 4
# (Env : 8 GPU, per RTX4090, 24GB )
# MAX Epoch : 300,000 is default
python -m src.main +experiment=re10k data_loader.train.batch_size=4
python -m src.main +experiment=acid data_loader.train.batch_size=4

# Train Resume
nohup python -m src.main +experiment=re10k data_loader.train.batch_size=4 checkpointing.load=/home/hskim/mvsplat_org/mvsplat/outputs/2025-01-09/08-01-43/checkpoints/epoch_9-step_300000.ckpt &
# If ckpt reach at max_steps, (ex : 300000) => Use Under 280000
python -m src.main +experiment=re10k data_loader.train.batch_size=4  \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-03-03/14-48-18/checkpoints/epoch_9-step_160000.ckpt

"/home/hskim/mvsplat/outputs/2025-03-16/14-38-54/checkpoints/epoch_9-step_160000.ckpt"



wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V" -O dtu_training.rar && rm -rf /tmp/cookies.txt


wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V}" -O {dtu_training.rar} && rm -rf ~/cookies.txt



https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view
https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view?usp=sharing


# --------------- Default Final Models ---------------

# HS Test Code
# "/home/hskim/mvsplat_org/mvsplat/outputs/2024-11-06/14-45-08/checkpoints/epoch_18-step_300000.ckpt"
#"/home/hskim/mvsplat/outputs/2024-11-24/07-53-36/checkpoints/epoch_13-step_300000.ckpt"
"/home/hskim/mvsplat/outputs/2025-01-25/09-38-58/checkpoints/epoch_6-step_140000.ckpt"
"/home/hskim/mvsplat/outputs/2025-02-02/07-44-26/checkpoints/epoch_16-step_280000.ckpt"
"/home/hskim/mvsplat/outputs/2025-02-04/07-13-34/checkpoints/epoch_18-step_300000.ckpt"
"/home/hskim/mvsplat/outputs/2025-02-06/08-05-17/checkpoints/epoch_18-step_300000.ckpt"
"/home/hskim/mvsplat/outputs/2025-02-10/09-13-39/checkpoints/epoch_18-step_300000.ckpt"
"/home/hskim/mvsplat/outputs/2025-02-15/11-11-43/checkpoints/epoch_16-step_220000.ckpt"
"/home/hskim/mvsplat/outputs/2025-02-21/05-21-55/checkpoints/epoch_2-step_40000.ckpt"

"/home/hskim/mvsplat/outputs/2025-02-22/09-08-01/checkpoints/epoch_13-step_220000.ckpt"
"/home/hskim/mvsplat/outputs/2025-02-22/09-08-01/checkpoints/epoch_16-step_280000.ckpt"
"/home/hskim/mvsplat/outputs/2025-03-16/14-38-54/checkpoints/epoch_18-step_300000.ckpt"


# :: RCNN, => No strict with DINO
"/home/hskim/mvsplat/outputs/2025-03-03/14-48-18/checkpoints/epoch_9-step_160000.ckpt"
"/home/hskim/mvsplat/outputs/2025-03-03/14-48-18/checkpoints/epoch_16-step_280000.ckpt"
    # :: BEST!!
"/home/hskim/mvsplat/outputs/2025-03-24/09-56-51/checkpoints/epoch_18-step_300000.ckpt"


# :: RCNN+acid => 1splat!
"/home/hskim/mvsplat/outputs/2025-03-28/08-12-56/checkpoints/epoch_54-step_300000.ckpt"
# :: RCNN+acid, 2splat
"/home/hskim/mvsplat/outputs/2025-04-03/10-24-37/checkpoints/epoch_108-step_300000.ckpt"


# :: DINOv2, => No strict with RCNN
    # :: BEST !!
"/home/hskim/mvsplat/outputs/2025-03-16/14-38-54/checkpoints/epoch_18-step_300000.ckpt"
"/home/hskim/mvsplat/outputs/2025-03-19/13-41-26/checkpoints/epoch_18-step_300000.ckpt"
"/home/hskim/mvsplat/outputs/2025-03-21/07-31-28/checkpoints/epoch_18-step_300000.ckpt"

# :: DINOv2+acid => 1splat !
"/home/hskim/mvsplat/outputs/2025-03-31/16-19-45/checkpoints/epoch_54-step_300000.ckpt"
# :: DINOv2 + acid => 2splat !
"/home/hskim/mvsplat/outputs/2025-04-07/05-43-48/checkpoints/epoch_108-step_300000.ckpt"


python -m src.main +experiment=acid \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-04-03/10-24-37/checkpoints/epoch_108-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

python -m src.main +experiment=acid \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-04-03/10-24-37/checkpoints/epoch_108-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true


python -m src.main +experiment=re10k \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-03-24/09-56-51/checkpoints/epoch_18-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# generate video
python -m src.main +experiment=re10k \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-03-16/14-38-54/checkpoints/epoch_18-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=true



# Table 1: re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# Table 1: acid
python -m src.main +experiment=acid \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-04-07/05-43-48/checkpoints/epoch_108-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true

# generate video
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false


# --------------- Cross-Dataset Generalization ---------------

# Table 2: RealEstate10K -> ACID
python -m src.main +experiment=acid \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-03-24/09-56-51/checkpoints/epoch_18-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true

# Table 2: RealEstate10K -> DTU (2 context views)
python -m src.main +experiment=dtu \
checkpointing.load=/home/hskim/mvsplat/outputs/2025-03-24/09-56-51/checkpoints/epoch_18-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
test.compute_scores=true

# RealEstate10K -> DTU (3 context views)
python -m src.main +experiment=dtu \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx3.json \
dataset.view_sampler.num_context_views=3 \
wandb.name=dtu/views3 \
test.compute_scores=true


# --------------- Ablation Models ---------------

# Table 3: base
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_base \
model.encoder.wo_depth_refine=true 

# Table 3: w/o cost volume
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wocv.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wocv \
model.encoder.wo_depth_refine=true \
model.encoder.wo_cost_volume=true

# Table 3: w/o cross-view attention
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wobbcrossattn_best.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_backbone_cross_attn \
model.encoder.wo_depth_refine=true \
model.encoder.wo_backbone_cross_attn=true

# Table 3: w/o U-Net
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wounet.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_unet \
model.encoder.wo_depth_refine=true \
model.encoder.wo_cost_volume_refine=true

# Table B: w/ Epipolar Transformer
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wepitrans.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_w_epipolar_trans \
model.encoder.wo_depth_refine=true \
model.encoder.use_epipolar_trans=true

# Table C: 3 Gaussians per pixel
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_gpp3.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_gpp3 \
model.encoder.gaussians_per_pixel=3

# Table D: w/ random init (300K)
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_wopretrained.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_pretrained 

# Table D: w/ random init (450K)
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_wopretrained_450k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_pretrained_450k 
