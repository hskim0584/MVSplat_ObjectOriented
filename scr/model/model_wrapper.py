from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from matplotlib import pyplot as plt
import moviepy.editor as mpy
import torch
import torch.nn as nn
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import torchvision


import lpips

# For Faster R-CNN + LPIPS
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import random
# For DINOv2 + LPIPS

from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T
import requests
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw


from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

from scipy.linalg import eigh


"""
#*=============== hb's code ===============*#
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #output.color.size(): torch.Size([2, 4, 3, 256, 256])
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 3, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):       
        
        return self.model(img)
#*=========================================*#

"""

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    splat: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        
        
        
        self.fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to("cuda")
        self.fasterRCNN.eval()  # 평가 모드 설정
        self.loss_fasterRCNN = lpips.LPIPS(net='vgg').to("cuda")
        
        
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dinoModel = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda")
        self.dinoModel.eval()
        
        self.preprocess =  T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        # Set model to eval mode

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0
        self.Flag = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            








    def compute_iou(self, box1, box2):
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
    
        inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area.clamp(min=1e-6)
        
        
    def detect_objects(self, batch_images):
        """ Batch 단위 객체 탐지 수행 """
        batch_images = [ToPILImage()(img.cpu()) for img in batch_images]  # B, C, H, W → 개별 이미지
        batch_tensors = [ToTensor()(img).to("cuda") for img in batch_images]
    
        with torch.no_grad():
            preds = self.fasterRCNN(batch_tensors)  # Batch prediction 수행
    
        batch_boxes = []
        for i, pred in enumerate(preds):
            scores = pred['scores']
            boxes = pred['boxes']
            
            # 0.8 이상 confidence 필터링
            keep = scores > 0.8
            scores = scores[keep]
            boxes = boxes[keep]
            
            # score 기준 정렬 후 상위 3개 선택
            if scores.numel() > 0:
                topk = min(3, scores.size(0))
                top_indices = torch.topk(scores, topk).indices
                boxes = boxes[top_indices]
    
                # 박스 확장
                img_h, img_w = batch_tensors[i].shape[1:]  # (C, H, W) → H, W
                new_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    pad_w = w * 0.2
                    pad_h = h * 0.2
                    x1_new = max(0, x1 - pad_w)
                    y1_new = max(0, y1 - pad_h)
                    x2_new = min(img_w, x2 + pad_w)
                    y2_new = min(img_h, y2 + pad_h)
                    new_boxes.append(torch.tensor([x1_new, y1_new, x2_new, y2_new], dtype=torch.float32))
                
                boxes = torch.stack(new_boxes)
    
            batch_boxes.append(boxes.cpu())  # 각 sample당 박스
    
        return batch_boxes
        
        
    
    def extract_object_patch(self, image, box):
        """ Bounding Box 좌표를 기반으로 객체 영역 crop """
        # RCNN Crop
        #x1, y1, x2, y2 = map(int, box.tolist())
        
        #Random Crop
        x1, y1, x2, y2 = map(int, box)
        return image[:, y1:y2, x1:x2]  # C, H, W 형태 유지
        
        
        
    def compute_lpips_loss_woRCNN_RandomBB(self, images_before, images_after, box_size=(40, 40)):
        B, _, H, W = images_before.shape
        
        print("size : ", H, ", ", W)
        lpips_losses = torch.tensor(0.0, device="cuda", requires_grad=True)

        for i in range(B):
            image_before = images_before[i]
            image_after = images_after[i]
            
            lpips_loss = torch.zeros(1, device="cuda", requires_grad=True)
    
            for _ in range(3):
                # 랜덤 바운딩 박스 좌표 생성
                box_w, box_h = box_size
                x1 = random.randint(32, 96 - box_w)
                y1 = random.randint(32, 96 - box_h)
                x2 = x1 + box_w
                y2 = y1 + box_h
                box = (x1, y1, x2, y2)
    
                # 이미지에서 해당 위치 Crop
                patch_before = self.extract_object_patch(image_before, box)
                patch_after = self.extract_object_patch(image_after, box)
    
                # LPIPS 계산 후 누적
                
                loss = self.loss_fasterRCNN(patch_before, patch_after)
                lpips_loss = lpips_loss + loss
    
            lpips_losses = lpips_losses + lpips_loss
    
        return lpips_losses / B
    
    
    def draw_boxes_on_image(self, image_tensor, boxes, color="red", width=2):
        """
        이미지 텐서 위에 바운딩박스를 그려주는 함수
        - image_tensor: [3, H, W] torch.Tensor
        - boxes: [N, 4] torch.Tensor or list of [x1, y1, x2, y2]
        - color: 박스 색상 (기본값: 빨간색)
        - width: 박스 두께
        """
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.device != torch.device("cpu"):
                image_tensor = image_tensor.cpu()
            img_pil = to_pil_image(image_tensor)
        else:
            raise ValueError("image_tensor must be a torch.Tensor")
    
        draw = ImageDraw.Draw(img_pil)
    
        if boxes is None or isinstance(boxes, torch.Tensor) and boxes.numel() == 0:
            return img_pil  # 박스 없음, 원본 이미지 반환
    
        if isinstance(boxes, torch.Tensor):
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)  # [4] → [1, 4]
    
        for box in boxes:
            box = [float(coord) for coord in box]
            draw.rectangle(box, outline=color, width=width)
    
        return img_pil
        
        
        
        
    def compute_lpips_loss(self, images_before, images_after, batch_boxes_before, batch_boxes_after):
        B = images_before.shape[0]
        
        """
        print("==== Shape Check ====")

        # images: Tensor [B, C, H, W] or list of tensors
        if isinstance(images_before, torch.Tensor):
            print(f"images_before shape: {images_before.shape}")  # torch.Size([B, C, H, W])
        else:
            print(f"images_before: list of {len(images_before)} elements")
            for idx, img in enumerate(images_before):
                print(f"  - image_before[{idx}]: {img.shape}")
        
        if isinstance(images_after, torch.Tensor):
            print(f"images_after shape: {images_after.shape}")
        else:
            print(f"images_after: list of {len(images_after)} elements")
            for idx, img in enumerate(images_after):
                print(f"  - image_after[{idx}]: {img.shape}")
        
        # boxes: list of list of boxes → List[B] of Tensor[N_i, 4]
        print(f"batch_boxes_before: list of {len(batch_boxes_before)} elements")
        for idx, boxes in enumerate(batch_boxes_before):
            print(f"  - batch_boxes_before[{idx}]: {boxes.shape}")
        
        print(f"batch_boxes_after: list of {len(batch_boxes_after)} elements")
        for idx, boxes in enumerate(batch_boxes_after):
            print(f"  - batch_boxes_after[{idx}]: {boxes.shape}")
            
        """
            
    
        lpips_losses = torch.tensor(0.0, device="cuda", requires_grad=True)
        for i in range(B):
            boxes_before = batch_boxes_before[i]
            boxes_after = batch_boxes_after[i]
    
            if len(boxes_before) == 0 or len(boxes_after) == 0:
              continue
    
            
    
            lpips_loss = torch.zeros(1, device="cuda", requires_grad=True)  # 초기화 수정

            for box1, box2 in zip(boxes_before, boxes_after):
            
            
                img = images_after[i]  # [3, H, W] 텐서
                boxes = box2 # [N, 4] 텐서
                img_with_boxes = self.draw_boxes_on_image(img, boxes)
                img_with_boxes.save("debug_box_output.png")
            
            
            
                obj1 = self.extract_object_patch(images_before[i], box1)
                obj2 = self.extract_object_patch(images_after[i], box2)
                # LPIPS Loss 계산
                
                #print("obj1 : ", obj1)
                #print("obj2 : ", obj2)
                """
                topilimage = torchvision.transforms.ToPILImage()
                
                img1 = topilimage(images_before[i])
                img_pil1 = topilimage(obj1[0])
                img_pil2 = topilimage(obj1[1])
                img_pil3 = topilimage(obj1[2])
                
                img2 = topilimage(images_after[i])
                img_pil11 = topilimage(obj2[0])
                img_pil12 = topilimage(obj2[1])
                img_pil13 = topilimage(obj2[2])
                
                img_array0 = np.array(img1)
                img_array1 = np.array(img_pil1)
                img_array2 = np.array(img_pil2)
                img_array3 = np.array(img_pil3)
                
                img_array00 = np.array(img2)
                img_array11 = np.array(img_pil11)
                img_array12 = np.array(img_pil12)
                img_array13 = np.array(img_pil13)
                
                plt.figure()
                plt.subplot(2, 4, 1)
                plt.imshow(img_array0)
                plt.subplot(2, 4, 2)
                plt.imshow(img_array1)
                plt.subplot(2, 4, 3)
                plt.imshow(img_array2)
                plt.subplot(2, 4, 4)
                plt.imshow(img_array3)
                
                plt.subplot(2, 4, 5)
                plt.imshow(img_array00)
                plt.subplot(2, 4, 6)
                plt.imshow(img_array11)
                plt.subplot(2, 4, 7)
                plt.imshow(img_array12)
                plt.subplot(2, 4, 8)
                plt.imshow(img_array13)
                plt.show()
                
                """
                min_h, min_w = min(obj1.shape[1], obj2.shape[1]), min(obj1.shape[2], obj2.shape[2])
                obj1 = F.interpolate(obj1.unsqueeze(0), size=(min_h, min_w), mode="bilinear", align_corners=False)
                obj2 = F.interpolate(obj2.unsqueeze(0), size=(min_h, min_w), mode="bilinear", align_corners=False)

                #print("lpips_losses", obj1)
                #print("obj1.shape", obj1.shape)
                if(min_h >= 32 and min_w >= 32):
                    lpips_loss = lpips_loss + self.loss_fasterRCNN(obj1, obj2)
    
            lpips_losses = lpips_losses + lpips_loss / len(boxes_before)  # 평균 LPIPS Loss 저장
        #print("lpips_losses", lpips_losses / B)
        
        
        return lpips_losses / B
        
        """
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dinoModel = DINOv2Model.from_pretrained("facebook/dinov2-base").to("cuda")
        self.dinoModel.eval()
        """
        
    def crop_object(self, image_tensor, box):
        # image_tensor: (3, H, W), box: (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.tolist())
        cropped = image_tensor[:, y1:y2, x1:x2]
        cropped_pil = T.ToPILImage()(cropped.cpu())
        processed = self.preprocess(cropped_pil).to("cuda")
        
        return processed
    
    
    def extract_dino_feature(self, img_tensor_batch):
        # img_tensor_batch: (B, 3, 224, 224)
        img_tensor_batch = img_tensor_batch.clamp(0, 1)
        inputs = self.processor(images=img_tensor_batch, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.dinoModel(**inputs)
        return outputs.last_hidden_state[:, 0]  # CLS token (B, 768)
    
    
    
    def compute_dino_loss_batch(self, pre_images, post_images, pre_boxes, post_boxes):
        all_pre_crops = []
        all_post_crops = []
    
        for b in range(len(pre_boxes)):
            pre_img = pre_images[b]
            post_img = post_images[b]
            boxes1 = pre_boxes[b]
            boxes2 = post_boxes[b]
    
            if boxes1 is None or boxes2 is None or len(boxes1) == 0 or len(boxes2) == 0:
                continue
    
            N = min(len(boxes1), len(boxes2))  # 짝수만 비교
            for i in range(N):
                all_pre_crops.append(self.crop_object(pre_img, boxes1[i]))
                all_post_crops.append(self.crop_object(post_img, boxes2[i]))
    
        if len(all_pre_crops) == 0:
            return torch.tensor(0.0, device="cuda", requires_grad=True)
    
        pre_batch = torch.stack(all_pre_crops)  # (M, 3, 224, 224)
        post_batch = torch.stack(all_post_crops)
    
        feats_pre = self.extract_dino_feature(pre_batch)  # (M, 768)
        feats_post = self.extract_dino_feature(post_batch)
    
        cos_sim = F.cosine_similarity(feats_pre, feats_post, dim=1)  # (M,)
        loss = 1 - cos_sim.mean()
    
        return loss
        

    def training_step(self, batch, batch_idx):#, optimizer_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )

        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        #*===== code for extrapolated samples =====*#
        
        
        
        
        if(self.global_step > 100000):
            _, v, _, _ = batch["context"]["extrinsics"].shape
    
            num_frames = 600
            t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
            
            target_sample = ['context', 'target'][1]
            B = batch[target_sample]["extrinsics"].shape[0]
            
            # 원래 extrinsics
            E0 = batch[target_sample]["extrinsics"][:, 0]  # (B, 4, 4)
            E1 = batch[target_sample]["extrinsics"][:, 1]  # (B, 4, 4)
            
            origin_a = E0[:, :3, 3].clone()
            origin_b = E1[:, :3, 3].clone()
            delta = (origin_a - origin_b).norm(dim=-1)  # (B,)
            
            factor_radius = 3.0
            depth = 3.0  # 시야선 상에서 중심으로 삼을 거리
            
            # 중심점 계산: cam_pos + (-Z) * depth
            cam_pos = E0[:, :3, 3]  # (B, 3)
            cam_dir = -E0[:, :3, 2]  # (B, 3)  ← camera's forward
            centers = cam_pos + cam_dir * depth  # (B, 3)
            
            # 회전축 생성
            axis = torch.randn((B, 3), device=self.device)
            axis = axis / axis.norm(dim=-1, keepdim=True)  # 정규화 (B, 3)
            
            # 회전 각도: ±5도 내에서
            angles = (torch.rand((B,), device=self.device) * 2 - 1) * (torch.pi / 36)  # 약 ±5도
            
            # Rodrigues' rotation matrix 만들기
            K = torch.zeros((B, 3, 3), device=self.device)
            K[:, 0, 1] = -axis[:, 2]
            K[:, 0, 2] = axis[:, 1]
            K[:, 1, 0] = axis[:, 2]
            K[:, 1, 2] = -axis[:, 0]
            K[:, 2, 0] = -axis[:, 1]
            K[:, 2, 1] = axis[:, 0]
            
            eye = torch.eye(3, device=self.device).expand(B, 3, 3)
            R = eye + torch.sin(angles).view(B, 1, 1) * K + (1 - torch.cos(angles).view(B, 1, 1)) * (K @ K)  # (B, 3, 3)
            
            # 4x4 회전 행렬로 확장
            R_full = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)  # (B, 4, 4)
            R_full[:, :3, :3] = R
            
            # 이동 행렬: 중심점 기준 회전
            T_to_origin = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
            T_to_origin[:, :3, 3] = -centers
            
            T_back = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
            T_back[:, :3, 3] = centers
            
            # 최종 wobble 변환: T_back @ R @ T_to_origin
            tf = T_back @ R_full @ T_to_origin  # (B, 4, 4)
            
            # 기존 방식 그대로 interpolation 수행
            extrinsics_new = interpolate_extrinsics(E0[0].clone(), (E1[0] if v == 2 else E0[0]), t * 5 - 2)
            intrinsics_new = interpolate_intrinsics(batch[target_sample]["intrinsics"][0, 0].clone(), (batch[target_sample]["intrinsics"][0, 1] if v == 2 else batch[target_sample]["intrinsics"][0, 0]), t * 5 - 2)
            
            # Wobble 적용
            extrinsics_new = extrinsics_new @ tf.unsqueeze(1)  # (B, 600, 4, 4)
            intrinsics_new = intrinsics_new[None]  # (1, num_frames, 3, 3)
            
            # 프레임 샘플링
            indices = torch.randint(0, num_frames, (4,))
            extrinsics_new = extrinsics_new[:, indices, :, :]
            
            
            output_extrapolate = self.decoder.forward(
                gaussians,
                extrinsics_new,
                batch[target_sample]["intrinsics"],
                batch[target_sample]["near"],
                batch[target_sample]["far"],
                (h, w), 
                depth_mode=self.train_cfg.depth_mode,#"depth"
            )
        
        
        
        
        
        
        target_gt = batch["target"]["image"]
        
        
        
        
        
        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())


        #*===== original loss function =====*#
        """
        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)
        """    
        #*==================================*#


        #*===== modified loss function including perceptual loss regarding extrapolated views =====*#
        # Compute and log loss.
        """
        self.use_loss_extrapolate = [False, True][0]

        total_loss = 0
        for loss_fn in self.losses:
            if loss_fn.name == 'lpips':
                loss_perceptual_org = loss_fn.forward(output, batch, gaussians, self.global_step)
                self.log(f"loss/lpips_org", loss_perceptual_org)
                total_loss = total_loss + loss_perceptual_org

                if self.use_loss_extrapolate:
                    loss_perceptual_extrapolate = loss_fn.forward(output_extrapolate, batch, gaussians, self.global_step)
                    self.log(f"loss/lpips_extrapolate", loss_perceptual_extrapolate)
                    total_loss = total_loss + loss_perceptual_extrapolate * 2
            else:
                loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss
        self.log("loss/total", total_loss)
        #*=========================================================================================*#
        """
        
        
        
        
        
        
        
        
        
        
                
            
        
        
        
        
        
        
        total_loss = 0
        for loss_fn in self.losses:
            if loss_fn.name == 'lpips':
                loss_lpips_org = loss_fn.forward(output, batch, gaussians, self.global_step)
                self.log(f"loss loss_lpips_org", loss_lpips_org)
                total_loss = total_loss + loss_lpips_org
                
                
                if(self.global_step > 100000):
                    bb, vv, cc, hh, ww = target_gt.shape
                    combine_target_gt = target_gt.view(bb*vv, cc, hh, ww)
                    combine_extrapolate = output_extrapolate.color.view(bb*vv, cc, hh, ww)
                    batch_boxes_before = self.detect_objects(combine_target_gt)
                    
                    dino_loss = self.compute_dino_loss_batch(combine_target_gt, 
                        combine_extrapolate, 
                        batch_boxes_before, 
                        batch_boxes_before)
                    total_loss = total_loss + dino_loss
                    print("dino_loss", dino_loss)
                    self.log(f"loss dino_loss_fasterRCNN", dino_loss)
                    
                    
                # Wobble + Random Crop
                """
                if(self.global_step > 0):
                
                    bb, vv, cc, hh, ww = target_gt.shape
                    combine_extrapolate = output_extrapolate.color.view(bb*vv, cc, hh, ww)
                    combine_target_gt = target_gt.view(bb*vv, cc, hh, ww)
                    
                    lpips_loss = self.compute_lpips_loss_woRCNN_RandomBB(combine_target_gt, combine_extrapolate).item()
                    #lpips_loss = self.compute_lpips_loss(combine_target_gt, combine_extrapolate).item()
                    #print("lpips_loss", lpips_loss)
                    self.log(f"loss lpips_loss_fasterRCNN", lpips_loss)
                    #print("LPIPS Loss per Batch:", lpips_loss)
                    total_loss = total_loss + lpips_loss
                    
                """
                
                # Wobble + RCNN Object Crop
                """
                if(self.global_step > 0):
                
                    bb, vv, cc, hh, ww = target_gt.shape
                    combine_extrapolate = output_extrapolate.color.view(bb*vv, cc, hh, ww)
                    combine_target_gt = target_gt.view(bb*vv, cc, hh, ww)
                    
                    lpips_loss = self.compute_lpips_loss(combine_target_gt, combine_extrapolate).item()
                    #print("lpips_loss", lpips_loss)
                    self.log(f"loss lpips_loss_fasterRCNN", lpips_loss)
                    #print("LPIPS Loss per Batch:", lpips_loss)
                    total_loss = total_loss + lpips_loss
                """
                
                # Wobble + DINOv2
                """
                if(self.global_step > 100000):
                
                    bb, vv, cc, hh, ww = target_gt.shape
                    combine_extrapolate = output_extrapolate.color.view(bb*vv, cc, hh, ww)
                    combine_target_gt = target_gt.view(bb*vv, cc, hh, ww)
                    
                    if combine_extrapolate.max() > 1 or combine_extrapolate.min() < 0:
                        combine_extrapolate = (combine_extrapolate - combine_extrapolate.min()) / (combine_extrapolate.max() - combine_extrapolate.min())
                        
                    if combine_target_gt.max() > 1 or combine_target_gt.min() < 0:
                        combine_target_gt = (combine_target_gt - combine_target_gt.min()) / (combine_target_gt.max() - combine_target_gt.min())
                    
                    inputs1 = self.processor(images=combine_extrapolate, return_tensors="pt").to("cuda")
                    inputs2 = self.processor(images=combine_target_gt, return_tensors="pt").to("cuda")
                    outputs1 = self.dinoModel(**inputs1)
                    outputs2 = self.dinoModel(**inputs2)
                    
                    features1 = outputs1.last_hidden_state  # (batch, seq_len, hidden_dim)
                    features2 = outputs2.last_hidden_state  # (batch, seq_len, hidden_dim)
                    
                    rinoloss = self.contrastive_loss(features1, features2).item() / 5
                    
                    #print("rinoloss", rinoloss)
                    self.log(f"loss rinoloss", rinoloss)
                    #print("LPIPS Loss per Batch:", lpips_loss)
                    total_loss = total_loss + rinoloss
                    
                    """
                    
            else:
                loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                #print(f"{loss_fn.name} : {loss}")
                
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss * 2
        self.log("loss/total", total_loss)
        
        
        
        

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
                f"loss_lpips_org = {loss_lpips_org:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        #*=====================*#

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]
        
        
        
        
        """
        
        _, v22, _, _ = batch["context"]["extrinsics"].shape
        num_frames = 600
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)

        target_sample = ['context', 'target'][1]

        origin_a = batch[target_sample]["extrinsics"][:, 0, :3, 3].clone()
        origin_b = batch[target_sample]["extrinsics"][:, 1, :3, 3].clone()
        delta = (origin_a - origin_b).norm(dim=-1)
        
        factor_radius = 1.0
        
        tf = generate_wobble_transformation(delta * factor_radius, t, 5, scale_radius_with_t=False,)
        extrinsics_new = interpolate_extrinsics(batch[target_sample]["extrinsics"][0, 0].clone(), (batch[target_sample]["extrinsics"][0, 1] if v22 == 2 else batch[target_sample]["extrinsics"][0, 0]), t * 5 - 2,)
        intrinsics_new = interpolate_intrinsics(batch[target_sample]["intrinsics"][0, 0].clone(), (batch[target_sample]["intrinsics"][0, 1] if v22 == 2 else batch[target_sample]["intrinsics"][0, 0]), t * 5 - 2,)
        extrinsics_new, intrinsics_new = extrinsics_new @ tf, intrinsics_new[None]
        
        indices = torch.randint(0, num_frames, (3,))
        extrinsics_new = extrinsics_new[:, indices, :, :]
        
        
        output_extrapolate = self.decoder.forward(
            gaussians,
            extrinsics_new,
            batch[target_sample]["intrinsics"],
            batch[target_sample]["near"],
            batch[target_sample]["far"],
            (h, w), 
            depth_mode=self.train_cfg.depth_mode,#"depth"
        )
        
        
        lpips_loss = self.compute_lpips_loss(rgb_gt, output_extrapolate.color[0]).item()
        
        
        
        """
        
        
        
        
        
        
        #*************************************    HS Code begin    *************************************#
        #
        #
        #
        #
        #HS's Code begin
        if self.test_cfg.splat and self.Flag < 10:
        
            gaussians_output = {
                'means': gaussians.means[0],  # 10 Gaussians, random positions
                'covariances': gaussians.covariances[0],  # Random covariance matrices
                'harmonics': gaussians.harmonics[0],  # Random colors
                'opacities': gaussians.opacities[0]  # Random transparency
            }
            output_file_output = "output%d.splat" %self.Flag
            
            SH_C0 = 0.28209479177387814
            with open(output_file_output, 'w', encoding='utf-8') as f:
                for i in range(65536):
                    position = np.array(gaussians_output['means'][i].tolist(), dtype=np.float32) 
                    
                    eigenvalues, eigenvectors = eigh(gaussians_output['covariances'][i].cpu())
                    scale = np.sqrt(eigenvalues) 
                    
                    m00, m01, m02 = eigenvectors[0, 0], eigenvectors[0, 1], eigenvectors[0, 2]
                    m10, m11, m12 = eigenvectors[1, 0], eigenvectors[1, 1], eigenvectors[1, 2]
                    m20, m21, m22 = eigenvectors[2, 0], eigenvectors[2, 1], eigenvectors[2, 2]
                    qw = np.sqrt(1 + m00 + m11 + m22) / 2
                    qx = (m21 - m12) / (4 * qw)
                    qy = (m02 - m20) / (4 * qw)
                    qz = (m10 - m01) / (4 * qw)
                    
                    quaternion  = np.array([qw, qx, qy, qz])
                    norm = np.linalg.norm(quaternion)
                    q_normalized = quaternion / norm
                    q_weight = np.array([1.0, 0.85, 0.85, 0.85])
                    q_scaled = ((q_normalized*q_weight + 1) / 2 * 255).astype(np.uint8)
                    
                    
                    r = gaussians_output['harmonics'][i].cpu()[0].tolist()
                    g = gaussians_output['harmonics'][i].cpu()[1].tolist()
                    b = gaussians_output['harmonics'][i].cpu()[2].tolist()
                    
                    r = np.array([r], dtype=np.float32)
                    g = np.array([g], dtype=np.float32)
                    b = np.array([b], dtype=np.float32)
                    
                    r = 0.5 + SH_C0 * r[0][0]
                    g = 0.5 + SH_C0 * g[0][0]
                    b = 0.5 + SH_C0 * b[0][0]
                    
                    
                    r = int(np.clip(r * 255, 0, 255))
                    g = int(np.clip(g * 255, 0, 255))
                    b = int(np.clip(b * 255, 0, 255))
                    
                    a = gaussians_output['opacities'][i].cpu().item()
                    a = int(np.clip(a * 255, 0, 255))
                    
                    
                    color = np.array([r, g, b, np.uint8(a)], dtype=np.uint8)
                    
                    # Write Gaussian parameters in .splat format
                    position.tofile(f, format='f4') #float32 * 3 : 4byte * 3 = 12 byte
                    scale.tofile(f, format='f4')    #float32 * 3 : 4byte * 3 = 12 byte
                    color.tofile(f, format='u1')
                    q_scaled.tofile(f, format='u1') #uint8 * 4 : 1byte * 4 = 4byte
            self.Flag += 1
            
            
        
            """
              0)
              
            print("gaussians", gaussians.means.shape)
            print("gaussians", gaussians.covariances.shape)
            
            result :
              gaussians torch.Size([1, 131072, 3])
              gaussians torch.Size([1, 131072, 3, 3])

            """
            
            
            
            """
             1) 
            print("  batch[target][extrinsics] size", batch["target"]["extrinsics"].size)
            
            print("  batch[target][intrinsics] size", batch["target"]["intrinsics"].size)
            
            print("  batch[target][near] size", batch["target"]["near"].size)
            
            print("  batch[target][far] size", batch["target"]["far"].size)
            
            
            result  : 
              batch[target][extrinsics] size <built-in method size of Tensor object at 0x7f384b751d50>
              batch[target][intrinsics] size <built-in method size of Tensor object at 0x7f384b753740>
              batch[target][near] size <built-in method size of Tensor object at 0x7f384b751e40>
              batch[target][far] size <built-in method size of Tensor object at 0x7f384b751e90>

            """
            
            
            """
              2)
              print("  batch[target][extrinsics] .detach().cpu().numpy() size", batch["target"]["extrinsics"].detach().cpu().numpy().size)
            
              print("  batch[target][intrinsics] .detach().cpu().numpy() size", batch["target"]["intrinsics"].detach().cpu().numpy().size)
            
              print("  batch[target][near] .detach().cpu().numpy() size", batch["target"]["near"].detach().cpu().numpy().size)
            
              print("  batch[target][far] .detach().cpu().numpy() size", batch["target"]["far"].detach().cpu().numpy().size)
              
              result :
                batch[target][extrinsics] .detach().cpu().numpy() size 48
                batch[target][intrinsics] .detach().cpu().numpy() size 27
                batch[target][near] .detach().cpu().numpy() size 3
                batch[target][far] .detach().cpu().numpy() size 3

            """
            
            """
              3)
              print("  batch[target][extrinsics] .detach().cpu().numpy() ", batch["target"]["extrinsics"].detach().cpu().numpy())
            
              print("  batch[target][intrinsics] .detach().cpu().numpy() ", batch["target"]["intrinsics"].detach().cpu().numpy())
              
              print("  batch[target][near] .detach().cpu().numpy() ", batch["target"]["near"].detach().cpu().numpy())
              
              print("  batch[target][far] .detach().cpu().numpy() ", batch["target"]["far"].detach().cpu().numpy())
              
              result : 
                 batch[target][extrinsics] .detach().cpu().numpy()  [[[[ 8.6715531e-01  6.8841316e-02 -4.9325708e-01  1.7136855e-01]
                 [-8.1695691e-02  9.9664700e-01 -4.5257350e-03 -1.6762537e-01]
                 [ 4.9129164e-01  4.4221494e-02  8.6987180e-01  1.3776296e+00]
                 [-1.0538393e-08 -5.5100258e-10 -2.9493954e-09  1.0000000e+00]]
              
                [[ 5.5278713e-01  9.0844058e-02 -8.2835609e-01 -5.3638875e-01]
                 [-1.2826531e-01  9.9146986e-01  2.3137068e-02 -2.8170142e-01]
                 [ 8.2339197e-01  9.3459472e-02  5.5972391e-01  2.0730231e+00]
                 [-2.5378295e-08  5.2249360e-09  1.7585258e-09  1.0000000e+00]]
              
                [[ 5.5065548e-01  9.0679370e-02 -8.2979262e-01 -5.4911715e-01]
                 [-1.2842415e-01  9.9144971e-01  2.3122126e-02 -2.8280824e-01]
                 [ 8.2479441e-01  9.3833089e-02  5.5759263e-01  2.0795655e+00]
                 [-2.5346139e-08 -5.9379772e-09 -4.1416115e-09  9.9999994e-01]]]]
                batch[target][intrinsics] .detach().cpu().numpy()  [[[[0.86214495 0.         0.5       ]
                 [0.         0.8623555  0.5       ]
                 [0.         0.         1.        ]]
              
                [[0.86214495 0.         0.5       ]
                 [0.         0.8623555  0.5       ]
                 [0.         0.         1.        ]]
              
                [[0.86214495 0.         0.5       ]
                 [0.         0.8623555  0.5       ]
                 [0.         0.         1.        ]]]]
                batch[target][near] .detach().cpu().numpy()  [[1. 1. 1.]]
                batch[target][far] .detach().cpu().numpy()  [[100. 100. 100.]]

            """
            
            
            """
              4)
              print("  batch[target][extrinsics] .detach().cpu().numpy() ", np.array(batch["target"]["extrinsics"].detach().cpu().numpy()).shape)
            
              print("  batch[target][intrinsics] .detach().cpu().numpy() ", np.array(batch["target"]["intrinsics"].detach().cpu().numpy()).shape)
              
              print("  batch[target][near] .detach().cpu().numpy() ", np.array(batch["target"]["near"].detach().cpu().numpy()).shape)
              
              print("  batch[target][far] .detach().cpu().numpy() ", np.array(batch["target"]["far"].detach().cpu().numpy()).shape)
              
              result :
                batch[target][extrinsics] .detach().cpu().numpy()  (1, 3, 4, 4)
                batch[target][intrinsics] .detach().cpu().numpy()  (1, 3, 3, 3)
                batch[target][near] .detach().cpu().numpy()  (1, 3)
                batch[target][far] .detach().cpu().numpy()  (1, 3)

            """
        #HS's Code end
        #
        #
        #
        #
        #*************************************    HS Code end    *************************************#

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]



        #TRAIN STEP CODE for Reference
        #gaussians = self.encoder(batch["context"], self.global_step, False, scene_names=batch["scene"])
        #output = self.decoder.forward(
        #    gaussians,
        #    batch["target"]["extrinsics"],
        #    batch["target"]["intrinsics"],
        #    batch["target"]["near"],
        #    batch["target"]["far"],
        #    (h, w),
        #    depth_mode=self.train_cfg.depth_mode,
        #)
        #target_gt = batch["target"]["image"]





        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        projections = hcat(*render_projections(
                                gaussians_softmax,
                                256,
                                extra_label="(Softmax)",
                            )[0])
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        self.render_video_interpolation_exaggerated(batch)#hb's comment
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                #delta * 0.25,
                delta * 1.0,#hb's comment
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")


    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                #delta * 0.5,
                delta * 1.0,#hb's comment
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        #return self.render_video_generic(batch, trajectory_fn, "interpolation_exagerrated", num_frames=300, smooth=False, loop_reverse=False,)#original code
        return self.render_video_generic(batch, trajectory_fn, "interpolation_exagerrated", num_frames=60, smooth=False, loop_reverse=False,)#hb's comment


    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        #print(f"type(self.parameters()): {type(self.parameters())}")
        #print(f"self.parameters(): {self.parameters()}")
        print(self.optimizer_cfg.lr)
        print(f"len(list(self.parameters()): {len(list(self.parameters()))}")

        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        #optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.optimizer_cfg.lr)

        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.optimizer_cfg.lr, self.trainer.max_steps + 10, pct_start=0.01, cycle_momentum=False, anneal_strategy='cos',)
            #warm_up_D = torch.optim.lr_scheduler.OneCycleLR(optimizer_D, self.optimizer_cfg.lr, self.trainer.max_steps + 10, pct_start=0.01, cycle_momentum=False, anneal_strategy='cos',)
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(optimizer, 1/warm_up_steps, 1, total_iters=warm_up_steps,)
            #warm_up_D = torch.optim.lr_scheduler.LinearLR(optimizer_D, 1/warm_up_steps, 1, total_iters=warm_up_steps,)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": warm_up, "interval": "step", "frequency": 1,},},
        #[{"optimizer": optimizer, "lr_scheduler": {"scheduler": warm_up, "interval": "step", "frequency": 1,},}, {"optimizer": optimizer_D, "lr_scheduler": {"scheduler": warm_up_D, "interval": "step", "frequency": 1,},}]
