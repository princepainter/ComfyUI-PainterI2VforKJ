import torch
import torch.nn.functional as F

# 模拟 ComfyUI-WanVideoWrapper 中的常量
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

def add_noise_to_reference_video(image, ratio):
    """
    给参考视频添加噪声，增强运动效果
    image: [C, T, H, W]
    ratio: 噪声强度
    """
    return image + torch.randn_like(image) * ratio
