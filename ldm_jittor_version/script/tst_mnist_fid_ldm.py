import sys
import os
# 获取当前脚本所在目录（script/）
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# 获取上级目录（ddpm_jittor/）
parent_dir = os.path.dirname(script_dir)
# 将上级目录添加到 Python 路径
sys.path.append(parent_dir)

import jittor as jt
import numpy as np
import matplotlib.pyplot as plt
from jittor.misc import make_grid
from tqdm import tqdm
import yaml
import time
from jittor.dataset.mnist import MNIST
from jittor.transform import Compose, Resize, Gray, ImageNormalize
from jittor import nn
import cv2  # 用于保存图像文件

from model.gaussian_diffusion import *
from model.unet import UNet

from model.autoencoder import *


#模型参数
latent_size =16 #压缩后的特征维度
hidden_size = 128 #encoder和decoder中间层的维度
input_size= output_size = 32*32 #原始图片和生成图片的维度

#确定模型，导入已训练模型（如有）
vae_name = './result/ldm/autoencoder/the_best_ever/best_vae_mnist.p'

vae = VAE(in_channels=1, latent_dim=4, hidden_dims=[32, 64])
vae.load_state_dict(jt.load(vae_name))

# 添加保存图像到文件夹的函数
def save_images_to_folder(images, folder_path, is_real=False):
    """
    将图像张量保存到指定文件夹
    
    :param images: 图像张量 (B, C, H, W) in [0,1]
    :param folder_path: 目标文件夹路径
    :param is_real: 是否为真实图像（需要特殊处理）
    """
    os.makedirs(folder_path, exist_ok=True)
    
    # 对于真实图像，需要从MNIST数据集中获取
    if is_real:
        transform = Compose([
            Resize(32),  # 与生成图像尺寸一致
            Gray(),
            ImageNormalize(mean=[0.5], std=[0.5])  # 与生成图像相同的归一化
        ])
        testset = MNIST(train=False, transform=transform)
        test_loader = testset.set_attrs(batch_size=100, shuffle=False)
        
        idx = 0
        for batch_idx, (imgs, _) in enumerate(test_loader):
            for i in range(imgs.shape[0]):
                if idx >= len(images):
                    return
                
                # 转换到[0,1]范围
                img = (imgs[i] + 1) * 0.5
                
                # 转换为numpy数组并调整维度顺序
                img_np = img.numpy().transpose(1, 2, 0)
                
                # 转换为3通道（灰度图复制为三通道）
                if img_np.shape[2] == 1:
                    img_np = np.repeat(img_np, 3, axis=2)
                
                # 保存为PNG文件
                img_np = (img_np * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(folder_path, f"real_{idx:05d}.png"), img_np)
                idx += 1
    else:
        # 生成图像的处理
        for i, img in enumerate(images):
            # 转换为numpy数组并调整维度顺序
            img_np = img.transpose(1, 2, 0)
            
            # 转换为3通道（灰度图复制为三通道）
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            import time
            # 保存为PNG文件
            img_np = (img_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(folder_path, f"fake_{i:05d}_{time.time()}.png"), img_np)

# ...（您原有的load_yaml、sample_images、visualize_sampling_process、save_image_grid函数保持不变）...


def load_yaml(config_path):
    """
    加载YAML配置文件并将其转换为可用的Python字典

    :param config_path: YAML配置文件路径
    :return: 解析后的配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def sample_images(model, vae, diffusion, device=None, num_images=16, image_size=(32, 32), channels=1):
    """
    使用训练好的模型采样生成新图像

    :param model: 训练好的UNet模型
    :param diffusion: 初始化好的扩散模型
    :param device: 计算设备 (cuda或cpu)
    :param num_images: 要生成的图像数量
    :param image_size: 生成图像的尺寸 (height, width)
    :param channels: 图像通道数 (MNIST为1)
    :return: 生成图像的Tensor
    """
    # 将模型设置为评估模式
    model.eval()

    # 创建初始随机噪声 (T时刻)
    x = jt.randn(num_images, channels, *image_size)
    print(x.shape)
    # 逆过程逐步去噪
    all_images = []
    steps = diffusion.num_timesteps
    pbar = tqdm(reversed(range(steps)), total=steps, desc="Sampling")

    with jt.no_grad():
        for i in pbar:
            # 创建时间步张量 (所有图像当前都是同一个时间步)
            t = jt.full((num_images,), i)

            # 应用去噪步骤
            out = diffusion.p_sample(
                model=model,
                x=x,
                t=t,
                clip_denoised=True  # 将结果限制在[-1, 1]范围内
            )

            # 获取新的采样图像
            x = out["sample"]


            # 如果需要保存中间过程
            if i % 20 == 0 or i == steps - 1:
                img = vae.decoder.decode_by_x(x).cpu()
                print(img.shape)
                all_images.append(img)

    # 可视化采样过程
    visualize_sampling_process(all_images, steps)

    # 返回最终生成的图像
    generated_images = (x + 1) * 0.5  # 转换为[0,1]范围
    return generated_images


def visualize_sampling_process(all_images, steps):
    """
    可视化采样过程中不同时间点的图像

    :param all_images: 采样过程保存的图像列表
    :param steps: 总步数
    """
    fig, axes = plt.subplots(1, len(all_images), figsize=(20, 3))

    # 设置每个子图的标题为对应的时间步
    for i, img in enumerate(all_images):
        # 计算当前时间步 (从大到小)
        current_step = steps - (len(all_images) - i - 1) * 20
        if current_step < 0:
            current_step = 0

        # 仅显示一批图像中的第一张
        axes[i].imshow(img[0].permute(1, 2, 0), cmap='gray')
        axes[i].set_title(f"t={current_step}")
        axes[i].axis('off')

    plt.suptitle("DDPM Sampling Process (T to 0)", fontsize=16)
    plt.tight_layout()
    plt.savefig("./result/ddpm/mnist/sample/sampling_process.png")
    plt.close()


def save_image_grid(images, grid_size=(4, 4), filename="generated_images.png"):
    """
    将生成的图像保存为网格图

    :param images: 生成的图像张量 (B, C, H, W)
    :param grid_size: 网格尺寸 (rows, cols)
    :param filename: 保存文件名
    """
    # 创建图像网格
    grid = make_grid(images, nrow=grid_size[0], padding=2, pad_value=1)

    # 转换为numpy并显示
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.title("Generated MNIST Images", fontsize=16)
    plt.savefig(f"./result/ddpm/mnist/sample/{filename}")
    plt.close()


if __name__ == "__main__":
    # 设置FID目录
    fid_dir = "./result/ldm/mnist/fid"
    real_path = os.path.join(fid_dir, "real")
    fake_path = os.path.join(fid_dir, "fake")
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)
    
    # 加载配置（与训练时相同的配置）
    config = load_yaml("./config/standard_ddpm.yml")
    
    # 初始化扩散模型（与训练时相同）
    diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule(
            config["diffusion"]["schedule_name"],
            config["diffusion"]["num_diffusion_timesteps"]
        ),
        model_mean_type=ModelMeanType[config["diffusion"]["model_mean_type"]],
        model_var_type=ModelVarType[config["diffusion"]["model_var_type"]],
        loss_type=LossType[config["diffusion"]["loss_type"]],
        rescale_timesteps=config["diffusion"]["rescale_timesteps"]
    )
    
    # 初始化UNet模型（与训练时相同结构）
    model = UNet(
        img_channels=64,
        base_channels=64,
        channel_mults=(1,),
        num_res_blocks=2,
        time_emb_dim=128,
        num_classes=None,
        dropout=0.1,
        attention_resolutions=[8],
        norm="gn",
        num_groups=8,
        initial_pad=0
    )
    
    # 加载训练好的模型权重
    model_path = "./result/ldm/unet/ldm_mnist_final.p"
    model.load_state_dict(jt.load(model_path))
    print(f"Loaded trained model from {model_path}")
    
    # ===== 1. 生成真实图像数据集 =====
    print("准备真实MNIST测试集图像...")
    # 创建10000个空张量作为占位符
    real_images_placeholder = [jt.zeros((1, 1, 32, 32)) for _ in range(10000)]
    save_images_to_folder(real_images_placeholder, real_path, is_real=True)
    print(f"已保存真实图像到: {real_path}")
    
    # ===== 2. 生成10000张图像 =====
    num_images_total = 10000
    batch_size = 256  # 每次生成的图像数量（根据GPU内存调整）
    num_batches = num_images_total // batch_size
    
    print(f"生成{num_images_total}张图像用于FID计算...")
    
    all_generated_images = []
    
    for batch_idx in tqdm(range(num_batches), desc="生成图像批次"):
        # 生成一批图像
        generated_images = sample_images(
            model=model,
            vae=vae,
            diffusion=diffusion,
            device=None,
            num_images=16,
            image_size=(1, 1),
            channels=64
        )

        generated_images = vae.decoder.decode_by_x(generated_images)
        
        # 保存当前批次的生成图像
        save_images_to_folder(generated_images.numpy(), fake_path)
        
        # 如果是最后一批，可能数量不足batch_size
        if batch_idx == num_batches - 1 and num_images_total % batch_size != 0:
            remaining = num_images_total % batch_size
            if remaining > 0:
                generated_images_remaining = sample_images(
                    model=model,
                    vae=vae,
                    diffusion=diffusion,
                    device=None,
                    num_images=16,
                    image_size=(1, 1),
                    channels=64
                )
                generated_images_remaining = vae.decoder.decode_by_x(generated_images_remaining)
                save_images_to_folder(generated_images_remaining.numpy(), fake_path)
    
