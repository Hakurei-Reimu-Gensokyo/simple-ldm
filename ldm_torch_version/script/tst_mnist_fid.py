import sys
import os

# 获取当前脚本所在目录（script/）
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# 获取上级目录（ddpm_jittor/）
parent_dir = os.path.dirname(script_dir)
# 将上级目录添加到 Python 路径
sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import yaml
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_fid import fid_score

from model.gaussian_diffusion import *
from model.unet import UNet


# 添加保存图像到文件夹的函数
def save_images_to_folder(images, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(images):

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        save_image(img, os.path.join(folder_path, f"image_{i:05d}.png"))


# 添加计算FID的函数
def calculate_fid(real_folder, gen_folder, device, num_images=10000):
    # 检查两个文件夹中的图像数量是否足够
    real_count = len(os.listdir(real_folder))
    gen_count = len(os.listdir(gen_folder))

    if real_count < num_images or gen_count < num_images:
        print(f"警告：文件夹中图像数量不足（真实:{real_count}, 生成:{gen_count}），推荐至少{num_images}张")
        num_images = min(real_count, gen_count)

    print(f"计算FID，使用{num_images}张图像...")

    # 使用pytorch_fid计算FID[1,7](@ref)
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[real_folder, gen_folder],
        batch_size=50,  # 根据GPU内存调整
        device=device,
        dims=2048,  # Inception-v3特征维度
        num_workers=4  # 加速数据加载
    )
    return fid_value


# 加载MNIST测试集并保存到文件夹
def prepare_mnist_testset(folder_path, num_images=10000):
    os.makedirs(folder_path, exist_ok=True)

    # 定义MNIST数据转换
    transform = transforms.Compose([
        transforms.Resize(32),  # 与生成图像尺寸一致
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1)  # 转换为[-1,1]范围
    ])

    # 加载MNIST测试集
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建数据加载器
    loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # 保存图像到文件夹
    count = 0
    for images, _ in loader:
        for i in range(images.shape[0]):
            if count >= num_images:
                return
            img = images[i].unsqueeze(0)
            # 将灰度图复制为三通道
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            save_image(img, os.path.join(folder_path, f"mnist_{count:05d}.png"))
            count += 1


def load_yaml(config_path):
    """
    加载YAML配置文件并将其转换为可用的Python字典

    :param config_path: YAML配置文件路径
    :return: 解析后的配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def sample_images(model, diffusion, device, num_images=16, image_size=(32, 32), channels=1):
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
    x = torch.randn(num_images, channels, *image_size, device=device)

    # 逆过程逐步去噪
    all_images = []
    steps = diffusion.num_timesteps
    pbar = tqdm(reversed(range(steps)), total=steps, desc="Sampling")

    with torch.no_grad():
        for i in pbar:
            # 创建时间步张量 (所有图像当前都是同一个时间步)
            t = torch.full((num_images,), i, device=device, dtype=torch.long)

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
                img = (x + 1) * 0.5  # 从[-1,1]转换到[0,1]
                all_images.append(img.cpu())

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
    plt.savefig("../result/ddpm/mnist/sample/sampling_process.png")
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
    plt.savefig(f"../result/ddpm/mnist/sample/{filename}")
    plt.close()


if __name__ == "__main__":
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建FID目录结构
    fid_dir = "../result/ddpm/mnist/fid"
    real_images_dir = os.path.join(fid_dir, "real")
    generated_images_dir = os.path.join(fid_dir, "generated")
    os.makedirs(real_images_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    # 准备MNIST测试集（仅需运行一次）
    if not os.listdir(real_images_dir):
        print("准备MNIST测试集图像...")
        prepare_mnist_testset(real_images_dir, num_images=10000)
        print(f"已保存MNIST测试集图像到: {real_images_dir}")

    # 加载配置（与训练时相同的配置）
    config = load_yaml("../config/standard_ddpm.yml")

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
        img_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=128,
        num_classes=None,
        dropout=0.1,
        attention_resolutions=[8],
        norm="gn",
        num_groups=8,
        initial_pad=0
    ).to(device)

    # 加载训练好的模型权重
    model_path = "../result/ddpm/mnist/ddpm_mnist_epoch_50.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained model from {model_path}")

    # ===== 采样生成新图像 =====
    # 生成少量图像用于可视化（16张）
    generated_images_vis = sample_images(
        model=model,
        diffusion=diffusion,
        device=device,
        num_images=16,
        image_size=(32, 32),
        channels=1
    )

    # 保存可视化图像
    timestamp = int(time.time())
    save_image_grid(generated_images_vis.cpu(), grid_size=(4, 4),
                    filename=f"generated_mnist_{timestamp}.png")

    # ===== 生成大量图像用于FID计算 =====
    # 推荐10,000张图像以获得可靠FID分数[5](@ref)
    num_fid_images = 10000

    # 分批生成以避免内存溢出
    batch_size = 256  # 根据GPU内存调整
    num_batches = (num_fid_images + batch_size - 1) // batch_size

    print(f"生成{num_fid_images}张图像用于FID计算（分批处理，每批{batch_size}张）...")

    all_generated_images = []

    for batch_idx in tqdm(range(num_batches)):
        # 计算当前批次大小
        current_batch_size = min(batch_size, num_fid_images - batch_idx * batch_size)

        # 生成图像
        generated_images_batch = sample_images(
            model=model,
            diffusion=diffusion,
            device=device,
            num_images=current_batch_size,
            image_size=(32, 32),
            channels=1
        )

        # 保存当前批次到内存
        all_generated_images.append(generated_images_batch.cpu())

    # 合并所有生成的图像
    all_generated_images = torch.cat(all_generated_images, dim=0)

    # 保存生成的图像到文件夹
    save_images_to_folder(all_generated_images, generated_images_dir)
    print(f"已保存生成图像到: {generated_images_dir}")

    # ===== 计算FID分数 =====
    fid_value = calculate_fid(real_images_dir, generated_images_dir, device, num_images=num_fid_images)
    print(f"FID Score: {fid_value:.2f}")

    # 保存FID结果到文件
    with open(os.path.join(fid_dir, f"fid_result_{timestamp}.txt"), "w") as f:
        f.write(f"FID Score: {fid_value:.2f}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Number of images: {num_fid_images}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("FID计算完成！结果保存到", os.path.join(fid_dir, f"fid_result_{timestamp}.txt"))
    print("所有操作完成！")