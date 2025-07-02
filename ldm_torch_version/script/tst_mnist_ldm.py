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
from torchvision.utils import make_grid
from tqdm import tqdm
import yaml
from model.gaussian_diffusion import *
from model.unet import UNet

from model.autoencoder import *


#模型参数
latent_size =16 #压缩后的特征维度
hidden_size = 128 #encoder和decoder中间层的维度
input_size= output_size = 32*32 #原始图片和生成图片的维度

#训练参数
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备
#确定模型，导入已训练模型（如有）
vae_name = '../result/ldm/autoencoder/the_best_ever/best_vae_mnist.pth'

vae = VAE(in_channels=1, latent_dim=4, hidden_dims=[32, 64]).to(device)
vae.load_state_dict(torch.load(vae_name))


def load_yaml(config_path):
    """
    加载YAML配置文件并将其转换为可用的Python字典

    :param config_path: YAML配置文件路径
    :return: 解析后的配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def sample_images(model, vae, diffusion, device, num_images=16, image_size=(32, 32), channels=1):
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
    print(x.shape)
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
        print(img[0].shape)
        axes[i].imshow(img[0].permute(1, 2, 0), cmap='gray')
        axes[i].set_title(f"t={current_step}")
        axes[i].axis('off')

    plt.suptitle("DDPM Sampling Process (T to 0)", fontsize=16)
    plt.tight_layout()
    plt.savefig("../result/ldm/mnist/sample/sampling_process.png")
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
    plt.savefig(f"../result/ldm/mnist/sample/{filename}")
    plt.close()


if __name__ == "__main__":
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
        img_channels=64,  # MNIST是灰度图像，只有1个通道
        base_channels=64,  # 基础通道数
        channel_mults=(1,),  # 通道数倍增器
        num_res_blocks=2,  # 每个分辨率级别的残差块数量
        time_emb_dim=128,  # 时间嵌入维度
        num_classes=None,  # 不进行分类条件的图像生成
        dropout=0.1,  # 随机失活率
        attention_resolutions=[8],  # 在分辨率大小为8时应用注意力
        norm="gn",  # 使用组归一化
        num_groups=8,  # 组归一化的组数
        initial_pad=0  # 不需要额外填充
    ).to(device)

    # 加载训练好的模型权重
    model_path = "../result/ldm/unet/ldm_mnist_final.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained model from {model_path}")

    # 采样生成新图像
    generated_images = sample_images(
        model=model,
        vae=vae,
        diffusion=diffusion,
        device=device,
        num_images=16,
        image_size=(1, 1),
        channels=64
    )
    import time
    # 保存生成的图像

    generated_images = vae.decoder.decode_by_x(generated_images)

    save_image_grid(generated_images.cpu(), grid_size=(4, 4), filename=f"generated_mnist_{time.time()}.png")

    print("Image generation complete! Results saved in ./result/ddpm/mnist/")