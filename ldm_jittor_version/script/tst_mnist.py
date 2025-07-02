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
from model.gaussian_diffusion import *
from model.unet import UNet

def load_yaml(config_path):
    """
    加载YAML配置文件并将其转换为可用的Python字典

    :param config_path: YAML配置文件路径
    :return: 解析后的配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def sample_images(model, diffusion, device=None, num_images=16, image_size=(32, 32), channels=1):
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
    # # 设置设备
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

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
    )

    # 加载训练好的模型权重
    model_path = "./result/ddpm/mnist/ddpm_mnist_epoch_50.p"
    model.load_state_dict(jt.load(model_path))
    print(f"Loaded trained model from {model_path}")

    # 采样生成新图像
    generated_images = sample_images(
        model=model,
        diffusion=diffusion,
        num_images=16,
        image_size=(32, 32),
        channels=1
    )
    import time
    # 保存生成的图像
    save_image_grid(generated_images, grid_size=(4, 4), filename=f"generated_mnist_{time.time()}.png")

    print("Image generation complete! Results saved in ./result/ddpm/mnist/")