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
from torch.optim import Adam

from model.unet import *
from model.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from model.autoencoder import VAE

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

import yaml


def load_yaml(config_path):
    """
    加载YAML配置文件并将其转换为可用的Python字典

    :param config_path: YAML配置文件路径
    :return: 解析后的配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_yaml("../config/standard_ddpm.yml")

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

def get_loss(model, x_0, y, t,  device):
    # x_0 = x_0.to(device)
    # print(x_0.device)
    x_noisy, noise = diffusion.q_sample(x_0, t, None)
    noise_pred = model(x_noisy.to(device), t)
    # return F.l1_loss(noise, noise_pred)
    return F.mse_loss(noise, noise_pred)


if __name__ == "__main__":
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 根据MNIST数据集特点初始化UNet
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

    # 获取扩散步数
    T = diffusion.num_timesteps
    print(f"Diffusion timesteps: {T}")

    # 训练参数
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 20

    # 准备MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Resize((32, 32))  # 调整为统一尺寸
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 创建优化器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    prev_noise = None

    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, y) in enumerate(dataloader):
            # bs = images.shape[0]
            optimizer.zero_grad()
            images = images.to(device)
            # print(images.shape)
            # 生成随机时间步
            t = torch.randint(0, T, (images.size(0),), device=device).long()
            # 计算损失

            # mu, log_var = vae.encode(images)
            # images = vae.reparameterize(mu, log_var)
            images = vae.encode(images)
            # images = images.squeeze(0)
            # print(images.shape)

            x = vae.decoder.fc(images)
            x = x.view(images.size(0), -1, 1, 1)
            # print(x.shape)
            loss = get_loss(model, x, y, t, device)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 定期打印训练信息
            if batch_idx % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_idx:04d}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")

        # 每个epoch结束时保存模型
        torch.save(model.state_dict(), f"../result/ldm/unet/ldm_mnist_epoch_{epoch + 1}.pth")
        print(f"Model saved after epoch {epoch + 1}")

    # 保存最终模型
    torch.save(model.state_dict(), "../result/ldm/unet/ldm_mnist_final.pth")
    print("Training complete!")




