import sys
import os
# 获取当前脚本所在目录（script/）
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# 获取上级目录（ddpm_jittor/）
parent_dir = os.path.dirname(script_dir)
# 将上级目录添加到 Python 路径
sys.path.append(parent_dir)


import csv
import torch
from torch.optim import Adam
from model.unet import *
from model.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml


def load_yaml(config_path):
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


def get_loss(model, x_0, t, device):
    x_0 = x_0.to(device)
    x_noisy, noise = diffusion.q_sample(x_0, t, None)
    noise_pred = model(x_noisy.to(device), t)
    return F.mse_loss(noise, noise_pred)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化UNet
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

    T = diffusion.num_timesteps
    print(f"Diffusion timesteps: {T}")

    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
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

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # ===== CSV日志记录功能 =====
    CSV_PATH = "../result/ddpm/mnist/loss/loss_log.csv"

    # 创建CSV文件并写入表头
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'global_batch', 'batch_loss', 'epoch_avg_loss'])
    # =========================

    global_batch_count = 0  # 全局batch计数器

    for epoch in range(EPOCHS):
        total_loss = 0
        epoch_batches = len(dataloader)

        for batch_idx, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()
            t = torch.randint(0, T, (images.size(0),), device=device).long()
            loss = get_loss(model, images, t, device)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value
            global_batch_count += 1

            # ===== 记录每个batch的损失 =====
            # with open(CSV_PATH, 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([
            #         epoch + 1,
            #         batch_idx,
            #         global_batch_count,
            #         loss_value,
            #         total_loss / (batch_idx + 1)  # 当前epoch平均损失
            #     ])
            # =============================

            if batch_idx % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_idx:04d}/{len(dataloader)} | "
                      f"Loss: {loss_value:.4f} | Avg Loss: {avg_loss:.4f}")

        # 每个epoch结束时保存模型
        epoch_avg_loss = total_loss / epoch_batches
        torch.save(model.state_dict(), f"../result/ddpm/mnist/ddpm_mnist_epoch_{epoch + 1}.pth")

        # ===== 记录epoch总结 =====
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                'EPOCH_END',
                global_batch_count,
                '',
                epoch_avg_loss
            ])
        # ========================

        print(f"Model saved after epoch {epoch + 1}, Avg Loss: {epoch_avg_loss:.4f}")

    torch.save(model.state_dict(), "../result/ddpm/mnist/ddpm_mnist_final.pth")
    print("Training complete!")