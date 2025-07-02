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
import jittor as jt
from jittor.optim import Adam
from jittor import nn, dataset, transform
from jittor.dataset import DataLoader
import yaml
import numpy as np
from model.unet import *
from model.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType

# 确保可视化目录存在
VIS_DIR = "./result/ddpm/mnist/visual/"
os.makedirs(VIS_DIR, exist_ok=True)

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 可视化函数：显示原始图像、加噪图像和预测图像
def visualize_denoising(original, noisy, predicted, epoch, batch_idx, nrow=4):
    """
    可视化加噪和去噪过程
    :param original: 原始图像 [B, C, H, W]
    :param noisy: 加噪图像 [B, C, H, W]
    :param predicted: 预测图像 [B, C, H, W]
    :param epoch: 当前epoch
    :param batch_idx: 当前batch索引
    :param nrow: 每行显示的图像数量
    """
    # 转换为numpy并调整值范围到[0, 1]
    def to_numpy(tensor):
        tensor = (tensor + 1) * 0.5  # [-1,1] -> [0,1]
        return tensor.numpy().transpose(0, 2, 3, 1)
    
    orig_np = to_numpy(original)[:nrow]
    noisy_np = to_numpy(noisy)[:nrow]
    pred_np = to_numpy(predicted)[:nrow]
    
    # 组合图像：原始 | 加噪 | 预测
    combined = []
    for i in range(nrow):
        combined.extend([orig_np[i], noisy_np[i], pred_np[i]])
    
    # 创建图像网格
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(nrow, 3, figsize=(12, 4*nrow))
    
    for i in range(nrow):
        axes[i, 0].imshow(orig_np[i].squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy_np[i].squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Noisy (t={t[i].item()})")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_np[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}epoch_{epoch}_batch_{batch_idx}.png", dpi=120)
    plt.close()
    return combined

if __name__ == "__main__":
    # 路径设置和初始化
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.append(parent_dir)

    # 加载配置
    config = load_yaml("./config/standard_ddpm.yml")
    
    # 初始化扩散模型
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

    # 修改损失函数：添加可视化逻辑
    def get_loss(model, x_0, t, epoch, batch_idx, visualize=False):
        """
        计算损失并可选地可视化加噪过程
        """
        # 生成加噪样本
        x_0 = x_0[:, 1, :, :]
        x_0 = x_0.unsqueeze(1)
        x_noisy, noise = diffusion.q_sample(x_0, t, None)
        
        # 模型预测
        noise_pred = model(x_noisy, t)
        
        # 计算预测的原始图像
        alpha_bar = jt.array(diffusion.alphas_cumprod)[t].reshape(-1, 1, 1, 1)
        pred_x0 = (x_noisy - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
        
        # 定期可视化
        if visualize and batch_idx % 100 == 0:
            visualize_denoising(x_0, x_noisy, pred_x0, epoch, batch_idx)
        
        return nn.mse_loss(noise, noise_pred)

    # 初始化UNet模型
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

    T = diffusion.num_timesteps
    print(f"Diffusion timesteps: {T}")

    # 训练参数
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-5
    EPOCHS = 50
    VIS_INTERVAL = 100

    # 数据准备（添加归一化处理）
    trans = transform.Compose([
        # transform.ToTensor(),
        transform.Resize((32, 32)),
        # transform.ImageNormalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
    ])

    train_dataset_loader = dataset.MNIST(
            data_root="./script/data/MNIST/raw/",
            train=True, 
            transform=trans,
            download=False
        ).set_attrs(
            batch_size=128,           # 每批16张图像
            shuffle=True              # 随机打乱顺序
        )


    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    global_batch_count = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        epoch_batches = len(train_dataset_loader)

        for batch_idx, (images, _) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            t = jt.randint(0, T, (images.shape[0],)).long()
            # visualize_denoising(images, images, images, epoch, batch_idx, nrow=4)
            
            # 计算损失并可视化
            visualize_flag = (batch_idx % VIS_INTERVAL == 0)
            loss = get_loss(model, images, t, epoch+1, batch_idx, visualize_flag)
            
            optimizer.step(loss)
            loss_value = loss.item()
            total_loss += loss_value
            global_batch_count += 1

            # 日志记录（保持原有CSV日志功能）
            if batch_idx % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx:04d}/{len(train_dataset_loader)} | "
                      f"Loss: {loss_value:.4f} | Avg Loss: {avg_loss:.4f}")
        
        # 保存模型（保持原有模型保存功能）
        epoch_avg_loss = total_loss / epoch_batches
        jt.save(model.state_dict(), f"./result/ddpm/mnist/ddpm_mnist_epoch_{epoch + 1}.p")
        print(f"Model saved after epoch {epoch + 1}, Avg Loss: {epoch_avg_loss:.4f}")

    # 训练完成
    jt.save(model.state_dict(), "./result/ddpm/mnist/ddpm_mnist_final.pth")
    print("Training complete!")