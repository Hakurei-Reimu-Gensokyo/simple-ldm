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

from model.unet import *
from model.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from model.autoencoder import VAE

from jittor import dataset, transform


import yaml


def load_yaml(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_yaml("./config/standard_ddpm.yml")

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

#确定模型，导入已训练模型（如有）
vae_name = './result/ldm/autoencoder/the_best_ever/best_vae_mnist.p'

vae = VAE(in_channels=1, latent_dim=4, hidden_dims=[32, 64])
vae.load_state_dict(jt.load(vae_name))

def get_loss(model, x_0, y, t,  device=None):
    # x_0 = x_0.to(device)
    # print(x_0.device)
    x_noisy, noise = diffusion.q_sample(x_0, t, None)
    noise_pred = model(x_noisy, t)
    # return F.l1_loss(noise, noise_pred)
    return jt.nn.mse_loss(noise, noise_pred)


if __name__ == "__main__":
    # 设置设备


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
    )

    # 获取扩散步数
    T = diffusion.num_timesteps
    print(f"Diffusion timesteps: {T}")

    # 训练参数
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 50

    # 准备MNIST数据集
    trans = transform.Compose([
        transform.Resize((32, 32))  # 调整为统一尺寸
    ])

    dataloader = dataset.MNIST(
        data_root='./script/data/MNIST/raw/',
        train=True,
        download=False,
        transform=trans
    ).set_attrs(batch_size=128, shuffle=True)

    CSV_PATH = "./result/ldm/mnist/loss/loss_log.csv"

    # 创建CSV文件并写入表头
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'batch_loss', 'epoch_avg_loss'])
        

    # 创建优化器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    prev_noise = None

    # 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        epoch_batches = len(dataloader)
        for batch_idx, (images, y) in enumerate(dataloader):
            # bs = images.shape[0]
            optimizer.zero_grad()
            # print(images.shape)
            # 生成随机时间步
            t = jt.randint(0, T, (images.size(0),)).long()
            # 计算损失

            # mu, log_var = vae.encode(images)
            # images = vae.reparameterize(mu, log_var)
            images = images[:, 0, :, :]
            images = images.unsqueeze(1)

            images = vae.encode(images)
            # images = images.squeeze(0)
            # print(images.shape)

            x = vae.decoder.fc(images)
            x = x.view(images.size(0), -1, 1, 1)
            # print(x.shape)
            loss = get_loss(model, x, y, t)

            # 反向传播和优化
            optimizer.step(loss)

            total_loss += loss.item()

            # 定期打印训练信息
            if batch_idx % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_idx:04d}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
        epoch_avg_loss = total_loss / epoch_batches
        # 每个epoch结束时保存模型
        jt.save(model.state_dict(), f"./result/ldm/unet/ldm_mnist_epoch_{epoch + 1}.p")
        print(f"Model saved after epoch {epoch + 1}")

        # ===== 记录epoch总结 =====
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                'EPOCH_END',
                '',
                epoch_avg_loss
            ])
        # ========================
    

    # 保存最终模型
    jt.save(model.state_dict(), "./result/ldm/unet/ldm_mnist_final.p")
    print("Training complete!")




