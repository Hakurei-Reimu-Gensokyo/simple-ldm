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
import jittor.nn as nn
import jittor.optim as optim
from jittor import dataset, transform
import matplotlib.pyplot as plt
from model.autoencoder import VAE

import csv


# 数据预处理
transform = transform.Compose([
    transform.Resize((32,32))
])

# 加载MNIST数据集[4,7](@ref)
train_dataset = dataset.MNIST(data_root='./script/data/MNIST/raw/', train=True, download=True, transform=transform)
test_dataset = dataset.MNIST(data_root='./script/data/MNIST/raw/', train=False, download=True, transform=transform)

train_loader = train_dataset.set_attrs(batch_size=128, shuffle=True)
test_loader = test_dataset.set_attrs(batch_size=128, shuffle=True)

# 检查GPU可用性
# print(f"使用设备: {device}")

# 初始化模型[4](@ref)
model = VAE(in_channels=1, latent_dim=4, hidden_dims=[32, 64])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练参数
epochs = 15
best_loss = float('inf')


def train(epoch):
    model.train()
    train_loss = 0
    epoch_batches = len(train_loader)
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()

        # 前向传播
        data = data[:, 0, :, :]
        data = data.unsqueeze(1)
        recon_batch, mu, logvar = model(data)
        # print(recon_batch.shape)

        loss = model.loss_function(recon_batch, data, mu, logvar)

        # 反向传播
        train_loss += loss.item()
        optimizer.step(loss)

        # 每100批次打印进度
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item() / len(data):.4f}")

    # 计算平均损失
    avg_loss = train_loss / epoch_batches
    print(f"====> Epoch {epoch} Average Loss: {avg_loss:.4f}")
    return avg_loss


# 测试函数[3](@ref)
def test():
    model.eval()
    test_loss = 0
    with jt.no_grad():
        for data, _ in test_loader:
            data = data[:, 0, :, :]
            data = data.unsqueeze(1)
            recon, mu, logvar = model(data)
            test_loss += model.loss_function(recon, data, mu, logvar).item()

    avg_loss = test_loss / len(test_loader.dataset)
    print(f"====> 测试集损失: {avg_loss:.4f}")
    return avg_loss





# 在训练循环前初始化CSV文件
with open('./result/ldm/autoencoder/the_best_ever/training_log.csv', 'w', newline='') as f:  # 创建/覆盖CSV文件
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'test_loss'])  # 写入列标题[2](@ref)

best_loss = float('inf')
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)  # 训练并返回训练损失
    test_loss = test()          # 测试并返回测试损失

    # 实时记录到CSV（追加模式）
    with open('./result/ldm/autoencoder/the_best_ever/training_log.csv', 'a', newline='') as f:  # 追加模式写入[2](@ref)
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, test_loss])  # 写入当前epoch数据

    # 保存最佳模型
    if test_loss < best_loss:
        best_loss = test_loss
        jt.save(model.state_dict(), './result/ldm/autoencoder/the_best_ever/best_vae_mnist.p')
        print(f"Epoch {epoch}: Save the model! Test Loss: {test_loss:.4f}")


def generate_and_visualize():
    model.eval()
    with jt.no_grad():
        # 从潜在空间采样
        z = jt.randn(64, model.latent_dim)
        samples = model.decode(z).cpu()

        # 可视化生成结果[2](@ref)
        fig, ax = plt.subplots(8, 8, figsize=(10, 10))
        for i in range(8):
            for j in range(8):
                ax[i][j].imshow(samples[i * 8 + j][0], cmap='gray')
                ax[i][j].axis('off')
        plt.savefig('vae_generated.png')
        plt.show()

        # 可视化重建结果
        data, _ = next(iter(test_loader))
        data = data[:8]
        recon, _, _ = model(data)

        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        for i in range(8):
            axes[0][i].imshow(data[i][0].cpu(), cmap='gray')
            axes[0][i].set_title("RAW")
            axes[0][i].axis('off')

            axes[1][i].imshow(recon[i][0].cpu(), cmap='gray')
            axes[1][i].set_title("REC")
            axes[1][i].axis('off')
        plt.savefig('vae_reconstructed.png')
        plt.show()


# 训练完成后生成样本
generate_and_visualize()

