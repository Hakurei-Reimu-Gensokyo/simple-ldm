import jittor as jt
import jittor.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], latent_dim=4):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        current_dim = in_channels

        # 构建下采样路径
        for h_dim in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    ResnetBlock(in_channels=current_dim, out_channels=h_dim,
                                dropout=0.0, temb_channels=0),
                    Downsample(h_dim, with_conv=True)
                )
            )
            current_dim = h_dim

        # 最终残差块（无下采样）
        self.final_block = ResnetBlock(in_channels=current_dim, out_channels=current_dim,
                                       dropout=0.0, temb_channels=0)

        # 输出潜在变量的均值与方差
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)

    def execute(self, x):
        # 通过下采样层
        for layer in self.layers:
            x = layer(x)[:, :, :, :]  # 兼容ResnetBlock的temb参数（设为None）

        # 最终残差块
        x = self.final_block(x, temb=None)

        # 全局平均池化并展平
        x = jt.mean(x, dims=[2,3])
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dims=[512, 256, 128, 64, 32], latent_dim=4):
        super(Decoder, self).__init__()
        hidden_dims.reverse()  # 反转维度（从高维到低维）
        current_dim = hidden_dims[0]

        # 初始全连接层（将潜在变量映射到特征图）
        self.fc = nn.Linear(latent_dim, hidden_dims[0])

        # 构建上采样路径
        self.layers = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims):
            self.layers.append(
                nn.Sequential(
                    ResnetBlock(in_channels=current_dim, out_channels=h_dim,
                                dropout=0.0, temb_channels=0),
                    Upsample(h_dim, with_conv=True) # if i < len(hidden_dims) - 1
                    #else nn.Identity()  # 最后一层不上采样
                )
            )
            current_dim = h_dim

        # 输出层（重建图像）
        self.final_conv = nn.Conv2d(current_dim, out_channels, kernel_size=3, padding=1)

    def execute(self, z):
        # 全连接层并重塑为特征图
        x = self.fc(z)
        x = x.view(z.size(0), -1, 1, 1)  # 重塑为 [B, C, 1, 1]
        x = nn.interpolate(x, scale_factor=8, mode='nearest')  # 初始上采样

        # 通过上采样层
        for layer in self.layers:
            x = layer(x)[:, :, :, :]  # 兼容ResnetBlock的temb参数

        # 最终卷积输出（Sigmoid归一化到[0,1]）
        x = self.final_conv(x)
        return jt.sigmoid(x)

    def decode_by_x(self, x):
        x = nn.interpolate(x, scale_factor=8, mode='nearest')  # 初始上采样

        # 通过上采样层
        for layer in self.layers:
            x = layer(x)[:, :, :, :]  # 兼容ResnetBlock的temb参数

        # 最终卷积输出（Sigmoid归一化到[0,1]）
        x = self.final_conv(x)
        return jt.sigmoid(x)



class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, hidden_dims=None):
        super(VAE, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(in_channels, hidden_dims.copy(), latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return mu + eps * std

    def execute(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return self.decoder(z)

    def loss_function(self, x_recon, x, mu, logvar):
        # 重建损失（MSE或交叉熵）
        recon_loss = nn.mse_loss(x_recon, x, reduction='sum')
        # KL散度（正则化潜在空间）
        kl_loss = -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss


def nonlinearity(x):
    # swish
    return x*jt.sigmoid(x)


def Normalize(in_channels, num_groups=1):
    # print(f"num_groups:{num_groups}, in_channels:{in_channels}")
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def execute(self, x):
        
        x = nn.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def execute(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = nn.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def execute(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h



# 损失函数
def vae_loss(recon_x, x, mu, logvar, kl_weight=1e-4):
    """
    VAE损失 = 重建损失 + KL散度
    KL_weight: KL项的权重 (LDM中通常设为0.00001)
    """
    # 像素级均方误差
    recon_loss = jt.nn.mse_loss(recon_x, x, reduction='sum') / x.shape[0]

    # KL散度 (相对标准正态分布)
    kl_loss = -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    return recon_loss + kl_weight * kl_loss