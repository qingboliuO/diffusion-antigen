import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 32
image_size = 64
channels = 3
timesteps = 1000
beta_start = 0.0001
beta_end = 0.02


# Define the diffusion betas and alphas
def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(timesteps, beta_start, beta_end)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Move to device
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
posterior_variance = posterior_variance.to(device)


# Define the UNet model for diffusion
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4,
                                                2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=32):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block(64, 128, time_emb_dim),
            Block(128, 256, time_emb_dim),
            Block(256, 256, time_emb_dim),
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Block(256, 256, time_emb_dim, up=True),
            Block(256, 128, time_emb_dim, up=True),
            Block(128, 64, time_emb_dim, up=True),
        ])

        # Final output
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


# Forward diffusion process functions
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t,
                                              x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Reverse diffusion sampling functions
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t,
                                              x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use predicted noise to predict x_0
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step',
                  total=timesteps):
        img = p_sample(model, img,
                       torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model,
                         shape=(batch_size, channels, image_size, image_size))


# Data loading and preprocessing for dog images
def get_data_loader(batch_size, image_size):
    # 数据预处理转换
    transform = transforms.Compose([
        transforms.Resize(image_size),  # 调整图片大小
        transforms.CenterCrop(image_size),  # 中心裁剪为正方形
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化
    ])

    # 从自定义文件夹加载狗图片
    # 注意：图片需要按以下结构组织：
    # dog_images/
    #   └── dogs/  (这个子文件夹名称可以是任何名称)
    #       ├── dog1.jpg
    #       ├── dog2.jpg
    #       └── ...
    dataset = datasets.ImageFolder(root='dog_images', transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Training function
def train(model, dataloader, epochs, lr=1e-3):
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (images, _) in enumerate(tqdm(dataloader)):
            images = images.to(device)

            # Sample random timesteps
            t = torch.randint(0, timesteps, (images.shape[0],),
                              device=device).long()

            # Calculate loss
            loss = p_losses(model, images, t)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 每隔几个周期采样并保存图像
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            samples = sample(model, image_size=image_size, batch_size=4)
            samples = samples[-1]  # 获取最后一步（去噪程度最高的）

            # 转换样本为显示格式
            samples = (samples * 0.5 + 0.5).clip(0, 1)
            samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC

            # 显示或保存样本
            fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(samples[i])
                ax.axis('off')
            plt.savefig(f"outputs/dog_samples_epoch_{epoch + 1}.png")
            plt.close()


# Main function to run the training
def main():
    # 创建输出文件夹
    import os
    os.makedirs("outputs", exist_ok=True)

    # 初始化模型
    model = UNet(in_channels=channels, out_channels=channels).to(device)
    print("模型已初始化")

    # 获取数据加载器
    try:
        dataloader = get_data_loader(batch_size, image_size)
        print(f"数据加载器已准备，每批次包含 {batch_size} 张图片")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print(
            "请确保您已经创建了dog_images文件夹，并在其中包含至少一个子文件夹(如dogs)，子文件夹中放入图片文件")
        return

    # 训练模型
    print(f"开始训练，将运行 50 个训练周期...")
    train(model, dataloader, epochs=50)

    # 保存训练好的模型
    torch.save(model.state_dict(), "outputs/dog_diffusion_model.pt")
    print("训练完成，模型已保存到 outputs/dog_diffusion_model.pt")

    # 使用训练好的模型生成样本
    print("正在生成图片样本...")
    samples = sample(model, image_size, batch_size=16)
    samples = samples[-1]

    # 转换样本为显示格式
    samples = (samples * 0.5 + 0.5).clip(0, 1)
    samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC

    # 显示样本
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i])
        ax.axis('off')
    plt.savefig("outputs/final_dog_samples.png")
    print("样本已生成并保存到 outputs/final_dog_samples.png")


if __name__ == "__main__":
    main()