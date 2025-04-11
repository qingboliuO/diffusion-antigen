import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
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


# Move all tensors to device after creation
betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# Define the UNet model for diffusion
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=time.device) * -embeddings)
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
    # Move t to CPU for gather operation, then move the result back to t's device
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


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
def get_data_loader(batch_size, image_size, max_images=200):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Using CIFAR10 and filter out dog class (class 5 in CIFAR10)
    cifar = datasets.CIFAR10(root='./data', train=True, download=True,
                             transform=transform)

    # Filter for dog class (class 5 in CIFAR10)
    indices = [i for i, (_, label) in enumerate(cifar) if label == 5]

    # Limit to only max_images
    indices = indices[:max_images]

    dog_dataset = Subset(cifar, indices)

    print(f"Loading only {len(dog_dataset)} dog images")

    dataloader = DataLoader(dog_dataset, batch_size=batch_size, shuffle=True)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10 dataset
    cifar = datasets.CIFAR10(root='./data', train=True, download=True,
                             transform=transform)

    # Filter for dog class (class 5 in CIFAR-10)
    dog_indices = [i for i, (_, label) in enumerate(cifar) if label == 5]

    # Select the first 16 dog images to display
    dog_indices = dog_indices[:16]

    # Create a figure to display the images
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(dog_indices):
        image, _ = cifar[idx]
        # Convert from tensor [3, 32, 32] to numpy array [32, 32, 3]
        img = image.permute(1, 2, 0).numpy()

        # Plot in a 4x4 grid
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("original_cifar10_dogs.png")
    plt.show()
    print(f"Saved 16 original dog images to 'original_cifar10_dogs.png'")
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

        # Sample and save images every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            samples = sample(model, image_size=image_size, batch_size=4)
            samples = samples[-1]  # Get the last step (most denoised)

            # Convert samples to display format
            samples = (samples * 0.5 + 0.5).clip(0, 1)
            samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC

            # Display or save samples
            fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(samples[i])
                ax.axis('off')
            plt.savefig(f"dog_samples_epoch_{epoch + 1}.png")
            plt.close()


# Main function to run the training
def main():
    # Initialize model
    model = UNet(in_channels=channels, out_channels=channels).to(device)
    print("Model initialized")

    # Get data loader with only 100 images
    dataloader = get_data_loader(batch_size, image_size, max_images=200)
    print("Data loader ready")

    # Train the model
    print("Starting training...")
    train(model, dataloader, epochs=300)

    # Save the trained model
    torch.save(model.state_dict(), "dog_diffusion_model.pt")
    print("Training complete and model saved")

    # Generate samples with the trained model
    print("Generating samples...")
    samples = sample(model, image_size, batch_size=16)
    samples = samples[-1]

    # Convert samples to display format
    samples = (samples * 0.5 + 0.5).clip(0, 1)
    samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC

    # Display samples
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i])
        ax.axis('off')
    plt.savefig("final_dog_samples.png")
    print("Samples generated and saved")


if __name__ == "__main__":
    main()