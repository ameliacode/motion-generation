import numpy as np
import torch
import torch.optim as optim
from config import *
from model_torch import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

encoder = Encoder()
decoder = Decoder()
autoencoder = Autoencoder(encoder, decoder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = np.load("./data/01_data.npz")
X = data["clips"].astype(np.float32)
Xmean = data["mean"].astype(np.float32)
Xstd = data["std"].astype(np.float32)
train_data, val_data = train_test_split(X, test_size=0.1, random_state=42)

train_data = (train_data - Xmean) / Xstd
val_data = (val_data - Xmean) / Xstd

train_tensor = torch.FloatTensor(train_data)
val_tensor = torch.FloatTensor(val_data)

train_dataset = TensorDataset(train_tensor, train_tensor)
val_dataset = TensorDataset(val_tensor, val_tensor)

train_loader = DataLoader(train_dataset)
val_loader = DataLoader(val_dataset)

writer = SummaryWriter("runs/01-autoencoder")

autoencoder = autoencoder.to(device)


def loss_fn(x_real, x_pred):
    l2_loss = torch.mean(torch.square(x_real - x_pred), dim=[1, 2])
    l1_loss = ALPHA * torch.mean(torch.abs(x_pred))
    return l2_loss + l1_loss


optimizer = optim.Adam(autoencoder.parameters())


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = torch.mean(loss_fn(target, output))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = torch.mean(loss_fn(target, output))
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


for epoch in range(EPOCHS):
    train_loss = train_epoch(autoencoder, train_loader, optimizer, loss_fn, device)
    val_loss = validate_epoch(autoencoder, val_loader, loss_fn, device)

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)

    current_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("Learning_Rate", current_lr, epoch)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
    )

writer.close()

torch.save(autoencoder.state_dict(), "01_weights.pth")
