import os
import time
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from english_to_marathi_translator.models.transformer import Transformer
from english_to_marathi_translator.utils.path_utils import PROCESSED_DATA_DIR, PROJECT_ROOT
from english_to_marathi_translator.utils.dataset import TranslationDataset, collate_fn

def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, filepath):
    """Save model checkpoint."""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "loss": loss,
    }
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(filepath, model, optimizer):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    batch_idx = checkpoint["batch_idx"]
    loss = checkpoint["loss"]
    print(f"Loaded checkpoint from {filepath} at epoch {epoch}, batch {batch_idx}, loss {loss:.4f}")
    return epoch, batch_idx, loss

def plot_loss(epoch_losses, save_path):
    """Plot training loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label="Training Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Loss plot saved at {save_path}")

def train(resume=False):
    config = load_config(PROJECT_ROOT / "configs/default_config.yaml")
    train_loader = torch.load(PROCESSED_DATA_DIR / "train_loader.pth")

    # Initialize model, optimizer, and criterion
    device = torch.device(config["training"]["device"])
    model = Transformer(**config["model"], device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Logging config details
    print(f"Training Configuration: {config['training']}")
    print(f"Model Configuration: {config['model']}")

    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float("inf")
    checkpoint_dir = PROCESSED_DATA_DIR / config["training"]["checkpoint_dir"]
    checkpoint_dir.mkdir(exist_ok=True)

    if resume:
        checkpoint_path = checkpoint_dir / "best_policy.pth"
        if checkpoint_path.exists():
            start_epoch, _, best_loss = load_checkpoint(checkpoint_path, model, optimizer)
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")

    # Resume training
    model.train()
    save_frequency = config["training"]["save_frequency"]
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        for batch_idx, (src, tgt) in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            # Timing and logging
            progress_bar.set_postfix(loss=loss.item())

            # Save checkpoint every `save_frequency` steps
            if (batch_idx + 1) % save_frequency == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth"
                save_checkpoint(model, optimizer, epoch, batch_idx, loss.item(), checkpoint_path)

        # Save the best policy
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_policy_path = checkpoint_dir / "best_policy.pth"
            save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, best_policy_path)

if __name__ == "__main__":
    # Set `resume=True` to continue training from the last best policy checkpoint
    train(resume=False)
