import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.flow_matching_v1.dataset import *
from model.flow_matching_v1.transformer import FlowMatchingTransformer
from model.flow_matching_v1.flow_matching import flow_matching_loss
from model.flow_matching_v1.config import *

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
VAL_FRACTION = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/v1/model.pt"
LOG_PATH = "checkpoints/v1/loss_log.csv"

def main():
    print(f"Using device: {DEVICE}")

    train_episodes, val_episodes = load_and_split_episodes(DATA_DIR, val_fractions=VAL_FRACTION)
    print(f"Train episodes: {len(train_episodes)}  Val episodes: {len(val_episodes)}")

    train_ds = LandingDataset(train_episodes)
    val_ds = LandingDataset(val_episodes)
    print(f"Train examples: {len(train_ds)}  Val examples: {len(val_ds)}")
    # val_size = int(len(full_dataset) * VAL_FRACTION)
    # train_size = len(full_dataset) - val_size
    # train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- model ---
    model = FlowMatchingTransformer(
        pose_dim=POSE_DIM, action_dim=ACTION_DIM, history_len=HISTORY_LEN, chunk_len=CHUNK_LEN
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    with open(LOG_PATH, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss\n")

        for epoch in range(EPOCHS):
            # --- training ---
            model.train()
            train_loss_sum = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
            for history, chunk in pbar:
                history, chunk = history.to(DEVICE), chunk.to(DEVICE)

                loss = flow_matching_loss(model, chunk, history)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * history.shape[0]
                pbar.set_postfix(loss=f"{loss.item():.5f}")  # live per-batch loss

            train_loss = train_loss_sum / len(train_ds)

            # --- validation ---
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for history, chunk in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False):
                    history, chunk = history.to(DEVICE), chunk.to(DEVICE)
                    loss = flow_matching_loss(model, chunk, history)
                    val_loss_sum += loss.item() * history.shape[0]
            val_loss = val_loss_sum / len(val_ds)

            print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
            log_file.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  ↳ saved new best checkpoint (val_loss={val_loss:.5f})")
            else: 
                epochs_since_improvement += 1

            if epochs_since_improvement >= PATIENCE:
                print(f"Early stopping: no improvement in {PATIENCE} epochs. "
                      f"Best val_loss={best_val_loss:.5f}")
                break


if __name__ == "__main__":
    main()