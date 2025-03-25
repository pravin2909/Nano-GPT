import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datset import get_dataloader
from config import config
from model import GPT
from transformers import get_scheduler
from tqdm import tqdm
import os
import glob

def generate_text(model, prompt, max_new_tokens=50, top_k=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(prompt).unsqueeze(0).to(config.device)
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)
        generated_text = config.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    model.train()
    return generated_text

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, f"{path}/checkpoint_epoch_{epoch}.pth")
    print(f"Checkpoint saved: {path}/checkpoint_epoch_{epoch}.pth")

def load_checkpoint(model, optimizer, scheduler, path="checkpoints"):
    checkpoint_files = sorted(glob.glob(f"{path}/checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return model, optimizer, scheduler, 0  

    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = checkpoint["epoch"] + 1  

    print(f"Resuming from {latest_checkpoint}, starting at epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch

def train():
    device = config.device
    model = GPT(config.vocab_size, config.seq_len, config.d_model, config.n_layers, config.n_heads, config.d_ff, config.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.epochs * config.gradient_accumulation_steps
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = get_dataloader(batch_size=config.batch_size, seq_len=config.seq_len)

    writer = SummaryWriter(log_dir="runs/nanogpt")
    prompt = config.tokenizer.encode("Once upon a time", add_special_tokens=False)

    model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler)

    model.train()
    for epoch in range(start_epoch, config.epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=True)
        for step, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 10 == 0:
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + step)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        generated_text = generate_text(model, prompt, max_new_tokens=config.max_new_tokens, top_k=config.top_k, temperature=config.temperature)
        writer.add_text("Generated Text", generated_text, epoch)
        print(f"Generated text after epoch {epoch+1}:\n{generated_text}\n")

        save_checkpoint(model, optimizer, scheduler, epoch, avg_loss)

    torch.save(model.state_dict(), "gpt_model.pth")
    print("Training complete. Final model saved.")
    writer.close()  

if __name__ == "__main__":
    train()
