import argparse
import os
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from utils import load_config, set_seed, setup_logging, save_checkpoint
from dataset import get_dataloader
from model import DiT
from flow import RectifiedFlow

def train(config_path):
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    logger = setup_logging(config["training"]["log_dir"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Global environment optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    # Initialize W&B
    wandb.init(project="DiT_ImageNet100", config=config)
    
    # Dataloaders
    logger.info("Initializing dataloaders...")
    # Make sure we use the pre-encoded latents cache
    train_latents_dir = os.path.join(config["dataset"]["cache_dir"], "train")
    dataloader, class_mapping = get_dataloader(
        train_latents_dir, 
        config["training"]["batch_size"], 
        shuffle=True
    )
    
    val_latents_dir = os.path.join(config["dataset"]["cache_dir"], "val")
    val_dataloader, _ = get_dataloader(
        val_latents_dir,
        config["training"]["batch_size"],
        shuffle=False
    )
    
    # Add + 1 to num_classes for the unconditional token (CFG)
    num_classes = config["dataset"]["num_classes"]
    
    logger.info("Initializing Model...")
    model = DiT(
        input_size=config["vae"]["latent_size"],
        patch_size=config["model"]["patch_size"],
        in_channels=config["vae"]["latent_channels"],
        hidden_size=config["model"]["hidden_dim"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        num_classes=num_classes + 1  # Extra class for unconditional
    ).to(device)
    
    logger.info("Compiling model for maximum throughput...")
    model = torch.compile(model)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )
    
    flow = RectifiedFlow()
    
    use_amp = config["training"]["mixed_precision"] in ["fp16", "bf16"]
    amp_dtype = torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["mixed_precision"] == "fp16")
    
    epochs = config["training"]["epochs"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for latents, labels in pbar:
            latents = latents.to(device)
            labels = labels.to(device)
            
            # Classifier-Free Guidance (CFG) dropout
            # Drop 10% of labels to unconditional class (id: num_classes)
            drop_mask = torch.rand(labels.shape[0], device=device) < 0.1
            labels = torch.where(drop_mask, torch.full_like(labels, num_classes), labels)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = flow.compute_loss(model, latents, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({"train_loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        
        # Validation Pass
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_latents, val_labels in val_dataloader:
                val_latents = val_latents.to(device)
                val_labels = val_labels.to(device)
                
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    val_loss = flow.compute_loss(model, val_latents, val_labels)
                    
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        logger.info(f"Epoch {epoch+1} completed. Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"train_loss_epoch": avg_loss, "val_loss_epoch": avg_val_loss, "epoch": epoch+1})
        
        # Checkpointing
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_checkpoint(model, optimizer, epoch+1, checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)
