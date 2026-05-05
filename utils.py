import yaml
import torch
import random
import numpy as np
import os
import logging

def load_config(config_path="config.yaml"):
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(log_dir):
    """Initialize logging."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger("DiT_Project")

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, name="model.pt"):
    """Save model and optimizer state to a checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, name)
    
    # Extract inner model if wrapped in DataParallel/DistributedDataParallel
    model_to_save = model.module if hasattr(model, "module") else model
    # Extract inner model if wrapped by torch.compile
    if hasattr(model_to_save, "_orig_mod"):
        model_to_save = model_to_save._orig_mod
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    logging.info(f"Saved checkpoint to {path}")

def load_checkpoint(model, optimizer, checkpoint_path, map_location="cpu"):
    """Load model and optimizer states from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model
    model_to_load = model.module if hasattr(model, "module") else model
    
    # Remove _orig_mod. prefix if checkpoint was saved while compiled
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model_to_load.load_state_dict(state_dict)
    
    # Load optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    return checkpoint.get("epoch", 0)
