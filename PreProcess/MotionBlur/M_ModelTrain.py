import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import local modules
from GetMotionBlurDataset import MotionBlurDataset
from DeConvModel import TinyFreqNet, CompactFreqLoss


def custom_collate_fn(batch):
    """
    Custom collate function to handle PIL images and kernel data
    """
    original_imgs = []
    blurred_imgs = []
    kernels = []
    person_ids = []
    
    for item in batch:
        if len(item) == 4:  # With kernel estimation
            orig, blur, kernel, person_id = item
            original_imgs.append(orig)
            blurred_imgs.append(blur)
            kernels.append(kernel)
            person_ids.append(person_id)
        else:  # Without kernel estimation
            orig, blur, person_id = item
            original_imgs.append(orig)
            blurred_imgs.append(blur)
            kernels.append(None)
            person_ids.append(person_id)
    
    return original_imgs, blurred_imgs, kernels, person_ids


class MotionBlurTrainer:
    """
    Motion Blur Kernel Estimation Model Trainer
    Trains a CNN to predict frequency domain transformation matrix from blurred images
    """
    
    def __init__(self, 
                 model_config=None,
                 training_config=None,
                 device=None):
        """
        Initialize trainer
        
        Args:
            model_config: Model configuration dict
            training_config: Training configuration dict  
            device: Training device
        """
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Model configuration
        self.model_config = model_config or {
            'kernel_size': 25,
        }
        
        # Training configuration
        self.training_config = training_config or {
            'batch_size': 8,
            'learning_rate': 1e-4,
            'num_epochs': 50,
            'train_split': 0.8,
            'val_split': 0.2,
            'save_interval': 5,
            'log_interval': 10,
        }
        
        # Initialize model and loss
        self.model = TinyFreqNet(kernel_size=self.model_config['kernel_size'])
        self.model.to(self.device)
        
        self.criterion = CompactFreqLoss(
            magnitude_weight=1.0,
            phase_weight=0.1,
            scale_factor=1000.0  # 增加损失数值以便于观察
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.training_config['learning_rate']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create output directories
        self.output_dir = 'training_output'
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def prepare_data(self, original_root, blurred_root, max_images_per_folder=3):
        """
        Prepare training and validation datasets
        
        Args:
            original_root: Path to original images
            blurred_root: Path to blurred images  
            max_images_per_folder: Max images per person
        """
        print("Preparing dataset...")
        
        # Create dataset with kernel estimation enabled
        dataset = MotionBlurDataset(
            original_root=original_root,
            blurred_root=blurred_root,
            max_images_per_folder=max_images_per_folder,
            estimate_kernels=True
        )
        
        print(f"Total dataset size: {len(dataset)}")
        
        # Split dataset
        train_size = int(self.training_config['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=0,  # Reduced to avoid multiprocessing issues with PIL images
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=0,  # Reduced to avoid multiprocessing issues with PIL images
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=custom_collate_fn
        )
        
        print("Dataset preparation completed!")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tensor: Preprocessed tensor [1, H, W]
        """
        if hasattr(image, 'convert'):  # PIL Image
            image = np.array(image.convert('L'))  # Convert to grayscale
        elif len(image.shape) == 3:  # RGB numpy array
            image = np.mean(image, axis=2)  # Convert to grayscale
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        
        return tensor
    
    def prepare_target_kernel(self, kernel):
        """
        Prepare target kernel for training
        
        Args:
            kernel: Estimated kernel (numpy array)
            
        Returns:
            kernel_fft: Frequency domain kernel tensor
        """
        # Ensure kernel is correct size
        target_size = self.model_config['kernel_size']
        if kernel.shape != (target_size, target_size):
            # Resize kernel to target size
            from scipy.ndimage import zoom
            scale_h = target_size / kernel.shape[0]
            scale_w = target_size / kernel.shape[1]
            kernel = zoom(kernel, (scale_h, scale_w))
        
        # Normalize kernel to ensure sum = 1
        kernel = kernel / (np.sum(kernel) + 1e-8)
        
        # Add small regularization to avoid zeros
        kernel = kernel + 1e-8
        kernel = kernel / np.sum(kernel)
        
        # Convert to tensor and compute FFT
        kernel_tensor = torch.from_numpy(kernel.astype(np.float32))
        
        # Pad to larger size for FFT computation to avoid artifacts
        pad_size = max(64, target_size * 2)
        padded_kernel = torch.zeros(pad_size, pad_size)
        start_idx = (pad_size - target_size) // 2
        padded_kernel[start_idx:start_idx+target_size, start_idx:start_idx+target_size] = kernel_tensor
        
        # Compute FFT and extract center region
        kernel_fft_full = torch.fft.fft2(padded_kernel)
        kernel_fft_full = torch.fft.fftshift(kernel_fft_full)
        
        # Extract center region matching target size
        center = pad_size // 2
        half_size = target_size // 2
        kernel_fft = kernel_fft_full[
            center-half_size:center+half_size+1, 
            center-half_size:center+half_size+1
        ]
        
        # Ensure proper size
        if kernel_fft.shape[0] != target_size:
            kernel_fft = torch.fft.fft2(kernel_tensor)
        
        return kernel_fft
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch data
            if len(batch_data) == 4:  # With kernel estimation
                original_imgs, blurred_imgs, kernels, person_ids = batch_data
            else:  # Without kernel estimation
                print("Error: Dataset must have kernel estimation enabled")
                continue
            
            batch_size = len(blurred_imgs)
            batch_loss = 0.0
            valid_samples = 0
            
            for i in range(batch_size):
                try:
                    # Preprocess input image
                    blurred_tensor = self.preprocess_image(blurred_imgs[i])
                    blurred_tensor = blurred_tensor.to(self.device)
                    
                    # Prepare target kernel
                    if kernels[i] is not None:
                        target_kernel_fft = self.prepare_target_kernel(kernels[i])
                        target_kernel_fft = target_kernel_fft.to(self.device)
                    else:
                        continue  # Skip if no kernel available
                    
                    # Forward pass
                    pred_kernel_fft = self.model(blurred_tensor.unsqueeze(0))
                    pred_kernel_fft = pred_kernel_fft.squeeze(0)  # Remove batch dim
                    
                    # Debug: Print data ranges for first batch
                    if batch_idx == 0 and i == 0 and epoch == 0:
                        print(f"Debug - Blurred image range: [{blurred_tensor.min():.6f}, {blurred_tensor.max():.6f}]")
                        print(f"Debug - Target kernel FFT magnitude range: [{target_kernel_fft.abs().min():.6f}, {target_kernel_fft.abs().max():.6f}]")
                        print(f"Debug - Predicted kernel FFT magnitude range: [{pred_kernel_fft.abs().min():.6f}, {pred_kernel_fft.abs().max():.6f}]")
                    
                    # Compute loss
                    loss = self.criterion(pred_kernel_fft, target_kernel_fft)
                    
                    # Debug: Print loss components for first batch
                    if batch_idx == 0 and i == 0 and epoch == 0:
                        print(f"Debug - Loss value: {loss.item():.6f}")
                    
                    batch_loss += loss
                    valid_samples += 1
                    
                except Exception as e:
                    print(f"Error processing sample {i} in batch {batch_idx}: {e}")
                    continue
            
            if valid_samples > 0:
                # Average loss over valid samples in batch
                batch_loss = batch_loss / valid_samples
                
                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'Loss': batch_loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate_epoch(self):
        """
        Validate for one epoch
        
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # Unpack batch data
                if len(batch_data) == 4:  # With kernel estimation
                    original_imgs, blurred_imgs, kernels, person_ids = batch_data
                else:
                    continue
                
                batch_size = len(blurred_imgs)
                batch_loss = 0.0
                valid_samples = 0
                
                for i in range(batch_size):
                    try:
                        # Preprocess input image
                        blurred_tensor = self.preprocess_image(blurred_imgs[i])
                        blurred_tensor = blurred_tensor.to(self.device)
                        
                        # Prepare target kernel
                        if kernels[i] is not None:
                            target_kernel_fft = self.prepare_target_kernel(kernels[i])
                            target_kernel_fft = target_kernel_fft.to(self.device)
                        else:
                            continue
                        
                        # Forward pass
                        pred_kernel_fft = self.model(blurred_tensor.unsqueeze(0))
                        pred_kernel_fft = pred_kernel_fft.squeeze(0)
                        
                        # Compute loss
                        loss = self.criterion(pred_kernel_fft, target_kernel_fft)
                        batch_loss += loss
                        valid_samples += 1
                        
                    except Exception as e:
                        continue
                
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    total_loss += batch_loss.item()
                    num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_model(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
    
    def plot_training_history(self):
        """
        Plot and save training history
        """
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.log_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history plot saved to: {plot_path}")
    
    def save_training_log(self):
        """
        Save training log as JSON
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'device': str(self.device)
        }
        
        log_path = os.path.join(self.log_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training log saved to: {log_path}")
    
    def train(self, original_root, blurred_root, max_images_per_folder=3):
        """
        Main training loop
        
        Args:
            original_root: Path to original images
            blurred_root: Path to blurred images
            max_images_per_folder: Max images per person
        """
        print("Starting motion blur kernel estimation model training...")
        print(f"Model configuration: {self.model_config}")
        print(f"Training configuration: {self.training_config}")
        
        # Prepare data
        self.prepare_data(original_root, blurred_root, max_images_per_folder)
        
        # Training loop
        for epoch in range(self.training_config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.training_config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            print(f"Training Loss: {train_loss:.6f}")
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save model checkpoint
            if (epoch + 1) % self.training_config['save_interval'] == 0 or is_best:
                self.save_model(epoch, is_best)
            
            # Plot training history periodically
            if (epoch + 1) % self.training_config['log_interval'] == 0:
                self.plot_training_history()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Final saves
        self.plot_training_history()
        self.save_training_log()
        self.save_model(self.training_config['num_epochs'] - 1, False)


def main():
    """
    Main training function
    """
    # Configuration
    model_config = {
        'kernel_size': 25,
    }
    
    training_config = {
        'batch_size': 4,  # Reduced for memory efficiency
        'learning_rate': 1e-3,
        'num_epochs': 15,
        'train_split': 0.8,
        'val_split': 0.2,
        'save_interval': 5,
        'log_interval': 5,
    }
    
    # Dataset paths
    original_root = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA"
    blurred_root = "/home/wenhao/CUHK/ECE4513/FinalProject/Data/CASIA_MotionBlurred"
    
    # Create trainer
    trainer = MotionBlurTrainer(
        model_config=model_config,
        training_config=training_config
    )
    
    # Start training
    trainer.train(
        original_root=original_root,
        blurred_root=blurred_root,
        max_images_per_folder=2  # Reduced for faster training
    )


if __name__ == "__main__":
    main()