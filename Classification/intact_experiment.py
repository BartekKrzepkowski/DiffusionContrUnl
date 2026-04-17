"""
InTAct (Interval-based Task Activation Consolidation) Unlearning Experiment for Classification Models

This script demonstrates InTAct unlearning on CIFAR-10 with ResNet18.
It pretrains a model, then applies InTAct to forget specified classes
while preserving performance on remaining classes.

Usage:
    python intact_experiment.py --unlearn_classes 0 --lambda_interval 100.0
    python intact_experiment.py --unlearn_classes 0 1 2 --unlearn_epochs 20
"""

import logging
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Add parent directory to path for InTAct import
sys.path.insert(0, str(Path(__file__).parent.parent))
from InTAct.intact import UnlearnIntervalProtection, classification_forward_fn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def get_cifar10_dataloaders(batch_size=128, data_dir='./data'):
    """Load CIFAR-10 with standard normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def create_class_subsets(dataset, classes):
    """Create a subset containing only specified classes."""
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in classes:
            indices.append(i)
    return Subset(dataset, indices)


# ============================================================================
# InTAct Training Functions
# ============================================================================

def intact_train_epoch(model, optimizer, criterion, forget_loader, device, protection=None):
    """Single epoch of InTAct unlearning for classification."""
    model.train()
    total_loss = total_unlearn = total_protect = 0.0
    n = 0

    for batch in forget_loader:
        X, y = batch[:2]
        X, y = X.to(device), y.to(device)

        out = model(X)
        # Unlearning via Gradient Ascent (Negative Loss)
        unlearn_loss = -criterion(out, y) 

        protect_loss = (
            protection.compute_protection_loss(model, device)
            if protection else torch.tensor(0.0, device=device)
        )

        loss = unlearn_loss + protect_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_unlearn += unlearn_loss.item()
        total_protect += protect_loss.item()
        n += 1

    return total_loss / n, total_unlearn / n, total_protect / n


def intact_unlearn(
    model, forget_loader, criterion, optimizer, num_epochs, device,
    targets=None,
    lambda_interval=10.0, 
    lower_percentile=0.05, 
    upper_percentile=0.95, 
    reduced_dim=32,
    infinity_scale=20.0
):
    """
    Run InTAct unlearning for classification models.
    
    Args:
        model: PyTorch model to unlearn
        forget_loader: DataLoader for forget data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of unlearning epochs
        device: Device to use
        targets: List of layer name patterns to protect (default: ["fc"])
        lambda_interval: Weight for protection loss
        lower_percentile: Lower bound percentile for safe zone
        upper_percentile: Upper bound percentile for safe zone
        reduced_dim: PCA dimension for efficiency
        infinity_scale: Scale for negative space intervals
    
    Returns:
        Unlearned model
    """
    if targets is None:
        targets = ["fc"]  # Default: protect final classifier layer
    
    protection = UnlearnIntervalProtection(
        targets=targets,
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        reduced_dim=reduced_dim,
        infinity_scale=infinity_scale
    )

    protection.setup_protection(model, forget_loader, device, forward_fn=classification_forward_fn)
    protection.freeze_non_target_params(model)

    for epoch in range(num_epochs):
        loss, unlearn, protect = intact_train_epoch(
            model, optimizer, criterion, forget_loader, device, protection
        )
        log.info(f"Epoch {epoch+1}/{num_epochs} | loss={loss:.4f} unlearn={unlearn:.4f} protect={protect:.4f}")

    return model


# ============================================================================
# Model Training & Evaluation
# ============================================================================

def pretrain_model(model, train_loader, criterion, device, epochs=10, lr=0.001):
    """Pretrain the model on all classes."""
    log.info(f"Pretraining model for {epochs} epochs...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        log.info(f"Pretrain Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")
    
    log.info("Pretraining complete!")


def test_model(model, test_loader, device, all_classes):
    """Test model and return per-class accuracy."""
    model.eval()
    
    class_correct = {cls: 0 for cls in all_classes}
    class_total = {cls: 0 for cls in all_classes}
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)
            
            for i in range(y.size(0)):
                label = y[i].item()
                if label in all_classes:
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
    
    per_class_acc = {}
    for cls in all_classes:
        if class_total[cls] > 0:
            per_class_acc[cls] = 100.0 * class_correct[cls] / class_total[cls]
        else:
            per_class_acc[cls] = 0.0
    
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    
    return per_class_acc, overall_acc


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='InTAct Unlearning for Classification')
    
    # Data & training
    parser.add_argument('--unlearn_classes', nargs='+', type=int, default=[0],
                        help='Classes to unlearn (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--unlearn_epochs', type=int, default=10)
    parser.add_argument('--pretrain_lr', type=float, default=0.001)
    parser.add_argument('--unlearn_lr', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # InTAct parameters
    parser.add_argument('--targets', nargs='+', type=str, default=['fc'],
                        help='Target layer patterns to protect (default: fc)')
    parser.add_argument('--lambda_interval', type=float, default=100.0,
                        help='Weight for interval protection loss')
    parser.add_argument('--lower_percentile', type=float, default=0.2,
                        help='Lower percentile for interval bounds')
    parser.add_argument('--upper_percentile', type=float, default=0.8,
                        help='Upper percentile for interval bounds')
    parser.add_argument('--reduced_dim', type=int, default=32,
                        help='PCA dimension for activation projection')
    parser.add_argument('--infinity_scale', type=float, default=100.0,
                        help='Scale factor for infinity bounds')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    log.info(f"Using device: {device}")
    
    # CIFAR-10 setup
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    all_classes = list(range(10))
    unlearn_classes = args.unlearn_classes
    retain_classes = [cls for cls in all_classes if cls not in unlearn_classes]
    
    log.info(f"Unlearn classes: {unlearn_classes} ({[class_names[c] for c in unlearn_classes]})")
    log.info(f"Retain classes: {retain_classes}")
    
    # Load data
    train_dataset, test_dataset = get_cifar10_dataloaders(args.batch_size, args.data_dir)
    
    full_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    forget_subset = create_class_subsets(train_dataset, unlearn_classes)
    forget_loader = DataLoader(forget_subset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    log.info(f"Train: {len(train_dataset)}, Forget: {len(forget_subset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Pretrain
    if args.pretrain_epochs > 0:
        pretrain_model(model, full_train_loader, criterion, device,
                       epochs=args.pretrain_epochs, lr=args.pretrain_lr)
    
    # Test before
    log.info("\n=== BEFORE Unlearning ===")
    per_class_before, overall_before = test_model(model, test_loader, device, all_classes)
    log.info(f"Overall: {overall_before:.2f}%")
    
    # Unlearn
    log.info("\n=== InTAct Unlearning ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.unlearn_lr)
    
    model = intact_unlearn(
        model=model,
        forget_loader=forget_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.unlearn_epochs,
        device=device,
        targets=args.targets,
        lambda_interval=args.lambda_interval,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        reduced_dim=args.reduced_dim,
        infinity_scale=args.infinity_scale
    )
    
    # Test after
    log.info("\n=== AFTER Unlearning ===")
    per_class_after, overall_after = test_model(model, test_loader, device, all_classes)
    log.info(f"Overall: {overall_after:.2f}%")
    
    # Summary
    log.info("\n=== Summary ===")
    forget_before = np.mean([per_class_before[c] for c in unlearn_classes])
    forget_after = np.mean([per_class_after[c] for c in unlearn_classes])
    retain_before = np.mean([per_class_before[c] for c in retain_classes])
    retain_after = np.mean([per_class_after[c] for c in retain_classes])
    
    log.info(f"Forget: {forget_before:.1f}% -> {forget_after:.1f}% (Δ{forget_after-forget_before:+.1f}%)")
    log.info(f"Retain: {retain_before:.1f}% -> {retain_after:.1f}% (Δ{retain_after-retain_before:+.1f}%)")


if __name__ == '__main__':
    main()
