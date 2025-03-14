import torch
from tqdm import tqdm
import torch.nn as nn
import sys

def train_model(model, dataloader, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # Ensure CUDA is used on the cluster
    model.to(device)

    if len(dataloader) == 0:
        print("Error: The dataloader is empty. Check your dataset and preprocessing.")
        return
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)
        for images, targets in progress_bar:
            optimizer.zero_grad()
            
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.long)

            output = model(images)
            pred_logits = output['pred_logits'].as_subclass(torch.Tensor)  # Ensure PyTorch tensor
            targets = targets.as_subclass(torch.Tensor)  # Convert targets to PyTorch tensor

            # Ensure `pred_logits` has the correct shape (batch_size, num_classes)
            pred_logits = pred_logits.view(targets.shape[0], -1)

            # Compute loss
            loss = criterion(pred_logits, targets)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
            sys.stdout.flush()
        print(f"Epoch {epoch+1} completed, Avg Loss: {epoch_loss / len(dataloader):.4f}")
        sys.stdout.flush()

def test_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(torch.float32) / 255.0
            output = model(images)
            pred_classes = output['pred_logits'].argmax(dim=-1)  # Get predicted class
            correct += (pred_classes == targets).sum().item()
            total += targets.numel()
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy