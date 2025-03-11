import torch
import torch.nn as nn

def train_model(model, dataloader, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for images, targets in dataloader:
            optimizer.zero_grad()
            output = model(images)
            pred_logits = output['pred_logits'].as_subclass(torch.Tensor)  # Ensure PyTorch tensor
            targets = targets.as_subclass(torch.Tensor)  # Convert targets to PyTorch tensor

            # Ensure `pred_logits` has the correct shape (batch_size, num_classes)
            pred_logits = pred_logits.view(targets.shape[0], -1)

            # Compute loss
            loss = criterion(pred_logits, targets)

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def test_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            output = model(images)
            pred_classes = output['pred_logits'].argmax(dim=-1)  # Get predicted class
            correct += (pred_classes == targets).sum().item()
            total += targets.numel()
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy