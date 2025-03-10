import torch
import torch.nn as nn

def train_model(model, dataloader, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for images, targets in dataloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output['pred_logits'].squeeze(), targets)
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