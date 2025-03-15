import torch
from tqdm import tqdm
import torch.nn as nn
import sys

def train_model(model, dataloader, epochs=3):

    torch.cuda.empty_cache()  # Frees unused memory
    torch.cuda.memory_summary(device=None, abbreviated=False)  # Shows memory usage

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # Ensure CUDA is used on the cluster
    model.to(device)

    if len(dataloader) == 0:
        print("Error: The dataloader is empty. Check your dataset and preprocessing.")
        return
    
    for batch in dataloader:
        print(f"Batch received: {type(batch)}, length: {len(batch)}")
        print(f"First batch shape: {batch[0].shape}")
        break
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True, file=sys.stdout, dynamic_ncols=True)
        print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU", flush=True)


        for images, targets in progress_bar:
            
            optimizer.zero_grad()
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)

            try:
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

            except RuntimeError as e:
                print(f"CUDA error: {e}")
                torch.cuda.empty_cache()
                sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            progress_bar.set_postfix(loss=loss.item())
            sys.stdout.flush()
        print(f"Epoch {epoch+1} completed, Avg Loss: {epoch_loss / len(dataloader):.4f}", flush=True)
        sys.stdout.flush()

def test_model(model, dataloader):
    """
    Evaluates the model on the test set using GPU.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0

    torch.cuda.empty_cache()  # Free unused GPU memory

    progress_bar = tqdm(dataloader, desc="Testing", leave=True, file=sys.stdout, dynamic_ncols=True)
    print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU", flush=True)

    with torch.no_grad():  # Disable gradient tracking for inference
        for images, targets in progress_bar:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)

            try:
                output = model(images)
                pred_logits = output['pred_logits'].as_subclass(torch.Tensor)  # Ensure PyTorch tensor
                pred_classes = pred_logits.argmax(dim=-1)  # Get predicted class

                correct += (pred_classes == targets).sum().item()
                total += targets.numel()

            except RuntimeError as e:
                print(f"CUDA error: {e}")
                torch.cuda.empty_cache()
                sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure GPU operations complete before moving forward

            progress_bar.set_postfix(accuracy=f"{(correct / total) * 100:.2f}%" if total > 0 else "N/A")
            sys.stdout.flush()

    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    sys.stdout.flush()
    return accuracy