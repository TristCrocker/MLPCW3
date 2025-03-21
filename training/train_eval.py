import torch
from tqdm import tqdm
import torch.nn as nn
import sys
from visualisations import visualisations
import os

def train_model(model, dataloader, epochs=1):

    torch.cuda.empty_cache()  # Frees unused memory
    torch.cuda.memory_summary(device=None, abbreviated=False)  # Shows memory usage

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=0)
    criterion_box = nn.L1Loss()

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


        for images, (target_boxes, target_labels) in progress_bar:
            
            optimizer.zero_grad()
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            target_boxes = target_boxes.to(device, dtype=torch.float32, non_blocking=True)
            target_labels = target_labels.to(device, dtype=torch.long, non_blocking=True)


            loss = None

            try:
                output = model(images)
                pred_logits = output['pred_logits']  # Shape: (batch, num_queries, num_classes+1)
                pred_boxes = output['pred_boxes']  # Shape: (batch, num_queries, 4)

                # Extract target labels and bounding boxes
                loss_cls = criterion_cls(pred_logits.view(-1, pred_logits.shape[-1]), target_labels.view(-1))
                loss_box = criterion_box(pred_boxes, target_boxes)

                # Total Loss
                loss = loss_cls + loss_box
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
                sys.stdout.flush()

            except RuntimeError as e:

                print(f"CUDA error: {e}")
                torch.cuda.empty_cache()
                sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            
            
        print(f"Epoch {epoch+1} completed, Avg Loss: {epoch_loss / len(dataloader):.4f}", flush=True)
        sys.stdout.flush()

def test_model(model, dataloader, n):
    """
    Evaluates the model on the test set using GPU.
    """
    torch.cuda.empty_cache()  # Frees unused memory
    torch.cuda.memory_summary(device=None, abbreviated=False)  # Shows memory usage

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient tracking for inference
        for index, batch in enumerate(tqdm(dataloader, desc="Testing", leave=True)):
            
            if index >= n:
                break 

            # Unpack batch correctly
            if isinstance(batch, tuple):  
                images = batch[0]  # Extract images
            else:
                images = batch  # If batch is just images, use it directly

            images = images.to(device, dtype=torch.float32, non_blocking=True)

            try:
                output = model(images)
                pred_logits = output['pred_logits'].softmax(-1)[0, :, :-1]   # Ensure PyTorch tensor
                pred_boxes = output["pred_boxes"]  # Get predicted class
                OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")  # Default to "output" if not set
                plot_path = os.path.join(OUTPUT_DIR, "bbox_plot_" + str(index) + ".png")

                visualisations.vis_bounding_boxes(images.cpu().numpy(), pred_logits.cpu().numpy(), pred_boxes.cpu().numpy(), plot_path)

            except RuntimeError as e:
                print(f"CUDA error: {e}")
                torch.cuda.empty_cache()
                sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure GPU operations complete before moving forward

            sys.stdout.flush()