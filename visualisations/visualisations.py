import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches
import os

def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from center-based format (cx, cy, w, h) 
    to (xmin, ymin, xmax, ymax).
    """
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    """
    Rescale bounding boxes from [0,1] range to actual image size.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def vis_bounding_boxes(images, pred_logits, pred_boxes, path):
    
    num_images = len(images)

    for idx in range(num_images):
        print(f"Image {idx} - Predicted Boxes: {pred_boxes[idx]}")
        img = images[idx]
        boxes = pred_boxes[idx]  # Get boxes for the specific image
        scores = pred_logits[idx]  # Get scores for the specific image

        img = np.moveaxis(img, 0, -1)
        height, width, _ = img.shape  # Get image dimensions

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Ensure tensor format
        boxes = rescale_bboxes(boxes, (width, height))  # Scale to image size
        
        unique_boxes = set()  # Reset per image to prevent duplicates

        for box, score in zip(boxes, scores):
            confidence = score.max().item()  # Get max confidence for the box

            if confidence > 0.7:  # Filter low confidence detections
                xmin, ymin, xmax, ymax = box.tolist()
                key = (xmin, ymin, xmax, ymax)  # Tuple key for uniqueness check

                if key in unique_boxes:  
                    continue  # Skip duplicate boxes
                unique_boxes.add(key)  # Store unique boxes

                # Draw rectangle
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=2)
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"{confidence:.2f}", color="white", fontsize=8,
                        bbox=dict(facecolor="red", alpha=0.5))

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)



def visualize_batch(dls, path, num_images=4, num_queries=10):
    """
    Visualizes multiple images from the DataLoader (`dls`) with their bounding boxes.
    """
    # Get a batch of images and labels
    batch = dls.one_batch()
    print(type(batch))  # Should be a tuple or list
    print(len(batch))   # Number of elements in batch
    print(type(batch[0])) 

    # Extract images & bounding boxes
    images_tensor = batch[0]  # Shape: (batch_size, 3, H, W)
    bboxes_tensor, labels_tensor = batch[1]  # Shape: (batch_size, num_queries, 4), (batch_size, num_queries)


    for i in range(num_images):
        image_tensor = images_tensor[i]  # Select an image
        image = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, 3)
        image_size = image.shape[0]
        # Get bounding boxes for the image
        bboxes = bboxes_tensor[i].cpu() # Convert from normalized to pixel coordinates

        # Convert to (xmin, ymin, width, height) format for visualization
        # bboxes = [(float(cx - w / 2), float(cy - h / 2), float(w), float(h)) for cx, cy, w, h in bboxes if w > 0 and h > 0]
        
        corrected_bboxes = [
            (
                torch.round(cx * image_size - (w * image_size) / 2).item(),  # Correctly scale cx
                torch.round(cy * image_size - (h * image_size) / 2).item(),  # Correctly scale cy
                torch.round(w * image_size).item(),  # width
                torch.round(h * image_size).item(),  # height
            )
            for cx, cy, w, h in bboxes if w > 0 and h > 0
        ]

        print(f"Bounding Boxes (before scaling): {bboxes}")
        print(f"Bounding Boxes (after scaling): {corrected_bboxes}")

        fig, ax = plt.subplots(figsize=(6, 6))
        # Plot the image
        # ax.imshow(image, extent=[0, image_size, image_size, 0])  # Force image to fill the correct area
        ax.imshow(image)
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)
        ax.set_aspect("auto")
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")

        # Draw bounding boxes
        for (xmin, ymin, width, height) in corrected_bboxes:
            if xmin >= 0 and ymin >= 0 and width > 0 and height > 0:
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none', alpha=0.8, zorder=2)
                ax.add_patch(rect)

        plt.savefig(os.path.join(path, f"bbox_image{i}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)