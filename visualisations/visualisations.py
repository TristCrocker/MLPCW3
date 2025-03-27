import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches
import os

def visualize_batch(dls, path, num_images=4, num_queries=10):

    # Get a batch of images and labels
    batch = dls.one_batch()

    # Extract images & bounding boxes
    images_tensor = batch[0]  
    bboxes_tensor, labels_tensor = batch[1] 


    for i in range(num_images):
        image_tensor = images_tensor[i] 
        image = image_tensor.permute(1, 2, 0).cpu().numpy() 
        image_size = image.shape[0]
        bboxes = bboxes_tensor[i].cpu()
        
        corrected_bboxes = [
            (
                torch.round(cx * image_size - (w * image_size) / 2).item(),  # scale cx
                torch.round(cy * image_size - (h * image_size) / 2).item(),  # scale cy
                torch.round(w * image_size).item(),  # scale width
                torch.round(h * image_size).item(),  # scale height
            )
            for cx, cy, w, h in bboxes if w > 0 and h > 0
        ]

        fig, ax = plt.subplots(figsize=(6, 6))
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

        plt.savefig(os.path.join(path, f"AA_bbox_image{i}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

    print("DONE")