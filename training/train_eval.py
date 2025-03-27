import torch
from tqdm import tqdm
import torch.nn as nn
import sys
from visualisations import visualisations
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from training.utils import *
from transformers import DetrForObjectDetection, DetrFeatureExtractor



def train_model(model, dataloader, epochs=10):

    torch.cuda.empty_cache() 
    torch.cuda.memory_summary(device=None, abbreviated=False)  

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=0)
    criterion_box = nn.L1Loss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
    model.to(device)

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = SetCriterion(
        num_classes=1,  # one ship class
        matcher=matcher,
        eos_coef=0.1,
        weight_dict=weight_dict,
        losses=['labels', 'boxes']
    ).to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True, file=sys.stdout, dynamic_ncols=True)
        print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU", flush=True)

        model.train()
        for images, (target_boxes, target_labels) in progress_bar:
            
            optimizer.zero_grad()
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            target_boxes = target_boxes.to(device, dtype=torch.float32, non_blocking=True)
            target_labels = target_labels.to(device, dtype=torch.long, non_blocking=True)


            batch_targets = []
            for i in range(images.size(0)):
                labels = target_labels[i]
                boxes = target_boxes[i]
                keep = labels == 1
                boxes = boxes[keep]
                labels = labels[keep]
                batch_targets.append({"labels": labels, "boxes": boxes})

            output = model(images)
            loss_dict = criterion(output, batch_targets)
            loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(f"Epoch {epoch+1} completed, Avg Loss: {epoch_loss / len(dataloader):.4f}", flush=True)
        sys.stdout.flush()


def train_model_pretrained(dataloader, epochs=10):
    DATASET_DIR = os.getenv("DATASET_DIR", "data")  
    
    model_path = os.path.join(DATASET_DIR, "pretrained_model")
    print("Model path:", model_path)
    print("Contents:", os.listdir(model_path))
    processor = DetrFeatureExtractor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path)

    torch.cuda.empty_cache() 
    torch.cuda.memory_summary(device=None, abbreviated=False)  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True, file=sys.stdout, dynamic_ncols=True)
        print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using CPU", flush=True)

        for images, (target_boxes, target_labels) in progress_bar:
            
            optimizer.zero_grad()
            images = [image.cpu() for image in images]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

            target_boxes = target_boxes.to(device, dtype=torch.float32, non_blocking=True)
            target_labels = target_labels.to(device, dtype=torch.long, non_blocking=True)

            batch_targets = []
            for i in range(len(images)):
                labels = target_labels[i]
                boxes = target_boxes[i]
                keep = labels == 1
                boxes = boxes[keep]
                labels = labels[keep]

                cx, cy, w, h = boxes.unbind(-1)
                x1 = cx - 0.5 * w
                y1 = cy - 0.5 * h
                x2 = cx + 0.5 * w
                y2 = cy + 0.5 * h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                boxes_xyxy[:, [0, 2]] /= images[i].shape[-1]
                boxes_xyxy[:, [1, 3]] /= images[i].shape[-2]
                batch_targets.append({"class_labels": labels, "boxes": boxes_xyxy})

            outputs = model(**inputs, labels=batch_targets)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            sys.stdout.flush()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(f"Epoch {epoch+1} completed, Avg Loss: {epoch_loss / len(dataloader):.4f}", flush=True)
        sys.stdout.flush()
    return model

def box_cxcywh_to_xywh(boxes, img_size):
    img_h, img_w = img_size
    cx, cy, w, h = boxes.unbind(-1)
    xmin = (cx - 0.5 * w) * img_w
    ymin = (cy - 0.5 * h) * img_h
    width = w * img_w
    height = h * img_h
    return torch.stack([xmin, ymin, width, height], dim=-1)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    unionArea = boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea
    return interArea / unionArea if unionArea != 0 else 0.0

def test_model(model, dataloader, n, output_dir):

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    total_seen = 0
    correct_binary = 0
    f2_scores = []
    total_time = []

    with torch.no_grad():
        for batch in dataloader:
            if total_seen >= n:
                print(f"Processed {total_seen} images.")
                break

            images, (true_bbox, true_label) = batch
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            true_bboxes = true_bbox.cpu()
            true_labels = true_label.cpu()
            batch_size = images.size(0)

            model.eval()

            torch.cuda.synchronize()
            start_event.record()

            output = model(images)

            end_event.record()
            torch.cuda.synchronize()
            total_time.append(start_event.elapsed_time(end_event))

            logits = output['pred_logits']  
            probs = logits.softmax(-1)        
            scores, labels = probs[:, :, :-1].max(dim=2)
            pred_boxes = output["pred_boxes"]

            for i in range(batch_size):
                if total_seen >= n:
                    break

                image_tensor = images[i].cpu()
                image = image_tensor.permute(1, 2, 0).numpy()
                img_h, img_w = image.shape[:2]

                true_box = true_bboxes[i]
                true_label = true_labels[i]
                
                #Ensure has bbox
                valid_flag = true_label == 1
                true_box_scaled = box_cxcywh_to_xywh(true_box[valid_flag], (img_h, img_w)).tolist()

                boxes = pred_boxes[i].cpu()
                scaled_boxes = box_cxcywh_to_xywh(boxes, (img_h, img_w))
                image_scores = scores[i].cpu()

                # Filter boxes based on confidence
                corrected_bboxes = [
                    (
                        round(x.item()),
                        round(y.item()),
                        round(w.item()),
                        round(h.item()),
                    )
                    for (x, y, w, h), score in zip(scaled_boxes, image_scores)
                    if score.item() > 0.0 and w > 0 and h > 0
                ]

                print("Boxes:", scaled_boxes)
                print("Scores:", image_scores)
                print("Corrected Boxes:", corrected_bboxes)


                #Get F2
                beta = 2
                f2 = f2_calc(corrected_bboxes, np.arange(0.5, 1.0, 0.05), true_box_scaled, beta)
                f2_scores.append(f2)

                #Get binary class score
                correct = binary_classif(true_box_scaled, corrected_bboxes)
                correct_binary += correct

                #Get visuals
                # visual_pred(image, corrected_boxes, img_h, img_w, output_dir, total_seen)
                total_seen += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            sys.stdout.flush()

    avg_f2 = np.mean(f2_scores)
    acc = correct_binary / total_seen
    print("Final F2 Score: ", avg_f2, " ------- Final Binary Class Acc Score: ", acc, ".")
    
    # Calc timing stats
    total_time_np = np.array(total_time)
    mean_time = np.mean(times)
    std_time = np.std(times, ddof=1)  # Sample standard deviation
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * std_time / np.sqrt(n)
    ci_low = mean_time - margin_of_error
    ci_high = mean_time + margin_of_error

    #Print timing stats
    print("Timing (Mean): ", mean_time/n)
    print("Timing (STD): ", std_time)
    print("Timing (CI): (", ci_low, ",", ci_high,")")


def f2_calc(corrected_bboxes, thresholds, true_box_scaled, beta):
    f2s = []
    for t in thresholds:
        TP, FP, FN = 0, 0, 0
        matched = set()
        for bbox in corrected_bboxes:
            matched_true = False
            for j, truth in enumerate(true_box_scaled):
                if j in matched:
                    continue
                iou = compute_iou(bbox, truth)
                if iou >= t:
                    TP += 1
                    matched.add(j)
                    matched_true = True
                    break
            if not matched_true:
                FP += 1
        FN = len(true_box_scaled) - len(matched)
        if TP + FP + FN == 0:
            f2 = 1.0
        else:
            f2 = (1 + beta**2) * TP / ((1 + beta**2) * TP + beta**2 * FN + FP)
        f2s.append(f2)
    return np.mean(f2s)

def visual_pred(image, corrected_boxes, img_h, img_w, output_dir, total_seen):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("auto")
    ax.set_title(f"Test Sample {total_seen}")
    ax.axis("off")

    for (xmin, ymin, width, height) in corrected_bboxes:
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                    edgecolor='r', facecolor='none', alpha=0.8, zorder=2)
        ax.add_patch(rect)

    plot_path = os.path.join(output_dir, f"A_bbox_plot_{total_seen}.png")
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

def binary_classif(true_box_scaled, corrected_bboxes):
    true_has_ship = len(true_box_scaled) > 0
    pred_has_ship = len(corrected_bboxes) > 0
    correct = int(true_has_ship == pred_has_ship)
    return correct