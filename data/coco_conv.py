import os
import json
import pandas as pd

def convert_to_coco(csv_path, image_dir, output_json):
    df = pd.read_csv(csv_path)
    df.fillna('', inplace=True)
    grouped = df.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "ship", "supercategory": "object"}]
    }
    
    annotation_id = 1
    for idx, row in grouped.iterrows():
        image_id = idx + 1
        file_name = row['ImageId']
        image_path = os.path.join(image_dir, file_name)
        
        height, width = 768, 768 
        
        coco_data['images'].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })
        
        for rle in row['EncodedPixels']:
            if rle: 
                coco_data['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": {"size": [height, width], "counts": rle},
                    "iscrowd": 0
                })
                annotation_id += 1
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO JSON saved to {output_json}")
    
    
convert_to_coco("data/train_ship_segmentations_v2.csv", "data/train_v2", "data/coco.json")
