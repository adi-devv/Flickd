
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import json
import clip
import torch
import pandas as pd
import faiss
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import spacy
from jsonschema import validate
import logging
import pickle
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_file_checksum(file_path):
    """Compute MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def extract_frames(video_path, output_dir, interval=10):
    """Extract keyframes from a video at specified intervals."""
    logger.info(f"Extracting frames from {video_path}")
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    os.makedirs(output_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        logger.error("Could not open video file")
        vid.release()
        return False
    count = 0
    frames_saved = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_saved += 1
            logger.info(f"Saved frame {count} to {frame_path}")
        count += 1
    vid.release()
    logger.info(f"Total frames processed: {count}, saved: {frames_saved}")
    return True

def detect_objects(image_path, frame_number, model, detectedframepath):
    """Detect fashion items in a frame using YOLOv8."""
    logger.info(f"Detecting objects in frame {frame_number}: {image_path}")
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return []
    try:
        frame = cv2.imread(image_path)
        results = model.predict(source=image_path, conf=0.4, save=False, line_width=2)
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                bbox = box.xywh[0].tolist()
                confidence = float(box.conf)
                center_x, center_y, w, h = [int(v) for v in bbox]
                
                # Convert center coordinates to top-left coordinates
                x = max(0, center_x - w // 2)
                y = max(0, center_y - h // 2)
                
                # Ensure crop stays within frame bounds
                x_end = min(frame.shape[1], x + w)
                y_end = min(frame.shape[0], y + h)
                x = max(0, x)
                y = max(0, y)
                
                # Check if crop is too small
                if (x_end - x) < 20 or (y_end - y) < 20:
                    logger.warning(f"Skipping small crop (w={x_end-x}, h={y_end-y}) for {class_name} in frame {frame_number}")
                    continue
                
                crop = frame[y:y_end, x:x_end]
                
                # Validate crop
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    logger.warning(f"Invalid crop (size: {crop.shape}) for {class_name} in frame {frame_number}")
                    continue
                
                detections.append({
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": confidence,
                    "frame_number": frame_number,
                    "crop": crop
                })
        if detections:
            annotated_frame = result.plot()
            save_path = os.path.join(detectedframepath, f"detected_frame_{frame_number}.jpg")
            cv2.imwrite(save_path, annotated_frame)
            logger.info(f"Saved detected frame to {save_path}")
        else:
            logger.warning(f"No objects detected in frame {frame_number}, skipping save")
        return detections
    except Exception as e:
        logger.error(f"Error running YOLO on {image_path}: {e}")
        return []

def setup_faiss_index(images_csv, product_data_csv, id_column="id", cache_dir="data/cache", max_product_ids=300):
    """Set up FAISS index for product matching with CLIP embeddings, with caching and ID limit."""
    logger.info(f"Setting up FAISS index with {images_csv} and {product_data_csv}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cache paths
    os.makedirs(cache_dir, exist_ok=True)
    cache_metadata_path = os.path.join(cache_dir, "cache_metadata.pkl")
    faiss_index_path = os.path.join(cache_dir, "faiss_index.bin")
    product_info_path = os.path.join(cache_dir, "product_info.pkl")
    product_id_to_indices_path = os.path.join(cache_dir, "product_id_to_indices.pkl")
    
    # Compute checksums
    try:
        images_checksum = get_file_checksum(images_csv)
        product_data_checksum = get_file_checksum(product_data_csv)
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        raise
    
    # Check cache
    if os.path.exists(cache_metadata_path):
        try:
            with open(cache_metadata_path, "rb") as f:
                cache_metadata = pickle.load(f)
            if (cache_metadata.get("images_checksum") == images_checksum and
                cache_metadata.get("product_data_checksum") == product_data_checksum):
                logger.info("Loading cached FAISS index and metadata")
                index = faiss.read_index(faiss_index_path)
                with open(product_info_path, "rb") as f:
                    product_info = pickle.load(f)
                with open(product_id_to_indices_path, "rb") as f:
                    product_id_to_indices = pickle.load(f)
                clip_model, preprocess = clip.load("ViT-B/32", device=device)
                logger.info(f"Loaded FAISS index with {index.ntotal} embeddings")
                return index, product_info, product_id_to_indices, clip_model, preprocess, device
            else:
                logger.info("Cache invalidated due to CSV changes")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Rebuilding index.")
    
    # Build new index
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    try:
        # Load CSVs
        images_df = pd.read_csv(images_csv)
        product_df = pd.read_csv(product_data_csv)
        logger.info(f"Images CSV columns: {list(images_df.columns)}")
        logger.info(f"Product Data CSV columns: {list(product_df.columns)}")
        
        # Validate id_column
        if id_column not in images_df.columns:
            raise ValueError(f"Column '{id_column}' not found in images CSV")
        if id_column not in product_df.columns:
            raise ValueError(f"Column '{id_column}' not found in product data CSV")
        
        # Check for ID mismatches
        image_ids = set(images_df[id_column].astype(str).unique())
        product_ids = set(product_df[id_column].astype(str).unique())
        missing_in_products = image_ids - product_ids
        missing_in_images = product_ids - image_ids
        if missing_in_products:
            logger.warning(f"Image IDs not found in product data: {missing_in_products}")
        if missing_in_images:
            logger.warning(f"Product IDs not found in images: {missing_in_images}")
        
        # Merge DataFrames
        catalog = images_df.merge(product_df, on=id_column, how="inner")
        if catalog.empty:
            raise ValueError("Merged catalog is empty. No matching IDs found.")
        logger.info(f"Merged catalog size: {len(catalog)} rows")
        
        embeddings = []
        product_info = []
        product_id_to_indices = {}
        current_index = 0
        successful_product_ids = 0
        
        # Group by product ID
        for product_id, group in catalog.groupby(id_column):
            if successful_product_ids >= max_product_ids:
                logger.warning(f"Reached limit of {max_product_ids} product IDs. Stopping analysis.")
                break
            product_indices = []
            product_data = group.iloc[0]
            valid_image_count = 0
            for _, row in group.iterrows():
                try:
                    response = requests.get(row['image_url'], timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    image = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = clip_model.encode_image(image).cpu().numpy()
                    embeddings.append(embedding)
                    product_indices.append(current_index)
                    current_index += 1
                    valid_image_count += 1
                except Exception as e:
                    logger.error(f"Error processing image {row['image_url']} for product ID {product_id}: {e}")
                    continue
            if valid_image_count > 0:
                product_info.append({
                    "id": str(product_id),
                    "product_type": product_data.get('product_type', 'unknown'),
                    "description": product_data.get('description', ''),
                    "product_tags": product_data.get('product_tags', '')
                })
                product_id_to_indices[str(product_id)] = product_indices
                successful_product_ids += 1
                logger.info(f"SUCCESS COUNT - {successful_product_ids} Processed {valid_image_count} images for product ID {product_id}")
            else:
                logger.warning(f"No valid images processed for product ID {product_id}")
        
        if not embeddings:
            raise ValueError("No valid embeddings generated from catalog images")
        
        # Create FAISS index
        embeddings = np.array(embeddings).squeeze()
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, embeddings.shape[0])
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"FAISS index created with {len(embeddings)} embeddings")
        
        # Save cache
        cache_metadata = {
            "images_checksum": images_checksum,
            "product_data_checksum": product_data_checksum
        }
        with open(cache_metadata_path, "wb") as f:
            pickle.dump(cache_metadata, f)
        faiss.write_index(index, faiss_index_path)
        with open(product_info_path, "wb") as f:
            pickle.dump(product_info, f)
        with open(product_id_to_indices_path, "wb") as f:
            pickle.dump(product_id_to_indices, f)
        logger.info("Saved FAISS index and metadata to cache")
        
        return index, product_info, product_id_to_indices, clip_model, preprocess, device
    
    except Exception as e:
        logger.error(f"Error setting up FAISS index: {e}")
        raise

FASHION_COLOR_MAP_RGB = {
    "Red": (255, 0, 0),
    "Green": (0, 128, 0),
    "Blue": (0, 0, 255),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Yellow": (255, 255, 0),
    "Gray": (128, 128, 128),
    "Orange": (255, 165, 0),
    "Pink": (255, 192, 203),
    "Purple": (128, 0, 128),
    "Brown": (139, 69, 19),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Navy Blue": (0, 0, 128),
    "Teal": (0, 128, 128),
    "Maroon": (128, 0, 0),
    "Olive Green": (128, 128, 0),
    "Beige": (245, 245, 220),
    "Cream": (255, 253, 208),
    "Ivory": (255, 255, 240),
    "Charcoal": (54, 69, 79),
    "Silver": (192, 192, 192),
    "Gold": (255, 215, 0),
    "Crimson": (220, 20, 60),
    "Indigo": (75, 0, 130),
    "Lavender": (230, 230, 250),
    "Coral": (255, 127, 80),
    "Khaki": (195, 176, 145),
    "Lime Green": (50, 205, 50),
    "Sky Blue": (135, 206, 235)
}

FASHION_COLOR_MAP_LAB = {}
for name, rgb in FASHION_COLOR_MAP_RGB.items():
    bgr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
    FASHION_COLOR_MAP_LAB[name] = lab

def get_dominant_color(crop):
    try:
        if crop is None or crop.size == 0:
            logger.warning("Input crop is empty or None.")
            return "Unknown"

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(crop_hsv)

        saturation_threshold_low = 30
        value_threshold_high = 220
        value_threshold_low = 30

        avg_s = np.mean(s)
        avg_v = np.mean(v)

        if avg_s < saturation_threshold_low:
            if avg_v > value_threshold_high:
                return "White"
            elif avg_v < value_threshold_low:
                return "Black"
            else:
                return "Gray"
        
        crop_hsv[:, :, 1] = np.clip(cv2.add(crop_hsv[:, :, 1], 120), 0, 255)
        crop_rgb_boosted = cv2.cvtColor(crop_hsv, cv2.COLOR_HSV2RGB)

        pixels = np.float32(crop_rgb_boosted).reshape(-1, 3)

        if len(pixels) < 20:
            logger.warning(f"Not enough pixels: {len(pixels)}")
            return "Unknown"

        k_clusters = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        
        _, labels, centers = cv2.kmeans(pixels, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        label_counts = np.bincount(labels.flatten())
        dominant_cluster_idx = np.argmax(label_counts)
        
        dominant_rgb = tuple(int(c) for c in centers[dominant_cluster_idx])
        logger.debug(f"K-Means Dominant RGB: {dominant_rgb}")

        dominant_bgr = np.uint8([[list(dominant_rgb)]])
        dominant_lab = cv2.cvtColor(dominant_bgr, cv2.COLOR_BGR2LAB)[0][0]
        
        min_dist = float("inf")
        color_name = "Unknown"

        for map_name, map_lab in FASHION_COLOR_MAP_LAB.items():
            dist = np.linalg.norm(dominant_lab - map_lab)
            if dist < min_dist:
                min_dist = dist
                color_name = map_name
        
        logger.debug(f"Selected Color: {color_name}, Distance: {min_dist}")
        return color_name

    except Exception as e:
        logger.error(f"Error detecting color: {e}")
        return "Unknown"
    
def match_products(detections, index, product_info, product_id_to_indices, clip_model, preprocess, device):
    """Match detected objects to catalog products using CLIP and FAISS."""
    logger.info("Matching products to detections")
    matches = []
    for i, detection in enumerate(detections):
        try:
            crop = detection['crop']
            crop_path = f"cropped_frames/crop_frame_{detection['frame_number']}_{detection['class']}_{i}.jpg"
            cv2.imwrite(crop_path, crop)
            logger.debug(f"Saved crop to {crop_path}")
            crop_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image_input = preprocess(crop_image).unsqueeze(0).to(device)
            with torch.no_grad():
                query_embedding = clip_model.encode_image(image_input).cpu().numpy()
            distances, indices = index.search(query_embedding, k=1)
            similarity = 1 - distances[0][0] / 2
            logger.debug(f"Detection similarity score: {similarity}")
            match_type = "exact" if similarity > 0.5 else "similar" if similarity > 0.75 else "no_match"
            
            matched_index = indices[0][0]
            matched_product_id = None
            for product_id, idx_list in product_id_to_indices.items():
                if matched_index in idx_list:
                    matched_product_id = product_id
                    break
            
            product = next((p for p in product_info if p['id'] == matched_product_id), None)
            if product and match_type != "no_match":
                matches.append({
                    "type": detection['class'],
                    "color": get_dominant_color(crop),
                    "match_type": match_type,
                    "matched_product_id": str(matched_product_id),
                    "confidence": float(similarity),
                    "crop_image_file": crop_path
                })
            else:
                matches.append({
                    "type": detection['class'],
                    "color": get_dominant_color(crop),
                    "match_type": "no_match",
                    "matched_product_id": None,
                    "confidence": 0.0,
                    "crop_image_file": crop_path
                })
        except Exception as e:
            logger.error(f"Error matching product: {e}")
            matches.append({
                "type": detection['class'],
                "color": "unknown",
                "match_type": "no_match",
                "matched_product_id": None,
                "confidence": 0.0
            })
    return matches

def classify_vibe(caption, product_info, vibe_taxonomy=None):
    """Classify video vibe based on caption and product metadata."""
    logger.info("Classifying vibe")
    nlp = spacy.load("en_core_web_sm")
    vibe_keywords = {
        'Coquette': ['darling', 'flirty', 'romance', 'dress', 'feminine', 'sweet', 'charm', 'lace', 'bow', 'heart', 'blush', 'cute'],
        'Boho': ['summer', 'flowy', 'earthy', 'outfit', 'date', 'bohemian', 'relaxed', 'fringe', 'natural', 'breezy', 'gypsy', 'vibes'],
        'Clean Girl': ['minimal', 'neutral', 'sleek', 'simple', 'clean', 'classic', 'chic', 'effortless', 'crisp', 'modern', 'subtle'],
        'Cottagecore': ['vintage', 'pastel', 'nature', 'rustic', 'floral', 'cozy', 'farmhouse', 'whimsical', 'meadow', 'homemade'],
        'Streetcore': ['urban', 'sneakers', 'graffiti', 'edgy', 'street', 'cool', 'grunge', 'skate', 'bold', 'raw', 'city'],
        'Y2K': ['glitter', 'metallic', 'retro', 'bold', 'sparkly', 'neon', 'futuristic', 'pop', 'shiny', 'trendy', '2000s'],
        'Party Glam': ['sparkle', 'sequin', 'bold', 'glam', 'shimmer', 'luxe', 'dazzle', 'fancy', 'evening', 'glitz', 'radiant']
    }
    if vibe_taxonomy:
        vibe_keywords = {vibe: vibe_keywords.get(vibe, []) for vibe in vibe_taxonomy}
    
    text = caption.lower()
    
    doc = nlp(text)
    scores = {vibe: 0 for vibe in vibe_keywords}
    for token in doc:
        for vibe, keywords in vibe_keywords.items():
            if token.text in keywords:
                scores[vibe] += 1
    logger.debug(f"Vibe scores: {scores}")
    top_vibes = [vibe for vibe, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3] if score > 0]
    return top_vibes or ["Unknown"]

def validate_output(output):
    """Validate JSON output against schema."""
    schema = {
        "type": "object",
        "properties": {
            "video_id": {"type": "string"},
            "vibes": {"type": "array", "items": {"type": "string"}},
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "color": {"type": "string"},
                        "match_type": {"type": "string", "enum": ["exact", "similar", "no_match"]},
                        "matched_product_id": {"type": ["string", "null"]},
                        "confidence": {"type": "number"}
                    },
                    "required": ["type", "color", "match_type", "matched_product_id", "confidence"]
                }
            }
        },
        "required": ["video_id", "vibes", "products"]
    }
    try:
        validate(instance=output, schema=schema)
        logger.info("Output JSON validated successfully")
    except Exception as e:
        logger.error(f"Output JSON validation failed: {e}")
        raise

def main(video_path, images_csv, product_data_csv, caption, video_id, output_json_path, vibe_taxonomy=None):
    """Main function to process video and generate output JSON."""
    logger.info(f"Starting processing for video ID: {video_id}")
    
    # Load YOLO model
    try:
        model = YOLO("D:/Aadit/ML/Flickd/runs/detect/train3/weights/best.pt")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return
    
    # Setup FAISS index
    try:
        index, product_info, product_id_to_indices, clip_model, preprocess, device = setup_faiss_index(images_csv, product_data_csv)
    except Exception as e:
        logger.error(f"Failed to setup FAISS index: {e}")
        return
    
    # Process video
    output_dir = "frames"
    detectedframepath = "detected_frames"
    cropped_frames_dir = "cropped_frames"

    if extract_frames(video_path, output_dir):
        detections = []
        os.makedirs(detectedframepath, exist_ok=True)
        os.makedirs(cropped_frames_dir, exist_ok=True)
        for frame_path in os.listdir(output_dir):
            if frame_path.endswith(".jpg"):
                frame_number = int(frame_path.split("_")[1].split(".")[0])
                full_frame_path = os.path.join(output_dir, frame_path)
                frame_detections = detect_objects(full_frame_path, frame_number, model, detectedframepath)
                detections.extend(frame_detections)
        
        # Match products
        matches = match_products(detections, index, product_info, product_id_to_indices, clip_model, preprocess, device)
        
        # Classify vibe
        vibes = classify_vibe(caption, product_info, vibe_taxonomy)
        
        # Format output
        output = {
            "video_id": video_id,
            "vibes": vibes,
            "products": matches
        }
        
        # Validate output
        validate_output(output)
        
        # Save output
        os.makedirs("outputs", exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved output to {output_json_path}")
    else:
        logger.error("Failed to extract frames. Exiting.")

if __name__ == "__main__":
    # Define paths and inputs
    video_path = "data/sample_video2.mp4"
    images_csv = "data/images.csv"
    product_data_csv = "data/product_data.csv"
    caption = '''hello darling trendy clothes right away lets gooooo #streetstyle #2000s'''
    video_id = "sample_video2"
    output_json_path = f"outputs/output_{video_id}.json"
    vibe_taxonomy = ["Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"]
    
    main(video_path, images_csv, product_data_csv, caption, video_id, output_json_path, vibe_taxonomy)
