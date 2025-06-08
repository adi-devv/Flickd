import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
from ultralytics import YOLO
import json
import clip
import torch
import pandas as pd
import faiss
import numpy as np
from PIL import Image
import io
from io import BytesIO
import requests
import spacy
from jsonschema import validate
import logging
import pickle
import hashlib
from colorthief import ColorThief
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

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
        results = model.predict(source=image_path, conf=0.5, save=False, line_width=2)
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                bbox = box.xywh[0].tolist()
                confidence = float(box.conf)
                
                center_x, center_y, w, h = [int(v) for v in bbox]
                
                x = max(0, center_x - w // 2)
                y = max(0, center_y - h // 2)
                x_end = min(frame.shape[1], x + w)
                y_end = min(frame.shape[0], y + h)
                x = max(0, x)
                y = max(0, y)
                
                if (x_end - x) < 20 or (y_end - y) < 20:
                    logger.warning(f"Skipping small crop (w={x_end-x}, h={y_end-y}) for {class_name} in frame {frame_number}")
                    continue
                
                crop = frame[y:y_end, x:x_end]
                
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
    
def setup_faiss_index(images_csv, product_data_csv, id_column="id", cache_dir="data/cache", max_product_ids=1000):
    """Set up FAISS index for product matching with CLIP embeddings, cropping images with YOLO to extract colors from fashion items and saving crops in cache."""
    logger.info(f"Setting up FAISS index with {images_csv} and {product_data_csv}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cache paths
    os.makedirs(cache_dir, exist_ok=True)
    cache_metadata_path = os.path.join(cache_dir, "cache_metadata.pkl")
    faiss_index_path = os.path.join(cache_dir, "faiss_index.bin")
    product_info_path = os.path.join(cache_dir, "product_info.pkl")
    product_id_to_indices_path = os.path.join(cache_dir, "product_id_to_indices.pkl")
    index_to_product_id_path = os.path.join(cache_dir, "index_to_product_id.json")
    cropped_images_dir = os.path.join(cache_dir, "cropped_images")
    os.makedirs(cropped_images_dir, exist_ok=True)
    
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
    
    # Load YOLO model
    try:
        yolo_model = YOLO("D:/Aadit/ML/Flickd/runs/detect/train3/weights/best.pt")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise
    
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
        index_to_product_id = {}
        current_index = 0
        successful_product_ids = 0
        invalid_product_ids = []
        
        # Group by product ID
        unique_product_ids = catalog[id_column].unique()
        logger.info(f"Total unique product IDs: {len(unique_product_ids)}")
        for product_id in unique_product_ids:
            if successful_product_ids >= max_product_ids:
                logger.warning(f"Reached limit of {max_product_ids} product IDs. Stopping analysis.")
                break
            group = catalog[catalog[id_column] == product_id]
            product_indices = []
            product_data = group.iloc[0]
            valid_image_found = False
            
            # Try processing one valid image
            for _, row in group.iterrows():
                if valid_image_found:
                    break
                try:
                    # Download and load image
                    response = requests.get(row['image_url'], timeout=10)
                    response.raise_for_status()
                    pil_image = Image.open(BytesIO(response.content))
                    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                    # Run YOLO to detect fashion item
                    results = yolo_model.predict(source=opencv_image, conf=0.5, save=False)
                    crop = opencv_image  # Default to full image
                    if results and results[0].boxes:
                        # Select highest-confidence detection
                        box = results[0].boxes[0]
                        bbox = box.xywh[0].tolist()
                        center_x, center_y, w, h = [int(v) for v in bbox]
                        x = max(0, center_x - w // 2)
                        y = max(0, center_y - h // 2)
                        x_end = min(opencv_image.shape[1], x + w)
                        y_end = min(opencv_image.shape[0], y + h)
                        if (x_end - x) >= 20 and (y_end - y) >= 20:
                            crop = opencv_image[y:y_end, x:x_end]
                            logger.debug(f"Product {product_id}: Cropped to bbox (x={x}, y={y}, w={x_end-x}, h={y_end-y})")
                        else:
                            logger.warning(f"Product {product_id}: Crop too small (w={x_end-x}, h={y_end-y}), using full image")
                    else:
                        logger.warning(f"Product {product_id}: No objects detected, using full image for color and embedding")
                    
                    # Save cropped image
                    crop_path = os.path.join(cropped_images_dir, f"product_{product_id}.jpg")
                    cv2.imwrite(crop_path, crop)
                    logger.debug(f"Product {product_id}: Saved cropped image to {crop_path}")
                    
                    # Extract dominant colors from cropped image
                    product_dominant_colors_rgb = get_dominant_color(crop, num_colors=5)
                    logger.debug(f"Product {product_id}: Extracted dominant RGBs: {product_dominant_colors_rgb}")
                    
                    # Process for CLIP embedding
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    image_for_clip = preprocess(crop_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = clip_model.encode_image(image_for_clip).cpu().numpy()
                        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                    
                    embeddings.append(embedding)
                    product_indices.append(current_index)
                    index_to_product_id[current_index] = str(product_id)
                    current_index += 1
                    valid_image_found = True
                except Exception as e:
                    logger.error(f"Error processing image {row['image_url']} for product ID {product_id}: {e}")
                    continue
            
            if valid_image_found:
                product_info.append({
                    "id": str(product_id),
                    "product_type": product_data.get('product_type', 'unknown'),
                    "description": product_data.get('description', ''),
                    "product_tags": product_data.get('product_tags', ''),
                    "dominant_colors_rgb": product_dominant_colors_rgb
                })
                product_id_to_indices[str(product_id)] = product_indices
                successful_product_ids += 1
                logger.info(f"Processed product ID {product_id} ({successful_product_ids}/{max_product_ids})")
            else:
                logger.warning(f"No valid images found for product ID {product_id}")
                invalid_product_ids.append(product_id)
                product_info.append({
                    "id": str(product_id),
                    "product_type": product_data.get('product_type', 'unknown'),
                    "description": product_data.get('description', ''),
                    "product_tags": product_data.get('product_tags', ''),
                    "dominant_colors_rgb": []
                })
            
        # Log if all product IDs are processed
        if successful_product_ids < max_product_ids:
            logger.info(f"Processed all {successful_product_ids} available product IDs")
        
        # Remove invalid product IDs from CSVs
        if invalid_product_ids:
            logger.info(f"Removing {len(invalid_product_ids)} invalid product IDs from CSVs")
            images_df = images_df[~images_df[id_column].isin(invalid_product_ids)]
            images_df.to_csv(images_csv, index=False)
            product_df = product_df[~product_df[id_column].isin(invalid_product_ids)]
            product_df.to_csv(product_data_csv, index=False)
        
        if not embeddings:
            raise ValueError("No valid embeddings generated")
        
        # Create FAISS index
        embeddings_array = np.vstack(embeddings)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        logger.info(f"FAISS index created with {index.ntotal} embeddings")
        
        # Save cache
        cache_metadata = {
            "images_checksum": get_file_checksum(images_csv),
            "product_data_checksum": get_file_checksum(product_data_csv)
        }
        with open(cache_metadata_path, "wb") as f:
            pickle.dump(cache_metadata, f)
        faiss.write_index(index, faiss_index_path)
        with open(product_info_path, "wb") as f:
            pickle.dump(product_info, f)
        with open(product_id_to_indices_path, "wb") as f:
            pickle.dump(product_id_to_indices, f)
        with open(index_to_product_id_path, "w") as f:
            json.dump(index_to_product_id, f, indent=2)
        logger.info("Saved FAISS index, metadata, and cropped images to cache")
        
        return index, product_info, product_id_to_indices, clip_model, preprocess, device
    
    except Exception as e:
        logger.error(f"Error setting up FAISS index: {e}")
        raise

FASHION_COLOR_MAP_RGB = {
    "Red": (255, 0, 0),
    "Scarlet": (255, 36, 0),
    "Crimson": (220, 20, 60),
    "Ruby": (224, 17, 95),
    "Maroon": (128, 0, 0),
    "Burgundy": (128, 0, 32),
    "Cherry": (222, 49, 99),
    "Green": (0, 128, 0),
    "Emerald": (0, 155, 119),
    "Forest Green": (34, 139, 34),
    "Olive Green": (128, 128, 0),
    "Sage Green": (159, 190, 147),
    "Mint Green": (152, 255, 152),
    "Lime Green": (50, 205, 50),
    "Jade": (0, 168, 107),
    "Blue": (0, 0, 255),
    "Navy Blue": (0, 0, 128),
    "Royal Blue": (65, 105, 225),
    "Sky Blue": (135, 206, 235),
    "Cerulean": (0, 123, 167),
    "Cobalt": (0, 71, 171),
    "Sapphire": (15, 82, 186),
    "Baby Blue": (137, 207, 240),
    "Turquoise": (64, 224, 208),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Charcoal": (54, 69, 79),
    "Slate Gray": (112, 128, 144),
    "Ash Gray": (178, 190, 181),
    "Silver": (192, 192, 192),
    "Ivory": (255, 255, 240),
    "Cream": (255, 253, 208),
    "Beige": (245, 245, 220),
    "Taupe": (139, 133, 112),
    "Khaki": (195, 176, 145),
    "Sand": (194, 178, 128),
    "Tan": (210, 180, 140),
    "Champagne": (247, 231, 206),
    "Off-White": (245, 245, 245),
    "Yellow": (255, 255, 0),
    "Canary Yellow": (255, 239, 0),
    "Mustard": (255, 219, 88),
    "Lemon": (255, 250, 124),
    "Buttercup": (250, 218, 94),
    "Orange": (255, 165, 0),
    "Tangerine": (255, 153, 102),
    "Peach": (255, 218, 185),
    "Apricot": (251, 206, 177),
    "Burnt Orange": (204, 85, 0),
    "Amber": (255, 191, 0),
    "Pink": (255, 192, 203),
    "Blush Pink": (255, 209, 220),
    "Millennial Pink": (243, 213, 213),
    "Rose": (255, 153, 204),
    "Fuchsia": (255, 0, 255),
    "Hot Pink": (255, 105, 180),
    "Bubblegum": (255, 193, 204),
    "Coral": (255, 127, 80),
    "Salmon": (250, 128, 114),
    "Peony": (237, 145, 166),
    "Purple": (128, 0, 128),
    "Lavender": (230, 230, 250),
    "Lilac": (200, 162, 200),
    "Violet": (148, 0, 211),
    "Indigo": (75, 0, 130),
    "Plum": (142, 69, 133),
    "Orchid": (218, 112, 214),
    "Mauve": (224, 176, 255),
    "Brown": (139, 69, 19),
    "Chocolate": (123, 63, 0),
    "Mocha": (150, 105, 80),
    "Caramel": (196, 132, 81),
    "Toffee": (176, 101, 54),
    "Sienna": (160, 82, 45),
    "Umber": (99, 81, 71),
    "Rust": (183, 65, 14),
    "Terracotta": (226, 114, 91),
    "Cyan": (0, 255, 255),
    "Teal": (0, 128, 128),
    "Aqua": (0, 255, 204),
    "Seafoam": (120, 219, 184),
    "Magenta": (255, 0, 255),
    "Berry": (153, 0, 76),
    "Gold": (255, 215, 0),
    "Rose Gold": (183, 110, 121),
    "Bronze": (205, 127, 50),
    "Copper": (184, 115, 51),
    "Platinum": (229, 228, 226),
    "Pastel Pink": (255, 224, 229),
    "Pastel Blue": (173, 216, 230),
    "Pastel Green": (198, 227, 199),
    "Pastel Yellow": (255, 245, 208),
    "Pastel Purple": (221, 204, 255),
    "Powder Blue": (176, 224, 230),
    "Mint": (189, 252, 201),
    "Pale Peach": (255, 229, 217),
    "Millennial Orange": (255, 179, 128),
    "Dusty Rose": (210, 144, 144),
    "Saffron": (244, 196, 48),
    "Periwinkle": (204, 204, 255),
    "Ochre": (204, 119, 34),
    "Celadon": (172, 225, 175),
    "Wisteria": (201, 160, 220),
    "Denim": (94, 138, 179),
    "Clay": (166, 123, 91)
}

FASHION_COLOR_MAP_LAB = {}
for name, rgb in FASHION_COLOR_MAP_RGB.items():
    bgr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)[0][0]
    FASHION_COLOR_MAP_LAB[name] = lab

def get_dominant_color(crop, num_colors=5, quality=10, center_region_percentage=0.5):
    """
    Extracts dominant RGB colors from a crop, prioritizing the center region.
    Returns a list of RGB tuples.
    """
    try:
        if crop is None or crop.size == 0:
            logger.warning("Input crop is empty or None.")
            return [(0,0,0)] * num_colors # Return black or a default color

        h, w, _ = crop.shape
        center_h = int(h * center_region_percentage)
        center_w = int(w * center_region_percentage)
        start_y = (h - center_h) // 2
        end_y = start_y + center_h
        start_x = (w - center_w) // 2
        end_x = start_x + center_w
        center_crop = crop[start_y:end_y, start_x:end_x]

        if center_crop.size == 0 or center_crop.shape[0] < 10 or center_crop.shape[1] < 10:
            logger.warning("Center crop is too small or invalid. Falling back to full crop for color detection.")
            target_crop = crop
        else:
            target_crop = center_crop

        target_crop_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(target_crop_rgb)
        
        temp_buffer = io.BytesIO()
        pil_image.save(temp_buffer, format="PNG")
        temp_buffer.seek(0)

        color_thief = ColorThief(temp_buffer)
        palette_rgb = color_thief.get_palette(color_count=num_colors, quality=quality)
        logger.debug(f"ColorThief Palette RGB: {palette_rgb}")

        # If ColorThief can't find enough colors, pad with black or most dominant
        while len(palette_rgb) < num_colors:
            palette_rgb.append(palette_rgb[0] if palette_rgb else (0,0,0))

        return palette_rgb

    except Exception as e:
        logger.error(f"Error detecting dominant RGB colors: {e}", exc_info=True)
        return [(0,0,0)] * num_colors # Return black or a default color on error

# --- Helper Function to Convert RGB to Color Names (for display/logging only) ---
def rgb_to_color_names(rgb_list, color_map_lab, max_colors=3):
    """
    Converts a list of RGB tuples to a list of approximate color names using LAB distance.
    Prioritizes distinct names and limits the output to max_colors.
    """
    named_colors = []
    seen_names = set()

    for rgb in rgb_list:
        rgb_array = np.uint8([[list(rgb)]])
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)[0][0] # Convert to LAB

        min_dist = float("inf")
        color_name = "Unknown"

        # Check for achromatic colors first (White, Black, Gray) using LAB values
        L, A, B = lab
        if L > 90 and abs(A) < 5 and abs(B) < 5: # High L, low A/B for white
            name_candidate = "White"
        elif L < 10 and abs(A) < 5 and abs(B) < 5: # Low L, low A/B for black
            name_candidate = "Black"
        elif abs(A) < 5 and abs(B) < 5: # Mid L, low A/B for gray
            name_candidate = "Gray"
        else:
            name_candidate = None

        if name_candidate and name_candidate not in seen_names:
            color_name = name_candidate
        else:
            # Fallback to map lookup for chromatic colors or if achromatic already seen
            for map_name, map_lab in color_map_lab.items():
                if map_name in ["White", "Black", "Gray"] and (L > 90 or L < 10 or (abs(A) < 5 and abs(B) < 5)):
                    continue 

                dist = np.linalg.norm(lab - np.array(map_lab)) # Calculate Euclidean distance in LAB
                if dist < min_dist:
                    min_dist = dist
                    color_name = map_name
        
        if color_name not in seen_names and len(named_colors) < max_colors:
            named_colors.append(color_name)
            seen_names.add(color_name)
        elif color_name in seen_names and len(named_colors) < max_colors:
            pass # Keep adding same color if palette has it and we haven't hit max distinct names
            
    while len(named_colors) < max_colors:
        named_colors.append(named_colors[0] if named_colors else "Unknown")

    return named_colors[:max_colors]

# --- NEW Helper Function for LAB Color Comparison ---
def compare_colors_by_lab(rgb_list1, rgb_list2, max_lab_distance=200.0):
    """
    Compares two lists of dominant RGB colors by converting them to LAB and calculating a bidirectional
    average minimum distance similarity. Returns a score from 0.0 (no similarity) to 1.0 (perfect similarity).
    
    Args:
        rgb_list1 (list): List of RGB tuples for the first image (e.g., [(255, 0, 0), ...]).
        rgb_list2 (list): List of RGB tuples for the second image.
        max_lab_distance (float): Maximum LAB distance for normalization (default 200.0).
    
    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    if not rgb_list1 or not rgb_list2:
        logger.warning("One or both RGB lists are empty. Returning similarity 0.0.")
        return 0.0

    try:
        # Convert RGB lists to LAB
        lab_list1 = [cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2LAB)[0][0] for rgb in rgb_list1]
        lab_list2 = [cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2LAB)[0][0] for rgb in rgb_list2]
        logger.debug(f"LAB colors 1: {lab_list1}")
        logger.debug(f"LAB colors 2: {lab_list2}")

        # Calculate minimum distances from lab_list1 to lab_list2
        min_distances_1to2 = []
        for lab1 in lab_list1:
            min_dist = min(np.linalg.norm(lab1 - lab2) for lab2 in lab_list2)
            min_distances_1to2.append(min_dist)
        
        # Calculate minimum distances from lab_list2 to lab_list1
        min_distances_2to1 = []
        for lab2 in lab_list2:
            min_dist = min(np.linalg.norm(lab2 - lab1) for lab1 in lab_list1)
            min_distances_2to1.append(min_dist)

        # Combine distances
        all_distances = min_distances_1to2 + min_distances_2to1
        if not all_distances:
            logger.warning("No valid LAB distances computed. Returning similarity 0.0.")
            return 0.0
        
        avg_min_distance = sum(all_distances) / len(all_distances)
        logger.debug(f"LAB distances: {all_distances}, Average: {avg_min_distance:.2f}")

        # Non-linear normalization using exponential decay
        # Similarity drops quickly for larger distances, reflecting perceptual differences
        similarity = np.exp(-avg_min_distance / (max_lab_distance / 2.0))
        similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        logger.debug(f"Color similarity: {similarity:.4f} (avg distance: {avg_min_distance:.2f}, max_lab_distance: {max_lab_distance})")

        return similarity

    except Exception as e:
        logger.error(f"Error in compare_colors_by_lab: {e}", exc_info=True)
        return 0.0
    
# --- Main Matching Function ---
def match_products(detections, index, product_info, product_id_to_indices, clip_model, preprocess, device):
    """Match detected objects to catalog products, targeting confidence >= 0.9 for exact and >= 0.75 for similar, excluding no_match. Color matching disabled."""
    logger.info("Matching products to detections")
    if not detections:
        logger.warning("No detections provided to match_products")
        return []
    
    logger.debug(f"Product ID to indices mapping: {product_id_to_indices}")
    logger.debug(f"Product info: {[p['id'] for p in product_info]}")
    logger.debug(f"FAISS index size: {index.ntotal} embeddings")
    
    matches = []
    product_info_dict = {p['id']: p for p in product_info}

    # Define thresholds and parameters
    MIN_CLIP_SIMILARITY = 0.25  # Low to allow more candidates
    MIN_COLOR_SIMILARITY = 0.7  # Low to allow more candidates
    TOTAL_SIMILARITY_EXACT_THRESHOLD = 0.90  # Target for exact matches
    TOTAL_SIMILARITY_SIMILAR_THRESHOLD = 0.75  # Target for similar matches
    TOP_K = 20  # Consider more candidates

    for i, detection in enumerate(detections):
        try:
            logger.debug(f"Processing detection {i}: class={detection['class']}, frame={detection['frame_number']}")
            crop = detection['crop']
            # crop_path = f"cropped_frames/crop_frame_{detection['frame_number']}_{detection['class']}_{i}.jpg"
            # cv2.imwrite(crop_path, crop) # Removed saving individual crops for every detection
            crop_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # # Dynamic weights based on object type
            if detection['class'] in ['dress', 'skirt', 'shirt']:
                WEIGHT_CLIP = 1   # Slightly favor color for clothing
                WEIGHT_COLOR = 0.1
            else:  # e.g., bag, shoe
                WEIGHT_CLIP = 1   # Slightly favor CLIP for accessories
                WEIGHT_COLOR = 0.05
            # WEIGHT_CLIP = 1.0

            # --- Visual (CLIP) Embedding & Search ---
            image_input = preprocess(crop_image).unsqueeze(0).to(device)
            with torch.no_grad():
                query_embedding = clip_model.encode_image(image_input).cpu().numpy()
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

            # Search for top-k matches
            distances, indices = index.search(query_embedding, k=TOP_K)
            logger.debug(f"Detection {i}: Top {TOP_K} FAISS indices: {indices[0]}, distances: {distances[0]}")

            # # --- Color Extraction for Detection ---
            detection_colors_rgb = get_dominant_color(crop)
            detection_colors_names = rgb_to_color_names(detection_colors_rgb, FASHION_COLOR_MAP_LAB, max_colors=3)
            logger.debug(f"Detection {i} dominant color names: {detection_colors_names}")

            # --- Evaluate Top-K Matches ---
            best_match = None
            best_total_similarity = -1.0
            best_clip_similarity = 0.0
            best_color_similarity = 0.0
            best_product_id = None
            best_match_type = "no_match"

            # Store all valid candidates for re-ranking
            candidates = []
            for j in range(TOP_K):
                if indices[0][j] == -1:
                    continue
                clip_similarity = distances[0][j]
                matched_index = indices[0][j]

                # Find corresponding product ID
                matched_product_id = None
                for product_id, idx_list in product_id_to_indices.items():
                    if matched_index in idx_list:
                        matched_product_id = product_id
                        break

                if matched_product_id is None:
                    logger.warning(f"No product ID found for matched index {matched_index}")
                    continue

                product = product_info_dict.get(matched_product_id)
                if not product:
                    logger.warning(f"No product info for product ID {matched_product_id}")
                    continue

                # # --- Color Similarity ---
                product_colors_rgb = product.get('dominant_colors_rgb', [])
                if not product_colors_rgb:
                    logger.warning(f"Product ID {matched_product_id} missing 'dominant_colors_rgb'")
                    color_similarity = 0.0
                else:
                    logger.debug(f"Product ID colors are {rgb_to_color_names(product_colors_rgb, FASHION_COLOR_MAP_LAB, max_colors=3) if product_colors_rgb else ['Unknown', 'Unknown', 'Unknown']} == {detection_colors_names}")
                    color_similarity = compare_colors_by_lab(detection_colors_rgb, product_colors_rgb, max_lab_distance=80.0)
                # color_similarity = 0.0  # Color similarity disabled

                # Apply minimum thresholds
                if clip_similarity < MIN_CLIP_SIMILARITY:  # Removed color threshold
                    logger.debug(f"Detection {i}: Index {matched_index} rejected (CLIP={clip_similarity:.4f})")
                    continue

                # Compute total similarity (now just CLIP similarity)
                total_similarity = WEIGHT_CLIP * clip_similarity  # Color weight is effectively 0
                total_similarity = min(total_similarity, 1.0)

                logger.debug(f"Detection {i}: Index {matched_index}, Product ID {matched_product_id}, "
                                f"CLIP Sim={clip_similarity:.4f}, Total Sim={total_similarity:.4f}")

                candidates.append({
                    "product_id": matched_product_id,
                    "product_type": product.get('product_type'),
                    "description": product.get('description'),
                    "product_tags": product.get('product_tags'),
                    "clip_similarity": clip_similarity,
                    "color_similarity": color_similarity,
                    "total_similarity": total_similarity,
                    "product_dominant_colors_rgb": product_colors_rgb,
                    "product_dominant_colors_names": rgb_to_color_names(product_colors_rgb, FASHION_COLOR_MAP_LAB)
                })
            
            # Re-rank candidates
            if candidates:
                candidates.sort(key=lambda x: x['total_similarity'], reverse=True)
                best_candidate = candidates[0]
                best_match = best_candidate
                best_total_similarity = best_candidate['total_similarity']
                best_clip_similarity = best_candidate['clip_similarity']
                best_color_similarity = best_candidate['color_similarity']
                best_product_id = best_candidate['product_id']

                if best_total_similarity >= TOTAL_SIMILARITY_EXACT_THRESHOLD:
                    best_match_type = "exact_match"
                elif best_total_similarity >= TOTAL_SIMILARITY_SIMILAR_THRESHOLD:
                    best_match_type = "similar_match"
                else:
                    best_match_type = "no_match" # Falls below similar threshold

            match_result = {
                "frame_number": detection['frame_number'],
                "detected_class": detection['class'],
                "bbox": detection['bbox'],
                "detection_confidence": detection['confidence'],
                "detection_dominant_colors_rgb": detection_colors_rgb,
                "detection_dominant_colors_names": detection_colors_names,
                "matched_product": {
                    "product_id": best_product_id,
                    "match_type": best_match_type,
                    "clip_similarity": float(f"{best_clip_similarity:.4f}") if best_match else 0.0,
                    "color_similarity": float(f"{best_color_similarity:.4f}") if best_match else 0.0,
                    "total_similarity": float(f"{best_total_similarity:.4f}") if best_match else 0.0,
                    "product_type": best_match['product_type'] if best_match else "N/A",
                    "description": best_match['description'] if best_match else "N/A",
                    "product_tags": best_match['product_tags'] if best_match else "N/A",
                    "dominant_colors_rgb": best_match['product_dominant_colors_rgb'] if best_match else [],
                    "dominant_colors_names": best_match['product_dominant_colors_names'] if best_match else []
                }
            }
            matches.append(match_result)
        except Exception as e:
            logger.error(f"Error processing detection {i}: {e}", exc_info=True)
            matches.append({
                "frame_number": detection['frame_number'],
                "detected_class": detection['class'],
                "bbox": detection['bbox'],
                "detection_confidence": detection['confidence'],
                "error": str(e)
            })
    return matches

def validate_json_output(data, schema_path="output_schema.json"):
    """Validates the output JSON against a predefined schema."""
    logger.info(f"Validating JSON output against schema: {schema_path}")
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        validate(instance=data, schema=schema)
        logger.info("JSON output validated successfully.")
        return True
    except FileNotFoundError:
        logger.error(f"Schema file not found at {schema_path}")
        return False
    except Exception as e:
        logger.error(f"JSON validation error: {e}")
        return False

# --- Main execution for all videos ---
def process_all_videos(data_dir="data", output_base_dir="output"):
    """
    Processes all videos in data/videos, extracts frames, detects objects,
    matches products, and saves results in separate video-named folders.
    """
    videos_dir = os.path.join(data_dir, "videos")
    captions_dir = os.path.join(data_dir, "captions")
    images_csv = os.path.join(data_dir, "images.csv")
    product_data_csv = os.path.join(data_dir, "product_data.csv")

    if not all(os.path.exists(d) for d in [videos_dir, captions_dir, images_csv, product_data_csv]):
        logger.error("Required data directories or files are missing. Please check 'data/' structure.")
        return

    # Setup FAISS index once for all videos
    try:
        index, product_info, product_id_to_indices, clip_model, preprocess, device = \
            setup_faiss_index(images_csv, product_data_csv)
    except Exception as e:
        logger.critical(f"Failed to set up FAISS index. Exiting: {e}")
        return

    yolo_model = YOLO("D:/Aadit/ML/Flickd/runs/detect/train3/weights/best.pt")

    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        logger.warning(f"No video files found in {videos_dir}. Exiting.")
        return

    for video_num, video_filename in enumerate(video_files):
        video_name = os.path.splitext(video_filename)[0]
        video_path = os.path.join(videos_dir, video_filename)
        caption_path = os.path.join(captions_dir, f"{video_name}.json") # Assuming caption file has same name as video

        video_output_dir = os.path.join(output_base_dir, f"video_{video_num + 1}")
        frames_output_dir = os.path.join(video_output_dir, "frames")
        detected_frames_output_dir = os.path.join(video_output_dir, "detected_frames")
        os.makedirs(frames_output_dir, exist_ok=True)
        os.makedirs(detected_frames_output_dir, exist_ok=True)

        logger.info(f"\n--- Processing Video: {video_filename} (Output to: {video_output_dir}) ---")

        # 1. Extract Frames
        if not extract_frames(video_path, frames_output_dir):
            logger.error(f"Skipping {video_filename} due to frame extraction failure.")
            continue

        # 2. Load Captions (if available)
        video_captions = {}
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r') as f:
                    video_captions = json.load(f)
                logger.info(f"Loaded captions for {video_filename}")
            except Exception as e:
                logger.error(f"Error loading captions for {video_filename}: {e}")
        else:
            logger.warning(f"No caption file found for {video_filename} at {caption_path}")

        all_detections = []
        frame_files = sorted([f for f in os.listdir(frames_output_dir) if f.endswith('.jpg')],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

        # 3. Detect Objects in Frames
        for frame_file in frame_files:
            frame_number = int(frame_file.split('_')[1].split('.')[0])
            frame_path = os.path.join(frames_output_dir, frame_file)
            detections_in_frame = detect_objects(frame_path, frame_number, yolo_model, detected_frames_output_dir)
            all_detections.extend(detections_in_frame)
            logger.info(f"Detected {len(detections_in_frame)} objects in frame {frame_number}")

        # 4. Match Products and Integrate Captions
        matched_results = match_products(all_detections, index, product_info, product_id_to_indices, clip_model, preprocess, device)

        final_output = []
        for result in matched_results:
            frame_number = result['frame_number']
            # Add caption if available for this frame
            result['caption'] = video_captions.get(str(frame_number), "No caption available for this frame.")
            final_output.append(result)

        # 5. Save Output
        output_json_path = os.path.join(video_output_dir, "output.json")
        try:
            with open(output_json_path, 'w') as f:
                json.dump(final_output, f, indent=4)
            logger.info(f"Analysis results saved to {output_json_path}")
            # 6. Validate Output
            if not validate_json_output(final_output):
                logger.warning(f"Validation failed for {output_json_path}")
        except Exception as e:
            logger.error(f"Failed to save or validate JSON for {video_filename}: {e}")

if __name__ == "__main__":
    process_all_videos(data_dir="data", output_base_dir="output")