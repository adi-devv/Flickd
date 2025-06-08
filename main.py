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
        results = model.predict(source=image_path, conf=0.4, save=False, line_width=2)
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
    
def setup_faiss_index(images_csv, product_data_csv, id_column="id", cache_dir="data/cache", max_product_ids=10):
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
def compare_colors_by_lab(rgb_list1, rgb_list2, max_lab_distance=100.0):
    """
    Compares two lists of dominant RGB colors by converting them to LAB and
    calculating an average minimum distance similarity.
    Returns a score from 0.0 (no similarity) to 1.0 (perfect similarity).
    """
    if not rgb_list1 or not rgb_list2:
        return 0.0

    lab_list1 = [cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2LAB)[0][0] for rgb in rgb_list1]
    lab_list2 = [cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2LAB)[0][0] for rgb in rgb_list2]

    # Calculate average minimum distance from list1 to list2
    min_distances_sum = 0
    count = 0
    for lab1 in lab_list1:
        min_dist_for_lab1 = float('inf')
        for lab2 in lab_list2:
            dist = np.linalg.norm(lab1 - lab2) # Euclidean distance in LAB
            if dist < min_dist_for_lab1:
                min_dist_for_lab1 = dist
        min_distances_sum += min_dist_for_lab1
        count += 1
    
    if count == 0:
        return 0.0
        
    avg_min_distance = min_distances_sum / count

    # Convert distance to similarity (0 to 1)
    # A max_lab_distance of ~100-150 is typical for significant differences.
    # The maximum possible delta E 2000 is ~100, but for Euclidean in Lab it can be higher (e.g., ~200-250 for black to white).
    # We cap it at 100 for normalization to make similarity scale more meaningfully.
    similarity = 1.0 - (avg_min_distance / max_lab_distance)
    similarity = max(0.0, similarity) # Ensure similarity is not negative
    
    return similarity

# --- Main Matching Function ---
def match_products(detections, index, product_info, product_id_to_indices, clip_model, preprocess, device):
    """Match detected objects to catalog products, targeting confidence >= 0.9 for exact and >= 0.75 for similar, excluding no_match."""
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
    MIN_COLOR_SIMILARITY = 0.15  # Low to allow more candidates
    TOTAL_SIMILARITY_EXACT_THRESHOLD = 0.90  # Target for exact matches
    TOTAL_SIMILARITY_SIMILAR_THRESHOLD = 0.6  # Target for similar matches
    TOP_K = 20  # Consider more candidates

    for i, detection in enumerate(detections):
        try:
            logger.debug(f"Processing detection {i}: class={detection['class']}, frame={detection['frame_number']}")
            crop = detection['crop']
            crop_path = f"cropped_frames/crop_frame_{detection['frame_number']}_{detection['class']}_{i}.jpg"
            cv2.imwrite(crop_path, crop)
            crop_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # Dynamic weights based on object type
            if detection['class'] in ['dress', 'skirt', 'shirt']:
                WEIGHT_CLIP = 0.45  # Slightly favor color for clothing
                WEIGHT_COLOR = 0.55
            else:  # e.g., bag, shoe
                WEIGHT_CLIP = 0.55  # Slightly favor CLIP for accessories
                WEIGHT_COLOR = 0.45

            # --- Visual (CLIP) Embedding & Search ---
            image_input = preprocess(crop_image).unsqueeze(0).to(device)
            with torch.no_grad():
                query_embedding = clip_model.encode_image(image_input).cpu().numpy()
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

            # Search for top-k matches
            distances, indices = index.search(query_embedding, k=TOP_K)
            logger.debug(f"Detection {i}: Top {TOP_K} FAISS indices: {indices[0]}, distances: {distances[0]}")

            # --- Color Extraction for Detection ---
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

                # --- Color Similarity ---
                product_colors_rgb = product.get('dominant_colors_rgb', [])
                if not product_colors_rgb:
                    logger.warning(f"Product ID {matched_product_id} missing 'dominant_colors_rgb'")
                    color_similarity = 0.0
                else:
                    logger.debug(f"Product ID colors are {rgb_to_color_names(product_colors_rgb, FASHION_COLOR_MAP_LAB, max_colors=3) if product_colors_rgb else ["Unknown", "Unknown", "Unknown"]} == {detection_colors_names}")
                    
                    color_similarity = compare_colors_by_lab(detection_colors_rgb, product_colors_rgb, max_lab_distance=80.0)
                
                # Apply minimum thresholds
                if clip_similarity < MIN_CLIP_SIMILARITY or color_similarity < MIN_COLOR_SIMILARITY:
                    logger.debug(f"Detection {i}: Index {matched_index} rejected (CLIP={clip_similarity:.4f}, Color={color_similarity:.4f})")
                    continue

                # Compute total similarity
                total_similarity = (WEIGHT_CLIP * clip_similarity) + (WEIGHT_COLOR * color_similarity)
                total_similarity = min(total_similarity, 1.0)

                logger.debug(f"Detection {i}: Index {matched_index}, Product ID {matched_product_id}, "
                            f"CLIP Sim={clip_similarity:.4f}, Color Sim={color_similarity:.4f}, Total Sim={total_similarity:.4f}")

                candidates.append({
                    "clip_similarity": clip_similarity,
                    "color_similarity": color_similarity,
                    "total_similarity": total_similarity,
                    "product_id": matched_product_id,
                    "match_type": "exact" if total_similarity > TOTAL_SIMILARITY_EXACT_THRESHOLD else \
                                  "similar" if total_similarity >= TOTAL_SIMILARITY_SIMILAR_THRESHOLD else "no_match"
                })

            # Re-rank candidates to prioritize high CLIP and color similarity
            if candidates:
                # Sort by total_similarity, prioritizing balanced CLIP and color similarity
                candidates = sorted(
                    candidates,
                    key=lambda x: (x["total_similarity"], min(x["clip_similarity"], x["color_similarity"])),
                    reverse=True
                )
                best_candidate = candidates[0]
                best_total_similarity = best_candidate["total_similarity"]
                best_clip_similarity = best_candidate["clip_similarity"]
                best_color_similarity = best_candidate["color_similarity"]
                best_product_id = best_candidate["product_id"]
                best_match_type = best_candidate["match_type"]

                if best_match_type in ["exact", "similar"]:
                    best_match = {
                        "type": detection['class'],
                        "colors": detection_colors_names,
                        "match_type": best_match_type,
                        "matched_product_id": str(best_product_id),
                        "confidence": float(best_total_similarity),
                        "clip_confidence": float(best_clip_similarity),
                        "color_confidence": float(best_color_similarity),
                        "crop_image_file": crop_path,
                        "matched_product_colors": rgb_to_color_names(product_colors_rgb, FASHION_COLOR_MAP_LAB, max_colors=3) if product_colors_rgb else ["Unknown", "Unknown", "Unknown"]                    }

            # Append only if a valid match is found
            if best_match and best_match_type in ["exact", "similar", "no_match"]:
                matches.append(best_match)
                logger.debug(f"Match found: product_id={best_product_id}, match_type={best_match_type}, "
                            f"total_confidence={best_total_similarity:.4f}")
            else:
                logger.debug(f"Skipping detection {i}: no_match, total_confidence={best_total_similarity:.4f}, "
                            f"clip_confidence={best_clip_similarity:.4f}, color_confidence={best_color_similarity:.4f}")

        except Exception as e:
            logger.error(f"Error matching product for detection {i}: {e}", exc_info=True)
            matches.append({
                "type": detection['class'],
                "colors": ["Unknown", "Unknown", "Unknown"],
                "match_type": "error",
                "matched_product_id": None,
                "confidence": 0.0,
                "clip_confidence": 0.0,
                "color_confidence": 0.0,
                "crop_image_file": crop_path
            })

    logger.debug(f"Final matches: {matches}")
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
                        "colors": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
                        "match_type": {"type": "string", "enum": ["exact", "similar", "no_match"]},
                        "matched_product_id": {"type": ["string", "null"]},
                        "confidence": {"type": "number"}
                    },
                    "required": ["type", "colors", "match_type", "matched_product_id", "confidence"]
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
    caption = '''darling How would you style this Where comfort meets chic.'''
    video_id = "sample_video2"
    output_json_path = f"outputs/output_{video_id}.json"
    vibe_taxonomy = ["Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"]
    
    main(video_path, images_csv, product_data_csv, caption, video_id, output_json_path, vibe_taxonomy)