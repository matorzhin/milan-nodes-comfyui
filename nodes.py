# by milan. Milan M.
# t.me/milandsgn
import os
import json
import hashlib
import numpy as np
import torch
from PIL import Image, ImageSequence, ImageOps, ExifTags
import folder_paths
import node_helpers

#! GLOBALS
NODES_FILE = os.path.abspath(__file__)
NODES_ROOT = os.path.dirname(NODES_FILE)
DATABASE = os.path.join(NODES_ROOT, 'database.json')

DATABASE = os.path.join(os.path.dirname(__file__), "database.json")

class Database:
    """
    This class is currently based on DATABASE class from WAS-node-suite. All credits go to the author <3
    """

    def __init__(self, filepath):
        self.filepath = filepath
        try:
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

    def _load(self):
        """Load data from JSON file. If the file doesn't exist, create an empty data structure."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON file {self.filepath}. Resetting database.")
        return {}

    def _save(self):
        """Save current data to the JSON file."""
        try:
            with open(self.filepath, 'w') as file:
                json.dump(self.data, file, indent=4)
        except Exception as e:
            print(f"Error: Failed to save database. Reason: {str(e)}")

    def category_exists(self, category):
        """Check if a category exists."""
        return category in self.data

    def key_exists(self, category, key):
        """Check if a key exists in a given category."""
        return self.category_exists(category) and key in self.data[category]

    def insert(self, category, key, value):
        """Insert a key-value pair into a category."""
        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()

    def update(self, category, key, value):
        """Update an existing key-value pair."""
        if self.key_exists(category, key):
            self.data[category][key] = value
            self._save()
        else:
            raise KeyError(f"Key '{key}' not found in category '{category}'.")

    def delete(self, category, key):
        """Delete a key-value pair."""
        if self.key_exists(category, key):
            del self.data[category][key]
            self._save()

    def get(self, category, key, default=None):
        """Retrieve a value for a given key in a category. Return default if not found."""
        return self.data.get(category, {}).get(key, default)

    def get_category(self, category):
        """Return all key-value pairs in a category."""
        if self.category_exists(category):
            return self.data[category]
        else:
            return {}

    def insert_category(self, category):
        """Create an empty category."""
        if not self.category_exists(category):
            self.data[category] = {}
            self._save()

class BaseImageLoader:
    @staticmethod
    def decode_exif_value(value):
        if isinstance(value, bytes):
            try:
                if b'\x00' in value:
                    return value.decode('utf-16-le').strip('\x00').strip()
                return value.decode('utf-8').strip('\x00').strip()
            except UnicodeDecodeError:
                return value.decode('latin-1', errors='ignore').strip('\x00').strip()
        return str(value).strip() if value else ''

    def extract_metadata(self, img):
        description = ""
        title = ""
        
        try:
            exif = img.getexif()
            if exif:
                # Обработка XP тегов
                xp_title_tag = 40090
                if xp_title_tag in exif:
                    title = self.decode_exif_value(exif[xp_title_tag])
                
                exif_tags = {
                    270: 'description',
                    37510: 'description',
                    40091: 'title',
                    40092: 'description'
                }

                for tag_id, meta_type in exif_tags.items():
                    if tag_id in exif:
                        value = self.decode_exif_value(exif[tag_id])
                        if value:
                            if meta_type == 'description' and not description:
                                description = value
                            elif meta_type == 'title' and not title:
                                title = value

            if hasattr(img, 'text'):
                if not description:
                    for field in ['Description', 'comment']:
                        if field in img.text:
                            description = img.text[field].strip()
                            break
                
                if not title:
                    for field in ['Title', 'Subject']:
                        if field in img.text:
                            title = img.text[field].strip()
                            break

        except Exception as e:
            description = f"Metadata error: {str(e)}"

        return title.strip(), description.strip()

    def process_image(self, img):
        img = ImageOps.exif_transpose(img)
        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image)[None,]
    
    def _load_image_data(self, image_path, need_mask=False):
        img = Image.open(image_path)
        title, description = self.extract_metadata(img)

        output_images = []
        output_masks = [] if need_mask else None
        excluded_formats = ['MPO']
        base_size = None

        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            
            if frame.mode == 'I':
                frame = frame.point(lambda i: i * (1 / 255))
            
            rgb_frame = frame.convert("RGB")
            

            if not base_size:
                base_size = rgb_frame.size
            elif rgb_frame.size != base_size:
                continue

  
            image_array = np.array(rgb_frame).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(image_array)[None,])
         
            if need_mask:
                mask = torch.zeros((base_size[1], base_size[0]), dtype=torch.float32)
                if 'A' in frame.getbands():
                    alpha = np.array(frame.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(alpha)
                output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            image_tensor = torch.cat(output_images, dim=0)
            mask_tensor = torch.cat(output_masks, dim=0) if need_mask else None
        else:
            image_tensor = output_images[0]
            mask_tensor = output_masks[0] if need_mask else None

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': os.path.splitext(os.path.basename(image_path))[0],
            'title': title or "",
            'description': description or ""
        }

class LoadImageExtended(BaseImageLoader):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "filename", "title", "description")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        data = self._load_image_data(image_path, need_mask=True)
        
        return (
            data['image'],
            data['mask'],
            data['filename'],
            data['title'],
            data['description']
        )

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class LoadImagesExtended(BaseImageLoader):
    db = Database(DATABASE)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"input_dir": ("STRING", {"default": '', "multiline": False})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "filename", "directory", "title", "description")
    FUNCTION = "load_images_batch"

    def load_images_batch(self, input_dir):
        if not os.path.exists(input_dir):
            raise ValueError(f"Directory '{input_dir}' does not exist.")

        valid_ext = {'.png', '.jpg', '.jpeg'}
        files = sorted([
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
            and os.path.isfile(os.path.join(input_dir, f))
        ])

        if not files:
            raise ValueError("No supported images found.")

        current_index = self.db.get("images", "current_index", default=0)
        current_index = current_index % len(files)

        image_filename = files[current_index]
        image_path = os.path.join(input_dir, image_filename)
        data = self._load_image_data(image_path, need_mask=False)

        new_index = (current_index + 1) % len(files)
        self.db.insert("images", "current_index", new_index)

        return (
            data['image'],
            data['filename'],
            input_dir, 
            data['title'],
            data['description']
        )
    @classmethod
    def IS_CHANGED(s, input_dir):
        current_index = s.db.get("images", "current_index", default=0)
        return hashlib.sha256(f"{input_dir}_{current_index}".encode()).hexdigest()
    @classmethod
    def VALIDATE_INPUTS(s, input_dir):
        if not os.path.isdir(input_dir):
            return f"Not a directory: {input_dir}"
        return True

NODE_CLASS_MAPPINGS = {
    "LoadOneImageExtended": LoadImageExtended,
    "LoadMultipleImagesExtended": LoadImagesExtended
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOneImageExtended": "Load One Image with Name, Title, Description",
    "LoadMultipleImagesExtended":  "Load Multiple Images with Name, Directory, Title, Description"
}

# by milan. Milan M.
# t.me/milandsgn