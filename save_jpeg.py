"""Save images as JPEG so large outputs stay under the RunPod response size limit."""
import os

import numpy as np
from PIL import Image

import folder_paths


class SaveImageJPEG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 92, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", quality=92):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = []
        for image in images:
            pixels = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
            file = f"{filename}_{counter:05}_.jpg"
            Image.fromarray(pixels).convert("RGB").save(
                os.path.join(full_output_folder, file), format="JPEG", quality=quality, optimize=True
            )
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"SaveImageJPEG": SaveImageJPEG}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImageJPEG": "Save Image (JPEG)"}
