"""Upload generated images to Salad Simple Storage Service (S4).

The RunPod worker only returns outputs listed under "images", so this node does not
try to send the URL back. The caller chooses the storage path in the workflow and
therefore already knows where the file will be:

    https://storage-api.salad.com/organizations/<SALAD_ORG>/files/<path>

Required environment variables on the endpoint: SALAD_API_KEY, SALAD_ORG.
"""
import io
import os

import numpy as np
import requests
from PIL import Image

S4_BASE = "https://storage-api.salad.com"
MAX_BYTES = 100 * 1024 * 1024  # S4 limit per file
FORMATS = {
    "jpeg": ("JPEG", "image/jpeg", ".jpg"),
    "png": ("PNG", "image/png", ".png"),
    "webp": ("WEBP", "image/webp", ".webp"),
}


class SaveImageS4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"default": "h3/output"}),
                "format": (list(FORMATS.keys()), {"default": "jpeg"}),
                "quality": ("INT", {"default": 92, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def upload(self, images, path="h3/output", format="jpeg", quality=92):
        api_key = os.environ.get("SALAD_API_KEY")
        org = os.environ.get("SALAD_ORG")
        if not api_key or not org:
            raise RuntimeError("SaveImageS4: SALAD_API_KEY and SALAD_ORG must be set")

        pil_format, mime, ext = FORMATS[format]
        base = path.strip("/")
        for known in (".jpg", ".jpeg", ".png", ".webp"):
            if base.lower().endswith(known):
                base = base[: -len(known)]
                break

        urls = []
        for index, image in enumerate(images):
            pixels = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
            buffer = io.BytesIO()
            picture = Image.fromarray(pixels)
            if pil_format == "PNG":
                picture.save(buffer, format="PNG", compress_level=4)
            else:
                picture.convert("RGB").save(buffer, format=pil_format, quality=quality)
            data = buffer.getvalue()
            if len(data) > MAX_BYTES:
                raise RuntimeError(
                    f"SaveImageS4: {len(data) / 1e6:.1f} MB exceeds the 100 MB S4 limit; use jpeg or webp"
                )

            # One image keeps the exact path; a batch gets _1, _2, ... suffixes.
            name = f"{base}{ext}" if len(images) == 1 else f"{base}_{index + 1}{ext}"
            url = f"{S4_BASE}/organizations/{org}/files/{name}"
            response = requests.put(
                url,
                headers={"Salad-Api-Key": api_key},
                data={"mimeType": mime},
                files={"file": (os.path.basename(name), data, mime)},
                timeout=120,
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"SaveImageS4: upload failed ({response.status_code}): {response.text[:300]}"
                )
            print(f"[s4] uploaded {name} ({len(data) / 1e6:.2f} MB)")
            urls.append(url)

        return {"ui": {"s4_urls": urls}}


NODE_CLASS_MAPPINGS = {"SaveImageS4": SaveImageS4}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImageS4": "Save Image (Salad S4)"}
