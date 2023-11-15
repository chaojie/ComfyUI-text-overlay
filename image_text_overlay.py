from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from typing import List, Optional, Union
    
# region TENSOR Utilities
def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageTextOverlay:
    def __init__(self, device="cpu"):
        self.device = device
    _alignments = ["left", "right", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING",{"multiline": True, "default": "Hello"}),
                "font_size": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "x": ("INT", {"default": 0}),
                "y": ("INT", {"default": 0}),
                "font": ("STRING", {"default": "wryh.ttf"}),  # Assuming it's a path to a .ttf or .otf file
                "alignment": (cls._alignments, {"default": "left"}),  # ["left", "right", "center"]
              #"color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                "r": ("INT", {"default": 0}),
                "g": ("INT", {"default": 0}),
                "b": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_text_on_image"
    CATEGORY = "image/text"
        
    def draw_text_on_image(self, images, text, font_size, x, y, font, alignment, r, g, b):
        # convert images to numpy
        #return images
        frames: List[Image.Image] = []
        outframes=[]
        for image in images:
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            frames.append(image)
            print(len(frames))
            # Convert tensor to numpy array and then to PIL Image
            #image_tensor = image
            #image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
            #image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image
    
            # Convert color from INT to RGB tuple
            #r = (color >> 16) & 0xFF
            #g = (color >> 8) & 0xFF
            #b = color & 0xFF
            color_rgb = (r, g, b)
    
            # Load font
            loaded_font = ImageFont.truetype(font, font_size)
    
            # Prepare to draw on image
            draw = ImageDraw.Draw(image)
    
            # Adjust x coordinate based on alignment
            w=image.width
            h=image.height
            print(w)
            print(h)
            text_width, text_height = draw.textsize(text, font=loaded_font)
            if alignment == "center":
                x = w/2-text_width/2
            elif alignment == "right":
                x = w-text_width
    
            # Draw text on the image
            draw.text((x, y), text, fill=color_rgb, font=loaded_font)
    
    
            # Convert back to Tensor if needed
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)  # Convert back to CxHxW
            image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
            
            #return (image_tensor_out,)
            outframes.append(image_tensor_out)
    
        return torch.cat(tuple(outframes), dim=0).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "Image Text Overlay": ImageTextOverlay,
}
