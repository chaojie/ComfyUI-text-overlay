from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from typing import List, Optional, Union
from pydub import AudioSegment
import requests
import folder_paths
import os
import subprocess
import time
from datetime import datetime
from urllib.parse import urlencode
import shutil
import re

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

class PrepareStringForSchedule:
    #def __init__(self, device="cpu"):
    #    self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING",{"multiline": True, "default": "Hello"}),
            }
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", )
    FUNCTION = "prepare_string_for_schedule"
    CATEGORY = "image/text"
    
    def prepare_string_for_schedule(self, texts):
        ret=''
        ind=0
        for txt in texts:
            txt=txt.replace('"','')
            if len(txt.split(','))>30:
                txt=','.join(txt.split(',')[:30])
            ret=ret+f'"{ind}":"{txt}", '
            ind=ind+1
        return (ret,)
    
class GPTTextSchedule:
    #def __init__(self, device="cpu"):
    #    self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING",{"multiline": True, "default": "Hello"}),
                "framelen": ("INT", {"default": 16}),
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING","INT","STRING")
    FUNCTION = "gpt_text_schedule"
    CATEGORY = "image/text"
    
    def gpt_text_schedule(self, texts,framelen):
        prompts=[]
        captions=[]
        plist=re.split('\.|。|\n|，|,|\?|？|；|！',texts)
        for pitem in plist:
            pitem=pitem.strip()
            if pitem!='' and len(pitem)>=2:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + os.getenv('OPENAI_API_KEY', 'sk-wJrbjKZkhDHMRe3T5a7b212eEbD3484b9d720059BeC6C2Fd'),
                }

                json_data = {
                    'model': 'gpt-4',
                    'messages': [
                        {
                            'role': 'system',
                            'content': '从现在开始你是一个人工智能绘画提示词生成器。根据我的中文描述生成更详细的英文标签提示词，以逗号分隔。最重要的是，你只能输出不超过25个单词。',
                        },
                        {
                            'role': 'user',
                            'content': pitem,
                        },
                    ],
                }
                
                gpttimes=0
                while gpttimes>=0 and gpttimes<10:
                    try:
                        response = requests.post('https://gptapi.us/v1/chat/completions', headers=headers, json=json_data)
                        pitemgpt=response.json()['choices'][0]['message']['content']
                        pitemgpt=pitemgpt.replace('"','')
                        print(pitemgpt)
                        if len(pitemgpt.split(' '))>30:
                            pitemgpt=' '.join(pitemgpt.split(' ')[:30])
                        prompts.insert(len(prompts),pitemgpt.strip())
                        captions.insert(len(captions),pitem.strip())
                        gpttimes=-1
                    except:
                        gpttimes=gpttimes+1
                if gpttimes==10:
                    continue
                    
        finalprompts=''
        finalcaptions=''
        finalmaskprompts=''
        frame1prompts=''
        
        #framelen=16
        framecal=1
        moveprompts=['zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout','zoomin','zoomout']
        for i in range(0,len(prompts),1):
            finalcaptions=finalcaptions+f'"{i}": "{captions[i]}",'
            finalprompts=finalprompts+f'"{i*framecal*framelen}": "{prompts[i]},(Tang Dynasty style:0.5),masterpiece, best quality, highres, original, perfect lighting, extremely detailed wallpaper, (extremely detailed CG: 1.2)",'
            frame1prompts=frame1prompts+f'"{i*framecal*1}": "{prompts[i]},(Tang Dynasty style:0.5),masterpiece, best quality, highres, original, perfect lighting, extremely detailed wallpaper, (extremely detailed CG: 1.2)",'
            finalmaskprompts=finalmaskprompts+f'"{i*framecal*framelen}": "writing, {moveprompts[i]}",'
            
        return (finalprompts,finalcaptions,finalmaskprompts,len(prompts),frame1prompts,)
    
class ImageTextOverlay:
    #def __init__(self, device="cpu"):
    #    self.device = device
    _alignments = ["left", "right", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "texts": ("STRING",{"multiline": True, "default": "Hello"}),
                "watermark": ("STRING",{"multiline": True, "default": "Hello"}),
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
        
    def draw_text_on_image(self, images, texts, watermark, font_size, x, y, font, alignment, r, g, b):
        # convert images to numpy
        #return images
        frames: List[Image.Image] = []
        outframes=[]
        ind=0
        for image in images:
            text=texts[int(ind/(len(images)/len(texts)))]
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            frames.append(image)
            #print(len(frames))
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
            loaded_font1 = ImageFont.truetype(font, 16)
    
            # Prepare to draw on image
            draw = ImageDraw.Draw(image)
    
            # Adjust x coordinate based on alignment
            w=image.width
            h=image.height
            #print(w)
            #print(h)
            text_width = draw.textlength(text, font=loaded_font)#text_width, text_height = draw.textsize(text, font=loaded_font)
            if alignment == "center":
                x = w/2-text_width/2
            elif alignment == "right":
                x = w-text_width
    
            # Draw text on the image
            draw.text((x, y), text, fill=color_rgb, font=loaded_font)
            draw.text((25, 25), watermark, fill=(255, 255, 255), font=loaded_font1)
    
    
            # Convert back to Tensor if needed
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)  # Convert back to CxHxW
            image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
            
            #return (image_tensor_out,)
            outframes.append(image_tensor_out)
            ind=ind+1
    
        return torch.cat(tuple(outframes), dim=0).unsqueeze(0)


class Image2AudioVideo:
    #def __init__(self, device="cpu"):
    #    self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "texts": ("STRING",{"multiline": True, "default": "Hello"}),
                "baidu_akey": ("STRING", {"default": ""}),
                "baidu_skey": ("STRING", {"default": ""}),
                "bgaudio": ("STRING", {"default": "bg.m4a"}),
                "fps": ("INT", {"default": 8}),
                "outfile": ("STRING", {"default": "lc_ad_"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "image_to_audiovideo"
    CATEGORY = "image/text"
        
    def image_to_audiovideo(self, images, texts, baidu_akey,baidu_skey,bgaudio,fps,outfile):
        output_dir = folder_paths.get_output_directory()
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(outfile, output_dir)
        counter=1
        file3 = f"{filename}_{counter:05}.mp3"
        file4 = f"{filename}_{counter:05}.mp4"
        sound_path = os.path.join(full_output_folder, file3)
        video_path = os.path.join(full_output_folder, file4)
        output_video_path = os.path.join(full_output_folder, f"{filename}.mp4")

        tmpdir=os.path.join(full_output_folder,filename)
        #os.rmdir(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)
        os.makedirs(tmpdir, exist_ok=True)
        ind=0
        for image in images:
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            tmpfile=os.path.join(tmpdir, f'{ind:05}.png')
            image.save(tmpfile)
            ind=ind+1
            
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),  # Set the framerate for the input files
            #'-pattern_type', 'glob',  # Enable pattern matching for filenames
            '-i', f'{tmpdir}/%5d.png',  # Input files pattern
            '-c:v', 'libx264',  # Set the codec for video
            '-pix_fmt', 'yuv420p',  # Set the pixel format
            '-crf', '17',  # Set the constant rate factor for quality
            video_path  # Output file
        ]
    
        # Run the ffmpeg command
        subprocess.run(cmd)
        
        diff_time=0
          
        while diff_time<10:
              time.sleep(10)
              output_video_path_mtime=int(os.path.getmtime(video_path))
              now_time=int(datetime.now().timestamp())
              diff_time=now_time-output_video_path_mtime
        time.sleep(2)
        
        #tmpdir=os.path.join(full_output_folder,filename)
        #os.makedirs(tmpdir, exist_ok=True)
        
        soundbg = AudioSegment.from_file(bgaudio)
        soundbg=soundbg*10
        #soundbg.export(sound_path, format="mp3")
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": baidu_akey, "client_secret": baidu_skey}
        access_token=str(requests.post(url, params=params).json().get("access_token"))
        
        url = "https://tsn.baidu.com/text2audio"
        pretext=""
        itxt=0
        for txt in texts:
            #print(txt)
            #if pretext!=txt:
            pretext=txt
            payload=urlencode({"tex":txt})+'&tok='+ access_token +'&cuid=lGt9yI9bVNxg8XpAIqLK5mXSUJ4zEHDB&ctp=1&lan=zh&spd=5&pit=5&vol=5&per=5003&aue=3'
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': '*/*'
            }
            print(payload)
            soundEx=1
            while soundEx==1:
                try:
                    resp = requests.request("POST", url, headers=headers, data=payload)
                    soundFile=f'output/{outfile}{itxt}.mp3'
                    with open(soundFile,'wb') as f:
                        f.write(resp.content)
                    soundcaption = AudioSegment.from_file(soundFile)

                    soundbg=soundbg.overlay(soundcaption, position=1000*itxt*len(images)/(len(texts)*fps))
                    soundEx=0
                except:
                    soundEx=1
            itxt=itxt+1
        #print(1000*len(texts)/fps)
        soundbg=soundbg[:1000*len(images)/fps]
        soundbg.export(sound_path, format="mp3")
        print(sound_path)
        cmd = [
              'ffmpeg',
              '-y',  # Overwrite output file if it exists
              '-i', video_path,  # Input files pattern
              '-i', sound_path,  # Input files pattern
              #'-filter_complex', "[v][a]concat=n=1:v=1:a=1",  # Input files pattern
              '-c:v', 'libx264',  # Set the codec for video
              '-c:a', 'aac',  # Set the codec for video
              '-movflags', '+faststart',  # Set the pixel format
              output_video_path  # Output file
          ]
  
          # Run the ffmpeg command
        subprocess.run(cmd)
        diff_time=0
          
        while diff_time<10:
              time.sleep(10)
              output_video_path_mtime=int(os.path.getmtime(output_video_path))
              now_time=int(datetime.now().timestamp())
              diff_time=now_time-output_video_path_mtime
        time.sleep(2)
        '''
        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output",
                "format": 'video/mp4',
            }
        ]
        '''
        return (output_video_path,)

NODE_CLASS_MAPPINGS = {
    "Image Text Overlay": ImageTextOverlay,
    "Image Audio Video": Image2AudioVideo,
    "GPT Text Schedule": GPTTextSchedule,
    "Prepare String For Schedule": PrepareStringForSchedule,
}
