import json
import os
import torch
import numpy as np
from PIL import Image, ImageOps

def read_json_file(file_path):
    """
    Reads a JSON file's content and returns it.
    Ensures content matches the expected format.
    """
    if not os.access(file_path, os.R_OK):
        print(f"Warning: No read permissions for file {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
            # Check if the content matches the expected format.
            if not all(['name' in item and 'prompt' in item and 'negative_prompt' in item for item in content]):
                print(f"Warning: Invalid content in file {file_path}")
                return None
            return content
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {str(e)}")
        return None
    
def read_sdxl_styles(json_data):
    """
    Returns style names from the provided JSON data.
    """
    if not isinstance(json_data, list):
        print("Error: input data must be a list")
        return []

    return [item['name'] for item in json_data if isinstance(item, dict) and 'name' in item]

def find_template_by_name(json_data, template_name):
    """
    Returns a template from the JSON data by name or None if not found.
    """
    for template in json_data:
        if template['name'] == template_name:
            return template
    return None

def replace_prompts_in_template(template, positive_prompt, negative_prompt):
    """
    Replace the placeholders in a given template with the provided prompts.
    
    Args:
    - template (dict): The template containing prompt placeholders.
    - positive_prompt (str): The positive prompt to replace '{prompt}' in the template.
    - negative_prompt (str): The negative prompt to be combined with any existing negative prompt in the template.
 
    Returns:
    - tuple: A tuple containing the replaced positive and negative prompts.
    """
    positive_result = template['prompt'].replace('{prompt}', positive_prompt)

    json_negative_prompt = template.get('negative_prompt', "")
    negative_result = f"{json_negative_prompt}, {negative_prompt}" if json_negative_prompt and negative_prompt else json_negative_prompt or negative_prompt

    return positive_result, negative_result

def load_image(filename: str):
    # 获取 images 文件夹路径
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
    
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Directory '{folder_path}' does not exist.")
    
    # 构造完整的文件路径
    file_path = os.path.join(folder_path, filename)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{filename}' does not exist in the '{folder_path}' directory.")

    # 加载并处理图片
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)  # 处理 EXIF 旋转
    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")  # 转换颜色模式

    # 转换为 NumPy 数组并归一化
    img = np.array(img, dtype=np.float32) / 255.0

    # 检查图像是否是二维（单通道灰度图），如果是则扩展维度
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)  # 扩展为 [H, W, 1]
    
    # 确保图像通道数为3或4
    if img.shape[-1] not in [3, 4]:
        raise ValueError(f"Unexpected number of channels: {img.shape[-1]}")

    # 转换为 PyTorch 张量，并调整形状为 [C, H, W]
    tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [C, H, W] 并添加 batch 维度
    
    return tensor_img  # 返回 PyTorch 张量

class OlympicsPrompt:

    sex_dict = { '男': 'man', '女': 'female', '男孩': 'boy', '女孩': 'girl' }

    def __init__(self):
        # self.dir = os.path.dirname(os.path.realpath(__file__))
        print(os.path.dirname(os.path.realpath(__file__)))
        pass

    @classmethod
    def INPUT_TYPES(self):
        self.prompts = read_json_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prompts.json'))
        
        return {
            "required": {
                "sports": ([item['name'] for item in self.prompts],),
                "sex": (list(self.sex_dict.keys()),),
                "signature": ("STRING", { "default": "Rainbow" })
            },
        }

    RETURN_TYPES = ('STRING', 'IMAGE', 'STRING')
    RETURN_NAMES = ('prompt', 'bg_image', 'signature')
    FUNCTION = 'prompt_styler'
    CATEGORY = 'utils'

    def prompt_styler(self, sports, sex, signature):
        # Process and combine prompts in templates
        # The function replaces the positive prompt placeholder in the template,
        # and combines the negative prompt with the template's negative prompt, if they exist.
        template = find_template_by_name(self.prompts, sports)
        text_positive_styled, text_negative_styled = replace_prompts_in_template(template, self.sex_dict[sex], '')

        background_img = load_image(f"{sports}.png")

        return text_positive_styled, background_img, signature
    
NODE_CLASS_MAPPINGS = {
    "OlympicsPrompt": OlympicsPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OlympicsPrompt": "OlympicsPrompt-PAI",
}
