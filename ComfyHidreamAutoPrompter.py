import os
import requests
import json
import time
import logging
from PIL import Image, PngImagePlugin
import numpy as np
import base64
import io # Added from script 1
import random
import torch # Keep for CLIP and other torch operations if any

# Transformers and Diffusers imports - potentially remove if HiDream pipe is fully replaced
# from transformers import PreTrainedTokenizerFast, LlamaForCausalLM # Keep if used by LLM or other parts
# from diffusers import HiDreamImagePipeline # This will be replaced for image generation

import open_clip
import tiktoken

# Set up logging (from script 2)
logging.basicConfig(filename='iterative_image_generation_with_comfyui.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Paths from script 2
image_dir = "Output/Folder_01/"
memory_file_path = "Memories/prompt_memory_01.json"
os.makedirs(image_dir, exist_ok=True)
os.makedirs("Memories", exist_ok=True)

# LMStudio server URL (from script 2)
server_url = "http://127.0.0.1:1234/v1/chat/completions" # For LLM calls

# --- ComfyUI API Configuration (from script 1) ---
COMFY_BASE_URL = "http://127.0.0.1:8188"
COMFY_PROMPT_URL = f"{COMFY_BASE_URL}/prompt"

# Themes list (expanded - excluding explicit content)
THEMES = [
    "ethereal dreamscapes",
    "futuristic cityscapes at dawn",
    "ancient enchanted forests",
    "surreal abstract geometry",
    "steampunk mechanical creatures",
    "art nouveau portraits",
    "bioluminescent underwater worlds",
    "minimalist architectural designs",
    "cosmic nebulas and galaxies",
    "vintage travel posters style",
    "cyberpunk noir alleyways",
    "impressionistic nature scenes",
    "detailed botanical illustrations",
    "whimsical fantasy characters",
    "serene zen gardens"
]

# Load OpenCLIP model and tokenizer (from script 2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = None
clip_tokenizer = None
preprocess = None
CLIP_AVAILABLE = False
try:
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=device)
    clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
    CLIP_AVAILABLE = True
    print("CLIP model loaded successfully.")
except Exception as e:
    print(f"ERROR: Error loading CLIP model: {e}. CLIP scoring will be unavailable.")
    CLIP_AVAILABLE = False

from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn

from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip  # Import the aesthetic predictor

# Load AVA model for scoring
AVA_AVAILABLE = False
ava_model = None
ava_preprocessor = None

try:
    # Load model and preprocessor
    ava_model, ava_preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    ava_model = ava_model.to(torch.bfloat16).cuda()  # Move model to GPU
    AVA_AVAILABLE = True
    print("Aesthetic predictor (v2.5) loaded successfully.")
except Exception as e:
    print(f"ERROR: Error loading aesthetic predictor: {e}. AVA scoring will be unavailable.")
    AVA_AVAILABLE = False

# --- Main HiDream Model Path Variable --- diffusion_models folder
MAIN_MODEL_NAME = "hidream_i1_fast_fp8.safetensors"  # Main HiDream model

# --- CLIP Model Path Variables (for ComfyUI workflow "in comfy") ---
# clip_l: Local CLIP-L model for main text encoding "text_encoders" folder
CLIP_NAME1 = "clip_l_hidream.safetensors"  # Main CLIP-L model "clip_l_hidream.safetensors"
# clip_g: Global CLIP-G model for global context
CLIP_NAME2 = "clip_g_hidream.safetensors"  # Global CLIP-G model "clip_g_hidream.safetensors"
# t5xxl: T5XXL language model for prompt understanding
CLIP_NAME3 = "t5xxl_fp8_e4m3fn_scaled.safetensors"  # T5XXL model "t5xxl_fp8_e4m3fn_scaled.safetensors"
# llama: Llama 3.1 8B Instruct for advanced prompt context
CLIP_NAME4 = "llama_3.1_8b_instruct_fp8_scaled.safetensors"  # Llama 3.1 8B Instruct "llama_3.1_8b_instruct_fp8_scaled.safetensors"

# --- ComfyUI Workflow Creation ---
def create_workflow(prompt, negative_prompt="", seed=None, width=1024, height=1024):
    """
    Creates a ComfyUI workflow for the HiDream fast model.
    
    Args:
        prompt (str): The positive prompt for image generation
        negative_prompt (str): The negative prompt (default empty)
        seed (int): Random seed (default None, which will generate a random seed)
        width (int): Image width (default 1024)
        height (int): Image height (default 1024)
    
    Returns:
        dict: A ComfyUI workflow configuration
    """
    if seed is None:
        import random
        seed = random.randint(0, 100000)
    
    return {
        "69": {  # UNETLoader
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": MAIN_MODEL_NAME,
                "weight_dtype": "default"
            }
        },
        "54": {  # QuadrupleCLIPLoader
            "class_type": "QuadrupleCLIPLoader",
            "inputs": {
                "clip_name1": CLIP_NAME1,  # Main CLIP-L
                "clip_name2": CLIP_NAME2,  # Global CLIP-G
                "clip_name3": CLIP_NAME3,  # T5XXL
                "clip_name4": CLIP_NAME4   # Llama 3.1 8B Instruct
            }
        },
        "55": {  # VAELoader
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ae.safetensors"  # VAE model for decoding
            }
        },
        "70": {  # ModelSamplingSD3
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["69", 0],
                "shift": 3.0
            }
        },
        "53": {  # EmptySD3LatentImage
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "16": {  # CLIPTextEncode - Positive Prompt
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["54", 0],
                "text": prompt
            }
        },
        "40": {  # CLIPTextEncode - Negative Prompt
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["54", 0],
                "text": negative_prompt
            }
        },
        "3": {  # KSampler
            "class_type": "KSampler",
            "inputs": {
                "model": ["70", 0],
                "positive": ["16", 0],
                "negative": ["40", 0],
                "latent_image": ["53", 0],
                "seed": seed,
                "steps": 16,
                "cfg": 1.0,
                "sampler_name": "lcm",
                "scheduler": "normal",
                "denoise": 1
            }
        },
        "8": {  # VAEDecode
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["55", 0]
            }
        },
        "9": {  # SaveImage
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "HiDream"
            }
        }
    }

def get_image_from_comfyui(prompt_id, save_image_node_id="9", timeout=240, poll_interval=3):
    """
    Polls the ComfyUI history for a specific prompt_id and retrieves the image
    generated by the specified SaveImage node.
    Uses node_id "9" which is the SaveImage node in the provided workflow.
    """
    prompt_history_url = f"{COMFY_BASE_URL}/history/{prompt_id}"
    start_time = time.time()

    print(f"Polling ComfyUI history for prompt_id: {prompt_id}, save_node: {save_image_node_id}")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(prompt_history_url)
            response.raise_for_status()
            history_data = response.json()

            if prompt_id in history_data:
                prompt_execution_data = history_data[prompt_id]
                # logging.debug(f"History data for {prompt_id}: {json.dumps(prompt_execution_data, indent=2)}")

                if "outputs" in prompt_execution_data and save_image_node_id in prompt_execution_data["outputs"]:
                    outputs = prompt_execution_data["outputs"][save_image_node_id]
                    if "images" in outputs and len(outputs["images"]) > 0:
                        image_info = outputs["images"][0]
                        filename = image_info.get("filename")
                        subfolder = image_info.get("subfolder", "")
                        image_type = image_info.get("type", "output") # usually 'output' or 'temp'

                        image_view_url = f"{COMFY_BASE_URL}/view?filename={requests.utils.quote(filename)}&subfolder={requests.utils.quote(subfolder)}&type={image_type}"
                        print(f"Fetching image from ComfyUI: {image_view_url}")
                        
                        image_response = requests.get(image_view_url)
                        image_response.raise_for_status()
                        
                        image_bytes = image_response.content
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        print(f"Successfully retrieved image: {filename} for prompt_id: {prompt_id}")
                        return image
                
                # Check for errors or incomplete status
                status = prompt_execution_data.get("status", {})
                if status.get("completed") is False and status.get("error") is True:
                     exception_message = status.get('exception_message', 'Unknown ComfyUI execution error')
                     print(f"ERROR: ComfyUI execution error for prompt {prompt_id}: {exception_message}")
                     # You might want to see node-specific errors if available
                     if "node_errors" in status: print(f"ERROR: Node errors: {status['node_errors']}")
                     raise Exception(f"ComfyUI error: {exception_message}")
                if status.get("status_str") == "error": # Simpler error check
                     exception_message = prompt_execution_data.get('exception_message', 'ComfyUI status string error')
                     print(f"ERROR: ComfyUI execution error (status_str) for prompt {prompt_id}: {exception_message}")
                     raise Exception(f"ComfyUI error: {exception_message}")
                if status.get("completed") is False and (time.time() - start_time) > (timeout - poll_interval*2) : # If nearing timeout and not complete
                    print(f"WARNING: Prompt {prompt_id} processing is slow. Status: {status.get('status_str', 'unknown')}, Progress: {status.get('progress', 'N/A')}/{status.get('progress_max', 'N/A')}")


            # else:
                # logging.debug(f"Prompt ID {prompt_id} not yet in history keys: {list(history_data.keys())}")


        except requests.exceptions.RequestException as e:
            print(f"ERROR: Error polling ComfyUI history for {prompt_id} (node {save_image_node_id}): {e}. Retrying...")
        except Exception as e: # Catch other errors like JSON parsing or unexpected structure
            print(f"ERROR: Error processing ComfyUI history for {prompt_id} (node {save_image_node_id}): {e}")
            raise # Re-raise critical errors to stop the attempt for this image

        time.sleep(poll_interval)
        
    raise TimeoutError(f"Timed out waiting for image from prompt {prompt_id} via SaveImage node {save_image_node_id}.")


# --- Utility Functions from Script 2 (keeping them as they are generally useful) ---
# save_image_with_metadata, remove_least_scored_image, load_prompt_memory, save_prompt_memory,
# find_next_available_filename, get_clip_embeddings, get_clip_image_similarity, score_prompt,
# compress_memory, review_and_enhance_annotation_with_system, extract_agent_score,
# generate_new_prompt, refine_prompt_for_token_limit, enhance_prompt_for_saving

def save_image_with_metadata(image, path, prompt, negative_prompt="", seed="", resolution="Unknown", model_info="ComfyUI", comfy_prompt_id="N/A"):
    """
    Saves the image with initial metadata.
    """
    meta = PngImagePlugin.PngInfo()
    meta.add_text("OriginalPrompt", prompt)
    meta.add_text("NegativePrompt", negative_prompt)
    meta.add_text("Seed", str(seed))
    meta.add_text("Resolution", resolution)
    meta.add_text("ModelInfo", model_info)
    meta.add_text("ComfyUIPromptID", str(comfy_prompt_id))
    meta.add_text("GenerationSystem", "ComfyUI_Iterative_Loop")
    meta.add_text("Timestamp", str(int(time.time())))
    image.save(path, "PNG", pnginfo=meta)
    print(f"Image saved with initial metadata: {path}")

def update_image_metadata(path, scores):
    """
    Updates the metadata of an existing image to include scores.
    """
    try:
        with Image.open(path) as img:
            meta = img.info
            updated_meta = PngImagePlugin.PngInfo()
            for key, value in meta.items():
                updated_meta.add_text(key, value)
            if scores:
                try:
                    updated_meta.add_text("Scores", json.dumps(scores))  # Save scores as JSON
                except TypeError:
                    updated_meta.add_text("Scores", str(scores))  # Fallback to string if JSON serialization fails
            img.save(path, "PNG", pnginfo=updated_meta)
            print(f"Metadata updated with scores for image: {path}")
    except Exception as e:
        print(f"ERROR: Failed to update metadata for {path}: {e}")

def remove_least_scored_image(memory):
    if not memory: return memory
    scored_entries = [(k, v) for k, v in memory.items() if 'scores' in v and isinstance(v['scores'], dict) and 'combined_score' in v['scores']]
    if not scored_entries:
        print("WARNING: No scored entries found in memory to remove.")
        return memory
    least_scored_prompt_key = min(scored_entries, key=lambda x: x[1]['scores']['combined_score'])[0]
    del memory[least_scored_prompt_key]
    print(f"Removed least scored prompt (key): {least_scored_prompt_key}")
    return memory

def load_prompt_memory(memory_file):
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r') as f: memory = json.load(f)
            print(f"Loaded prompt memory from {memory_file}")
            return memory
        except Exception as e:
            print(f"ERROR: Error loading memory file {memory_file}: {e}. Starting fresh.")
            return {}
    return {}

def save_prompt_memory(memory, file_path):
    try:
        with open(file_path, 'w') as f: json.dump(memory, f, indent=4)
        print(f"Prompt memory saved to {file_path}")
    except Exception as e:
        print(f"ERROR: Error saving prompt memory: {e}")

prompt_memory = load_prompt_memory(memory_file_path)

def find_next_available_filename(directory, start=0, end=15000, extension='.png'): # Increased end limit
    if not os.path.exists(directory): os.makedirs(directory, exist_ok=True)
    existing_files = []
    for f in os.listdir(directory):
        if f.endswith(extension) and f.split('.')[0].isdigit():
            try:
                existing_files.append(int(f.split('.')[0]))
            except ValueError:
                continue # Skip non-numeric filenames
    
    next_num = start
    if existing_files:
        existing_files.sort()
        # Find first gap or next number
        last_num = start -1
        found_gap = False
        for num in existing_files:
            if num > last_num + 1:
                next_num = last_num + 1
                found_gap = True
                break
            last_num = num
        if not found_gap:
            next_num = last_num + 1

    if next_num >= end:
         print(f"WARNING: Next file number {next_num} exceeds end limit {end-1}. Restarting search from {start} for a gap.")
         # More robust search for a gap if limit is hit
         all_nums = set(existing_files)
         for i in range(start, end):
             if i not in all_nums:
                 next_num = i
                 break
         else: # No gap found
             print(f"ERROR: Directory {directory} appears full up to {end-1}. Cannot find filename.")
             return None
    return next_num


def get_clip_embeddings(prompt_text): # Renamed prompt to prompt_text for clarity
    if not CLIP_AVAILABLE: return None
    try:
        if not isinstance(prompt_text, str): return None
        tokens = clip_tokenizer(prompt_text).to(device)
        tokens = tokens[:, :clip_model.positional_embedding.shape[0]] # Ensure not too long
        with torch.no_grad(): embeddings = clip_model.encode_text(tokens)
        return embeddings.float()
    except Exception as e:
        print(f"ERROR: Error in get_clip_embeddings for '{prompt_text[:30]}...': {e}")
        return None

def get_clip_image_similarity(image_obj, prompt_text): # Renamed for clarity
    if not CLIP_AVAILABLE: return 0.0
    try:
        if not isinstance(image_obj, Image.Image) or not isinstance(prompt_text, str): return 0.0
        image_input = preprocess(image_obj).unsqueeze(0).to(device)
        with torch.no_grad(): image_emb = clip_model.encode_image(image_input)
        prompt_emb = get_clip_embeddings(prompt_text)
        if prompt_emb is None: return 0.0
        similarity = torch.nn.functional.cosine_similarity(image_emb.float(), prompt_emb.float()).item()
        return similarity
    except Exception as e:
        print(f"ERROR: Error in get_clip_image_similarity for '{prompt_text[:30]}...': {e}")
        return 0.0

def get_ava_score(image_obj):
    """
    Computes the AVA score for the given image using the aesthetic-predictor-v2-5 package.
    Args:
        image_obj (PIL.Image.Image): The image to score.
    Returns:
        float: The AVA score.
    """
    if not AVA_AVAILABLE:
        return 0.0
    try:
        if not isinstance(image_obj, Image.Image):
            return 0.0
        # Preprocess the image
        pixel_values = (
            ava_preprocessor(images=image_obj, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )
        # Predict aesthetic score
        with torch.inference_mode():
            score_tensor = ava_model(pixel_values).logits.squeeze()
        # Convert to Python float
        score = float(score_tensor.float().cpu().numpy()) # <--- Explicit conversion to Python float
        return max(0.0, min(10.0, score))
    except Exception as e:
        print(f"ERROR: Error in get_ava_score: {e}") # Fixed f-string
        return 0.0

def score_prompt(prompt_text, image_obj, agent_score=0.0):
    """
    Scores the prompt and image based on their suitability.
    """
    clip_similarity_score = 0.0
    ava_score = 0.0

    if image_obj:
        try:
            # Ensure image mode and convert if necessary for CLIP (should be RGB or L)
            temp_image = image_obj
            if temp_image.mode not in ['RGB', 'L']:
                temp_image = temp_image.convert('RGB')
            clip_similarity_score = get_clip_image_similarity(temp_image, prompt_text)
            ava_score = get_ava_score(temp_image)
        except Exception as e:
            print(f"ERROR: Error processing image for scoring: {e}")
            clip_similarity_score = 0.0
            ava_score = 0.0

    clip_scaled = min(10.0, max(0.0, clip_similarity_score * 10.0))  # Scale [0,1] to [0,10]
    try:
        agent_score = min(10.0, max(0.0, float(agent_score)))
    except:
        agent_score = 0.0

    total_weight = 2 + 0.5 + 1  # CLIP weight 2, Agent weight 0.5, AVA weight 1
    combined_score = ((clip_scaled * 2) + (agent_score * 0.5) + (ava_score * 1)) / total_weight if total_weight > 0 else 0.0

    print(f"Scores for '{prompt_text[:30]}...': CLIP={clip_scaled:.2f}, AVA={ava_score:.2f}, Agent={agent_score:.2f}, Combined={combined_score:.2f}")
    return {"clip_similarity": clip_scaled, "ava_score": ava_score, "agent_score": agent_score, "combined_score": combined_score}

def compress_memory(memory, max_context_tokens, reserved_tokens):
    # Sort by combined_score, highest first
    scored_entries = sorted(
        [(k, v) for k, v in memory.items() if isinstance(v, dict) and 'scores' in v and isinstance(v['scores'], dict) and 'combined_score' in v['scores'] and 'prompt' in v],
        key=lambda x: x[1]['scores']['combined_score'], reverse=True
    )
    memory_context_list = []
    total_tokens = reserved_tokens
    try:
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception as e:
        print(f"WARNING: Tiktoken encoder for gpt-3.5-turbo not found, using basic split for token estimation: {e}")
        encoder = type('obj', (object,), {'encode' : lambda x: x.split()})()


    for key, data in scored_entries:
        context_line = f"Prompt: \"{data['prompt']}\", Score: {data['scores']['combined_score']:.2f}"
        line_tokens = len(encoder.encode(context_line))
        if total_tokens + line_tokens <= max_context_tokens:
            memory_context_list.append(context_line)
            total_tokens += line_tokens
        else:
            break
    return "\n".join(memory_context_list), total_tokens

def review_and_enhance_annotation_with_system(image_path, current_annotation, max_retries=3): # image_path is expected
    system_prompt = (
        "As an expert AI assistant, critically evaluate the provided image and its annotation. "
        "Assign an Agent_Score on a scale of 0 to 10, with 10 being the most aesthetic, explicit, unique, suited as a sticker, and well-structured. "
        "Avoid defaulting to common scores. Provide a nuanced score. "
        "Format your response ONLY as 'Agent_Score: X', where X is your rating (e.g., 'Agent_Score: 7.45'). Do NOT include any other text."
    )
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found for review: {image_path}"); return 0.0
    try:
        with Image.open(image_path) as img:
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buffered = io.BytesIO(); img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"ERROR: Error processing image {image_path} for review: {e}"); return 0.0

    print("Preparing to send image and annotation for review to LLM.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": f"Review this image based on the annotation: {current_annotation}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]}
    ]
    data = {"model": "local-model", "messages": messages, "max_tokens": 50, "temperature": 0.1}

    # Log the chat messages being sent to the LLM
    logging.debug(f"LLM Chat Messages: {json.dumps(messages, indent=2)}")

    for attempt in range(max_retries):
        try:
            response = requests.post(server_url, json=data, headers={"Content-Type": "application/json"}, timeout=240)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and result['choices']:
                response_text = result['choices'][0]['message']['content'].strip()
                agent_score = extract_agent_score(response_text)
                if agent_score is not None: return agent_score
                print(f"WARNING: Could not extract Agent_Score from '{response_text}'. Attempt {attempt + 1}")
        except Exception as e:
            print(f"ERROR: LLM review attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1: return 0.0
            time.sleep(2)
    return 0.0

def extract_agent_score(text):
    import re
    if not isinstance(text, str): return None
    # More specific pattern first
    match = re.search(r'Agent_Score\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        try: score = float(match.group(1)); return score if 0 <= score <= 10 else None
        except ValueError: pass
    # Fallback to any number if the specific pattern fails
    match = re.search(r'\b(\d+(?:\.\d+)?)\b', text) # A general number
    if match:
        try: score = float(match.group(1)); return score if 0 <= score <= 10 else None
        except ValueError: pass
    print(f"WARNING: Could not extract valid Agent_Score from text: '{text}'")
    return None

def generate_new_prompt(max_retries=5, target_tokens=77):
    MAX_CONTEXT_TOKENS = 3500
    RESERVED_TOKENS = 600
    memory_context, _ = compress_memory(prompt_memory, MAX_CONTEXT_TOKENS, RESERVED_TOKENS)
    random_theme= random.choice(THEMES)

    # Refined system promptC
    system_prompt = (
        "Your SOLE task is to generate a descriptive image generation prompt (ideally around 77 tokens or less)."
        "The generated image should be designed specifically for stickers, with bold, visually striking elements "
        "and a white outline designating the sticker's edge. Try to make the main subject unique from the other images in the memory. "
        "This prompt should incorporate bold, visual elements. This prompt should be distinct, visually appealing, high quality, unique, "
        "and incorporate elements inspired by this theme "+random_theme+
        "The generated image will later be printed as a sticker, so ensure the prompt describes a design that is "
        "dynamic, visually engaging, and suitable for high-quality printing. "
        "Consider the structures and subjects of high-scoring prompts from the provided memory examples below to inform your style, but DO NOT simply copy them or repeat subjects excessively. Aim for novelty.\n"
        "Memory examples (for inspiration only - DO NOT include this memory text in your response):\n"
        f"---MEMORY START---\n{memory_context}\n---MEMORY END---\n\n"
        f"Remember: Output ONLY the image prompt itself. Prepend the word 'sticker, ' to the prompt. "
        "Do NOT include any conversational text, explanations, markdown formatting, scores, or anything else. "
    )

    print("Preparing to generate a new prompt using LLM.")
    user_message = "Generate the image prompt now."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    data = {"model": "local-model", "messages": messages, "max_tokens": 150, "temperature": 1.3, "top_p": 0.9}

    # Log the chat messages being sent to the LLM
    logging.debug(f"LLM Chat Messages: {json.dumps(messages, indent=2)}")

    for attempt in range(max_retries):
        try:
            response = requests.post(server_url, json=data, headers={"Content-Type": "application/json"}, timeout=240)
            response.raise_for_status()
            result = response.json()

            # Log the raw response for debugging
            logging.debug(f"LLM Raw Response: {json.dumps(result, indent=2)}")

            if 'choices' in result and result['choices']:
                raw_response = result['choices'][0]['message']['content'].strip()

                # Clean and validate the response
                cleaned_prompt = raw_response.splitlines()
                cleaned_prompt = [
                    line for line in cleaned_prompt
                    if line.strip() and not line.lower().startswith(("here's a prompt", "prompt:", "sure, here", "certainly:"))
                ]
                cleaned_prompt = cleaned_prompt[-1] if cleaned_prompt else raw_response
                cleaned_prompt = cleaned_prompt.strip().strip('"').strip("'").strip()

                token_count = len(cleaned_prompt.split())
                if 1 <= token_count <= (target_tokens + 20):
                    print(f"LLM generated prompt: '{cleaned_prompt}' ({token_count} tokens)")
                    return cleaned_prompt
                else:
                    print(f"WARNING: Generated prompt token count {token_count} outside range for: '{cleaned_prompt}'. Retrying.")
                    data["messages"] = messages + [
                        {"role": "assistant", "content": raw_response},
                        {"role": "user", "content": f"That was {token_count} tokens. Please try again, ensuring the prompt is ONLY the prompt text and under {target_tokens} tokens."}
                    ]
            else:
                print(f"WARNING: Unexpected LLM response format: {result}. Retrying.")
        except Exception as e:
            print(f"ERROR: LLM prompt generation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("ERROR: All attempts to generate a prompt failed.")
                return None
            time.sleep(2)

    print("ERROR: Failed to generate a valid prompt after retries.")
    return None

def refine_prompt_for_token_limit(target_tokens=77):
    prompt = generate_new_prompt(target_tokens=target_tokens)
    if prompt:
        # Final check, not strictly trimming here, relying on LLM to adhere.
        token_count = len(prompt.split())
        if token_count > (target_tokens + 10): # A bit more generous than initial target
            print(f"WARNING: Final generated prompt '{prompt}' is {token_count} tokens, exceeding target {target_tokens}.")
        elif token_count < 1:
            print("WARNING: Final generated prompt is empty.")
            return None
    return prompt

def enhance_prompt_for_saving(image_path, original_prompt, max_retries=3): # image_path expected
    system_prompt = (
        "Review the image and its original prompt. Create a concise description of the image (under 25 words) "
        "capturing key visual elements, style, and colors. This is for memory. "
        "ONLY provide the description."
    )
    if not os.path.exists(image_path): return original_prompt
    try:
        with Image.open(image_path) as img:
            img.thumbnail((512,512), Image.Resampling.LANCZOS)
            buffered = io.BytesIO(); img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"ERROR: Error processing image {image_path} for description: {e}"); return original_prompt

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": f"Original prompt: {original_prompt}\n\nDescribe this image:"},
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}]
    data = {"model": "local-model", "messages": messages, "max_tokens": 100, "temperature": 0.5}
    for attempt in range(max_retries):
        try:
            response = requests.post(server_url, json=data, headers={"Content-Type": "application/json"}, timeout=240)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result and result['choices']:
                desc = result['choices'][0]['message']['content'].strip().splitlines()[-1].strip()
                print(f"Enhanced description for memory: '{desc}'")
                return desc
        except Exception as e:
            print(f"ERROR: LLM description enhance attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1: return original_prompt
            time.sleep(1)
    return original_prompt


# --- Main iterative process (adapted for ComfyUI) ---
NUM_ITERATIONS = 100 # From script 2

print("Starting iterative image generation process with ComfyUI.")
for i in range(NUM_ITERATIONS):
    print(f"--- Starting iteration {i + 1}/{NUM_ITERATIONS} ---")

    # Step 1: Get a new prompt from LLM
    current_prompt = refine_prompt_for_token_limit(target_tokens=77) # Renamed to current_prompt
    if not current_prompt:
        print("ERROR: Iteration failed: Could not generate a valid prompt. Skipping.")
        time.sleep(5) # Wait a bit before retrying next iteration
        continue

    print(f"Using prompt for ComfyUI generation: {current_prompt}")

    # Negative prompt (Using the one from Script 1, as it's tied to the ComfyUI workflow's expectations)
    negative_prompt = " "#"bad quality, low resolution, watermark, text, ugly, blurry, worst quality, low quality, jpeg artifacts, signature, username, poorly drawn face, mutation, deformed, extra limbs, extra fingers, too many fingers, fused fingers, long neck"
    seed = random.randint(0, 2**32 - 1)

    # Step 2: Generate an image using ComfyUI
    generated_image = None # Renamed
    comfy_prompt_id = None # To store ComfyUI's prompt ID
    try:
        print(f"Queuing prompt to ComfyUI. Seed: {seed}, Prompt: {current_prompt[:100]}...")
        workflow = create_workflow(current_prompt, negative_prompt, seed)
        
        client_id = f"iterative_gen_{int(time.time())}" # Optional client ID
        api_payload = {"prompt": workflow, "client_id": client_id}

        response = requests.post(COMFY_PROMPT_URL, json=api_payload, timeout=240) # Timeout for queueing
        response.raise_for_status()
        
        queue_response = response.json()
        comfy_prompt_id = queue_response.get("prompt_id")
        
        if not comfy_prompt_id:
            print(f"ERROR: Failed to get prompt_id from ComfyUI. Response: {queue_response}")
            continue # Skip to next iteration
        print(f"Prompt queued in ComfyUI with ID: {comfy_prompt_id}")

        # Node ID '9' is the SaveImage node in the provided workflow from script 1
        generated_image = get_image_from_comfyui(comfy_prompt_id, save_image_node_id="9", timeout=240) # Increased timeout for generation
        
        if generated_image is None:
            print(f"ERROR: Failed to retrieve image from ComfyUI for prompt_id {comfy_prompt_id}.")
            continue
        print(f"Image generated successfully via ComfyUI (prompt_id: {comfy_prompt_id}, seed: {seed})")

    except requests.exceptions.Timeout:
        print(f"ERROR: Timeout when trying to queue prompt or get image from ComfyUI.")
        continue
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Connection error with ComfyUI server at {COMFY_BASE_URL}. Is it running?")
        # Maybe stop or pause significantly if ComfyUI is down
        print("Stopping due to ComfyUI connection error.")
        break
    except requests.exceptions.RequestException as e:
        print(f"ERROR: ComfyUI API request error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"ERROR: Response status: {e.response.status_code}, Response text: {e.response.text[:300]}")
        continue
    except TimeoutError as e: # Timeout from get_image_from_comfyui
        print(f"ERROR: ComfyUI image retrieval timeout: {e}")
        continue
    except Exception as e: # Catch-all for other errors during ComfyUI interaction
        print(f"ERROR: Error during ComfyUI image generation: {e}", exc_info=True)
        continue

    # Step 2.5: Face Enhancement (if enabled and image was generated)
    # if face_enhancer_available and generated_image:
    #      generated_image = enhance_faces(generated_image, face_enhancer)

    # Step 3: Save the generated image
    image_num = find_next_available_filename(image_dir)
    if image_num is None:
         print(f"ERROR: Could not find an available filename in {image_dir}. Stopping.")
         break 
    image_filename = f"{image_num:05d}.png"
    image_path = os.path.join(image_dir, image_filename)
    
    image_resolution = "Unknown"
    if generated_image:
        image_resolution = f"{generated_image.width}x{generated_image.height}"
        save_image_with_metadata(
            generated_image, image_path, current_prompt,
            negative_prompt=negative_prompt, seed=seed,
            resolution=image_resolution,
            model_info="ComfyUI Workflow (Script1 SD3 type)", # More specific model info
            comfy_prompt_id=comfy_prompt_id
        )
    else:
        print(f"ERROR: Cannot save image as it was not generated: {image_path}")
        continue # Skip scoring and memory if no image

    # Step 4 & 5: Review, Score, and Store
    agent_score = review_and_enhance_annotation_with_system(image_path, current_prompt)
    prompt_scores = score_prompt(current_prompt, generated_image, agent_score)

    # Update metadata with scores
    update_image_metadata(image_path, prompt_scores)

    # Get enhanced description for memory
    memory_prompt_text = enhance_prompt_for_saving(image_path, current_prompt)
    
    # Store information in memory
    MAX_MEMORY_SIZE = 100 # From script 2
    prompt_memory_entry_key = f"{image_num:05d}"
    prompt_memory[prompt_memory_entry_key] = {
        "prompt": memory_prompt_text, # This is the enhanced description for memory
        "original_prompt": current_prompt, # The prompt used for generation
        "scores": prompt_scores,
        "seed": seed,
        "image_filename": image_filename,
        "comfy_prompt_id": comfy_prompt_id,
        "generation_parameters": {
            "workflow_source": "Script1_ComfyUI_SD3_type",
            "image_resolution": image_resolution,
            "negative_prompt": negative_prompt
            # You can add more specific params from the ComfyUI workflow if needed
        }
    }

    # Optional: Limit memory size
    while len(prompt_memory) > MAX_MEMORY_SIZE:
        print(f"WARNING: Memory size ({len(prompt_memory)}) exceeds limit ({MAX_MEMORY_SIZE}). Removing least scored.")
        prompt_memory = remove_least_scored_image(prompt_memory)

    # Step 6: Save the prompt memory
    save_prompt_memory(prompt_memory, memory_file_path)

    # Update image metadata with scores
    update_image_metadata(image_path, prompt_scores)

    print(f"--- Completed iteration {i + 1}/{NUM_ITERATIONS} ---")
    
    # Wait a bit before the next iteration
    iteration_wait_time = 5 # seconds
    print(f"Waiting {iteration_wait_time} seconds before next iteration...")
    time.sleep(iteration_wait_time)


print("Iterative image generation process with ComfyUI completed.")
