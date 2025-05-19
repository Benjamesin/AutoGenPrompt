Work in progress **
Vision enable llm running on LMStudio is required, I recommend Gemma-3
Also comfyUI needs to be running
I need to run the llm on a seperate computer to make this work, but there are less vram intesive models that can be used. 
I've had past iterations use flux and SDXL 
Currently this is setup to be a sticker design generator. I recommend adjusting the system prompt to better suite your needs. 


# Iterative AI Image Generation with LLM & ComfyUI

This project implements an advanced iterative loop for generating diverse and high-quality images. It leverages a Large Language Model (LLM) via an LMStudio-compatible API to craft creative and detailed prompts. These prompts are then processed by a ComfyUI instance to generate images. The system incorporates a scoring mechanism (CLIP similarity, aesthetic score, and an LLM-based quality assessment) to evaluate generated images. This feedback, along with the prompts, is stored in a "memory" to guide and refine future prompt generation, enabling exploration and discovery of visual styles.

The system is designed for users interested in automated art generation, prompt engineering research, and exploring the capabilities of generative AI in a structured, iterative manner.

## Features

*   **Iterative Generation:** Continuously generates images and refines prompting strategies based on feedback.
*   **LLM-Powered Prompt Engineering:** Utilizes an LLM for sophisticated and context-aware prompt creation and enhancement.
*   **ComfyUI Integration:** Employs ComfyUI as a flexible backend for image generation, using a configurable workflow.
*   **Multi-faceted Scoring & Feedback:**
    *   CLIP similarity score between the prompt and the generated image.
    *   Aesthetic score (AVA) using `aesthetic-predictor-v2_5`.
    *   LLM-based quality review: The LLM assesses the image based on the initial prompt, providing a quality score.
*   **Adaptive Prompt Memory:** Stores successful prompts, their scores, and associated metadata to improve future generation quality and diversity.
*   **Metadata-Rich Images:** Saves images with embedded PNG metadata, including prompts, seeds, scores, and ComfyUI workflow details.
*   **Configurable Framework:** Key parameters, paths, API endpoints, and model configurations are managed through a central `config.json` file.
*   **Externalized Prompting Logic:** System prompts for the LLM and thematic lists are loaded from external files, allowing for easy customization of creative direction.

## Prerequisites

1.  **Python 3.8+**
2.  **Running ComfyUI Instance:**
    *   Accessible via HTTP (e.g., `http://127.0.0.1:8188`).
    *   **Required ComfyUI Models:** The script's default workflow (`create_workflow` function) expects specific model files. Ensure these (or your customized alternatives specified in `config.json`) are correctly placed for ComfyUI:
        *   UNET Model (e.g., `hidream_i1_fast_fp8.safetensors`)
        *   CLIP Models (e.g., `hidream/clip_l_hidream.safetensors`, etc.)
        *   VAE Model (e.g., `ae.sft`)
        *   *Refer to the `comfyui_model_paths` section in `config.example.json` for default names.*
3.  **Running LMStudio (or compatible OpenAI API-like) Server:**
    *   Accessible via HTTP (e.g., `http://localhost:1234/v1/chat/completions`).
    *   A capable instruction-following model loaded, preferably one that also supports image input for the review stage.
4.  **OpenCLIP and Aesthetic Predictor Models:**
    *   The script uses pre-trained OpenCLIP (`ViT-H-14` from `laion2b_s32b_b79k`) and the `aesthetic-predictor-v2_5` model. An internet connection may be required for the first run if these are not cached.
