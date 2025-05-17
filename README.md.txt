Work in progress **

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

## User Responsibility

*   **Content Generation:** You are responsible for the content generated using this script, including the themes and system prompts you provide to the LLM.
*   **Ethical Use:** Ensure your use of this script complies with all applicable local laws, terms of service for any APIs or models used, and ethical AI guidelines. The developers of this script are not responsible for how it is used or the content it produces.
*   **Resource Usage:** Be mindful of the computational resources (GPU, API calls) consumed during generation.

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

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[Your GitHub Username]/[Your Repo Name].git
    cd [Your Repo Name]
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the script:**
    *   Copy `config.example.json` to `config.json`.
        ```bash
        cp config.example.json config.json
        ```
    *   Edit `config.json` with your specific settings:
        *   `lmstudio.server_url`, `lmstudio.model_name`.
        *   `lmstudio.system_prompt_path`: Path to your system prompt text file (see step 4).
        *   `lmstudio.themes_path`: Path to your themes JSON file (see step 5).
        *   `comfyui.base_url`.
        *   `paths.image_dir`, `paths.memory_file`, `paths.log_file`.
        *   `comfyui_model_paths`: Update if your ComfyUI model names/paths differ from the defaults.
        *   Adjust other settings like `generation_settings.num_iterations` as needed.

4.  **Create System Prompt File:**
    *   The script uses a system prompt to guide the LLM's prompt generation. An example path is `prompts/system_prompt.txt`.
    *   Create this file and populate it. **This prompt should define the desired style, quality, and constraints for the image prompts the LLM generates. Focus on creativity, artistic merit, and adherence to any specific project goals.**
    *   *Example content for `prompts/system_prompt.txt` (professional, SFW focus):*
        ```text
        You are 'ArtPrompt Architect,' an AI assistant specializing in generating highly detailed, creative, and visually evocative image prompts for use with generative AI art platforms. Your goal is to produce prompts that result in aesthetically pleasing, high-quality images suitable for diverse artistic applications like concept art, illustration, and digital media.

        PROMPT GENERATION GUIDELINES:
        1.  **Core Concept:** Clearly define the main subject, ensuring it is interesting and well-described.
        2.  **Descriptive Details:** Layer in rich details related to:
            *   **Composition & Framing:** (e.g., "dynamic angle," "close-up portrait," "wide landscape shot").
            *   **Artistic Style:** (e.g., "impressionistic oil painting," "detailed cel-shaded anime style," "photorealistic concept art," "art deco illustration").
            *   **Lighting & Atmosphere:** (e.g., "dramatic volumetric lighting," "soft morning glow," "mysterious twilight ambiance").
            *   **Color Palette:** (e.g., "vibrant contrasting colors," "monochromatic blue tones," "pastel dreamscape").
            *   **Keywords for Quality:** Include terms like "masterpiece," "highly detailed," "sharp focus," "intricate," "award-winning."
        3.  **Thematic Focus:** When provided with a theme, integrate it naturally and creatively into the prompt.
        4.  **Conciseness & Clarity:** Aim for prompts that are descriptive yet concise, ideally within 70-80 tokens.
        5.  **Avoid Ambiguity:** Strive for prompts that are unlikely to be misinterpreted by the image generation model.
        6.  **Professional & Safe Content:** All generated prompts must be suitable for a general audience. Do not generate prompts that are offensive, hateful, explicit, or harmful.

        INPUT FORMAT:
        You will receive a theme and context from past successful prompts.

        Your SOLE task is to generate an image generation prompt based on these inputs.
        Output ONLY the image prompt itself. Do NOT include any conversational text, explanations, or markdown.
        Consider the structures and subjects of high-scoring prompts from the provided memory examples to inform your style, but DO NOT simply copy them. Aim for novelty and high artistic quality.
        Memory examples (for inspiration only - DO NOT include this memory text in your response):
        ---MEMORY START---
        [Memory context will be injected here by the script]
        ---MEMORY END---

        Theme for current prompt: [Theme will be injected here by the script]

        Generate the image prompt now:
        ```

5.  **Create Themes File:**
    *   The script loads themes from a JSON file (e.g., `prompts/themes.json`) to provide creative direction.
    *   Create this file with a JSON list of SFW, professional, or artistically diverse theme strings.
    *   *Example content for `prompts/themes.json` (professional, SFW focus):*
        ```json
        [
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
        ```

6.  **Ensure Directories Exist:**
    *   The script will attempt to create the `image_dir` (and other output directories like for memory and logs) if they don't exist, as specified in `config.json`.

## Running the Script

Once configured, run the main script from your terminal:

```bash
python your_script_name.py