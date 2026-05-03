import json
import re
import numpy as np

def generate_caption(image):
    """
    Generates a descriptive caption of the plant image using basic image analysis.
    Analyzes color distribution and texture to produce a meaningful description
    that the language model can use for diagnosis.
    
    In a production system, this would be replaced by a full VLM
    (e.g., PaliGemma or LLaVA) for pixel-level understanding.
    """
    img_array = np.array(image)

    # Color channel analysis
    r_mean = img_array[:, :, 0].mean()
    g_mean = img_array[:, :, 1].mean()
    b_mean = img_array[:, :, 2].mean()

    # Brightness and contrast
    gray = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean
    brightness = "bright" if gray > 128 else "dark"

    # Dominant color analysis
    if g_mean > r_mean and g_mean > b_mean:
        if g_mean > 150:
            color_desc = "vibrant green leaves with healthy coloration"
        else:
            color_desc = "pale or faded green leaves with possible chlorosis"
    elif r_mean > g_mean:
        color_desc = "leaves with significant browning, reddish discoloration, or necrotic lesions"
    elif b_mean > g_mean:
        color_desc = "leaves with unusual bluish or grayish discoloration"
    else:
        color_desc = "leaves with mixed discoloration and irregular color patterns"

    # Spot/texture estimation via local variance
    if img_array.shape[0] > 10 and img_array.shape[1] > 10:
        sample = img_array[::10, ::10, 0].astype(float)
        variance = sample.var()
        if variance > 1200:
            texture_desc = "with high-contrast irregular spots, lesions, or fungal patches visible on the surface"
        elif variance > 400:
            texture_desc = "with moderate surface irregularities and possible early-stage disease symptoms"
        else:
            texture_desc = "with a relatively uniform texture suggesting early or no visible disease symptoms"
    else:
        texture_desc = "with visible surface texture"

    # Edge yellowing estimate
    top_strip = img_array[:max(1, img_array.shape[0]//8), :, :]
    edge_r = top_strip[:, :, 0].mean()
    edge_g = top_strip[:, :, 1].mean()
    if edge_r > edge_g * 1.1:
        edge_desc = "Leaf edges show yellowing or browning, consistent with nutrient deficiency or disease spread."
    else:
        edge_desc = "Leaf edges appear relatively intact."

    caption = (
        f"A {brightness} plant leaf image showing {color_desc}, "
        f"{texture_desc}. {edge_desc} "
        f"The overall appearance suggests this sample requires careful agronomic evaluation."
    )

    return caption


def build_prompt(caption, language="EN"):
    """
    Builds a deterministic, structured prompt for the Gemma 4 model.
    """
    if language == "TR":
        return f"""Sen uzman bir ziraat mühendisisin.
Aşağıdaki bitki yaprağı betimlemesine dayanarak hastalığı teşhis et ve bir tedavi planı sun.
Betimleme: {caption}

SADECE aşağıdaki formatta, başka hiçbir metin veya açıklama içermeyen geçerli bir JSON döndürmelisin:
{{
"disease": "hastalık adı",
"confidence": "yüzdelik oran",
"symptoms": "belirtiler",
"treatment": "tedavi adımları",
"severity": "low/medium/high"
}}"""
    else:
        return f"""You are an expert agricultural engineer.
Based on the following description of a plant leaf, diagnose the disease and provide a treatment plan.
Description: {caption}

You must return ONLY valid JSON in the exact format below, without any markdown formatting or extra text:
{{
"disease": "disease name",
"confidence": "percentage",
"symptoms": "symptoms description",
"treatment": "treatment steps",
"severity": "low/medium/high"
}}"""


def extract_json_from_response(response_text):
    """
    Extracts the JSON dictionary from the model's text response.
    """
    try:
        # Try to find a JSON block in the response using regex
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            # Fallback, try parsing the whole text
            return json.loads(response_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_output": response_text}