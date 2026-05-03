import json
import re

def generate_caption(image):
    """
    Lightweight placeholder function for image captioning.
    In a real-world scenario, this would use a VLM (Vision-Language Model)
    like PaliGemma or LLaVA to generate a description of the image.
    """
    # For now, we return a dummy caption describing a diseased plant.
    return "A plant leaf with brown circular spots and yellowing edges, indicating a potential fungal infection."

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
