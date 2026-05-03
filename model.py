import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    """
    Loads the Gemma 4 model.
    Uses unsloth FastLanguageModel if available for faster inference,
    with 4-bit quantization to fit on low hardware (Colab T4).
    Includes a CPU fallback if GPU is not present.
    """
    model_name = "unsloth/gemma-4-e4b-it"
    max_seq_length = 2048
    
    try:
        from unsloth import FastLanguageModel
        UNSLOTH_AVAILABLE = True
    except ImportError:
        UNSLOTH_AVAILABLE = False
        
    print(f"Loading {model_name}...")
    
    if torch.cuda.is_available() and UNSLOTH_AVAILABLE:
        print("Using Unsloth FastLanguageModel on GPU...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    else:
        # Fallback for standard transformers and CPU support
        print("Unsloth not available or no GPU detected. Using transformers fallback...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            print("GPU detected, using bitsandbytes 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16  # float16 → bfloat16 (Gemma 4 requirement)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            print("No GPU detected. Falling back to CPU mode (slower)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
        return model, tokenizer

def run_inference(model, tokenizer, prompt):
    """
    Executes real inference on the loaded model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            do_sample=True,      # Required when using temperature/top_p
            temperature=0.2,     # Low temperature keeps JSON format stable
            top_p=0.9,
        )
    
    # Sadece yeni üretilen token'ları çöz (prompt'u sil)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response