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
                bnb_4bit_compute_dtype=torch.float16
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
    Runs the inference pipeline with the given prompt.
    Forces deterministic generation by using a low temperature.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    except Exception:
        # Fallback if no chat template is available
        text_prompt = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(text_prompt, return_tensors="pt").input_ids
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.1,  # Low temperature for deterministic JSON structure
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    input_length = inputs.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response
