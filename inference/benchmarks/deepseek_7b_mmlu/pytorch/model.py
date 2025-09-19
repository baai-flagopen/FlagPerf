from transformers import AutoModelForCausalLM
import os


def create_model(config):
    model_path = os.path.join(config.data_dir, config.weight_dir)
    print(f"[DEBUG] Loading model from: {model_path}")
    print(f"[DEBUG] Model path exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"[DEBUG] Files in model path: {os.listdir(model_path) if os.path.exists(model_path) else 'Path not found'}")
    
    try:
        print("[DEBUG] Starting model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path).eval().cuda().float()
        print("[DEBUG] Model loaded successfully")
        
        if config.fp16:
            print("[DEBUG] Converting to FP16...")
            model.half()
            print("[DEBUG] FP16 conversion completed")

        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        raise
