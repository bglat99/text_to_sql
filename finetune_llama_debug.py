#!/usr/bin/env python3
"""
Debug Llama 3.1 8B Loading for Apple Silicon
Tests different approaches to get Llama working
"""

import os
import json
import logging
import torch
from typing import List, Dict
import warnings

# Suppress SSL warning
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def test_llama_loading_approaches():
    """Test different approaches to load Llama 3.1 8B"""
    
    logger.info("=" * 60)
    logger.info("Testing Llama 3.1 8B Loading Approaches")
    logger.info("=" * 60)
    
    # Check device
    if torch.backends.mps.is_available():
        logger.info("✅ Apple Silicon detected - MPS backend available")
    else:
        logger.info("⚠️ MPS not available")
    
    # Test 1: Try Unsloth approach
    logger.info("\n--- Test 1: Unsloth Approach ---")
    try:
        from unsloth import FastLanguageModel
        
        logger.info("Loading Unsloth Llama 3.1 8B...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        logger.info("✅ Unsloth approach SUCCESS!")
        return model, tokenizer, "unsloth"
        
    except Exception as e:
        logger.error(f"❌ Unsloth approach failed: {e}")
    
    # Test 2: Try standard transformers with authentication
    logger.info("\n--- Test 2: Standard Transformers with Auth ---")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Try to authenticate
        import huggingface_hub
        logger.info("Attempting Hugging Face authentication...")
        
        # Try loading with auth
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info("✅ Standard transformers approach SUCCESS!")
        return model, tokenizer, "standard"
        
    except Exception as e:
        logger.error(f"❌ Standard transformers approach failed: {e}")
    
    # Test 3: Try without quantization
    logger.info("\n--- Test 3: Without Quantization ---")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info("✅ No quantization approach SUCCESS!")
        return model, tokenizer, "no_quant"
        
    except Exception as e:
        logger.error(f"❌ No quantization approach failed: {e}")
    
    # Test 4: Try alternative Llama model
    logger.info("\n--- Test 4: Alternative Llama Model ---")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Try a different Llama model that might not require auth
        model_name = "NousResearch/Llama-2-7b-chat-hf"  # Alternative that's often available
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info("✅ Alternative Llama approach SUCCESS!")
        return model, tokenizer, "alternative"
        
    except Exception as e:
        logger.error(f"❌ Alternative Llama approach failed: {e}")
    
    logger.error("❌ All Llama loading approaches failed!")
    return None, None, None

def setup_huggingface_auth():
    """Try to set up Hugging Face authentication"""
    try:
        import huggingface_hub
        
        # Check if already logged in
        try:
            user = huggingface_hub.whoami()
            logger.info(f"✅ Already logged in as: {user}")
            return True
        except:
            logger.info("Not logged in, attempting to login...")
            
            # Try to login with token
            token = os.getenv("HUGGINGFACE_TOKEN")
            if token:
                huggingface_hub.login(token)
                logger.info("✅ Logged in with token from environment")
                return True
            else:
                logger.info("No token found in environment")
                logger.info("Please set HUGGINGFACE_TOKEN environment variable")
                return False
                
    except Exception as e:
        logger.error(f"Authentication setup failed: {e}")
        return False

def main():
    logger.info("🔍 Debugging Llama 3.1 8B Loading...")
    
    # Try to set up authentication first
    auth_success = setup_huggingface_auth()
    
    # Test loading approaches
    model, tokenizer, approach = test_llama_loading_approaches()
    
    if model and tokenizer:
        logger.info(f"🎉 SUCCESS! Loaded model using approach: {approach}")
        
        # Test basic functionality
        logger.info("Testing basic model functionality...")
        try:
            test_input = "Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ Model test successful! Response: {response}")
            
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
        
        return model, tokenizer
    else:
        logger.error("❌ Failed to load any Llama model")
        return None, None

if __name__ == "__main__":
    main() 