import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    ProductBaselines,
)
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config
eval_prompt = """the sentiment of 'I am happy' is positive or negative: the answer is"""
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')

model_input = tokenizer(eval_prompt, return_tensors="pt")
print(model_input["input_ids"])

# fa = FeatureAblation(model)
emb_layer = model.get_submodule("model.embed_tokens")
ig = LayerIntegratedGradients(model, emb_layer)
#ig = LayerIntegratedGradients()
llm_attr = LLMAttribution(ig, tokenizer)

inp = TextTokenInput(
    eval_prompt,
    tokenizer,
    skip_tokens=[1],  # skip the special token for the start of the text <s>
)
print("inp in 30------------->",dir(inp))
target = "playing guitar, hiking, and spending time with his family."

attr_res = llm_attr.attribute(inp)
attr_res.plot_token_attr(show=True)