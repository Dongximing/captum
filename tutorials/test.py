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
eval_prompt = """Heathrow airpot is located in the city of"""
model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
#print(model)
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
embeddings_output = []

# 定义一个hook函数
def get_embeddings(module, input, output):
    embeddings_output.append(output)

# 注册hook到embedding层
embedding_layer = model.get_input_embeddings()
handle = embedding_layer.register_forward_hook(get_embeddings)

model_input = tokenizer(eval_prompt, return_tensors="pt")
print(model_input["input_ids"])
with torch.no_grad():
    output_ids = model.generate(model_input["input_ids"], max_new_tokens=2)[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    handle.remove()
    print(response)
print(embeddings_output)
print("------------------------------------------------")
# fa = FeatureAblation(model)
emb_layer = model.get_submodule("transformer.wte")
ig = LayerIntegratedGradients(model, emb_layer)
#ig = LayerIntegratedGradients()
llm_attr = LLMGradientAttribution(ig, tokenizer)

inp = TextTokenInput(
    eval_prompt,
    tokenizer,
    skip_tokens=[1],  # skip the special token for the start of the text <s>
)
print("inp in 30------------->",dir(inp))
target = "playing guitar, hiking, and spending time with his family."

attr_res = llm_attr.attribute(inp)
attr_res.plot_token_attr(show=True)