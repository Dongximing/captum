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
eval_prompt = "I love you and"
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

model_input = tokenizer(eval_prompt, return_tensors="pt")
print(model_input["input_ids"])
# model.eval()
# with torch.no_grad():
#     output_ids = model.generate(model_input["input_ids"], max_new_tokens=15)[0]
#     response = tokenizer.decode(output_ids, skip_special_tokens=True)
#     print(response)
fa = FeatureAblation(model)

ig = LayerIntegratedGradients()
llm_attr = LLMAttribution(fa, tokenizer)

inp = TextTokenInput(
    eval_prompt,
    tokenizer,
    skip_tokens=[1],  # skip the special token for the start of the text <s>
)
print("inp in 30------------->",dir(inp))
target = "playing guitar, hiking, and spending time with his family."

attr_res = llm_attr.attribute(inp)
attr_res.plot_token_attr(show=True)