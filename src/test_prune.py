from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy
import torch.nn as nn

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

prompt = "What is the capital of France?"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# ==========================
# ORIGINAL MODEL
# ==========================

orig_text = model.generate(
    **inputs,
    max_new_tokens=50
)

print("\n===== ORIGINAL MODEL =====\n")

print(
    tokenizer.decode(
        orig_text[0],
        skip_special_tokens=True
    )
)

# ==========================
# PRUNE SELECTED LAYERS
# ==========================

pruned_model = copy.deepcopy(model)

layers = list(pruned_model.model.layers)

drop_layers = {
    3,4,8,10,
    12, 13, 14, 15,
    16, 17,
    19, 20, 21, 22
}
"""
observations
I thought what if its just that 2 layers consecutively might have soemthing 
so I tried removing in clusters
I removed 3,4 - no issue
8,10  - still no issue

12, 13, 14, 15, 16, 17
What is the capital of France? (in the other words)
- - The 100th is a 100-year commitment to raising the 100 in the 100
  (capital) of France
 100 10

 19, 20, 21, 22
What is the capital of France?**
1000009999999999999999999999999999999999999999999
"""


layers = [
    layer
    for idx, layer in enumerate(layers)
    if idx not in drop_layers
]

pruned_model.model.layers = nn.ModuleList(layers)

print(
    f"\nLayers before: {len(model.model.layers)}"
)

print(
    f"Layers after : {len(pruned_model.model.layers)}"
)

print(
    f"Layers removed: {sorted(drop_layers)}"
)

# ==========================
# GENERATE
# ==========================

pruned_text = pruned_model.generate(
    **inputs,
    max_new_tokens=50
)

print("\n===== PRUNED MODEL OUTPUT =====\n")

print(
    tokenizer.decode(
        pruned_text[0],
        skip_special_tokens=True
    )
)