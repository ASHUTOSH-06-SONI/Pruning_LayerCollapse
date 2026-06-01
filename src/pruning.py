from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
layer = model.model.layers[0]
#print(layer)
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
original_hidden = outputs.hidden_states[-1]
pruned_hidden = outputs.hidden_states[-1]
import torch.nn.functional as F
sim = F.cosine_similarity(original_hidden, pruned_hidden, dim=0)

#print(sim)
import copy 
import torch.nn as nn

pruned_model = copy.deepcopy(model)
layers = list(pruned_model.model.layers)
del layers[10]
pruned_model.model.layers = nn.ModuleList(layers)

with torch.no_grad():
    original_outputs = model(**inputs, output_hidden_states=True)
    pruned_outputs = pruned_model(**inputs, output_hidden_states=True)

original_hidden = original_outputs.hidden_states[-1]
pruned_hidden = pruned_outputs.hidden_states[-1]
sim = F.cosine_similarity(
    original_hidden.flatten(),
    pruned_hidden.flatten(),
    dim=0
)

print("Cosine similarity:", sim.item())
diff = original_hidden - pruned_hidden

print("Mean abs diff:",
      diff.abs().mean().item())

print("Max abs diff:",
      diff.abs().max().item())

orig_text = model.generate(
    **inputs,
    max_new_tokens=50
)

print(
    tokenizer.decode(
        orig_text[0],
        skip_special_tokens=True
    )
)
pruned_text = pruned_model.generate(
    **inputs,
    max_new_tokens=50
)

print(
    tokenizer.decode(
        pruned_text[0],
        skip_special_tokens=True
    )
)

#lets drop layers 0,5,10,15,20 and see the effect on the output
pruned_model = copy.deepcopy(model)
layers = list(pruned_model.model.layers)
drop_set = {0,5,10,15,20}

layers = [
    layer
    for idx, layer in enumerate(layers)
    if idx not in drop_set
]
pruned_model.model.layers = nn.ModuleList(layers)       
with torch.no_grad():
    original_outputs = model(**inputs, output_hidden_states=True)
    pruned_outputs = pruned_model(**inputs, output_hidden_states=True)
original_hidden = original_outputs.hidden_states[-1]
pruned_hidden = pruned_outputs.hidden_states[-1]
sim = F.cosine_similarity(
    original_hidden.flatten(),
    pruned_hidden.flatten(),    
    dim=0
)
print("Cosine similarity after dropping multiple layers:", sim.item())
diff = original_hidden - pruned_hidden
print("Mean abs diff after dropping multiple layers:",
        diff.abs().mean().item())
print("Max abs diff after dropping multiple layers:",
        diff.abs().max().item())
pruned_text = pruned_model.generate(
    **inputs,
    max_new_tokens=50
)
print(
    tokenizer.decode(
        pruned_text[0],
        skip_special_tokens=True
    )
)

for layer_idx in [0,5,10,15,20]:

    pruned_model = copy.deepcopy(model)

    layers = list(pruned_model.model.layers)

    del layers[layer_idx]

    pruned_model.model.layers = nn.ModuleList(layers)

    with torch.no_grad():
        pruned_outputs = pruned_model(
            **inputs,
            output_hidden_states=True
        )

    pruned_hidden = pruned_outputs.hidden_states[-1]

    sim = F.cosine_similarity(
        original_hidden.flatten(),
        pruned_hidden.flatten(),
        dim=0
    )

    print(
        f"Layer {layer_idx}: "
        f"{sim.item():.4f}"
    )