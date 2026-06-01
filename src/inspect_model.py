from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
layer = model.model.layers[0]

print(model)
print(type(model))
print("Number of Layers: \n")
print(len(model.model.layers))
print(model.model.layers[22])
print(model.model.layers[22].self_attn)
attn = layer.self_attn
print(attn.q_proj.weight.shape)
print(attn.k_proj.weight.shape)
print(attn.v_proj.weight.shape)
print(attn.o_proj.weight.shape)
print(layer.mlp)
layer_params = list(layer.parameters())
print("Number of parameters in the first layer: ", len(layer_params))
print("Shape of the first parameter: ", layer_params[1].shape)

l1 = model.model.layers[1]

diff = (
    l1.self_attn.q_proj.weight
    - layer.self_attn.q_proj.weight
)
print(diff)
layer_params = sum(
    p.numel()
    for p in layer.parameters()
)

print(layer_params)
layer0 = model.model.layers[0]

activations = {}
def hook_fwd(module, input, output):
    activations["layer0"]=output
handle = layer0.register_forward_hook(hook_fwd)
inputs = tokenizer("hello world", return_tensors="pt")

with torch.no_grad():
    model(**inputs)
print(activations["layer0"].shape)