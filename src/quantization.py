import numpy as np
import torch
def quantize_per_tensor_symmetric(x: torch.Tensor, bits: int=8):
    alpha = x.abs().max() # maximum magnitude 
    S = alpha/(2**(bits-1)-1) # scale factor 
    x_quantized = torch.clamp(torch.round(x/S),-(2**(bits-1)),2**(bits-1)-1)
    return x_quantized ,S
def dequantize(x_quantized: torch.Tensor,S:float):
    return x_quantized*S

x = torch.tensor([1.5,-2.3,0.7,-0.1,3.1])
x_quantized, S = quantize_per_tensor_symmetric(x)
x_deq = dequantize(x_quantized, S)
print(f"Original:     {x}")
print(f"Quantized:    {x_quantized}")
print(f"Dequantized:  {x_deq}")
print(f"Error:        {(x - x_deq).abs().max():.5f}")
