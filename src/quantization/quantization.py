import numpy as np
import torch
def quantize_per_tensor_symmetric(x: torch.Tensor, bits: int=8):
    alpha = x.abs().max() # maximum magnitude 
    S = alpha/(2**(bits-1)-1) # scale factor 
    x_quantized = torch.clamp(torch.round(x/S),-(2**(bits-1)),2**(bits-1)-1) #make a range by clamp
    return x_quantized ,S
def dequantize(x_quantized: torch.Tensor,S:float):
    return x_quantized*S

def quantize_per_channel_symmetric(x: torch.Tensor, bits:int = 8):
    alpha = x.abs().max(dim = 1, keepdim = True).values 
    # finding max for every row here coz now we're scaling per row 
    S = alpha/(2**(bits-1)-1)
    x_quant = torch.clamp(torch.round(x/S),-(2**(bits-1)),2**(bits-1)-1)
    return x_quant, S
def dequantize_per_channel(x_quant: torch.Tensor, S: torch.Tensor):
    return x_quant * S

def asymmetric_quantization(x,bits =8):
    S = (x.max()-x.min())/(2**bits-1) # scale factor 
    z = torch.round(-x.min()/S)
    x_quantized = torch.clamp(torch.round(x/S)+z,0,2**bits-1) 
    return x_quantized ,S,z

def asymmetric_dequantization(x_quantized,S,z):
    return (x_quantized-z)*S


# for single tensor
x = torch.tensor([1.5,-2.3,0.7,-0.1,3.1])
x_quantized, S = quantize_per_tensor_symmetric(x)
x_deq = dequantize(x_quantized, S)


#for entire channel 
x2 = torch.tensor([
    [0.01, -0.02, 0.005, -0.008],   # tiny range
    [50.0, -80.0, 30.0, -100.0]     # huge range
    #basically experimenting with a larger range 
])

x_q, S2 = quantize_per_channel_symmetric(x2)
x_deq2 = dequantize_per_channel(x_q, S2)

print(f"Scale factors: {S2.squeeze()}")
print(f"Original:\n{x2}")
print(f"Dequantized:\n{x_deq2}")
print(f"Per-row error: {(x2 - x_deq2).abs().max(dim=1).values}")
print(f"Original:     {x2}")
print(f"Quantized:    {x_q}")
print(f"Dequantized:  {x_deq2}")
print(f"Error:        {(x2 - x_deq2).abs().max():.5f}")


print("\n Asymmetric Dequantization : \n")
#asymmetric quantization/dequantization 
x3= torch.tensor([0.5, 1.2, 3.3, 2.1, 0.8])
xq3, s3,z = asymmetric_quantization(x3)
x_deq3 = asymmetric_dequantization(xq3,s3,z)
print(f"Original:    {x3}")
print(f"Quantized:   {xq3}")
print(f"Dequantized: {x_deq3}")
print(f"Error:       {(x3 - x_deq3).abs().max():.5f}")