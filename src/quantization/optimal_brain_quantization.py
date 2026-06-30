import numpy as np
from numpy import linalg 
#the standard way of going about this stuff is that we round off all the elements, 1.72 ->2 -0.63 -> -1 and 2.18 ->2
w = np.array([1.72, -0.63, 2.18])
wq_naive = np.round(w[0]) # Quantize only the first weight for now.

error = wq_naive-w[0] 
print("error = ", error)

#now in obq, our objective is to somehow distribute this error
# so basically, there are 2 other weights, hence, error/2 and then distribute that to both the elements 
# This is the intuition behind OBQ.
rem = len(w)-1
dist = error/rem
w[1]-=dist
w[2]-=dist
w[0]= wq_naive
print("Original:", w)
print("Quantized first weight:", wq_naive)

# now what we did was quantize then run into an error, then compensate for that error
# what we did was really just naive distribution, how do we know which remaining weight should absorb more of the error?
# we introduce hessian, coz that will tell us about the weights that can move the most while increasing the loss the least.
# importance of weight and hence the amount to be distributed are supposed to be inverse
wt = np.array([0.8,0.2])
inverse = 1/wt
inverse /= inverse.sum()
print("weights inverted based on how they move: ",inverse)

# now distribute
w[1] -= error * inverse[0]
w[2] -= error * inverse[1]
print("Quantization error:", error)
print("Updated weights:", w)

# according to OBTQ Paper, H=2*X*X_transpose
X = np.array([[1,2,3],[2,1,5],[4,0,1]], dtype=float)
H = 2*(X@X.T)
print("Hessian Matrix: \n",H)
H_inv = np.linalg.inv(H)
print("Inverted Hessian matrix:\n " ,H_inv)


# Actual OBQ Update equation- Delta(F) = -((W - W_quantized)/[diagonal_element])* [column of inverse Hessian]
weight_vector = np.array([1.72,-0.63,2.18])
q = 0
current_weight = weight_vector[q]
quantized_weight = np.round(current_weight)
print("Current Weight: ",current_weight)
print("Quantized weight: " ,quantized_weight)
quant_error = current_weight - quantized_weight
print("Quantization Error: ",quant_error)
column_q = H_inv[:, q]
diag_element = H_inv[q,q]
delta = (-(quant_error)/diag_element)*column_q
print("Delta = ",delta)


# once the updation is done, wF​←wF​+Delta_F​
updated_weight_vector = weight_vector + delta
updated_weight_vector[q] = quantized_weight # juss keep the quantized weight fixed
print("Updated Weight Vector: ", updated_weight_vector)
