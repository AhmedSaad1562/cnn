import random
import math

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2

random.seed(42) 
w1 = random.uniform(-0.5, 0.5)
w2 = random.uniform(-0.5, 0.5)
w3 = random.uniform(-0.5, 0.5)
w4 = random.uniform(-0.5, 0.5)

# Initialize biases
b1 = 0.5
b2 = 0.7

def forward_pass(x1, x2):
    h1_input = w1 * x1 + w2 * x2 + b1
    h1_output = tanh(h1_input)

    h2_input = w3 * h1_output + w4 * x2 + b2
    h2_output = tanh(h2_input)
    
    return h2_output

x1, x2 = 0.5, 0.3

# Perform the forward pass
output = forward_pass(x1, x2)
print(f"Output: {output}")