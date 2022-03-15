# Weights
wac = 0.9907079
wad = 0.01417504
wae = 0.88046944
 
wbc = 1.0264927
wbd = -0.8950311
wbe = 0.7524377
 
wcf = 0.794296
wdf = 1.1687347
wef = 0.2406084
 
# Biases
bc = -0.00070612
bd = -0.06846002
be = -0.00055442
bf = -0.00000929
 
def relu(x):
    return max(0, x)
 
def run(a, b):
    c = (a * wac) + (b * wbc) + bc
    d = (a * wad) + (b * wbd) + bd
    e = (a * wae) + (b * wbe) + be
    f = (relu(c) * wcf) + (relu(d) * wdf) + (relu(e) * wef) + bf
    return f
