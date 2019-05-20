# cy_sum.pyx

def cy_sum(data):
    s = 0.0
    for d in data:
        s += d
    return s