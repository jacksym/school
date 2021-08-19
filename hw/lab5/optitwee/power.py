import numpy as np

def power(current):
    power = 0.8154*current - 41.954
    return power

print(power(70))
print(power(58.5))
print(power(62))
