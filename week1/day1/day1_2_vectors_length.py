import numpy as np

v = np.array([3, 4])
# Pythagoras sats för att ta reda på längden av en vektor (c)
# a^2 + b^2 = roten ur c
# 3*3 + 4*4 = 25 = roten ur 25 = 5

length = np.linalg.norm(v)
print("Längd av v:", length)