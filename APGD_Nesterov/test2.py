import numpy as np

k=10
while k>2 and True:
	print(k)
	k-=1

r = [np.zeros([6, ])]
r[0][1]=4
print(r[0][1])
print(type(r[0]))