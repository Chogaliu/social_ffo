import numpy as np

record = [1, 2, 3, 4]
int_list = [int(i) for i in record]
idx = int_list.index(min(int_list))
print(idx)
