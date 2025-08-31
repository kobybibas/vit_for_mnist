import torch
a = torch.arange(3)  # tensor([0, 1, 2])

print(a[0:3])  # tensor([0, 1, 2])
print(a[2:5])  # tensor([2])       <-- shorter, not rotated
print(a[3:6])  # tensor([])        <-- empty