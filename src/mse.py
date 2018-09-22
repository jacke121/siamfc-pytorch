

import torch

a=torch.FloatTensor((0.1,1))
b=torch.FloatTensor((0.1,1))

loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)

print(loss_fn(a,b))