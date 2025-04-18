import torch
import torch.nn.functional as F

# x = torch.ones((1, 3072, 32, 64), dtype=torch.bfloat16, device='cuda')
attention_mask = torch.load('attention_mask.pt').to(device='cuda', dtype=torch.bfloat16)
# attention_mask = torch.ones((1, 3072), dtype=torch.bfloat16, device='cuda')
seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
max_seqlen_in_batch = seqlens_in_batch.max().item()
cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
print("attention_mask: ",attention_mask, ", shape: ", attention_mask.shape)
print("seqlens_in_batch: ", seqlens_in_batch, ", shape: ", seqlens_in_batch.shape)
print("indices: ", indices, ", shape: ", indices.shape)
print("max_seqlen_in_batch: ", max_seqlen_in_batch)
print("cu_seqlens: ", cu_seqlens, ", shape: ", cu_seqlens.shape)
# exit(0)



x = torch.load('gk.pt').to(device='cuda', dtype=torch.bfloat16)
# cu_seqlens = torch.load('cu_seqlens.pt').to(device='cuda', dtype=torch.int32)
print(x.shape)
print(cu_seqlens)

from fla.ops.utils import chunk_local_cumsum

y = chunk_local_cumsum(x.contiguous(), 64, cu_seqlens=cu_seqlens)

print("y:", y)