import torch
from megablocks.layers.arguments import Arguments
from megablocks.layers.router import LearnedRouter

args = Arguments(hidden_size=128, moe_num_experts=8, moe_top_k=2,
                 moe_latent_size=32, moe_num_anchors=4, fp16=False, bf16=False)
router = LearnedRouter(args).cuda()
x = torch.randn(4, 16, 128).cuda()
scores, logits, weights, indices = router(x)
print('Router OK!')
print('indices:', indices.shape)
print('weights:', weights.shape)
