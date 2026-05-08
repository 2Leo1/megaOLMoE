from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch

class _UniformExpertAssignment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        self.down_proj = torch.nn.Linear(
            args.hidden_size,
            args.moe_latent_size,
            bias=False,
            dtype=common.dtype(args),
            device=args.device,
        )
        args.init_method(self.down_proj.weight)

        self.anchors = torch.nn.Parameter(
            torch.empty(
                args.moe_num_experts,
                args.moe_num_anchors,
                args.moe_latent_size,
                dtype=common.dtype(args),
                device=args.device,
            )
        )
        args.init_method(self.anchors)

        self.gamma = torch.nn.Parameter(
            torch.ones(1, dtype=common.dtype(args), device=args.device)
        )
        self.beta = torch.nn.Parameter(
            torch.ones(1, dtype=common.dtype(args), device=args.device)
        )

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def input_rmsnorm(self, x):
        return x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) +
            self.args.moe_router_rmsnorm_eps
        )

    def project_q(self, x):
        if self.args.moe_router_use_input_rmsnorm:
            x = self.input_rmsnorm(x)
        return self.down_proj(x)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    #@torch.compile(mode="default")
    def _compute_logits(self, x):
        q = self.project_q(x)

        q_norm = torch.norm(q, p=2.0, dim=-1, keepdim=True)
        q_norm = q_norm.clamp_min(self.args.moe_norm_eps)

        k_norm = torch.norm(self.anchors, p=2.0, dim=-1, keepdim=True)
        k_norm = k_norm.clamp_min(self.args.moe_norm_eps)

        phi_q = self.gamma * (1.0 + self.beta * torch.tanh(q_norm))
        psi_k = 1.0 + (k_norm - 1.0) / self.args.sips_p

        q_dir = q / q_norm
        k_dir = self.anchors / k_norm

        q_scaled = q_dir * phi_q
        k_scaled = k_dir * psi_k
        k_scaled_flat = k_scaled.view(-1, self.args.moe_latent_size)
        z_raw = (q_scaled @ k_scaled_flat.T.contiguous()).view(q.shape[0], self.args.moe_num_experts, self.args.moe_num_anchors)
        logits = torch.logsumexp(z_raw, dim=-1)
        return logits

    def forward(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        flattened_x = x.view(-1, x.shape[-1])
        logits = self._compute_logits(flattened_x)
        scores = torch.softmax(logits, dim=-1)
        routing_logits = logits
        if self.training and self.args.moe_router_logit_jitter_eps is not None:
            eps = self.args.moe_router_logit_jitter_eps
            routing_logits = routing_logits + torch.empty_like(routing_logits).uniform_(-eps, eps)
        _, expert_indices = self._top_k(routing_logits)
        expert_logits = logits.gather(dim=-1, index=expert_indices)
        expert_weights = torch.softmax(expert_logits, dim=-1)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, logits, expert_weights, expert_indices
