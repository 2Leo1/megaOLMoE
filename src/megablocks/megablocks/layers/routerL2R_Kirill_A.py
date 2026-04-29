from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch
import torch.nn.functional as F

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

        #Сферическая инициализация якорей
        with torch.no_grad():
            self.anchors.data = F.normalize(self.anchors.data, p=2.0, dim=-1)

        self.gamma = torch.nn.Parameter(
            torch.ones(1, dtype=common.dtype(args), device=args.device) * args.sips_gamma
        )
        self.beta = torch.nn.Parameter(
            torch.ones(1, dtype=common.dtype(args), device=args.device) * args.sips_beta
        )

    #Нормальный jitter(предотвращает коллапс маршрутизации в первые эпохи, правильное распределение токенов)
    def jitter(self, x):
        eps = self.args.moe_jitter_eps
        if eps is None:
            return x
        low = 1.0 - eps
        high = 1.0 + eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def input_rmsnorm(self, x):
        #Более оптимальный расчет дисперсии
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.args.moe_router_rmsnorm_eps)

    def project_q(self, x):
        if self.args.moe_router_use_input_rmsnorm:
            x = self.input_rmsnorm(x)
        return self.down_proj(x)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    @torch.compile(mode="reduce-overhead") #тут подумать мб(просто чтобы не забыть)
    def _compute_logits(self, x):
        q = self.project_q(x)

        #torch.linalg.vector_norm работает лучше, чем torch.norm
        #.clamp_min_ (в отличие от .clamp_min) в рамках одного тензора это экономит аллокацию памяти под новый объект
        q_norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min_(self.args.moe_norm_eps)
        k_norm = torch.linalg.vector_norm(self.anchors, dim=-1, keepdim=True).clamp_min_(self.args.moe_norm_eps)

        phi_q = self.gamma * (1.0 + self.beta * torch.tanh(q_norm))
        psi_k = 1.0 + (k_norm - 1.0) / self.args.sips_p

        #Объединение деления и умножения(просто немного экономим память)
        q_scaled = (q / q_norm) * phi_q
        k_scaled = (self.anchors / k_norm) * psi_k
        k_scaled_flat = k_scaled.view(-1, self.args.moe_latent_size)

        #F.linear работает быстрее благодаря вызову cuBLAS напрямую под капотом
        z_raw = F.linear(q_scaled, k_scaled_flat)
        z_raw = z_raw.view(q.shape[0], self.args.moe_num_experts, self.args.moe_num_anchors)

        logits = torch.logsumexp(z_raw, dim=-1)
        return logits

    def forward(self, x):
        #Возвращен джиттер для регуляризации
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        flattened_x = x.view(-1, x.shape[-1])
        logits = self._compute_logits(flattened_x)

        #Использование F.softmax (fused kernel) вместо ручной математики(должно работать быстрее и лучше)
        scores = F.softmax(logits, dim=-1)
        expert_weights, expert_indices = self._top_k(scores)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.linalg.vector_norm(
                expert_weights,
                ord=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, logits, expert_weights, expert_indices
