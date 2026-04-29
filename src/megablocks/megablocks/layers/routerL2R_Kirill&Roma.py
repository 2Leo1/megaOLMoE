from megablocks.layers import common
from megablocks.layers.arguments import Arguments
import torch
import torch.nn.functional as F


class _UniformExpertAssignment(torch.autograd.Function):
    # Оптимизированная версия: генерируем повторяющийся паттерн [0,1,...,num_experts-1]
    # вместо огромного тензора torch.arange(x.numel())
    @staticmethod
    def forward(ctx, x, num_experts):
        batch_size, top_k = x.shape
        device = x.device
        pattern = torch.arange(num_experts, device=device)
        repeats = (batch_size * top_k + num_experts - 1) // num_experts
        flat = pattern.repeat(repeats)[:batch_size * top_k]
        return flat.view(batch_size, top_k)


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

        # Сферическая инициализация якорей
        with torch.no_grad():
            self.anchors.data = F.normalize(self.anchors.data, p=2.0, dim=-1)

        # Защита через getattr на случай, если параметров нет в Arguments
        gamma_val = getattr(args, 'sips_gamma', 1.0)
        beta_val = getattr(args, 'sips_beta', 1.0)
        self.p_val = getattr(args, 'sips_p', 4.0)

        self.gamma = torch.nn.Parameter(
            torch.ones(1, dtype=common.dtype(args), device=args.device) * gamma_val
        )
        self.beta = torch.nn.Parameter(
            torch.ones(1, dtype=common.dtype(args), device=args.device) * beta_val
        )

        # Элегантная регистрация tau.
        # Теперь tau всегда тензор, что убирает лишние if-проверки при forward
        if getattr(args, 'learnable_tau', False):
            self.tau = torch.nn.Parameter(torch.ones(1, dtype=common.dtype(args), device=args.device))
        else:
            self.register_buffer('tau', torch.ones(1, dtype=common.dtype(args), device=args.device))

        # Буфер для безопасного кэширования якорей при инференсе (генерации)
        self.register_buffer("_cached_k_scaled_flat", None)
        #Детектор изменения anchors для надежной инвалидации кэша
        self._anchors_version = None

    # Нормальный jitter (предотвращает коллапс маршрутизации в первые эпохи, правильное распределение токенов)
    def jitter(self, x):
        eps = self.args.moe_jitter_eps
        if eps is None:
            #Возвращаем скаляр 1.0. Умножение на скаляр не аллоцирует память,
            # в отличие от torch.ones_like(x), который сжирает VRAM на создание матрицы единиц.
            return 1.0
        low = 1.0 - eps
        high = 1.0 + eps
        # Более чистая генерация шума через rand_like
        noise = torch.rand_like(x)
        return low + noise * (high - low)

    def input_rmsnorm(self, x):
        # Более оптимальный расчет дисперсии
        # Вычисление RMSNorm в float32 для стабильности при обучении в fp16/bf16
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.args.moe_router_rmsnorm_eps).to(x.dtype)

    def project_q(self, x):
        if self.args.moe_router_use_input_rmsnorm:
            x = self.input_rmsnorm(x)
        return self.down_proj(x)

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    # Лосс балансировки
    def _compute_load_balance_loss(self, expert_indices):
        """Вспомогательный лосс балансировки экспертов (λ_bal = 0.01 по статье L2R)."""
        num_tokens = expert_indices.shape[0]
        num_experts = self.args.moe_num_experts
        flat = expert_indices.flatten()
        counts = torch.bincount(flat, minlength=num_experts)
        probs = counts.float() / (num_tokens * self.args.moe_top_k)
        target = 1.0 / num_experts
        return torch.mean((probs - target) ** 2)

    @torch.compile(mode="reduce-overhead")
    def _compute_logits(self, x):
        q = self.project_q(x)

        # Считаем нормы строго в float32, чтобы избежать underflow/overflow (нулей или бесконечностей)
        q_f32 = q.float()

        # torch.linalg.vector_norm работает лучше, чем torch.norm
        # .clamp_min_ (в отличие от .clamp_min) в рамках одного тензора это экономит аллокацию памяти под новый объект
        q_norm = torch.linalg.vector_norm(q_f32, dim=-1, keepdim=True).clamp_min_(self.args.moe_norm_eps)

        # Приводим параметры gamma и beta к float32 для стабильной математики
        phi_q = self.gamma.float() * (1.0 + self.beta.float() * torch.tanh(q_norm))

        # Объединение деления и умножения (просто немного экономим память)
        q_scaled = (q_f32 / q_norm) * phi_q
        q_scaled = q_scaled.to(q.dtype)  # Возвращаем в исходный тип (fp16/bf16) для быстрого matmul

        #Кэширование якорей с проверкой встроенного PyTorch счетчика _version
        if (not self.training and self._cached_k_scaled_flat is not None and
            self._anchors_version == self.anchors._version):
            k_scaled_flat = self._cached_k_scaled_flat
        else:
            # Считаем якоря в float32 (БЕЗ no_grad, чтобы градиенты текли!)
            anchors_f32 = self.anchors.float()
            k_norm = torch.linalg.vector_norm(anchors_f32, dim=-1, keepdim=True).clamp_min_(self.args.moe_norm_eps)
            psi_k = 1.0 + (k_norm - 1.0) / self.p_val

            k_scaled = (anchors_f32 / k_norm) * psi_k
            k_scaled_flat = k_scaled.view(-1, self.args.moe_latent_size).to(q.dtype)

            # Сохраняем кэш для режима оценки и запоминаем версию anchors
            if not self.training:
                self._cached_k_scaled_flat = k_scaled_flat.detach()
                self._anchors_version = self.anchors._version

        # F.linear работает быстрее благодаря вызову cuBLAS напрямую под капотом (быстрее чем einsum с 3D)
        z_raw = F.linear(q_scaled, k_scaled_flat)
        z_raw = z_raw.view(q.shape[0], self.args.moe_num_experts, self.args.moe_num_anchors)

        logits = torch.logsumexp(z_raw, dim=-1)

        logits = logits / self.tau.abs().clamp_min(1e-4)

        return logits

    def forward(self, x):
        # Возвращен джиттер для регуляризации
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        # Сохраняем исходную форму, чтобы вернуть корректные 3D/4D тензоры, если они были на входе
        original_shape = x.shape
        flattened_x = x.view(-1, original_shape[-1])

        logits = self._compute_logits(flattened_x)

        # Использование F.softmax (fused kernel) вместо ручной математики (должно работать быстрее и лучше)
        scores = F.softmax(logits, dim=-1)
        expert_weights, expert_indices = self._top_k(scores)

        #Лосс балансировки – вычисляем ДО изменения формы, жестко фиксируя размерность
        loss_balance = None
        if self.training:
            flat_indices = expert_indices.detach().view(-1, self.args.moe_top_k)
            loss_balance = self._compute_load_balance_loss(flat_indices)

        if self.args.moe_normalize_expert_weights:
            # Нормализация весов экспертов тоже в float32 для исключения NaN лоссов
            weights_f32 = expert_weights.float()
            expert_weights = expert_weights / torch.linalg.vector_norm(
                weights_f32,
                ord=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            ).to(expert_weights.dtype)

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if getattr(self.args, 'uniform_expert_assignment', False) else expert_indices
        )

        # Восстанавливаем batch-размерности, если вход был многомерным
        batch_shape = original_shape[:-1]
        if len(batch_shape) > 1:
            expert_weights = expert_weights.view(*batch_shape, -1)
            expert_indices = expert_indices.view(*batch_shape, -1)
            logits = logits.view(*batch_shape, -1)
            scores = scores.view(*batch_shape, -1)

        return scores, logits, expert_weights, expert_indices, loss_balance
