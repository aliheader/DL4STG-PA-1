import math

import torch
from torch import nn
from torch.nn import functional as F


class SeriesDecomp(nn.Module):
    """Centred moving average with replicate padding. seasonal + trend == x exactly."""

    def __init__(self, kernel):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x):  # [B,L,C] -> (seasonal, trend)
        radius = self.kernel // 2
        padded = F.pad(x.transpose(1, 2), (radius, radius), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """Period-level dependency discovery: FFT delay scores, top-k delays, time-delay aggregation."""

    def __init__(self, d, heads, factor=2.0):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.heads, self.factor = heads, factor
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(d, d) for _ in range(3))
        self.out = nn.Linear(d, d)

    def forward(self, q_in, k_in, v_in):
        batch, len_q, width = q_in.shape
        len_k = k_in.shape[1]
        heads, head_dim = self.heads, width // self.heads
        q = self.q_proj(q_in).view(batch, len_q, heads, head_dim)
        k = self.k_proj(k_in).view(batch, len_k, heads, head_dim)
        v = self.v_proj(v_in).view(batch, len_k, heads, head_dim)
        if len_k > len_q:
            k, v = k[:, :len_q], v[:, :len_q]
        elif len_k < len_q:
            pad = len_q - len_k
            k, v = F.pad(k, (0, 0, 0, 0, 0, pad)), F.pad(v, (0, 0, 0, 0, 0, pad))
        q, k, v = (t.permute(0, 2, 3, 1) for t in (q, k, v))

        qs, ks = q - q.mean(-1, keepdim=True), k - k.mean(-1, keepdim=True)
        corr = torch.fft.irfft(
            torch.fft.rfft(qs, dim=-1) * torch.fft.rfft(ks, dim=-1).conj(),
            n=len_q,
            dim=-1,
        )
        scores = corr.mean(2)

        top = max(1, min(int(self.factor * math.log(max(len_q, 2))), len_q))
        weight, delay = scores.topk(top, dim=-1)  # [B,H,K]
        weight = weight.softmax(-1)

        mixed = aggregate_delays(v, delay, weight)
        return self.out(mixed.permute(0, 3, 1, 2).reshape(batch, len_q, width))


def aggregate_delays(values, delays, weights):
    """Time-delay aggregation (same contract as Question 1's `aggregate_delays`).
    values [B,H,dh,L]; delays, weights [B,H,K]. Output position t reads values[t - delay]
    (circularly), weighted: z_t = sum_j weights_j * values[(t - delays_j) mod L]."""
    batch, heads, head_dim, length = values.shape
    positions = torch.arange(length, device=values.device)
    mixed = torch.zeros_like(values)
    for j in range(delays.shape[-1]):
        src = (positions.view(1, 1, length) - delays[..., j : j + 1]) % length
        index = src.long().unsqueeze(2).expand(batch, heads, head_dim, length)
        mixed = mixed + weights[..., j].view(batch, heads, 1, 1) * values.gather(
            -1, index
        )
    return mixed


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, kernel, dropout):
        super().__init__()
        self.corr = AutoCorrelation(d, heads)
        self.decomp1, self.decomp2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.drop(self.corr(x, x, x)))
        x, _ = self.decomp2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, kernel, dropout):
        super().__init__()
        self.self_corr, self.cross_corr = AutoCorrelation(d, heads), AutoCorrelation(
            d, heads
        )
        self.decomp1, self.decomp2, self.decomp3 = (
            SeriesDecomp(kernel) for _ in range(3)
        )
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )
        self.drop = nn.Dropout(dropout)
        self.trend_proj = nn.Linear(d, 1, bias=False)

    def forward(self, x, memory):
        x, t1 = self.decomp1(x + self.drop(self.self_corr(x, x, x)))
        x, t2 = self.decomp2(x + self.drop(self.cross_corr(x, memory, memory)))
        x, t3 = self.decomp3(x + self.drop(self.ff(x)))
        return x, self.trend_proj(t1 + t2 + t3)


class Autoformer(nn.Module):
    def __init__(
        self,
        enc_in=1,
        dec_in=1,
        d=32,
        heads=4,
        e_layers=1,
        d_layers=1,
        kernel=25,
        label_len=168,
        pred_len=168,
        dropout=0.1,
        damped_trend=False,
    ):
        super().__init__()
        self.label_len, self.pred_len = label_len, pred_len

        self.damping = (
            nn.Parameter(torch.logit(torch.tensor(0.8))) if damped_trend else None
        )
        self.decomp = SeriesDecomp(kernel)
        self.enc_embed = nn.Conv1d(
            enc_in, d, 3, padding=1, padding_mode="circular", bias=False
        )
        self.dec_embed = nn.Conv1d(
            dec_in, d, 3, padding=1, padding_mode="circular", bias=False
        )
        self.encoder = nn.ModuleList(
            [EncoderLayer(d, heads, kernel, dropout) for _ in range(e_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d, heads, kernel, dropout) for _ in range(d_layers)]
        )
        self.enc_norm, self.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.seasonal_proj = nn.Linear(d, 1)

    def forward(self, x_enc, x_dec_cov=None):

        batch = x_enc.shape[0]
        seasonal_hist, trend_hist = self.decomp(x_enc[..., :1])

        level = x_enc[:, -self.label_len :, :1].mean(1, keepdim=True)
        horizon_trend = level.expand(batch, self.pred_len, 1)
        if self.damping is not None:
            steps = torch.arange(
                1, self.pred_len + 1, device=x_enc.device, dtype=x_enc.dtype
            )
            decay = torch.sigmoid(self.damping) ** steps.view(1, -1, 1)
            horizon_trend = level + (x_enc[:, -1:, :1] - level) * decay
        trend_init = torch.cat([trend_hist[:, -self.label_len :], horizon_trend], dim=1)
        seasonal_init = torch.cat(
            [
                seasonal_hist[:, -self.label_len :],
                torch.zeros(
                    batch, self.pred_len, 1, device=x_enc.device, dtype=x_enc.dtype
                ),
            ],
            dim=1,
        )

        memory = self.enc_embed(x_enc.transpose(1, 2)).transpose(1, 2)
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.enc_norm(memory)

        dec_in = (
            seasonal_init
            if x_dec_cov is None
            else torch.cat([seasonal_init, x_dec_cov], -1)
        )
        x = self.dec_embed(dec_in.transpose(1, 2)).transpose(1, 2)
        trend = trend_init
        for layer in self.decoder:
            x, delta = layer(x, memory)
            trend = trend + delta
        out = self.seasonal_proj(self.dec_norm(x)) + trend
        return out[:, -self.pred_len :, 0]  # [B, pred_len]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------
# Self-tests (Step 3 gate): run these before training anything
# ---------------------------------------------------------------------------------------------
def run_self_tests(param_budget: int = 40_000) -> int:
    """Assertions (a)-(g). Raises AssertionError naming the failing test; returns P (10 covariates)."""
    # (a) decomposition reconstructs the input exactly
    x = torch.randn(2, 50, 3)
    s, t = SeriesDecomp(25)(x)
    assert torch.allclose(s + t, x, atol=1e-6), "(a) seasonal + trend != x"

    # (b) a constant series has a constant trend (replicate padding, not zero padding)
    c = torch.full((1, 40, 1), 3.7)
    assert torch.allclose(
        SeriesDecomp(25)(c)[1], c, atol=1e-6
    ), "(b) constant trend broken"

    # (c) target-only model
    torch.manual_seed(0)
    m_a = Autoformer(enc_in=1, dec_in=1)
    assert m_a(torch.randn(4, 336, 1)).shape == (4, 168), "(c) wrong output shape"

    # (d) model with past + future covariates
    m_c = Autoformer(enc_in=11, dec_in=11)
    x_enc, x_dec = torch.randn(4, 336, 11), torch.randn(4, 336, 10)
    out = m_c(x_enc, x_dec)
    assert out.shape == (4, 168), "(d) wrong output shape"

    # (e) every parameter receives a gradient
    out.pow(2).mean().backward()
    missing = [n for n, p in m_c.named_parameters() if p.grad is None]
    assert not missing, f"(e) no gradient for {missing}"

    # (f) same seed -> identical output
    def run(seed):
        torch.manual_seed(seed)
        m = Autoformer(enc_in=11, dec_in=11).eval()
        return m(x_enc, x_dec)

    assert torch.equal(run(1), run(1)), "(f) not deterministic"

    # (g) parameter budget
    p_a, p_c = count_params(m_a), count_params(m_c)
    print(
        f"P target-only = {p_a:,}   P with 10 covariates = {p_c:,}   budget = {param_budget:,}"
    )
    assert p_c <= param_budget, "(g) over parameter budget"

    print("All 7 Autoformer self-tests passed.")
    return p_c


if __name__ == "__main__":
    run_self_tests()
