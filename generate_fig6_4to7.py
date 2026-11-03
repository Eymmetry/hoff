"""Hoff Fig 6.4-6.7 の再現：MC と MCMC の違い、および MCMC 診断。

Fig 6.4: 3成分正規混合の真の密度と、独立モンテカルロ標本のヒストグラム
Fig 6.5: ギブス標本 1,000 個のヒストグラムとトレースプロット
Fig 6.6: ギブス標本 10,000 個のヒストグラムとトレースプロット
Fig 6.7: 6.4節の半共役正規モデルの θ, σ² のトレースプロット
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 自己相関と有効標本サイズ
# ============================================================
def acf(x, lag):
    """ラグ t の標本自己相関（Hoff の acf_t の定義）"""
    x = np.asarray(x, dtype=float)
    S = len(x)
    m = x.mean()
    num = np.sum((x[:S - lag] - m) * (x[lag:] - m)) / (S - lag)
    den = np.sum((x - m) ** 2) / (S - 1)
    return num / den


def effective_size(x):
    """有効標本サイズ。Geyer の initial positive sequence で自己相関和を打ち切る。
    coda の effectiveSize（AR スペクトル法）とは実装が違うので値は厳密には一致しない。"""
    x = np.asarray(x, dtype=float)
    S = len(x)
    xc = x - x.mean()
    f = np.fft.rfft(xc, 2 * S)
    acov = np.fft.irfft(f * np.conj(f))[:S] / S
    rho = acov / acov[0]
    total, t = 0.0, 1
    while t + 1 < S:
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
        t += 2
    return S / (1 + 2 * total)


# ============================================================
# 正規混合分布の設定
# ============================================================
pd_ = np.array([0.45, 0.10, 0.45])       # Pr(δ = 1, 2, 3)
mu = np.array([-3.0, 0.0, 3.0])
sd = np.sqrt(np.array([1 / 3, 1 / 3, 1 / 3]))

xs = np.linspace(-6, 6, 1000)
true_dens = sum(p * stats.norm.pdf(xs, m, s) for p, m, s in zip(pd_, mu, sd))

rng = np.random.default_rng(1)

# ------------------------------------------------------------
# (a) 独立モンテカルロ：δ を周辺から引き、その δ で θ を引く
# ------------------------------------------------------------
S_MC = 1000
d_mc = rng.choice(3, size=S_MC, p=pd_)
th_mc = rng.normal(mu[d_mc], sd[d_mc])

# ------------------------------------------------------------
# (b) ギブスサンプラー：θ と δ を交互に引く
# ------------------------------------------------------------
def gibbs_mixture(S, theta0=0.0, seed=1):
    r = np.random.default_rng(seed)
    th = np.zeros(S)
    theta = theta0
    for s in range(S):
        # δ | θ ∝ Pr(δ=d) × dnorm(θ, μ_d, σ_d)
        w = pd_ * stats.norm.pdf(theta, mu, sd)
        w = w / w.sum()
        d = r.choice(3, p=w)
        # θ | δ
        theta = r.normal(mu[d], sd[d])
        th[s] = theta
    return th


th_gibbs_1k = gibbs_mixture(1000)
th_gibbs_10k = gibbs_mixture(10000)

# ============================================================
# Fig 6.4：真の密度と独立 MC 標本
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(th_mc, bins=50, density=True, color='gray', alpha=0.45,
        edgecolor='none', label='1000 independent MC samples')
ax.plot(xs, true_dens, color='black', linewidth=1.6, label=r'exact $p(\theta)$')
ax.set_xlim(-6, 6)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p(\theta)$')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('fig6_4.pdf', bbox_inches='tight')
plt.savefig('fig6_4.png', bbox_inches='tight', dpi=150)
plt.close()


# ============================================================
# Fig 6.5 / 6.6：ギブス標本のヒストグラムとトレースプロット
# ============================================================
def hist_and_trace(th, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(th, bins=50, density=True, color='gray', alpha=0.45,
                 edgecolor='none')
    axes[0].plot(xs, true_dens, color='black', linewidth=1.6)
    axes[0].set_xlim(-6, 6)
    axes[0].set_xlabel(r'$\theta$')
    axes[0].set_ylabel(r'$p(\theta)$')
    axes[0].set_title(f'histogram ({title})')

    axes[1].plot(np.arange(1, len(th) + 1), th, color='black',
                 linewidth=0.3, marker='o', markersize=0.8, linestyle='-')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel(r'$\theta$')
    axes[1].set_title(f'traceplot ({title})')
    plt.tight_layout()
    plt.savefig(fname + '.pdf', bbox_inches='tight')
    plt.savefig(fname + '.png', bbox_inches='tight', dpi=150)
    plt.close()


hist_and_trace(th_gibbs_1k, 'fig6_5', '1,000 Gibbs samples')
hist_and_trace(th_gibbs_10k, 'fig6_6', '10,000 Gibbs samples')

# ============================================================
# Fig 6.7：6.4節の半共役正規モデルのトレースプロット
# ============================================================
mu0, t20, s20, nu0 = 1.9, 0.95 ** 2, 0.01, 1
y = np.array([1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.82, 1.90, 2.08])
n = len(y)
mean_y, var_y = y.mean(), y.var(ddof=1)

S = 1000
PHI = np.zeros((S, 2))
phi = np.array([mean_y, 1 / var_y])
PHI[0] = phi
r2 = np.random.default_rng(1)
for s in range(1, S):
    mun = (mu0 / t20 + n * mean_y * phi[1]) / (1 / t20 + n * phi[1])
    t2n = 1 / (1 / t20 + n * phi[1])
    phi[0] = r2.normal(mun, np.sqrt(t2n))
    nun = nu0 + n
    s2n = (nu0 * s20 + (n - 1) * var_y + n * (mean_y - phi[0]) ** 2) / nun
    phi[1] = r2.gamma(nun / 2, 1 / (nun * s2n / 2))
    PHI[s] = phi

sigma2 = 1 / PHI[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, v, lab in [(axes[0], PHI[:, 0], r'$\theta$'),
                   (axes[1], sigma2, r'$\sigma^2$')]:
    ax.plot(np.arange(1, S + 1), v, color='black', linewidth=0.4)
    ax.set_xlabel('iteration')
    ax.set_ylabel(lab)
plt.tight_layout()
plt.savefig('fig6_7.pdf', bbox_inches='tight')
plt.savefig('fig6_7.png', bbox_inches='tight', dpi=150)
plt.close()

# ============================================================
# 診断量
# ============================================================
print("=== 正規混合（ギブス 10,000）===")
print(f"  lag-10 の自己相関 = {acf(th_gibbs_10k, 10):.3f}   （Hoff: 0.93）")
print(f"  lag-50 の自己相関 = {acf(th_gibbs_10k, 50):.3f}   （Hoff: 0.812）")
print(f"  有効標本サイズ    = {effective_size(th_gibbs_10k):.2f}  （Hoff: 18.42）")
print()
print("=== 正規混合（独立 MC 1,000）===")
print(f"  lag-1 の自己相関  = {acf(th_mc, 1):.3f}")
print(f"  有効標本サイズ    = {effective_size(th_mc):.1f}")
print()
print("=== 半共役正規モデル（ギブス 1,000）===")
print(f"  θ  : lag-1 自己相関 = {acf(PHI[:, 0], 1):.3f}  "
      f"有効標本サイズ = {effective_size(PHI[:, 0]):.0f}   （Hoff: 0.031, 1000）")
print(f"  σ² : lag-1 自己相関 = {acf(sigma2, 1):.3f}  "
      f"有効標本サイズ = {effective_size(sigma2):.0f}   （Hoff: 0.147, 742）")
