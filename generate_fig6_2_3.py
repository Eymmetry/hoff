"""Hoff Fig 6.2 / 6.3 の再現：半共役正規モデルのギブスサンプラー。

Fig 6.2: 最初の 5, 15, 100 反復の軌跡
Fig 6.3: 1000 個の標本を離散近似の等高線に重ねたもの＋θ と σ̃² の密度推定
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

mu0, t20, s20, nu0 = 1.9, 0.95 ** 2, 0.01, 1
y = np.array([1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.82, 1.90, 2.08])
n = len(y)
mean_y = y.mean()
var_y = y.var(ddof=1)

# ============================================================
# ギブスサンプリング（Hoff の R コードと同じ手順）
# ============================================================
S = 1000
PHI = np.zeros((S, 2))
phi = np.array([mean_y, 1 / var_y])   # 出発点：標本平均と標本精度
PHI[0] = phi

rng = np.random.default_rng(1)
for s in range(1, S):
    # 1. θ をその完全条件付き分布から引く（φ[1] = 精度を固定）
    mun = (mu0 / t20 + n * mean_y * phi[1]) / (1 / t20 + n * phi[1])
    t2n = 1 / (1 / t20 + n * phi[1])
    phi[0] = rng.normal(mun, np.sqrt(t2n))

    # 2. 1/σ² をその完全条件付き分布から引く（いま引いた θ を固定）
    #    Σ(y_i-θ)² = (n-1)s² + n(ȳ-θ)² の恒等式を使う
    nun = nu0 + n
    s2n = (nu0 * s20 + (n - 1) * var_y + n * (mean_y - phi[0]) ** 2) / nun
    phi[1] = rng.gamma(nun / 2, 1 / (nun * s2n / 2))   # 第2引数は scale=1/rate

    PHI[s] = phi

# ============================================================
# Fig 6.2：最初の 5, 15, 100 反復
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, m in zip(axes, [5, 15, 100]):
    th, pr = PHI[:m, 0], PHI[:m, 1]
    # 階段状の軌跡：θ を更新（横移動）→ 精度を更新（縦移動）
    xs, ys = [th[0]], [pr[0]]
    for i in range(1, m):
        xs += [th[i], th[i]]
        ys += [pr[i - 1], pr[i]]
    ax.plot(xs, ys, color='gray', linewidth=0.6, zorder=1)
    if m <= 15:
        for i in range(m):
            ax.text(th[i], pr[i], str(i + 1), fontsize=7, ha='center',
                    va='center', zorder=3, clip_on=True,
                    bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    else:
        ax.scatter(th, pr, s=4, color='black', zorder=2)
    ax.set_xlim(1.63, 1.97)
    ax.set_ylim(0, 175)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\tilde{\sigma}^2$')
    ax.set_title(f'first {m} iterations')
plt.tight_layout()
plt.savefig('fig6_2.pdf', bbox_inches='tight')
plt.savefig('fig6_2.png', bbox_inches='tight', dpi=150)
plt.close()

# ============================================================
# Fig 6.3：1000 標本＋密度推定（背景は 6.2 節の離散近似）
# ============================================================
G = H = 100
mean_grid = np.linspace(1.505, 2.00, G)
prec_grid = np.linspace(1.75, 175, H)
post_grid = np.zeros((G, H))
for g, t in enumerate(mean_grid):
    for h, p in enumerate(prec_grid):
        post_grid[g, h] = (
            stats.norm.pdf(t, mu0, np.sqrt(t20))
            * stats.gamma.pdf(p, nu0 / 2, scale=1 / (s20 * nu0 / 2))
            * np.prod(stats.norm.pdf(y, t, 1 / np.sqrt(p)))
        )
post_grid /= post_grid.sum()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

ax1 = axes[0]
TH, PR = np.meshgrid(mean_grid, prec_grid, indexing='ij')
ax1.contour(TH, PR, post_grid, levels=10, colors='gray', linewidths=0.7)
ax1.scatter(PHI[:, 0], PHI[:, 1], s=2, color='black', alpha=0.4)
ax1.set_xlim(1.6, 2.0)
ax1.set_ylim(0, 175)
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'$\tilde{\sigma}^2$')
ax1.set_title('1000 Gibbs samples over discrete approx.')

panels = [
    (axes[1], 0, r'$\theta$', r'$p(\theta \mid y_1 \ldots y_n)$', (1.6, 2.0)),
    (axes[2], 1, r'$\tilde{\sigma}^2$',
     r'$p(\tilde{\sigma}^2 \mid y_1 \ldots y_n)$', (0, 175)),
]
for ax, col, xlab, ylab, xlim in panels:
    kde = stats.gaussian_kde(PHI[:, col])
    xs = np.linspace(*xlim, 400)
    ax.plot(xs, kde(xs), color='black', linewidth=1.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, None)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f'density estimate of {xlab}')

plt.tight_layout()
plt.savefig('fig6_3.pdf', bbox_inches='tight')
plt.savefig('fig6_3.png', bbox_inches='tight', dpi=150)
plt.close()

# ============================================================
# 経験分位点（Hoff の quantile() 出力に対応）
# ============================================================
q = [0.025, 0.5, 0.975]
print("                       2.5%        50%      97.5%")
print(f"θ（母平均）      {np.quantile(PHI[:, 0], q)}")
print(f"σ̃²（母精度）     {np.quantile(PHI[:, 1], q)}")
print(f"σ（母標準偏差）  {np.quantile(1 / np.sqrt(PHI[:, 1]), q)}")
print()
print("Hoff の R 出力（set.seed(1)）:")
print("θ   : 1.707282  1.804348  1.901129")
print("σ̃²  : 17.48020  53.62511  129.20020")
print("σ   : 0.08797701  0.13655763  0.23918408")
