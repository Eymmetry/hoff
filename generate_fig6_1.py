import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Hoff Fig 6.1 の再現：midge データの事後分布の離散（グリッド）近似
# 本文の R コードと同じ設定
mu0, t20, s20, nu0 = 1.9, 0.95 ** 2, 0.01, 1
y = np.array([1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.82, 1.90, 2.08])
n = len(y)

G = H = 100
mean_grid = np.linspace(1.505, 2.00, G)      # θ のグリッド
prec_grid = np.linspace(1.75, 175, H)        # σ̃² = 1/σ² のグリッド

# post_grid[g, h] ∝ p(θ_g, σ̃²_h, y_1...y_n)
#   = dnorm(θ, μ0, τ0) × dgamma(σ̃², ν0/2, ν0σ0²/2) × Π dnorm(y_i, θ, 1/√σ̃²)
# Hoff の dgamma は rate パラメータなので scipy では scale=1/rate
post_grid = np.zeros((G, H))
for g, th in enumerate(mean_grid):
    for h, prec in enumerate(prec_grid):
        post_grid[g, h] = (
            stats.norm.pdf(th, mu0, np.sqrt(t20))
            * stats.gamma.pdf(prec, nu0 / 2, scale=1 / (s20 * nu0 / 2))
            * np.prod(stats.norm.pdf(y, th, 1 / np.sqrt(prec)))
        )

# 正規化：グリッド上の和が 1 になるようにするだけ（p(y) は自動的に消える）
post_grid = post_grid / post_grid.sum()

# 周辺分布は単に足し合わせるだけで得られる
post_theta = post_grid.sum(axis=1)
post_prec = post_grid.sum(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
plt.rcParams['font.size'] = 9

# パネル1：同時事後分布
ax1 = axes[0]
TH, PR = np.meshgrid(mean_grid, prec_grid, indexing='ij')
ax1.contour(TH, PR, post_grid, levels=12, cmap='viridis')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'$\tilde{\sigma}^2$')
ax1.set_title(r'$p_D(\theta, \tilde{\sigma}^2 \mid y_1 \ldots y_n)$')

# パネル2：θ の周辺事後分布
ax2 = axes[1]
ax2.plot(mean_grid, post_theta, color='black', linewidth=1.5)
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'$p_D(\theta \mid y_1 \ldots y_n)$')
ax2.set_xlim(1.5, 2.0)
ax2.set_ylim(0, None)
ax2.set_title(r'marginal of $\theta$')

# パネル3：σ̃² の周辺事後分布
ax3 = axes[2]
ax3.plot(prec_grid, post_prec, color='black', linewidth=1.5)
ax3.set_xlabel(r'$\tilde{\sigma}^2$')
ax3.set_ylabel(r'$p_D(\tilde{\sigma}^2 \mid y_1 \ldots y_n)$')
ax3.set_xlim(0, 175)
ax3.set_ylim(0, None)
ax3.set_title(r'marginal of $\tilde{\sigma}^2 = 1/\sigma^2$')

plt.tight_layout()
plt.savefig('fig6_1.pdf', bbox_inches='tight')
plt.savefig('fig6_1.png', bbox_inches='tight', dpi=150)

# グリッドの端にどれだけ確率が漏れているかの確認（グリッド配置の妥当性チェック）
edge = post_theta[0] + post_theta[-1] + post_prec[0] + post_prec[-1]
print(f"fig6_1.pdf と fig6_1.png を保存しました")
print(f"θ の事後平均  = {np.sum(mean_grid * post_theta):.4f}")
print(f"σ̃² の事後平均 = {np.sum(prec_grid * post_prec):.2f}")
print(f"グリッド端の確率質量 = {edge:.2e}")
