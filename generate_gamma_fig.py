import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Hoffの記法 dgamma(y, a, b) はrateパラメータ b を使う:
#   p(y) = b^a / Γ(a) * y^{a-1} e^{-by}
# scipy では stats.gamma(a, scale=1/b) に対応する


def dgamma(y, a, b):
    return stats.gamma.pdf(y, a, scale=1 / b)


fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
plt.rcParams['font.size'] = 9

# ============================================================
# パネル1：βを固定してαを変える（形が変わる）
# ============================================================
ax1 = axes[0]
y = np.linspace(0.001, 10, 500)
alphas = [0.5, 1, 2, 3, 5]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(alphas)))

for a, color in zip(alphas, colors):
    ax1.plot(y, dgamma(y, a, 1), color=color, linewidth=2,
             label=rf'$\alpha={a}$')

ax1.set_xlabel('y')
ax1.set_ylabel('density')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1.5)
ax1.legend(fontsize=8)
ax1.set_title(r'Gamma($\alpha$, $\beta=1$): shape varies')

# ============================================================
# パネル2：αを固定してβを変える（スケールだけ変わる）
# ============================================================
ax2 = axes[1]
betas = [0.5, 1, 2]
colors2 = plt.cm.plasma(np.linspace(0, 0.7, len(betas)))

for b, color in zip(betas, colors2):
    ax2.plot(y, dgamma(y, 2, b), color=color, linewidth=2,
             label=rf'$\beta={b}$')

ax2.set_xlabel('y')
ax2.set_ylabel('density')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 0.8)
ax2.legend(fontsize=8)
ax2.set_title(r'Gamma($\alpha=2$, $\beta$): scale varies')

# ============================================================
# パネル3：カイ二乗分布 χ²_ν = Gamma(ν/2, 1/2)
# ============================================================
ax3 = axes[2]
x = np.linspace(0.001, 15, 500)
nus = [1, 2, 3, 5, 10]
colors3 = plt.cm.viridis(np.linspace(0, 0.85, len(nus)))

for nu, color in zip(nus, colors3):
    ax3.plot(x, dgamma(x, nu / 2, 1 / 2), color=color, linewidth=2,
             label=rf'$\nu={nu}$')

ax3.set_xlabel('y')
ax3.set_ylabel('density')
ax3.set_xlim(0, 15)
ax3.set_ylim(0, 0.6)
ax3.legend(fontsize=8)
ax3.set_title(r'$\chi^2_\nu$ = Gamma($\nu/2$, $1/2$)')

plt.tight_layout()
plt.savefig('gamma_chi2_fig.pdf', bbox_inches='tight')
plt.savefig('gamma_chi2_fig.png', bbox_inches='tight', dpi=150)
print("gamma_chi2_fig.pdf と gamma_chi2_fig.png を保存しました")
