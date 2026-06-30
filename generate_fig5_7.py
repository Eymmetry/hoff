import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ---- 母集団の生成（GSSデータの近似）----
# 実際のGSSデータは非公開なのでPoisson(2.42)で近似
# 右歪み・非負整数・平均2.42 という特徴を持つ
N = 921
population = np.random.poisson(2.42, N)
print(f"母集団平均: {population.mean():.3f}")
print(f"母集団標準偏差: {population.std():.3f}")
print(f"歪度: {stats.skew(population):.3f}")

S = 10000  # モンテカルロ繰り返し回数

# ---- 図の作成 ----
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
plt.rcParams['font.size'] = 9

# ============================================================
# パネル1：母集団分布
# ============================================================
ax1 = axes[0]
values, counts = np.unique(population, return_counts=True)
probs = counts / N
ax1.bar(values, probs, color='#888888', edgecolor='white', width=0.7)
ax1.set_xlabel('y = number of children')
ax1.set_ylabel('p(y)')
ax1.set_xlim(-0.5, 8.5)
ax1.set_ylim(0, 0.32)
ax1.set_xticks(range(9))
ax1.set_title('Population distribution')

# ============================================================
# パネル2：標本平均の標本分布（n=5, 15, 45）
# ============================================================
ax2 = axes[1]
colors = ['steelblue', 'tomato', 'seagreen']
ns = [5, 15, 45]

for n, color in zip(ns, colors):
    means = [np.mean(np.random.choice(population, n, replace=True))
             for _ in range(S)]
    x = np.linspace(0.3, 5.5, 400)
    kde = stats.gaussian_kde(means, bw_method=0.3)
    ax2.plot(x, kde(x), color=color, label=f'n={n}', linewidth=2)

ax2.axvline(population.mean(), color='gray', linestyle='--',
            linewidth=1.5, label=f'θ={population.mean():.2f}')
ax2.set_xlabel('number of children')
ax2.set_ylabel('density')
ax2.legend(fontsize=8)
ax2.set_xlim(0.3, 5.5)
ax2.set_ylim(bottom=0)
ax2.set_title('Distribution of sample mean')

# ============================================================
# パネル3：(ȳ, s²) の同時標本分布（n=45）
# ============================================================
ax3 = axes[2]
n = 45
ybar_vals, s2_vals = [], []
for _ in range(S):
    sample = np.random.choice(population, n, replace=True)
    ybar_vals.append(np.mean(sample))
    s2_vals.append(np.var(sample, ddof=1))

ybar_arr = np.array(ybar_vals)
s2_arr   = np.array(s2_vals)

# 2次元KDEで等高線を描画
x_range = np.linspace(1.6, 3.2, 120)
y_range = np.linspace(0.5, 7.0, 120)
X, Y = np.meshgrid(x_range, y_range)
kernel = stats.gaussian_kde(np.vstack([ybar_arr, s2_arr]))
Z = kernel(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

ax3.contour(X, Y, Z, levels=6, colors='gray', linewidths=1.2)
ax3.set_xlabel(r'$\bar{y}$')
ax3.set_ylabel(r'$s^2$')
ax3.set_xlim(1.6, 3.2)
ax3.set_ylim(0.5, 7.0)
ax3.set_title(r'Joint density of $(\bar{y}, s^2)$, $n=45$')

# 正の相関を注記
print(f"\nCorr(ȳ, s²) = {np.corrcoef(ybar_arr, s2_arr)[0,1]:.3f}")

plt.tight_layout()
plt.savefig('fig5_7.pdf', bbox_inches='tight')
plt.savefig('fig5_7.png', bbox_inches='tight', dpi=150)
print("\nfig5_7.pdf と fig5_7.png を保存しました")
