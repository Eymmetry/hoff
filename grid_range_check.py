"""グリッドの範囲を変えると離散近似がどう壊れるかを確かめる。

section6_2_seminar.tex の「弱点その1」の表を再現する。
θ 側のグリッドは固定し、σ̃² の上限だけを動かす（点数は H=100 のまま）。
"""
import numpy as np
from scipy import stats

mu0, t20, s20, nu0 = 1.9, 0.95 ** 2, 0.01, 1
y = np.array([1.64, 1.70, 1.72, 1.74, 1.82, 1.82, 1.82, 1.90, 2.08])


def grid_post(prec_hi, G=100, H=100, th_lo=1.505, th_hi=2.00, prec_lo=1.75):
    th = np.linspace(th_lo, th_hi, G)
    pr = np.linspace(prec_lo, prec_hi, H)
    post = np.zeros((G, H))
    for g, t in enumerate(th):
        for h, p in enumerate(pr):
            post[g, h] = (
                stats.norm.pdf(t, mu0, np.sqrt(t20))
                * stats.gamma.pdf(p, nu0 / 2, scale=1 / (s20 * nu0 / 2))
                * np.prod(stats.norm.pdf(y, t, 1 / np.sqrt(p)))
            )
    return pr, post / post.sum()


def quantile(grid, mass, q):
    return grid[np.searchsorted(np.cumsum(mass), q)]


print(f"{'σ̃² の範囲':<22}{'E[σ̃²|y]':>10}{'2.5%':>9}{'97.5%':>9}"
      f"{'格子幅':>9}{'上端の質量':>13}{'本体の点数':>11}")
for hi, label in [(60, '狭すぎ'), (175, 'Hoff'), (3000, '広すぎ')]:
    pr, post = grid_post(hi)
    marg = post.sum(axis=0)
    # 事後分布の本体 [30, 100] に何点入っているか
    n_bulk = ((pr >= 30) & (pr <= 100)).sum()
    print(f"{f'1.75-{hi}（{label}）':<22}"
          f"{(pr * marg).sum():>10.2f}"
          f"{quantile(pr, marg, 0.025):>9.2f}"
          f"{quantile(pr, marg, 0.975):>9.2f}"
          f"{pr[1] - pr[0]:>9.2f}"
          f"{marg[-1]:>13.1e}"
          f"{n_bulk:>11d}")
