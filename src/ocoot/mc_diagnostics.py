"""Conditional Monte Carlo variance diagnostics for fixed generated populations.

These estimates describe uncertainty in a generated pseudobulk mean. They do
not alter predictions or official metrics. Singleton strata have unidentified
conditional variances; a complete estimate from one population is unavailable
without additional assumptions when any such strata exist.
"""
import numpy as np


# 固定生成总体的条件 Monte Carlo 方差诊断，不改变预测或官方指标。
def mean_sampling_variance(values, strata):
    """Estimate per-gene-averaged MC variance of a fixed-composition mean.

    For repeated stratum b, n_b*S_b^2/n^2 is an unbiased contribution when
    cells are independent and identically distributed conditional on b. Their
    sum covers only repeated strata. The pooled proxy additionally assumes
    a common within-stratum variance and extrapolates to singleton strata.
    Stratum means can differ arbitrarily. Both average over all input genes.
    """
# 校验并把所有输入统一为 float64，避免统计计算受到输入精度影响。
    x = np.asarray(values, dtype=np.float64)
    labels = np.asarray(strata)
    if (x.ndim != 2 or not x.shape[0] or not x.shape[1] or labels.shape != (len(x),)
            or not np.isfinite(x).all()):
        raise ValueError('A finite nonempty cell-by-gene array and aligned stratum labels are required')
# 识别独立层级并统计重复层级与 singleton 层级。
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    total, genes = x.shape
    repeated = counts >= 2
    covered = int(counts[repeated].sum())
    degrees = int(np.sum(counts - 1))
    partial, within_ss = 0., 0.
    for group, count in enumerate(counts):
# 未重复的层级无法由该总体内数据估计条件方差，保留为缺失。
        if count < 2:
            continue
        group_x = x[inverse == group]
        centered = group_x - group_x.mean(axis=0)
        # Mean of the per-gene within-stratum sums of squares.
        ss = float(np.square(centered).sum(axis=0).mean())
        partial += int(count) * ss / (int(count) - 1) / total**2
        within_ss += ss
    return {
        'cells': total, 'genes': genes, 'strata': len(counts),
        'repeated_strata': int(repeated.sum()), 'singleton_strata': int((~repeated).sum()),
        'repeated_cells': covered, 'repeated_cell_fraction': covered / total,
        'within_stratum_degrees_of_freedom': degrees,
        'all_strata_have_repeats': covered == total,
        'mc_contribution_repeated_strata_only': partial if degrees else None,
        'pooled_common_within_stratum_variance_proxy': within_ss / degrees / total if degrees else None,
        'naive_iid_variance_over_n_includes_stratum_mean_spread':
            float(x.var(axis=0, ddof=1).mean() / total) if total > 1 else None,
    }
