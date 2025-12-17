from scipy.stats import pearsonr

def compute_statistical_significance(series1, series2):
    corr, pval = pearsonr(series1, series2)
    print(f'Correlation: {corr:.2f}, p-value: {pval:.3g}')
    return {'correlation': corr, 'p_value': pval}
