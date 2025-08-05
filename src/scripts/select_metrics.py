import argparse
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2


def backward_lrt_selection(df, label_col='label', alpha=0.05, verbose=True):
    """
    Perform backward elimination using likelihood-ratio tests based on model deviances.
    Returns list of selected delta-metric columns.
    """
    y = df[label_col]
    metric_cols = [c for c in df.columns if c.startswith('delta_')]
    X_full = sm.add_constant(df[metric_cols])
    model_full = sm.GLM(y, X_full, family=sm.families.Binomial()).fit()
    selected = set(metric_cols)

    if verbose:
        print("Initial full model fitted with metrics:", sorted(selected))

    improved = True
    while improved and len(selected) > 1:
        improved = False
        best_drop = None
        for m in sorted(selected):
            trial = sorted(selected - {m})
            X_trial = sm.add_constant(df[trial])
            model_trial = sm.GLM(y, X_trial, family=sm.families.Binomial()).fit()
            lr_stat = model_trial.deviance - model_full.deviance
            df_diff = int(model_full.df_model - model_trial.df_model)
            p_val = chi2.sf(lr_stat, df_diff)
            if verbose:
                print(f"Trying drop {m:20s} | LR stat={lr_stat:.2f}, p-value={p_val:.4f}")
            if p_val > alpha:
                if best_drop is None or p_val > best_drop[1]:
                    best_drop = (m, p_val, model_trial)
        if best_drop:
            m, p_val, new_model = best_drop
            selected.remove(m)
            model_full = new_model
            improved = True
            if verbose:
                print(f"Dropped {m}, new model has {len(selected)} metrics (p={p_val:.4f})")
    return sorted(selected)


def rfecv_selection(df, label_col='label', cv_folds=5, scoring='average_precision', verbose=True):
    """
    Recursive feature elimination with cross-validation.
    Uses logistic regression with minimal regularization to approximate unpenalized fit.
    """
    metric_cols = [c for c in df.columns if c.startswith('delta_')]
    X = df[metric_cols].values
    y = df[label_col].values

    # Use very small regularization strength to mimic unpenalized logistic regression
    estimator = LogisticRegression(penalty='l2', C=1e12, solver='lbfgs', max_iter=5000)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
    selector = RFECV(estimator, cv=cv, scoring=scoring, verbose=verbose)
    selector.fit(X, y)

    keep = [m for m, use in zip(metric_cols, selector.support_) if use]
    if verbose:
        print("RFECV support mask:", selector.support_)
        print(f"Selected {selector.n_features_} of {len(metric_cols)} features.")
    return sorted(keep)


def main():
    parser = argparse.ArgumentParser(
        description="Select necessary delta-metrics via LRT-based backward elimination or RFECV."
    )
    parser.add_argument('input_csv', help='Path to train_pairs.csv with delta_* columns')
    parser.add_argument('--method', choices=['lrt', 'rfecv'], default='lrt',
                        help='Selection method: lrt (backward) or rfecv')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for LRT elimination')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of CV folds for RFECV')
    parser.add_argument('--output', help='Optional path to write selected metrics')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.method == 'lrt':
        selected = backward_lrt_selection(df, alpha=args.alpha)
    else:
        selected = rfecv_selection(df, cv_folds=args.cv)

    print("Selected metrics:")
    for m in selected:
        print(m)

    if args.output:
        with open(args.output, 'w') as f:
            for m in selected:
                f.write(m + '\n')
        print(f"Written selection to {args.output}")

if __name__ == '__main__':
    main()
