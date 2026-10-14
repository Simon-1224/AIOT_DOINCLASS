"""Exhaustive subset selection: rank 31 schemes using training-only CV."""
from itertools import combinations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

NUM = ['R&D Spend', 'Administration', 'Marketing Spend']
NAMES = NUM + ['State_Florida', 'State_New York']

def make_model(indices):
    # Category vocabulary is the declared dataset schema, not learned from test rows.
    prepare = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUM),
        ('state', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
          ('encode', OneHotEncoder(categories=[['California', 'Florida', 'New York']],
             drop='first', sparse_output=False, handle_unknown='error'))]), ['State'])])
    select = ColumnTransformer([('subset', 'passthrough', list(indices))])
    return Pipeline([('prepare', prepare), ('select', select), ('regression', LinearRegression())])

def run(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    out = root / 'outputs'
    out.mkdir(exist_ok=True)
    df = pd.read_csv(root / 'data' / '50_Startups.csv')
    X = df[NUM + ['State']]
    y = df['Profit']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    records = []
    for k in range(1, 6):
        for indices in combinations(range(5), k):
            fold_mse = -cross_val_score(make_model(indices), Xtr, ytr, cv=cv,
                scoring='neg_mean_squared_error', error_score='raise')
            records.append({'features': ' + '.join(NAMES[i] for i in indices),
                'indices': indices, 'k': k, 'cv_mse': float(fold_mse.mean()),
                'cv_rmse': float(np.sqrt(fold_mse.mean())),
                'cv_mse_std': float(fold_mse.std(ddof=1))})
    ranked = pd.DataFrame(records).sort_values(['cv_mse', 'k', 'features']).reset_index(drop=True)
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    ranked.to_csv(out / 'all_31_schemes.csv', index=False)
    ranked.head(20).to_csv(out / 'top20_schemes.csv', index=False)
    # Freeze the top five BEFORE evaluating the holdout; do not rerank by holdout MSE.
    selected = ranked.head(5).copy()
    predictions = pd.DataFrame({'row_id': yte.index, 'actual': yte.to_numpy()})
    for row in selected.itertuples():
        fitted = make_model(row.indices).fit(Xtr, ytr)
        pred = fitted.predict(Xte)
        selected.loc[row.Index, 'test_mse'] = mean_squared_error(yte, pred)
        selected.loc[row.Index, 'test_rmse'] = np.sqrt(mean_squared_error(yte, pred))
        selected.loc[row.Index, 'test_mae'] = mean_absolute_error(yte, pred)
        selected.loc[row.Index, 'test_r2'] = r2_score(yte, pred)
        predictions[f'rank_{row.rank}_prediction'] = pred
        joblib.dump(fitted, out / f'top5_rank_{row.rank}_evaluation.joblib')
        joblib.dump(clone(fitted).fit(X, y), out / f'top5_rank_{row.rank}_deployment.joblib')
    selected.to_csv(out / 'top5_schemes.csv', index=False)
    predictions.to_csv(out / 'top5_test_predictions.csv', index=False)
    original = make_model(range(5)).fit(Xtr, ytr)
    original_mse = float(mean_squared_error(yte, original.predict(Xte)))
    summary = {'original_test_mse': original_mse, 'selection': 'Training-only 5-fold mean MSE',
        'split_seed': 42, 'cv_seed': 42, 'train_rows': 40, 'test_rows': 10,
        'best_features': selected.iloc[0]['features'],
        'best_test_mse': float(selected.iloc[0]['test_mse'])}
    (out / 'feature_selection_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = '# Feature selection schemes (CRISP-DM extension)\n\n'
    report += 'Schemes are predictor subsets, not 20 distinct algorithms. Exhaustive search evaluates all 31 nonempty subsets.\n\n'
    report += 'Business/data understanding and preparation follow analysis.py. Modeling uses LinearRegression for every subset.\n'
    report += 'Ranking uses training-only 5-fold mean MSE; smaller is better. CV scores are selection scores and may be optimistic after searching.\n'
    report += 'The holdout was already viewed in the baseline run; it is not a fresh independent confirmation. No holdout-driven tuning or reranking is performed.\n\n'
    report += f'Original full-model test MSE: {original_mse:.6f}\n\n'
    report += '## Top 20 by CV MSE\n\n```text\n' + ranked.head(20).drop(columns='indices').to_string(index=False) + '\n```\n\n'
    report += '## Top 5 holdout evaluation\n\n```text\n' + selected.drop(columns='indices').to_string(index=False) + '\n```\n\n'
    report += 'Deployment: saved evaluation models use 40 training rows; deployment models are refit on all 50 rows and have no fresh test score.\n'
    (root / 'FEATURE_SELECTION.md').write_text(report, encoding='utf-8')
    print('Original test MSE:', original_mse)
    print('TOP 20 (training CV ranking):')
    print(ranked.head(20).drop(columns='indices').to_string(index=False))
    print('TOP 5 (holdout results, still ordered by CV):')
    print(selected.drop(columns='indices').to_string(index=False))
    return selected

if __name__ == '__main__':
    run()
