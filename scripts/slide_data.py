from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/results'
cv=pd.read_csv(OUT/'grid_search.csv')
curves=cv[(cv.param_knn__metric=='euclidean') & (cv.param_knn__weights=='uniform')].sort_values('param_knn__n_neighbors')
obj={'run':json.loads((OUT/'run_summary.json').read_text()),'counts':pd.read_csv(OUT/'class_counts.csv').to_dict('records'),'metrics':pd.read_csv(OUT/'metrics.csv').fillna('N/A').to_dict('records'),'per_class':pd.read_csv(OUT/'classification_report.csv').iloc[:3].to_dict('records'),'curves':curves[['param_knn__n_neighbors','mean_train_accuracy','mean_test_accuracy','mean_train_macro_f1','mean_test_macro_f1']].to_dict('records'),'noise':pd.read_csv(OUT/'noise_mean.csv').fillna('N/A').to_dict('records')}
(ROOT/'.build/slides').mkdir(parents=True, exist_ok=True)
(ROOT/'.build/slides/data.json').write_text(json.dumps(obj))
