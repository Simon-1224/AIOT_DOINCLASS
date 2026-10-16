# 研究來源與比較邊界

1. Kaggle Flood Prediction Dataset，Kamrul Hasan，version 1，2026-10-16 下載。
https://www.kaggle.com/datasets/kamrulhasan0077/flood-prediction-dataset
官方 API metadata 已保存 data/kaggle_metadata.json。資料頁標示 CC0，並承認原始外部來源不明。實際下載 50,000 筆、20 個輸入特徵，目標為 float64，與頁面「全部 int64」的敘述不符，以 CSV 為準。

2. Sakshi Satre：Flood Prediction Using Linear Regression (0.84558)。
https://www.kaggle.com/code/sakshisatre/flood-prediction-using-linear-regression-0-84558/notebook
用於對照「EDA → 切分 → 線性回歸 → 評估」的教學流程。它使用 Playground S4E5 競賽資料。標題分數與 Kaggle 顯示版本分數可能不同，報告不將其數字當成本作業的公平對照。

3. broccoli beef：Sorting along the feature axis as feature engineering。
https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499579
作者提出同題競賽的特徵總和與排序等想法；不是本作業重跑結果。此作業另用開發集核對近似等權線性和，不直接搬用其模型成績。

4. scikit-learn Feature selection 官方文件。
https://scikit-learn.org/stable/modules/feature_selection.html
SelectKBest、RFE、Sequential Feature Selection、L1 模型與 Pipeline 用法。

5. Angelopoulos, A. N. & Bates, S. (2021, revised 2022).
A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.
https://arxiv.org/abs/2107.07511
本作業採用獨立校準集絕對殘差分位數的 split conformal；交換性假設下為邊際涵蓋率，非條件保證。

6. 本地公平方法比較：HistGradientBoostingRegressor。
使用同一資料版本、同一切分、同一五折 CV。此模型僅為非線性對照，未提交 Kaggle，沒有名次。

以上是 GPT 整理的可追溯研究素材。只有真正匯入 NotebookLM 並完成研究後，才能在報告標為「NotebookLM 摘要」。
