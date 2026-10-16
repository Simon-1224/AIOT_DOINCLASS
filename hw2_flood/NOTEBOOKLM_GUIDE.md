# NotebookLM 實作與證據（尚待完成）
學號：5115056023

這份文件是 GPT 協助準備的操作指引，不是 NotebookLM 研究摘要。

## 操作步驟
1. 開啟 https://notebooklm.google.com/ 並由本人登入 Google 帳戶。
2. 建立筆記本，命名「5115056023 HW2 洪水預測研究」。
3. 加入下列公開來源；若網站無法擷取，從原頁複製可見文字或加入官方文件，保留網址。不要繞過存取限制。
4. 貼入下方提問，等待 NotebookLM 帶來源引用的回答。
5. 核對來源、保留原始回答與引用。將實際使用的來源連結寫進 evidence/notebooklm_sources.txt，一行一個網址。
6. 將原始回答全文另存為 UTF-8 的 evidence/notebooklm_raw.txt；需含來源引用、日期、實際研究結果，不要只填入 GPT 寫的指引。
7. 把自己確認過、100 字以上的繁體中文摘要存為 evidence/notebooklm_summary.txt。
8. 保存筆記本連結至 evidence/notebooklm_url.txt；若不分享筆記本，可以改附本人操作截圖或匯出 PDF 作佐證。
9. 重新執行 build_report.py 和 package_submission.py。腳本會檢查文件；不會憑空產生 NotebookLM 研究紀錄。

## 候選來源（必須實際加入後才列為已研究）
- 實際使用的 Kaggle 資料：https://www.kaggle.com/datasets/kamrulhasan0077/flood-prediction-dataset
- 同題線性回歸範例：https://www.kaggle.com/code/sakshisatre/flood-prediction-using-linear-regression-0-84558/notebook
- 同題特徵工程討論（原競賽版本，非本作業的 50,000 筆版本）：https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499579
- 特徵選擇官方文件：https://scikit-learn.org/stable/modules/feature_selection.html
- 區間方法原始論文：https://arxiv.org/abs/2107.07511

## 可直接貼入 NotebookLM 的提問
請只根據我加入的來源，以繁體中文撰寫至少 200 字研究摘要並逐項附來源引用：
1. 用 LinearRegression 預測 FloodProbability 時，有哪些資料品質與生成機制的限制？區分 50,000 筆教學版本和 Kaggle Playground S4E5 競賽版本。
2. 比較 OLS、Ridge、Lasso，以及特徵總和、排序等工程方法的思路。說明為何刪除特徵未必提升表現。
3. 解釋五折交叉驗證、特徵選擇和校準資料如何避免洩漏，並解釋 95% split-conformal 預測區間需要的交換性假設。
4. 我本機實驗發現目標幾乎是 20 個特徵的等權線性和，完整模型 R² 約 1。這個結果為什麼不能當成真實洪災預測能力？如果完整模型的殘差僅剩浮點誤差，應如何處理回歸假設檢定？
不要補造來源未提供的 Kaggle 名次、資料蒐集地點或分數。外部方法的分數不能直接與本機不同版本／切分比較。
