import pandas as pd

lgb = pd.read_csv("submission_lgb.csv")
xgb = pd.read_csv("submission_xgb.csv")
cat = pd.read_csv("submission_cat.csv")

submission = pd.read_csv('sample_submission.csv')
submission['revenue'] = cat["revenue"]*0.32 + xgb["revenue"]*0.35 + lgb["revenue"]*0.33
submission.to_csv("submission_xgb_lgb_cat5.csv", index=False)