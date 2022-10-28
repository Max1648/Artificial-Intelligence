from sklearn.metrics import confusion_matrixconfusion_matrix(df.actual_label.values, df.predicted_RF.values)


import pandas as pd
df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()