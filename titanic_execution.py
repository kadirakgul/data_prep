
import pandas as pd
from sklearn.preprocessing import StandardScaler
from eda import *
from data_prep import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

df = pd.read_csv("titanic.csv")
df.columns = [col.upper() for col in df.columns]
df.head()


df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.head()