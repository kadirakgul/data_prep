
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from eda import *
from data_prep import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

df = pd.read_csv("titanic.csv")
df.columns = [col.upper() for col in df.columns]
df.head()