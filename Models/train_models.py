import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('heart_balanced.csv')
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
xgb = XGBClassifier(scale_pos_weight=1.0, random_state=42)
svm = make_pipeline(StandardScaler(), SVC(probability=True, C=1.0, kernel="rbf", random_state=42))


# Train RandomForest
rf.fit(X_train, y_train)
joblib.dump(rf, 'random_forest.joblib')

# Train XGBoost
xgb.fit(X_train, y_train)
joblib.dump(xgb, 'xgboost.joblib')

svm.fit(X_train, y_train)
joblib.dump(svm, 'svm.joblib')