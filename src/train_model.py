import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data/raw/pd.csv")


selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'HNR', 'NHR']
df = df[selected_features + ['status']]

x = df[selected_features]
y = df['status']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, class_weight='balanced')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]

print("accuracy: ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred, 'Probability': y_prob})
# print(results.head())

cv_scores = cross_val_score(model, x_scaled, y, cv=5, scoring='accuracy')
print(f"CV scores: {cv_scores}")
print(f"average cv: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)