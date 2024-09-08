import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Changeable parameters
target = "output"
test_size = 0.2

# Load and preprocess the data
df = pd.read_csv("HEART_MODEL/Heart Attack/heart_attack_dataset.csv")

# Extracting X and y
y = df[target].values
x = df.drop([target], axis=1).values

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, stratify=y, random_state=42)

# Ensemble models for classification
model1 = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
model2 = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
model3 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
model4 = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
model5 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)

ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', model1),
        ('svm', model2),
        ('rf', model3),
        ('knn', model4),
        ('gb', model5)
    ],
    voting='soft',
    weights=[1, 2, 1, 1, 2]
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('HEART_MODEL/Heart Attack/heart_attack_prediction.pkl', 'wb') as model_file:
    pickle.dump(ensemble_model, model_file)

print("Model trained and saved successfully.")
