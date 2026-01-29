import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load your dataset
data = pd.read_csv(r"C:\Users\sadha\OneDrive\Desktop\Crop_Recommendation Project\Data\crop_recommendation.csv")

# Split features and labels
X = data.drop('label', axis=1)
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save new model (compatible with sklearn 1.5.2)
with open("models/model2.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved successfully!")
