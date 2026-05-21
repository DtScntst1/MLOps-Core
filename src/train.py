import os
import json
import pickle
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    print("Loading data...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # We use a fixed random state for reproducibility in CI/CD
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    metrics = {
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4)
    }
    
    print(f"Metrics: {metrics}")
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Save the model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # Save metrics to a markdown file for GitHub Actions to read
    with open("reports/metrics.md", "w", encoding="utf-8") as f:
        f.write("## Model Performance 🚀\n")
        f.write(f"- **Accuracy:** {metrics['Accuracy']}\n")
        f.write(f"- **F1 Score:** {metrics['F1 Score']}\n")
        
    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_model()
