import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test, class_names=['glioma', 'meningioma', 'notumor', 'pituitary']):
    # Evaluates the model and prints classification report.

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return y_pred

def plot_confusion_matrix(y_true, y_pred, class_names=['glioma', 'meningioma', 'notumor', 'pituitary'], save_path=None):
    # Plots the confusion matrix.

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        
    plt.show()
