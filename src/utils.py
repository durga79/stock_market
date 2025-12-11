import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Cohen_Kappa': kappa
    }
    
    return results

def print_evaluation_metrics(results):
    print(f"\n{'='*60}")
    print(f"  {results['Model']} - Performance Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy:      {results['Accuracy']:.4f}")
    print(f"  F1-Score:      {results['F1-Score']:.4f}")
    print(f"  Precision:     {results['Precision']:.4f}")
    print(f"  Recall:        {results['Recall']:.4f}")
    print(f"  Cohen's Kappa: {results['Cohen_Kappa']:.4f}")
    print(f"{'='*60}\n")

def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def compare_models_visualization(results_list):
    models = [r['Model'] for r in results_list]
    accuracy = [r['Accuracy'] for r in results_list]
    f1 = [r['F1-Score'] for r in results_list]
    kappa = [r['Cohen_Kappa'] for r in results_list]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, accuracy, width, label='Accuracy', color='skyblue')
    ax.bar(x, f1, width, label='F1-Score', color='lightcoral')
    ax.bar(x + width, kappa, width, label="Cohen's Kappa", color='lightgreen')
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison Across All Members', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    for i, v in enumerate(accuracy):
        ax.text(i - width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(kappa):
        ax.text(i + width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig
