import seaborn as sns
import numpy as np
cm = np.array([[50, 10, 5], [8, 40, 7], [3, 6, 30]])  # Dummy data
labels = ['Negative', 'Neutral', 'Positive']
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Naive Bayes)')
plt.savefig('visualizations/confusion_matrix.png')
plt.close()