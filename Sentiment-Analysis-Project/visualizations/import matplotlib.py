import matplotlib.pyplot as plt
labels = ['Positive', 'Negative', 'Neutral']
sizes = [40, 35, 25]
colors = ['#36A2EB', '#FF6384', '#FFCE56']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.savefig('visualizations/sentiment_distribution.png')
plt.close()
