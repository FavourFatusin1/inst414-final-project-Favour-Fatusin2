import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Ensure the 'visualizations' directory exists
os.makedirs('data/visualizations', exist_ok=True)

# Load the data
file_path1 = 'data/processed/loaded_firstchart.csv'
file_path2 = 'data/processed/loaded_secondchart.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Print column names to identify correct names
print("Columns in df1:")
print(df1.columns)
print("\nColumns in df2:")
print(df2.columns)

# Matplotlib Line Plot
plt.figure(figsize=(12, 6))
plt.plot(df1['Age'], df1['Number of reports, 2024 Q1-Q2'], marker='o', linestyle='-', color='b', label='Reports by Age')
plt.title('Number of Reports by Age')
plt.xlabel('Age')
plt.ylabel('Number of Reports')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/visualizations/line_plot.png')
plt.show()

# Seaborn Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='Number of reports, 2024 Q1-Q2', data=df1, palette='viridis')
plt.title('Bar Plot of Reports by Age')
plt.xlabel('Age')
plt.ylabel('Number of Reports')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/visualizations/bar_plot.png')
plt.show()

# Plotly Interactive Bar Plot
fig = px.bar(df1, x='Age', y='Number of reports, 2024 Q1-Q2', title='Interactive Bar Plot of Reports by Age', labels={'Age': 'Age', 'Number of reports, 2024 Q1-Q2': 'Number of Reports'})
fig.write_html('data/visualizations/interactive_bar_plot.html')
fig.show()

# Matplotlib Histogram
plt.figure(figsize=(12, 6))
plt.hist(df1['Number of reports, 2024 Q1-Q2'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Number of Reports')
plt.xlabel('Number of Reports')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('data/visualizations/histogram.png')
plt.show()

# Seaborn Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Age', y='Number of reports, 2024 Q1-Q2', data=df1, palette='coolwarm')
plt.title('Boxplot of Reports by Age')
plt.xlabel('Age')
plt.ylabel('Number of Reports')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/visualizations/boxplot.png')
plt.show()

# Plotly Interactive Scatter Plot for Credit Card Fraud per Capita
fig2 = px.scatter(df2, x='State', y='Credit Card Fraud per Capita', size='Credit Card Fraud Victim Count', color='Credit Card Fraud Victim Loss Amount', hover_name='State', title='Interactive Scatter Plot of Credit Card Fraud per Capita by State')
fig2.write_html('data/visualizations/interactive_scatter_plot.html')
fig2.show()

print("Visualizations have been created and saved in the 'data/visualizations' directory.")
