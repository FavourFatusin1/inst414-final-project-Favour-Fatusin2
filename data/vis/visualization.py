import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Ensure the 'visualizations' directory exists
os.makedirs('data/visualizations', exist_ok=True)

# Load the transformed data
file_path1 = 'data/outputs/transformed_loaded_firstchart.csv'
file_path2 = 'data/outputs/transformed_secondchart.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Basic Matplotlib visualization example
plt.figure(figsize=(10, 6))
plt.plot(df1['unique_id'], df1['some_numeric_column'], label='Some Data')
plt.title('Basic Line Plot of Some Data')
plt.xlabel('Unique ID')
plt.ylabel('Value')
plt.legend()
plt.savefig('data/visualizations/basic_line_plot.png')
plt.show()

# Seaborn example for a more advanced plot
plt.figure(figsize=(10, 6))
sns.barplot(x='unique_id', y='some_numeric_column', data=df1, palette='viridis')
plt.title('Bar Plot using Seaborn')
plt.xlabel('Unique ID')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/visualizations/seaborn_bar_plot.png')
plt.show()

# Plotly example for an interactive plot
fig = px.scatter(df2, x='unique_id', y='another_numeric_column', color='category_column', title='Interactive Scatter Plot')
fig.write_html('data/visualizations/interactive_scatter_plot.html')
fig.show()

# Additional visualizations
# Histogram using Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(df1['some_numeric_column'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Some Numeric Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('data/visualizations/histogram.png')
plt.show()

# Boxplot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='category_column', y='another_numeric_column', data=df2, palette='coolwarm')
plt.title('Boxplot using Seaborn')
plt.xlabel('Category')
plt.ylabel('Value')
plt.savefig('data/visualizations/boxplot.png')
plt.show()

print("Visualizations have been created and saved in the 'data/visualizations' directory.")
