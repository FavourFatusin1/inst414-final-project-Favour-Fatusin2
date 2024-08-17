import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Ensure the 'visualizations' directory exists
os.makedirs('data/visualizations', exist_ok=True)

# Load data function
def load_data(file_path1, file_path2):
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    return df1, df2

# Function to print column names
def print_column_names(df1, df2):
    print("Columns in df1:")
    print(df1.columns)
    print("\nColumns in df2:")
    print(df2.columns)

# Matplotlib Line Plot
def line_plot(df1):
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
def bar_plot(df1):
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
def interactive_bar_plot(df1):
    fig = px.bar(df1, x='Age', y='Number of reports, 2024 Q1-Q2', title='Interactive Bar Plot of Reports by Age', labels={'Age': 'Age', 'Number of reports, 2024 Q1-Q2': 'Number of Reports'})
    fig.write_html('data/visualizations/interactive_bar_plot.html')
    fig.show()

# Matplotlib Histogram
def histogram(df1):
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
def boxplot(df1):
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
def interactive_scatter_plot(df2):
    fig2 = px.scatter(df2, x='State', y='Credit Card Fraud per Capita', size='Credit Card Fraud Victim Count', color='Credit Card Fraud Victim Loss Amount', hover_name='State', title='Interactive Scatter Plot of Credit Card Fraud per Capita by State')
    fig2.write_html('data/visualizations/interactive_scatter_plot.html')
    fig2.show()

# Plotly Choropleth Map for Credit Card Fraud per Capita
def choropleth_map(df2):
    fig3 = px.choropleth(
        df2,
        locations='State',  # Assuming your DataFrame contains state abbreviations or full names
        locationmode='USA-states',  # Use 'USA-states' for state abbreviations
        color='Credit Card Fraud per Capita',  # The data to color code by
        hover_name='State',  # Column to show when hovering over states
        color_continuous_scale='Viridis',  # Color scale
        scope='usa',  # Focus on the USA map
        labels={'Credit Card Fraud per Capita': 'Credit Card Fraud per Capita'}
    )
    fig3.write_html('data/visualizations/choropleth_map.html')
    fig3.show()

