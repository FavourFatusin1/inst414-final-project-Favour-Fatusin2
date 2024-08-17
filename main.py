import logging
from etl.extracted import extract_table
from etl.transform import transform
from etl.load import load
from analysis.model import train_and_save_model
from analysis.evaluate import evaluate_and_save_predictions
from vis.visualizations import bar_plot, boxplot, histogram, print_column_names, interactive_bar_plot, interactive_scatter_plot, choropleth_map, load_data, line_plot

# Configure logging
logging.basicConfig(
    filename='data_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Main function to execute all visualizations
def main():
    logging.info("Starting the data pipeline workflow.")
    
    try:
        # Step 1: Extract data
        file_path1 = 'data/statistic1.csv'
        file_path2 = 'data/statistic2.csv'
        url1 = 'https://www.fool.com/the-ascent/research/identity-theft-credit-card-fraud-statistics/'
        url2 = 'https://www.bankrate.com/credit-cards/news/credit-card-fraud-statistics/#fraud'
        
        logging.info("Extracting data from URLs.")
        df1 = extract_table(url1, file_path1)
        df2 = extract_table(url2, file_path2)

        # Step 2: Transform data
        logging.info("Transforming data.")
        transform()

        # Step 3: Load data
        logging.info("Loading data.")
        load()

        # Step 4: Train model
        logging.info("Training and saving the model.")
        train_and_save_model()

        # Step 5: Evaluate model
        logging.info("Evaluating and saving predictions.")
        evaluate_and_save_predictions()

        # Load data for visualization
        logging.info("Loading data for visualizations.")
        df1, df2 = load_data(file_path1, file_path2)
        
        # Print column names
        logging.info("Printing column names.")
        print_column_names(df1, df2)
        
        # Generate and save visualizations
        logging.info("Generating visualizations.")
        line_plot(df1)
        bar_plot(df1)
        interactive_bar_plot(df1)
        histogram(df1)
        boxplot(df1)
        interactive_scatter_plot(df2)
        choropleth_map(df2)

        logging.info("Data pipeline workflow completed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Execute the main function
if __name__ == "__main__":
    main()
