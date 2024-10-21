import pandas as pd
import matplotlib.pyplot as plt

def plot_graph_table():
    # Load the CSV file
    file_path = 'results.csv'  # Update with the correct file path
    df = pd.read_csv(file_path)

    # Create a pivot table for easier plotting
    pivot_table = df.pivot(index='Variation', columns='Model', values=['MAP', 'MRR'])

    # Creating a figure with six subplots (3 rows, 2 columns)
    _, axs = plt.subplots(1, 4, figsize=(18, 10))

    # Plotting MAP for different models and variations (subplot 1)
    pivot_table['MAP'].plot(ax=axs[0], marker='o', title='MAP for different Models and Variations')
    axs[0].set_xlabel('Variation')
    axs[0].set_ylabel('MAP')

    # Plotting MRR for different models and variations (subplot 2)
    pivot_table['MRR'].plot(ax=axs[1], marker='x', title='MRR for different Models and Variations')
    axs[1].set_xlabel('Variation')
    axs[1].set_ylabel('MRR')

    # Plotting MAP as a bar plot for different models (subplot 3)
    pivot_table['MAP'].plot(kind='bar', ax=axs[2], title='MAP for Models with Different Variations', legend=True)
    axs[2].set_xlabel('Variation')
    axs[2].set_ylabel('MAP')
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45)

    # Plotting MRR as a bar plot for different models (subplot 4)
    pivot_table['MRR'].plot(kind='bar', ax=axs[3], title='MRR for Models with Different Variations', legend=True)
    axs[3].set_xlabel('Variation')
    axs[3].set_ylabel('MRR')
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=45)

    plt.tight_layout()

    # Save the combined plots as an image
    plt.savefig("output/Variation_for_Model_Combined_Plots", bbox_inches='tight', dpi=300)

    pivot_table = df.pivot(index='Model', columns='Variation', values=['MAP', 'MRR'])

    fig, axs = plt.subplots(1, 4, figsize=(18, 10))

    # Plotting MAP for different models and variations (subplot 1)
    pivot_table['MAP'].plot(ax=axs[0], marker='o', title='MAP for different Models and Variations')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('MAP')

    # Plotting MRR for different models and Models (subplot 2)
    pivot_table['MRR'].plot(ax=axs[1], marker='x', title='MRR for different Models and Models')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('MRR')

    # Plotting MAP as a bar plot for different models (subplot 3)
    pivot_table['MAP'].plot(kind='bar', ax=axs[2], title='MAP for Models with Different Models', legend=True)
    axs[2].set_xlabel('Model')
    axs[2].set_ylabel('MAP')
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45)

    # Plotting MRR as a bar plot for different models (subplot 4)
    pivot_table['MRR'].plot(kind='bar', ax=axs[3], title='MRR for Models with Different Models', legend=True)
    axs[3].set_xlabel('Model')
    axs[3].set_ylabel('MRR')
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=45)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plots as an image
    plt.savefig("output/Model_for_Variation_Combined_Plots", bbox_inches='tight', dpi=300)

    def get_unique_for_column(df, column):
        unique_table = df.copy()
        last_val = None
        for i in range(len(df)):
            if df[column][i] == last_val:
                unique_table.at[i, column] = ''  # Remove repeated values
            else:
                last_val = df[column][i]
        return unique_table

    # Specify the column where you want to remove duplicate values (e.g., 'Model')
    column_to_uniquify = 'Model'  # Change this to the column you want

    # Applying the function to remove repeated values only in the specified column
    unique_table = get_unique_for_column(df, column_to_uniquify)

    # Convert the unique table to an image with expanded row span effect for the specified column
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create the table, showing unique values for the specified column and normal values for others
    table = ax.table(cellText=unique_table.values, colLabels=unique_table.columns, cellLoc='center', loc='center')

    # Save the table with unique values only for the specified column
    plt.savefig("output/unique_table_for_column.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    plot_graph_table()