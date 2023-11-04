# Data preparation and understanding code
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Function including docstring that will print information that allows understanding of the raw DataFrame
def understand_df(dfdata):
    """Prints information that allows understanding of the DataFrame

    Args:
        dfdata: A pandas DataFrame containing the data
    """
    print('\nRAW DATASET INFORMATION\n')

    #Printing the basic information on the raw dataset
    print("\nThe shape of the dataset is: \n", dfdata.shape)
    print("\nThe columns of the dataset are: \n", dfdata.columns)
    print("\nThe first 5 rows of the dataset are: \n", dfdata.head(5))
    print("\nThe last 5 rows of the dataset are: \n", dfdata.tail(5))

    #Printing the unique values in the Country and Regionn columns to get an idea of the univeristies included in the dataset
    print("The unique values in the column 'Country of HE provider' are: \n", dfdata['Country of HE provider'].unique())
    print("The unique values in the column 'Region of HE provider' are: \n", dfdata['Region of HE provider'].unique())

    #Printing the data types of the columns in the dataset
    print("\nThe data types of the dataset are: \n", dfdata.dtypes)
    
    #Printing the number of null values in each column
    dfdata_null_columns = dfdata.isnull().sum()
    print("\nThe number of null values in each column is: \n", dfdata_null_columns)

    #Printing the rows with alphatical values in the 'Value' column
    print("\nThese rows have alphatical values in the 'Value' column: \n", dfdata[dfdata['Value'].str.isalpha()])


#Function including docstring that will prepare the data for use in the data visualisation dashboard
def prepare_df(dfdata):
    """The raw dataset is prepared for use in the data visualisation dashboard
    
    The data is cleaned and the output is saved to a csv file

    Args:
        dfdata: A pandas DataFrame containing the raw data

    Returns:
        df_clean: A pandas DataFrame containing the cleaned data

    """

    print('\nPREPARED DATASET INFORMATION\n')

    #The rows with null values are removed and to check this, the number of null values in each column is printed
    df_dropna = dfdata.dropna()
    print("\nThe number of null values in each column is: \n", df_dropna.isnull().sum())

    #Remove the rows where the 'Academic Year' is '2015/16' or '2016/17' or '2017/18'
    df_dropna = df_dropna[df_dropna['Academic Year'] != '2015/16']
    df_dropna = df_dropna[df_dropna['Academic Year'] != '2016/17']
    df_dropna = df_dropna[df_dropna['Academic Year'] != '2017/18']

    #Remove the rows where 'Country of HE provider' is not 'England'
    df_dropna = df_dropna[df_dropna['Country of HE provider'] == 'England']

    #Print the number or rows and columns in the dataset after removing the rows of universities 
    # not in England and removing all the data from before 2018/19
    print('\n The shape of the prepared dataset is: \n', df_dropna.shape)

    #Rename the 'Table' column to 'Class' and replace the values 
    # in the 'Class' column with the names of the classes
    df_class_column = df_dropna.rename(columns={'Table':'Class'})
    df_table_replaced = df_class_column.replace({'Table-1': 'Building and spaces',
                                            'Table-2': 'Energy',
                                            'Table-3': 'Emissions and waste',
                                            'Table-4': 'Transport and environment',
                                            'Table-5': 'Finances and people'})
    
    #Print the columns of the dataset after renaming the 'Table' 
    # column to 'Class' and replacing the values
    print('\nThe columns of the prepared dataset are: \n', df_table_replaced.columns)

    #Remove the percentage sign from the end of the values in the 'Value' column
    df_table_replaced.loc[:, 'Value'] = df_table_replaced['Value'].str.rstrip('%')
    
    #TODO: Explain this in the pdf
    #There are som rows with alphabetical values in the 'Value' column. The 
    # majority of these rows will be replaced with a numerical rating system to 
    # make them easier to analyse.

    #The method used to calculated scope 3 emissions is rated from Basic to Detailed 
    # but this is replaced with numbers to make it easier to analyse
    df_table_replaced['Value'] = df_table_replaced['Value'].apply(lambda x: 
                                                                  1 if x == 'Basic' 
                                                                    else (2 if x == 'Medium' 
                                                                    else (3 if x == 'Detailed' 
                                                                    else x)))
    
    #There are various categories related to HEIs environmental management with 'Yes' or'No' values. 
    # The Fairtrasde accreditation existence category also includes the value 'Working towards 
    # accreditation'. Again, these are replaced with numbers to make it easier to analyse.
    df_table_replaced['Value'] = df_table_replaced['Value'].apply(lambda x: 
                                                                  0 if x == 'No' 
                                                                    else (0.5 if x == 'Working towards accreditation' 
                                                                    else (1 if x == 'Yes' 
                                                                    else x)))
    

    #TODO:Explain this in the pdf
    #The remaining rows with alphabetical values in the 'Value' column are printed
    non_numeric_rows = df_table_replaced[pd.to_numeric(df_table_replaced['Value'], errors='coerce').isna()]
    print("\n The rows with non numeric values in the 'Vaue' column are", non_numeric_rows)
    
    df_prepared = df_table_replaced

    #Print the data types of the columns in the dataset after cleaning
    print("\nThe data types of the prepared dataset are: \n", df_prepared.dtypes)

    #Save the prepared dataset to a csv file
    prepared_dataset_filepath = Path(__file__).parent.joinpath('dataset','dataset_prepared.csv')
    df_prepared.to_csv(prepared_dataset_filepath, index=False)

    return df_prepared

#Function including docstring that will explore the data by drawing stripplots to undertand if there are any outliers
def explore_data(df_prepared):
    """ The data is explored by drawing plots to understand if there are any outliers
    
    Args:
        dfdata: A pandas DataFrame containing the data
        
        """
    #Remove rows where the 'Category' is Environmental management system external verification
    df_prepared = df_prepared[df_prepared['Category'] != 'Environmental management system external verification']
    df_numeric_values = df_prepared.astype({'Value': 'float64'})

    #TODO: Explain the following code in the pdf
    # Separate the dataset by 'Class'
    class_data = {}
    for class_name, group in df_numeric_values.groupby('Class'):
        class_data[class_name] = group

    # Determine the number of columns for subplots
    num_columns = 5

    # Specify the maximum number of subplots per figure
    max_subplots_per_figure = 20

    # Specify the directory where you want to save the figures
    save_dir = Path(__file__).parent.joinpath('figures')

    # Create separate figures for each 'Class' with varying numbers of rows and columns
    for class_name, class_df in class_data.items():
        num_categories = class_df['Category'].nunique()
        num_rows = (num_categories + num_columns - 1) // num_columns

        # Determine the number of figures needed
        num_figures = (num_categories + max_subplots_per_figure - 1) // max_subplots_per_figure

        for figure_num in range(num_figures):
            # Calculate the number of subplots for this figure
            subplots_in_figure = min(max_subplots_per_figure, num_categories - figure_num * max_subplots_per_figure)

            # Calculate the number of rows needed for the subplots
            num_rows_in_figure = (subplots_in_figure + num_columns - 1) // num_columns

            # Create a new figure with adjusted size for this batch of subplots
            fig, axes = plt.subplots(
                nrows=num_rows_in_figure,
                ncols=num_columns,
                figsize=(15, 5 * num_rows_in_figure))  # Adjust the height

            # Counter for the current subplot
            current_subplot = 0

            for i, (category, category_group) in enumerate(class_df.groupby('Category')):
                if figure_num * max_subplots_per_figure <= i < (figure_num + 1) * max_subplots_per_figure:
                    ax = axes[current_subplot // num_columns, current_subplot % num_columns]

                    # Create a Seaborn stripplot for the category
                    sns.stripplot(data=category_group, x=[0] * len(category_group), y='Value', ax=ax, alpha=0.5, jitter=True, color='green', size=7)

                    ax.set_title(f'{category}')
                    ax.set_ylabel('Value')

                    # Remove x-tick labels
                    ax.set_xticks([])

                    current_subplot += 1

            # Hide any remaining empty subplots
            for i in range(current_subplot, num_rows_in_figure * num_columns):
                fig.delaxes(axes[i // num_columns, i % num_columns])

            # Adjust the layout of subplots for the current figure
            plt.tight_layout()

            # Specify the file name and format for the saved figure
            figure_filename = f'{class_name}_figure{figure_num}.png'
            figure_filepath = save_dir.joinpath(figure_filename)

            # Save the figure
            plt.savefig(figure_filepath)

            # Close the current figure to release resources
            plt.close(fig)
    


if __name__ == "__main__":
    raw_data = Path(__file__).parent.joinpath('dataset','dataset.csv')
    data_df = pd.read_csv(raw_data, skiprows=10)
    #understand_df(data_df)      
    cleaned_data = prepare_df(data_df)
    explore_data(cleaned_data)

