# Data preparation and understanding code
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)

def understand_df(dfdata):
    """
    Prints information about the given dataframe, including its shape, columns, first and last 5 rows,
    unique values in the 'Country of HE provider' and 'Region of HE provider' columns, data types of columns,
    and number of null values in each column. Also identifies any rows with non-numeric values in the 'Value' column.
    
    Args:
        dfdata (pandas.DataFrame): The raw dataframe to be analysed.
    """
    # Print basic information about the dataframe
    print('\nRAW DATASET INFORMATION')
    print(f"\nThe shape of the dataset is:\n {dfdata.shape}")
    print(f"\nThe columns of the dataset are:\n {dfdata.columns}")
    print(f"\nThe first 5 rows of the dataset are:\n {dfdata.head(5)}")
    print(f"\nThe last 5 rows of the dataset are:\n {dfdata.tail(5)}")
    
    # Print unique values in specific columns
    print(f"\nThe unique values in the column 'Country of HE provider' are:\n {dfdata['Country of HE provider'].unique()}")
    print(f"\nThe unique values in the column 'Region of HE provider' are:\n {dfdata['Region of HE provider'].unique()}")
    
    # Print data types of columns and number of null values in each column
    print(f"\nThe data types of the columns in the dataset are:\n {dfdata.dtypes}")
    print(f"\nThe number of null values in each column is:\n {dfdata.isnull().sum()}")
    
    # Identify any rows with non-numeric values in the 'Value' column
    non_numeric_rows = dfdata[~dfdata['Value'].apply(is_numeric_or_nan)]
    print("\nThese rows have non-numeric values in the 'Value' column:\n")
    print(non_numeric_rows)

def is_numeric_or_nan(value):
    """
    Check if a given value is numeric or NaN.

    Args:
        value (any): The value to check.

    Returns:
        bool: True if the value is numeric or NaN, False otherwise.
    """
    try:
        # Attempt to convert the value to a float
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def prepare_df(dfdata):
    """
    This function prepares the input dataframe by cleaning, renaming, rearranging, converting and saving the data.
    It prints out information about the prepared dataset such as the number of null values and columns.

    Args:
        dfdata (pandas.DataFrame): The input dataframe to be prepared.

    Returns:
        pandas.DataFrame: The prepared dataframe.
    """
    print('\nPREPARED DATASET INFORMATION')
    # Clean the data
    df_clean = clean_data(dfdata)
    # Rename the columns
    df_clean = rename_columns(df_clean)
    # Rearrange the columns
    df_clean = rearrange_columns(df_clean)
    # Convert the value column
    df_clean = convert_value_column(df_clean)
    # Save the cleaned data
    df_clean = save_cleaned_data(df_clean)
    # Print the number of null values in each column
    print(f"\nThe number of null values in each column of the prepared  dataset is:\n {df_clean.isnull().sum()}")
    # Print the columns of the prepared dataset
    print(f"\nThe columns of the prepared dataset are:\n {df_clean.columns}")
    return df_clean

def clean_data(dfdata):
    """
    This function takes a pandas dataframe as input and performs the following operations:
    1. Removes any rows with missing values
    2. Removes any rows that do not correspond to England
    3. Removes any rows that correspond to academic years before 2018/19
    4. Returns the cleaned dataframe
    
    Args:
        dfdata (pandas.DataFrame): The input dataframe to be cleaned
    
    Returns:
        pandas.DataFrame: The cleaned dataframe
    """
    # Remove any rows with missing values
    df_clean = dfdata.dropna()
    
    # Remove any rows that correspond to academic years before 2018/19
    df_clean = remove_years(df_clean)
    
    # Remove any rows that do not correspond to England
    df_clean = keep_england_data(df_clean)
    
    # Print the shape of the prepared dataset
    print(f"\nThe shape of the prepared dataset is:\n {df_clean.shape}")
    
    # Return the cleaned dataframe
    return df_clean

def remove_years(dfdata):
    """
    Removes data from specific academic years from the input dataframe.

    Args:
        dfdata (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The input dataframe with data from specific academic years removed.
    """
    return dfdata[~dfdata['Academic Year'].isin(['2015/16', '2016/17', '2017/18'])]

def keep_england_data(dfdata):
    """
    Returns a new DataFrame containing only the rows where the 'Country of HE provider' column is 'England'.
    
    Args:
        dfdata (pandas.DataFrame): The DataFrame to filter.
    
    Returns:
        pandas.DataFrame: A new DataFrame containing only the rows where the 'Country of HE provider' column is 'England'.
    """
    return dfdata[dfdata['Country of HE provider'] == 'England']

def rename_columns(dfdata):
    """
    Renames the 'Table' column in the given DataFrame to 'Class', using a mapping of table names to class names.
    Then drops the 'Table' column from the DataFrame and returns the modified DataFrame.

    Args:
        dfdata (pandas.DataFrame): The DataFrame to modify.

    Returns:
        pandas.DataFrame: The modified DataFrame with the 'Table' column renamed to 'Class' and dropped.
    """
    # Create a new column 'Class' in the DataFrame by replacing the values in 'Table' column with the corresponding class names
    dfdata['Class'] = dfdata['Table'].replace({
        'Table-1': 'Building and spaces',
        'Table-2': 'Energy',
        'Table-3': 'Emissions and waste',
        'Table-4': 'Transport and environment',
        'Table-5': 'Finances and people'
    })
    # Drop the 'Table' column from the DataFrame
    dfdata = dfdata.drop(columns=['Table'])
    # Return the modified DataFrame
    return dfdata

def rearrange_columns(dfdata):
    """
    Rearranges the columns of a given dataframe such that the 'Class' column is moved to the 5th index.

    Args:
        dfdata (pandas.DataFrame): The dataFrame to rearrange.

    Returns:
        pandas.DataFrame: The rearranged dataFrame.
    """
    # Get the 'Class' column
    class_column = dfdata['Class']

    # Drop the 'Class' column from the dataframe
    dfdata = dfdata.drop(columns=['Class'])
 
    # Set the desired index for the 'Class' column
    desired_column_index = 5

    # Insert the 'Class' column at the desired index
    dfdata.insert(desired_column_index, 'Class', class_column)

    # Return the rearranged dataframe
    return dfdata


def convert_value_column(dfdata):
    """
    This function converts the 'Value' column of a given dataframe to numeric values.
    It removes the percentage sign from the values and applies the 'convert_to_numeric'
    function to convert them to numeric values. It then identifies any rows 
    with non-numeric values in the 'Value' column and prints them to the console.
    
    Args:
        dfdata (pandas.DataFrame): The Dataframe containing the data to be processed
    
    Returns:
        pandas.DataFrame: The Dataframe with the 'Value' column converted to numeric values
    """
    
    # Remove percentage sign from 'Value' column
    dfdata['Value'] = dfdata['Value'].str.rstrip('%')
    
    #Using the convert to numeric function to convert the majority of alphabetical values to numeric values
    dfdata['Value'] = dfdata['Value'].apply(convert_to_numeric)

    # Identify rows with non-numeric values in 'Value' column
    non_numeric_rows = dfdata[~pd.to_numeric(dfdata['Value'], errors='coerce').notna()]
    # Print rows with non-numeric values in 'Value' column
    print("\nThe remaining rows with non-numeric values in the 'Value' column in the prepared dataset are\n", non_numeric_rows)
    
    return dfdata

def convert_to_numeric(value):
    """
    Converts a string value to a numeric value based on predefined mappings.

    Args:
        value (str): The string value to be converted.

    Returns:
        int or float or str: The converted numeric value, or the original string value if no mapping is found.
    """
    # Check if the value is one of the predefined string values
    if value in ['Basic', 'Medium', 'Detailed']:
        # If it is, return the corresponding numeric value
        return {'Basic': 1, 'Medium': 2, 'Detailed': 3}[value]
    # Check if the value is one of the predefined string values
    if value in ['No', 'Working towards accreditation', 'Yes']:
        # If it is, return the corresponding numeric value
        return {'No': 0, 'Working towards accreditation': 0.5, 'Yes': 1}[value]
    # If the value is not in any of the predefined mappings, return the original string value
    return value

def save_cleaned_data(dfdata):
    """
    Saves the cleaned dataset to a CSV file.

    Args:
        dfdata (pandas.DataFrame): The cleaned dataset.

    Returns:
        pandas.DataFrame: The cleaned dataset.
    """

    heidf_with_duplicates = dfdata[['UKPRN', 'HE Provider', 'Region of HE provider']].copy()
    hei_df = heidf_with_duplicates.drop_duplicates()
    hei_df[['lat', 'lon']] = None
    hei_dataset_filepath = Path(__file__).parent.joinpath('dataset', 'hei_data.csv')
    hei_df.to_csv(hei_dataset_filepath, index=False)

    entrydf = dfdata[['Academic Year', 'Class', 'Category marker', 'Category', 'Value', 'UKPRN', 'HE Provider']].copy()
    entrydf.insert(0, 'id', range(1, 1 + len(entrydf)), )
    entry_dataset_filepath = Path(__file__).parent.joinpath('dataset', 'entry_data.csv')
    entrydf.to_csv(entry_dataset_filepath, index=False)

    return dfdata

def explore_data(dfdata):
    """
    Prepare the input dataframe by removing non-numerical data and create visualizations for the prepared data.

    Args:
        dfdata (pandas.DataFrame): The input DataFrame to be explored.
    """
    df_prepared = remove_non_numerical(dfdata)
    create_data_visualizations(df_prepared)

def remove_non_numerical(dfdata):
    """
    Removes rows from the input dataframe that have 'Environmental management system 
    external verification' as the Category value.
    Converts the 'Value' column of the resulting dataframe to float64 data type.
    
    Args:
        dfdata (pandas.DataFrame): The input DataFrame
    
    Returns:
        pandas.DataFrame: The Dataframe with non-numerical rows removed and 'Value' column converted to float64 data type.
    """
    df_prepared = dfdata[dfdata['Category'] != 'Environmental management system external verification']
    df_numeric_values = df_prepared.astype({'Value': 'float64'})
    return df_numeric_values

def create_data_visualizations(df_numeric_values):
    """
    Creates visualizations for the given numeric DataFrame.

    Args:
        df_numeric_values (pandas.DataFrame): The numeric DataFrame to visualize.
    """
    # Set the number of columns to display in each row of subplots
    num_columns = 5
    # Set the maximum number of subplots to display in each figure
    max_subplots_per_figure = 20
    # Set the directory to save the figures
    save_dir = Path(__file__).parent.joinpath('figures')

    # Separate the data by class
    class_data = separate_data_by_class(df_numeric_values)
    # Create subplots for each class and save the figures
    create_subplots(class_data, num_columns, max_subplots_per_figure, save_dir)

def separate_data_by_class(dfdata):
    """
    Separates the input dataframe into separate dataframes based on the unique values in the 'Class' column.

    Args:
        dfdata (pandas.DataFrame): The input DataFrame to be separated.

    Returns:
        dict: A dictionary where the keys are the unique values in the 'Class' column and the values are the corresponding DataFrames.
    """
    # Create an empty dictionary to store the separated dataframes
    class_data = {}

    # Group the input dataframe by the 'Class' column and iterate over the resulting groups
    for class_name, group in dfdata.groupby('Class'):
        # Add the current group to the dictionary with the key being the current class name
        class_data[class_name] = group

    # Return the dictionary of separated dataframes
    return class_data

def create_subplots(class_data, num_columns, max_subplots_per_figure, save_dir):
    """
    Create subplots for each category in the given class data and save them as separate figures.

    Args:
        class_data (dict): A dictionary containing the data for each class.
        num_columns (int): The number of columns to use for the subplots.
        max_subplots_per_figure (int): The maximum number of subplots to include in each figure.
        save_dir (str): The directory to save the figures in.
    """
    # Loop through each class in the class data
    for class_name, class_df in class_data.items():
        # Get the number of unique categories in the class data
        num_categories = class_df['Category'].nunique()

        # Calculate the number of rows needed for the subplots
        num_rows = (num_categories + num_columns - 1) // num_columns

        # Calculate the number of figures needed to display all the subplots
        num_figures = (num_categories + max_subplots_per_figure - 1) // max_subplots_per_figure

        # Loop through each figure
        for figure_num in range(num_figures):
            # Calculate the number of subplots to include in this figure
            subplots_in_figure = min(max_subplots_per_figure, num_categories - figure_num * max_subplots_per_figure)

            # Calculate the number of rows needed for the subplots in this figure
            num_rows_in_figure = (subplots_in_figure + num_columns - 1) // num_columns

            # Create a new figure with adjusted size for this batch of subplots
            fig, axes = plt.subplots(
                nrows=num_rows_in_figure,
                ncols=num_columns,
                figsize=(15, 5 * num_rows_in_figure))  # Adjust the height

            # Counter for the current subplot
            current_subplot = 0

            # Loop through each category in the class data
            for i, (category, category_group) in enumerate(class_df.groupby('Category')):
                # Check if this category should be included in the current figure
                if figure_num * max_subplots_per_figure <= i < (figure_num + 1) * max_subplots_per_figure:
                    # Get the current subplot
                    ax = axes[current_subplot // num_columns, current_subplot % num_columns]

                    # Create a Seaborn stripplot for the category
                    sns.stripplot(data=category_group, x=[0] * len(category_group), y='Value',  
                                  ax=ax, alpha=0.5, jitter=True, color='green', size=7)

                    # Set the title and y-axis label for the subplot
                    ax.set_title(f'{category}')
                    ax.set_ylabel('Value')

                    # Remove x-tick labels
                    ax.set_xticks([])

                    # Increment the current subplot counter
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
    data_df = pd.read_csv(raw_data, skiprows=10) #First 10 rows are not needed as they contain metadata
    understand_df(data_df)      
    cleaned_data = prepare_df(data_df)
    explore_data(cleaned_data)

