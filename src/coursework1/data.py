# Data preparation and understanding code
from pathlib import Path
import pandas as pd

def understand_df(dfdata):
    """Prints information that allows understanding of the DataFrame

    Args:
        dfdata: A pandas DataFrame containing the data
    """
    print('\nRAW DATASET INFORMATION\n')
    #The following code will provide information on the raw dataset
    print("\nThe shape of the dataset is: \n", dfdata.shape)
    print("\nThe columns of the dataset are: \n", dfdata.columns)
    print("\nThe first 5 rows of the dataset are: \n", dfdata.head(5))
    print("\nThe last 5 rows of the dataset are: \n", dfdata.tail(5))
    print("\nThe data types of the dataset are: \n", dfdata.dtypes)
    dfdata_null_rows = dfdata[dfdata.isnull().any(axis=1)]
    print("\nThe rows containing null values are: \n", dfdata_null_rows)
    dfdata_null_columns =dfdata.isnull().sum()
    print("\nThe number of null values in each column is: \n", dfdata_null_columns)

    # TODO: Add code to check for unique values of region, country, category only has one category marker


def prepare_df(dfdata):
    """The raw dataset is prepared for use in the data visualisation dashboard
    
    The data is cleaned and the output is saved to a csv file

    Args:
        dfdata: A pandas DataFrame containing the raw data

    Returns:
        df_clean: A pandas DataFrame containing the cleaned data

    """

    print('\nPREPARED DATASET INFORMATION\n')
    df_dropna = dfdata.dropna()
    print("\nThe number of null values in each column is: \n", df_dropna.isnull().sum())
    df_class_column = df_dropna.rename(columns={'Table':'Class'})
    df_replaced = df_class_column.replace({'Table-1': 'Building and spaces',
                                            'Table-2': 'Energy',
                                            'Table-3': 'Emissions and waste',
                                            'Table-4': 'Transport and environment',
                                            'Table-5': 'Finances and people'})
    print('\nThe columns of the prepared dataset are: \n', df_replaced.columns)

    df_prepared = df_replaced

    prepared_dataset_filepath = Path(__file__).parent.joinpath('dataset','dataset_prepared.csv')
    df_prepared.to_csv(prepared_dataset_filepath, index=False)

    return df_prepared


if __name__ == "__main__":
    raw_data = Path(__file__).parent.joinpath('dataset','dataset.csv')
    data_df = pd.read_csv(raw_data, skiprows=10)
    understand_df(data_df)
    cleaned_data = prepare_df(data_df)
