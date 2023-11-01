# Data preparation and understanding code
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
    #TODO: Explain the folowing two lines in pdf
    print("The unique values in the column 'Country of HE provider' are: \n", dfdata['Country of HE provider'].unique())
    print("The unique values in the column 'Region of HE provider' are: \n", dfdata['Region of HE provider'].unique())


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
    # TODO: Explain this in the pdf
    df_numeric_values = df_replaced[pd.to_numeric(df_replaced['Value'], errors='coerce').notnull()]
    df_numeric_values.loc[:, 'Value'] = df_numeric_values['Value'].str.rstrip('%')
    df_numeric_values = df_numeric_values.astype({'Value': 'float64'})

    df_prepared = df_numeric_values

    print("\nThe data types of the prepared dataset are: \n", df_prepared.dtypes)

    prepared_dataset_filepath = Path(__file__).parent.joinpath('dataset','dataset_prepared.csv')
    df_prepared.to_csv(prepared_dataset_filepath, index=False)

    return df_prepared

#Function including docstring that will explore the data by drawing plots to undertand if there are any outliers
def explore_data(df_prepared):
    """ The data is explored by drawing plots to understand if there are any outliers
    
    Args:
        dfdata: A pandas DataFrame containing the data
        
        """
    
    #TODO: Explain the following code in the pdf + the image generated
    plt.rc('axes', titlesize=4)
    plt.rc('axes', labelsize=4)
    grouped = df_prepared.groupby('Category marker')
    #rowlength = int((grouped.ngroups+3)/4)
    #fig, axs = plt.subplots(4, rowlength, gridspec_kw=dict(wspace=0.5, hspace=0.5))

    #Create a new figure for each boxplot for value in each category marker group, where each figure is not a subplot
    for grp in grouped['Value']:
        plt.figure()
        grp[1].plot(kind='box', title=grp[0])
        plt.title(grp[0])
        plt.tick_params(axis='both', which='major', labelsize=7)
    

    #for grp, ax in zip(grouped['Value'], axs.flatten()):
        #grp[1].plot(kind='hist', ax=ax, title=grp[0])
        #ax.title.set_size(6)
        #ax.tick_params(axis='both', which='major', labelsize=7)
    
    plt.show()

if __name__ == "__main__":
    raw_data = Path(__file__).parent.joinpath('dataset','dataset.csv')
    data_df = pd.read_csv(raw_data, skiprows=10)
    #understand_df(data_df)      
    cleaned_data = prepare_df(data_df)
    explore_data(cleaned_data)

