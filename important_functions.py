import itertools
import os
import pickle
import pandas as pd
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import csv 
from sklearn.calibration import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LassoCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
import joblib

def process_data(excel_path: str, sample_number: int):
    
    """Process data from an Excel file and combine it with the categoric encoded variables.

    This function reads data from an Excel file specified by the 'excel_path' argument,
    extracts information related to a particular sample identified by 'sample_number',
    and combines it with encoded variables from another sheet in the same Excel file.

    Args:
        excel_path (str): The file path to the Excel file containing the data.
        sample_number (int): The identifier of the sample to be processed, but 
        only the number of the sample as the "SF_" is already included inside the function. 

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data, where each row
        corresponds to a specific sample, and columns represent different features
        extracted from the Excel file and encoded variables. The columns format is 
        "offline feature"_"day".

    Example:
        sample = process_data(excel_path = "Clean_data_(SF)\\New Data.xlsx", sample_number = 0)
    """

    sheet_name = "SF_" + str(sample_number)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    new_df = pd.DataFrame()
    codes = "Variable_Encoding"
    df_encoded = pd.read_excel(excel_path, sheet_name=codes)
    new_df_encoded = pd.DataFrame()

    for column in df.columns:
        if column != 'Days':
            for day in df['Days']: 
                new_column_names = f"{column}_{day}"
                new_value = df.loc[df['Days'] == day, column].values[0]
                new_df.loc[sheet_name, new_column_names] = new_value

    new_df = new_df.drop(columns=["Mel_0", "Mel_1_2", "Lip_0", "Lip_1_2"])

    for column in df_encoded.columns:
        if column != 'ID':
            new_value = df_encoded.loc[df_encoded['ID'] == sheet_name, column].values[0]
            new_df_encoded.loc[sheet_name, column] = new_value

    sf_combined = pd.concat((new_df, new_df_encoded), axis=1)
    return sf_combined

def process_multiple_samples(excel_path: str, num_samples: int):

    """Process data for multiple samples from an Excel file and concatenate them into a single DataFrame.

    This function processes data for a specified number of samples from an Excel file,
    combines each sample into a DataFrame, and concatenates them into a single DataFrame.

    Args:
        excel_path (str): The file path to the Excel file containing the data.
        num_samples (int): The number of samples to be processed.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data for all samples,
        where each row corresponds to a specific sample, and columns represent different features
        extracted from the Excel file corresponding to the values of the offline features in each day of
        the process and categoric encoded variables.

    Example:
        df = process_multiple_samples(excel_path = "Clean_data_(SF)\\New Data.xlsx" ,num_samples = 36)
    """

    all_samples = []
    
    for i in range(1, num_samples + 1):
        sample = process_data(excel_path, i)
        all_samples.append(sample)
    
    df_final = pd.concat(all_samples, axis=0)
    return df_final

def get_differences_between_days(excel_path,sample_number):
    sheet_name = "SF_" + str(sample_number)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    new_df = pd.DataFrame()
    for column in df.columns:
        if column != 'Days' and column != "Bio":
            for i in range(len(df["Days"])-1):
                day_a = df["Days"].iloc[-(i+1)]
                day_b = df["Days"].iloc[-(i+2)] 
                new_column_name = f"diff_{column}_{day_b}_to_{day_a}"
                first_value = df[column].iloc[-(i+1)]
                second_value = df[column].iloc[-(i+2)]
                new_value = first_value - second_value
                new_df.loc[sheet_name, new_column_name] = new_value
    
    new_df = new_df.drop(columns=['diff_Mel_1_2_to_4', 'diff_Mel_0_to_1_2','diff_Lip_1_2_to_4', 'diff_Lip_0_to_1_2'])
    return new_df

def multiple_get_differences_between_days(excel_path: str, num_samples: int):
    all_samples = []
    
    for i in range(1, num_samples + 1):
        sample = get_differences_between_days(excel_path, i)
        all_samples.append(sample)
    
    df_final = pd.concat(all_samples, axis=0)
    return df_final
    
def carbon_nitrogen_ratio(dataframe: pd.DataFrame, days: list, rows: list = None):

    """Calculate the carbon and nitrogen ratio for specific days and samples in a DataFrame.

    This function calculates the carbon-to-nitrogen (C/N) ratio for each specified day and sample 
    in a given DataFrame. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.

    Args:
        dataframe (DataFrame): The DataFrame containing the data.
        days (list): A list of days for which the C/N ratio is calculated.
            If one wants to calculate the C/N ratio for day "1_2" the list should contain the days in str form,
            otherwise it can be passed as a list of int.
        rows (list, optional): A list of row labels (samples) for which the C/N ratio will be calculated. 
            Defaults to None, which means all rows in the DataFrame are used if no rows are specified.

    Returns:
        DataFrame: A DataFrame containing the calculated C/N ratios for each specified day, indexed by the row labels.

    Example:
        carbon_nitrogen_ratio(df,[4,7],["SF_1","SF_2"])
        carbon_nitrogen_ratio(df,["1_2","4","7"],["SF_1","SF_2"])
    """    

    ratio_values = {}

    if rows is None:
        rows = dataframe.index
    
    for row in rows:
        sample = dataframe.loc[row]  

        for day in days:
            value_carbon = sample[f"Sugar_{day}"]
            value_nitrogen = sample[f'NaNO3_{day}']
            value = value_carbon / value_nitrogen
            ratio_values.setdefault(row, {})[f'C/N_{day}'] = value
    
    return pd.DataFrame(ratio_values).T

def biomass_specific_growth_rate(dataframe: pd.DataFrame, start_day: Union[int, str], end_day: int, rows: list = None):
    
    """Calculates the specific growth rate for specific rows (samples).
    
    This function calculates the specific growth rate for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        start_day (int or str): The first day where there is biomass measurements.
            If the start day is "1_2" it should be in str format, otherwise int format.
        end_day (int): The last day where there is biomass measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the growth rate.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the specific growth rate for each specified row,
        where each row represents a sample, and the column is labeled 'GrowthRate'.
    
    Example:    
        biomass_specific_growth_rate(df,"1_2",7,["SF_1","SF_2"])
    """

    if start_day  == "1_2":
        start_day = 1.5
        time = (end_day - start_day) * 24
        start_day = "1_2"
    if end_day == "1_2":
        end_day = 1.5
        time = (end_day - start_day) * 24
        end_day = "1_2"
    else:
        time = (end_day - start_day) * 24


    growth_rates = {}

    if rows is None:
        rows = dataframe.index
    
    for row_idx in rows:  
        x_0 = dataframe.at[row_idx, f'Bio_{start_day}']
        if x_0 == 0.0:
            x_0 = 1
        x_t = dataframe.at[row_idx, f'Bio_{end_day}']
        growth_rate = np.log(x_t / x_0) / time
        growth_rates[row_idx] = growth_rate
    
    return pd.DataFrame(growth_rates.values(), index=growth_rates.keys(), columns=['GrowthRate'])

def carbon_and_nitrogen_consumption_rate(dataframe: pd.DataFrame, start_day: Union[int, str], end_day: int, rows: list = None):
        
    """Calculates the substrate consupation rate for specific rows (samples).
    
    This function calculates the carbon and nitrogen consumption rate for each specified sample 
    in a given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        start_day (int or str): The first day where there is substrates measurements.
            If the start day is "1_2" it should be in str format, otherwise int format.
        end_day (int): The last day where there is substrate measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the consumption rate.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the consumption rate for each specified row,
        where each row represents a sample, and there's two column one for each substrate.
    
    Example:    
        carbon_and_nitrogen_consumption_rate(df,0,18,["SF_4","SF_5"])   
    """
    if start_day  == "1_2":
        start_day = 1.5
        time = (end_day - start_day) * 24
        start_day = "1_2"
    if end_day == "1_2":
        end_day = 1.5
        time = (end_day - start_day) * 24
        end_day = "1_2"
    else:
        time = (end_day - start_day) * 24

    consumption_rates = {"CarbonConsumptionRate": [], "NitrogenConsumptionRate": []}
    
    if rows is None:
        rows = dataframe.index
    
    for row_idx in rows:  
        sample = dataframe.loc[row_idx] 
        carbon_0 = sample["Sugar_{}".format(start_day)]
        nitrogen_0 = sample['NaNO3_{}'.format(start_day)]
        carbon_7 = sample["Sugar_{}".format(end_day)]
        nitrogen_7 = sample['NaNO3_{}'.format(end_day)]

        carbon_consumption_rate = (carbon_7 - carbon_0) / time
        nitrogen_consumption_rate = (nitrogen_7 - nitrogen_0) / time

        consumption_rates["CarbonConsumptionRate"].append(carbon_consumption_rate)
        consumption_rates["NitrogenConsumptionRate"].append(nitrogen_consumption_rate)

    return pd.DataFrame(consumption_rates, index=rows)

def biomass_substrate_yield(dataframe: pd.DataFrame, start_day: Union[int, str], end_day: int, rows: list = None):
        
    """Calculates the biomass substrate yield for specific rows (samples).
    
    This function calculates the biomass substrate yield for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        start_day (int or str): The first day where there is sugar measurements.
            If the start day is "1_2" it should be in str format, otherwise int format.
        end_day (int): The last day where there is biomass and sugar measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the biomass substrate yield.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the biomass substrate yield for each specified row,
        where each row represents a sample, and the column is labeled 'Yield'.
    
    Example:    
        biomass_substrate_yield(df,0,18,["SF_4","SF_5"])
    """
    
    yields = {}

    if rows is None:
        rows = dataframe.index
    
    for row_idx in rows:  
        x_t = dataframe[f"Bio_{end_day}"].loc[row_idx]
        s_i = dataframe[f"Sugar_{start_day}"].loc[row_idx]
        s_o = dataframe[f"Sugar_{end_day}"].loc[row_idx]
        if s_o == 0 and s_i == 0:
            b_s_yield = 0
        else:
            b_s_yield = x_t/(s_i - s_o)

        yields[row_idx] = b_s_yield
    
    return pd.DataFrame(yields.values(), index=yields.keys(), columns=['Yield'])

def specific_rate_of_substrate_consumption(dataframe: pd.DataFrame, time: list, substrate: str, end_day: Union[int, str], rows: list = None):
             
    """Calculates the specific rate of substrate consumption for specific rows (samples).
    
    This function calculates the specific rate of substrate consumption for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        time (list): List of time intervals in days from day 0 to the last day of production.
            Instead of "1_2" in the list, one should put 1.5.
        substrate (str): The substrate for which the calculation should be made ("Sugar" or "NaNO3")
        end_day (int): The last day where there is substrate measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the specific rate of substrate consumption.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the specific rate of substrate consumtpion for each specified row,
        where each row represents a sample, and the column is labeled 'SpecificRateOf{substrate}Consumption'.
    
    Example:    
        specific_rate_of_substrate_consumption(df,[0,1.5,4,7,10,14,18], substrate= "Sugar", end_day=18,rows=["SF_4","SF_5"])
    """

    times = [t * 24 for t in time]

    if rows is None:
        rows = dataframe.index

    substrate_consumptions = {}

    for row_idx in rows:
        biomass = dataframe[f"Bio_{end_day}"].loc[row_idx]
        substrates = []
        sample = dataframe.loc[row_idx] 

        for day in time:
            if day == 1.5:
                day = "1_2"
            
            substrate_values = sample[f"{substrate}_{day}"]
            substrates.append(substrate_values)     

        time_interval = pd.Series(times).diff().mean()
        substrate_interval = pd.Series(substrates).diff().mean()

        substrate_consumption = (1/biomass)*(substrate_interval/time_interval)
        substrate_consumptions[row_idx] = substrate_consumption
        
    return pd.DataFrame (substrate_consumptions.values(), index=substrate_consumptions.keys(), columns=[f"SpecificRateOf{substrate}Consumption"])

#####Feature engeneering functions

##PCA functions used

def create_pca_features(n_components:int,train_x_scaled:pd.DataFrame,columns,train_x):

    pca = PCA(n_components=n_components)
    pca.fit_transform(train_x_scaled)
    components = pca.components_
    components = np.mean(components, axis=0)
    features_subset = train_x.drop(columns[components <= 0], axis=1)
    return features_subset

def pca_explained_variance(n_components:int,train_x_scaled:pd.DataFrame):
    pca = PCA(n_components=n_components)
    pca.fit_transform(train_x_scaled)
    exp_var_pca = sum(pca.explained_variance_ratio_)

    print(exp_var_pca)

def plot_of_the_cumulative_sum_of_eigenvalues(train_x_scaled: pd.DataFrame,  type_of_experience: str = None, number_of_case: int = None,save: str = None,):
    pca = PCA()

    pca.fit_transform(train_x_scaled)

    exp_var_pca = pca.explained_variance_ratio_

    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\pca_explained_variance.png",dpi=300)


### ANOVA functions

def create_anova_features(k_features:int, train_x: pd.DataFrame, train_y:pd.DataFrame, train_x_columns ):
    selector_anova = SelectKBest(score_func=f_classif, k=k_features)

    X_anova_array = selector_anova.fit_transform(pd.DataFrame.to_numpy(train_x), np.ravel( pd.DataFrame.to_numpy(train_y)))

    feature_cols_anova = train_x_columns[selector_anova.get_support()].tolist()
    features_anova = pd.DataFrame(X_anova_array, columns=feature_cols_anova)
    return features_anova

def plot_ANOVA_F_values(train_x: pd.DataFrame, train_y_array, train_x_columns, type_of_experience: str = None, number_of_case: int = None, save: str = None,):
    fig, (ax1) = plt.subplots(1,figsize=(14, 6))
    f_values, _ = f_classif(train_x, train_y_array)
    ax1.bar(train_x_columns, f_values)
    ax1.set_title('ANOVA F-values')
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_ylabel('F-value')
    plt.tight_layout()
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\ANOVA_f_score.png",dpi=300)


### Lasso functions
def create_lasso_features(cv:int,max_iter:int,train_x_scaled,train_y_array,train_x_columns,train_x,random_state=42):
    lasso = LassoCV(cv=cv,max_iter=max_iter,random_state=random_state)
    lasso.fit(train_x_scaled, train_y_array)

    selected_columns = train_x_columns[np.abs(lasso.coef_) > 1e-9]
    features_lasso = train_x[selected_columns.tolist()]
    return features_lasso

def plot_lasso_coef_values(cv, max_iter,train_x_scaled,train_y_array,train_x_columns,type_of_experience: str = None, number_of_case: int = None, save: str = None,):
    lasso = LassoCV(cv=cv,max_iter=max_iter)
    lasso.fit(train_x_scaled, train_y_array)
    plt.figure(figsize=(18, 6))
    plt.bar(train_x_columns, np.abs(lasso.coef_))
    plt.title('Lasso Coefficients')
    plt.xticks(rotation=90)
    plt.ylabel('Coefficient')
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\lasso_coeficient.png",dpi=300)

###Correlations functions
def create_correlation_features(train_x,correlation_threshold):
    corr_matrix = train_x.corr()

    correlation_threshold=correlation_threshold

    correlation_matrix = train_x.corr(numeric_only=True).abs()

    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper.columns:
        for idx, val in enumerate(upper[column]):
            if val > correlation_threshold:
                if column not in to_drop and upper.index[idx] not in to_drop:
                    to_drop.append(column)

    features_correlation = train_x.drop(columns=to_drop)
    return features_correlation


### RFE functions

def create_RFE_features(n_features_to_select,train_x_scaled, train_y_scaled_array,train_x):
    estimator = SVR(kernel='linear')
    selector = RFE(estimator, n_features_to_select=n_features_to_select)

    selector = selector.fit(train_x_scaled, train_y_scaled_array)

    features_column = selector.get_feature_names_out()

    features_rfe = train_x[features_column.tolist()]

    return features_rfe

def plot_RFE_ranking(train_x_columns,n_features_to_select,train_x_scaled, train_y_scaled_array,type_of_experience: str = None, number_of_case: int = None, save: str = None,):

    estimator = SVR(kernel='linear')
    selector = RFE(estimator, n_features_to_select=n_features_to_select)

    selector = selector.fit(train_x_scaled, train_y_scaled_array)
    plt.figure(figsize=(18, 6))
    plt.bar(train_x_columns, selector.ranking_)
    plt.title('RFE Rankings')
    plt.xticks(rotation=90)
    plt.ylabel('Ranking')
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\RFE_rankings.png",dpi=300)

### Grid Search

#Grid Search functions

def get_scores(y_true,y_pred,metric):
    score = metric(y_true,y_pred)
    return score.mean()

def sanitize_args_dict(args_dict):
    return "_".join(str(v) for v in args_dict.values())

def save_checkpoint(store, filename='store_checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(store, f)

def load_checkpoint(filename='store_checkpoint.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return {}
    
def stage_experiments(dict_params:dict):
    #get list of lengths of each value associated with a key
    lens = [len(x) for x in dict_params.values()]
    #get a list of arrays with a random permutation for each value 
    permutes = [np.random.permutation(l) for l in lens]
    # make all possible combinations for the indexs
    combinations = list(itertools.product(*permutes))
    return combinations

def save_model(model,mode,case,algorithm,subset,args):
    joblib.dump(model,f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{mode}\\Case_{case}\\GridSearch\\{algorithm}\\{subset}_{args}.pkl")

def load_model(file_path):
    loaded_model = joblib.load(file_path)
    return loaded_model

def search_ML(dict_param, algorithm, x_val_train, y_val_train, x_val_test, y_val_test, train_y_index, train_y_columns, scaler, train_y_as_array, scoring, mode, subset, case,regressor, scale=None,  checkpoint_file='checkpoint.pkl', arg_to_skip=None):
    grid = stage_experiments(dict_param)
    store = load_checkpoint(checkpoint_file)
    print("Initial checkpoint data:", store)
    count = 0

    for experiment in grid:
        args = [v[experiment[i]] for i, v in enumerate(dict_param.values())]
        count += 1
        print(f"Experiment {count}/{len(grid)}: {args}")
        args_dict = {k: args[i] for i, k in enumerate(dict_param)}

        if str(args_dict) in store:
            print(f"Skipping experiment {count}/{len(grid)} (already completed)")
            continue
        else:
            try:
                model = algorithm(**args_dict)
                model.fit(x_val_train,y_val_train)
                args_str = sanitize_args_dict(args_dict)
                save_model(model=model, mode=mode, case=case, algorithm=regressor, subset=subset, args=args_str)
                y_val_test_pred = model.predict(x_val_test) 
                

                if scale == "y":
                    y_val_test_pred_scaled = pd.DataFrame(y_val_test_pred, index=train_y_index, columns=train_y_columns)
                    y_val_test_pred = pd.DataFrame(scaler.inverse_transform(y_val_test_pred_scaled), index=train_y_index, columns=train_y_columns)
                    y_val_test_pred = y_val_test_pred.values.ravel()

                for metric in scoring:
                    value = get_scores(y_true=y_val_test, y_pred=y_val_test_pred, metric=scoring[metric])
                    if str(args_dict) not in store:
                        store[str(args_dict)] = []
                    store[str(args_dict)].append(value)
                print(f"Results for experiment {count}: {store[str(args_dict)]}")
                save_checkpoint(store, checkpoint_file)

            except Exception as e:
                store[str(args_dict)] = []
                save_checkpoint(store, checkpoint_file)
                print(f"Error encountered with args: {args_dict}")
                print(f"Exception: {e}")
                continue
    
    print("Final store contents:", store)
    return store

def transform_csv_into_dict(path: str) -> dict:
    data_dict = {}

    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            if len(row) == 2:
                # Convert the parameter string to a dictionary
                param_dict = ast.literal_eval(row[0])
                
                # Convert any list in the parameter dictionary to a tuple
                for key, value in param_dict.items():
                    if isinstance(value, list):
                        param_dict[key] = tuple(value)
                
                # Convert the scores string to a list of floats
                scores = ast.literal_eval(row[1])
                
                # Convert the param_dict to a tuple of tuples
                key = tuple(sorted(param_dict.items()))

                # Store in the dictionary
                data_dict[key] = scores

    return data_dict

def get_best_param(dict: dict):
    # best_value = 0
    best_value = float('inf')
    for key, value in dict.items():
        if isinstance(value, list) and len(value) > 0:
            if value[0] < best_value:
                best_value = value[0]
                best_param = key
    return best_param, best_value

def bar_for_metrics(dictionary: dict, first_label: str, second_label:str, y_label:str, save: str= None,):
    labels = list(dictionary.keys())
    first_values = [value[0] for value in dictionary.values()]
    second_values = [value[1] for value in dictionary.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_height = 0.35
    index = range(len(labels))
    first_bars = ax.barh(index, first_values, bar_height, label=first_label, color='lightblue')
    second_bars = ax.barh([i + bar_height for i in index], second_values, bar_height, label=second_label, color='lightcoral')

    ax.set_ylabel(y_label)
    ax.set_xlabel('Error')
    ax.set_title(f'{first_label} and {second_label} for Different {y_label}')
    ax.set_yticks([i + bar_height / 2 for i in index])
    ax.set_yticklabels(labels)
    ax.legend()

    for bars in [first_bars, second_bars]:
        for bar in bars:
            width = bar.get_width()
            ax.annotate('{}'.format(round(width, 2)),
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha='left', va='center')
    if save == "y":
         plt.savefig("results.png", dpi=300)
    plt.tight_layout()
    plt.show()

def plot_y_true_pred(true_y: pd.DataFrame, target: str, pred_y:dict, model: str, save: str = None):
    y_true_labels = true_y.index.tolist()
    y_true_values = true_y[target].values

    plt.figure(figsize=(10, 6))

    plt.scatter(y_true_labels, y_true_values,color="red", label='Experimental Values',marker='D',s=80)

    colors = ['b', 'g', 'c', 'm', 'y', 'orange']

    for i, (key, y_pred) in enumerate(pred_y.items()):
        plt.scatter(y_true_labels, y_pred, color=colors[i % len(colors)],label=key)

    plt.ylim(bottom=0)


    # plt.title(f'Experimental vs Predicted Values for different feature subsets using {model}')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Samples',fontsize = 15)
    plt.ylabel('MELs concentration values (g/L)', fontsize = 15)
    plt.legend(fontsize = 10)
    plt.grid(False)
    if save == "y":
         plt.savefig(f"plot_y_pred_and_true_for_{model}.png", dpi=300)
    plt.show()

def percentage_histogram(true_y: pd.DataFrame, target: str, pred_y: dict, model: str, save: str = None):
    y_true_values = true_y[target].values

    num_subplots = len(pred_y)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (key, y_pred) in enumerate(pred_y.items()):
        values = []
        for actual, predicted in zip(y_true_values, y_pred):
            if np.isclose(actual, 0, 1, 1e-3):
                values.append(round((np.abs(actual - predicted)) * 100, 0))
            else:
                values.append(round((np.abs((actual - predicted) / actual)) * 100, 0))
        
        sns.histplot(values, kde=True, ax=axes[idx])
        axes[idx].set_title(f'{model} - {key}')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xlabel('Percentage (%)')
    
    plt.tight_layout()
    if save == "y":
        plt.savefig(f"percentage_histogram_for_{model}.png", dpi=300)
    plt.show()

def residual_histogram(true_y: pd.DataFrame, target: str, pred_y: dict, model: str, save: str = None, xlim=None, ylim=None):
    y_true_values = true_y[target].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (key, y_pred) in enumerate(pred_y.items()):
        residuals = y_true_values - y_pred
        
        sns.histplot(residuals, kde=True, ax=axes[idx])
        axes[idx].set_title(f'{model} - {key}')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xlabel('Residuals')
        
        # Set x and y limits if specified
        if xlim:
            axes[idx].set_xlim(xlim)
        if ylim:
            axes[idx].set_ylim(ylim)

    plt.tight_layout()
    if save == "y":
        plt.savefig(f"residual_histogram_for_{model}.png", dpi=300)
    plt.show()
