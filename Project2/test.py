import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft, rfft
from joblib import dump, load


def get_no_meal_data_feature_matrix(no_meal_data):
    index_to_remove_non_meal = no_meal_data.isna().sum(axis=1).replace(0, np.nan).dropna().where(
        lambda x: x > 5).dropna().index
    no_meal_data_cleaned = no_meal_data.drop(no_meal_data.index[index_to_remove_non_meal]).reset_index().drop(
        columns='index')
    no_meal_data_cleaned = no_meal_data_cleaned.interpolate(method='linear', axis=1)
    index_to_drop_again = no_meal_data_cleaned.isna().sum(axis=1).replace(0, np.nan).dropna().index
    no_meal_data_cleaned = no_meal_data_cleaned.drop(
        no_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    no_meal_feature_matrix = pd.DataFrame()
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    for i in range(len(no_meal_data_cleaned)):
        array = abs(rfft(no_meal_data_cleaned.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array = abs(rfft(no_meal_data_cleaned.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        power_third_max.append(sorted_array[-4])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    no_meal_feature_matrix['power_second_max'] = power_second_max
    no_meal_feature_matrix['power_third_max'] = power_third_max
    first_differential_data = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(no_meal_data_cleaned)):
        first_differential_data.append(np.diff(no_meal_data_cleaned.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(no_meal_data_cleaned.iloc[:, 0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(no_meal_data_cleaned.iloc[i]))
    no_meal_feature_matrix['2ndDifferential'] = second_differential_data
    no_meal_feature_matrix['standard_deviation'] = standard_deviation
    return no_meal_feature_matrix


def get_results():
    data = pd.read_csv('test.csv', header=None)
    feature_matrix = get_no_meal_data_feature_matrix(data)

    with open('RandomForestClassifier.pickle', 'rb') as pre_trained_model_file:
        model = load(pre_trained_model_file)
        predictions = model.predict(feature_matrix)
        pre_trained_model_file.close()

    pd.DataFrame(predictions).to_csv('Result.csv', index=False, header=False)


if __name__ == '__main__':
    get_results()
