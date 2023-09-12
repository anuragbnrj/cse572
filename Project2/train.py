import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from datetime import timedelta
from scipy.fftpack import fft, ifft, rfft
from sklearn.utils import shuffle


def create_meal_data(insulin_data_original, cgm_data_original):
    insulin_data = insulin_data_original.copy()
    insulin_data = insulin_data.set_index('date_time')
    insulin_data = insulin_data.sort_values(by='date_time', ascending=True).dropna().reset_index()
    insulin_data['BWZ Carb Input (grams)'].replace(0.0, np.nan, inplace=True)
    insulin_data = insulin_data.dropna()
    insulin_data = insulin_data.reset_index().drop(columns='index')
    timestamps = insulin_data['date_time'].values
    timestamps_with_2_hrs_diff = []

    for i in range(len(timestamps) - 1):
        curr_timestamp = insulin_data['date_time'][i]
        next_timestamp = insulin_data['date_time'][i + 1]
        diff_with_next_timestamp = (next_timestamp - curr_timestamp).seconds
        if diff_with_next_timestamp >= 7200:
            timestamps_with_2_hrs_diff.append(curr_timestamp)
    timestamps_with_2_hrs_diff.append(insulin_data['date_time'][len(timestamps) - 1])

    meal_data = []

    for idx, i in enumerate(timestamps_with_2_hrs_diff):
        start = pd.to_datetime(i - timedelta(seconds=1800))
        end = pd.to_datetime(i + timedelta(seconds=7199))
        curr_date = i.date()

        meal_data.append(
            cgm_data_original[cgm_data_original['Date'].dt.date == curr_date]
            .set_index('date_time')
            .between_time(start_time=start.time(), end_time=end.time())['Sensor Glucose (mg/dL)']
            .values
            .tolist()
        )
    return pd.DataFrame(meal_data)


def create_no_meal_data(insulin_data_original, cgm_data_original):
    insulin_data = insulin_data_original.copy()
    insulin_data = insulin_data.sort_values(by='date_time', ascending=True).replace(0.0, np.nan).dropna().copy()
    insulin_data = insulin_data.reset_index().drop(columns='index')

    timestamps = insulin_data['date_time'].values
    timestamps_with_4_hrs_diff = []

    cgm_data = cgm_data_original.copy()

    for i in range(len(timestamps) - 1):
        curr_timestamp = insulin_data['date_time'][i]
        next_timestamp = insulin_data['date_time'][i + 1]
        diff_with_next_timestamp = (next_timestamp - curr_timestamp).seconds
        if diff_with_next_timestamp >= 14400:
            timestamps_with_4_hrs_diff.append(curr_timestamp)
    timestamps_with_4_hrs_diff.append(insulin_data['date_time'][len(timestamps) - 1])

    no_meal_data = []
    for idx, i in enumerate(timestamps_with_4_hrs_diff):
        counter = 1
        try:
            number_of_no_meal_data_sets = len(cgm_data.loc[(cgm_data['date_time'] >= (
                    timestamps_with_4_hrs_diff[idx] + pd.Timedelta(hours=2))) & (cgm_data['date_time'] <
                                                                                 timestamps_with_4_hrs_diff[
                                                                                     idx + 1])]) // 24
            while counter <= number_of_no_meal_data_sets:
                if counter == 1:
                    no_meal_data.append(cgm_data.loc[(cgm_data['date_time'] >= (
                            timestamps_with_4_hrs_diff[idx] + pd.Timedelta(hours=2))) & (
                                                                 cgm_data['date_time'] < timestamps_with_4_hrs_diff[
                                                             idx + 1])]['Sensor Glucose (mg/dL)'][
                                        :counter * 24].values.tolist())
                    counter += 1
                else:
                    no_meal_data.append(cgm_data.loc[(cgm_data['date_time'] >= timestamps_with_4_hrs_diff[
                        idx] + pd.Timedelta(hours=2)) & (cgm_data['date_time'] < timestamps_with_4_hrs_diff[idx + 1])][
                                            'Sensor Glucose (mg/dL)'][
                                        (counter - 1) * 24:(counter) * 24].values.tolist())
                    counter += 1
        except IndexError:
            break
    return pd.DataFrame(no_meal_data)


def get_meal_data_feature_matrix(meal_data):
    index = meal_data.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    meal_data_cleaned = meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned = meal_data_cleaned.interpolate(method='linear', axis=1)
    index_to_drop_again = meal_data_cleaned.isna().sum(axis=1).replace(0, np.nan).dropna().index
    meal_data_cleaned = meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    meal_data_cleaned = meal_data_cleaned.dropna().reset_index().drop(columns='index')
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    for i in range(len(meal_data_cleaned)):
        array = abs(rfft(meal_data_cleaned.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sorted_array = abs(rfft(meal_data_cleaned.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        power_third_max.append(sorted_array[-4])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    meal_feature_matrix = pd.DataFrame()
    meal_feature_matrix['power_second_max'] = power_second_max
    meal_feature_matrix['power_third_max'] = power_third_max
    tm = meal_data_cleaned.iloc[:, 22:25].idxmin(axis=1)
    maximum = meal_data_cleaned.iloc[:, 5:19].idxmax(axis=1)
    list1 = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(meal_data_cleaned)):
        list1.append(np.diff(meal_data_cleaned.iloc[:, maximum[i]:tm[i]].iloc[i].tolist()).max())
        second_differential_data.append(
            np.diff(np.diff(meal_data_cleaned.iloc[:, maximum[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(meal_data_cleaned.iloc[i]))
    meal_feature_matrix['2ndDifferential'] = second_differential_data
    meal_feature_matrix['standard_deviation'] = standard_deviation
    return meal_feature_matrix


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
    insulin_data_patient_1 = pd.read_csv('InsulinData.csv', low_memory=False,
                                         usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
    insulin_data_patient_1['Date'] = pd.to_datetime(insulin_data_patient_1['Date'])
    cgm_data_patient_1 = pd.read_csv('CGMData.csv', low_memory=False,
                                     usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgm_data_patient_1['Date'] = pd.to_datetime(cgm_data_patient_1['Date'])

    insulin_data_patient_2 = pd.read_csv('Insulin_patient2.csv', low_memory=False,
                                         usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
    insulin_data_patient_2['Date'] = pd.to_datetime(insulin_data_patient_2['Date'])
    cgm_data_patient_2 = pd.read_csv('CGM_patient2.csv', low_memory=False,
                                     usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgm_data_patient_2['Date'] = pd.to_datetime(cgm_data_patient_2['Date'])

    insulin_data_patient_1["date_time"] = pd.to_datetime(
        insulin_data_patient_1['Date'].astype(str) + ' ' + insulin_data_patient_1['Time'])
    insulin_data_patient_2["date_time"] = pd.to_datetime(
        insulin_data_patient_2['Date'].astype(str) + ' ' + insulin_data_patient_2['Time'])
    cgm_data_patient_1["date_time"] = pd.to_datetime(
        cgm_data_patient_1['Date'].astype(str) + ' ' + cgm_data_patient_1['Time'])
    cgm_data_patient_2["date_time"] = pd.to_datetime(
        cgm_data_patient_2['Date'].astype(str) + ' ' + cgm_data_patient_2['Time'])

    meal_data_patient_1 = create_meal_data(insulin_data_patient_1, cgm_data_patient_1)
    meal_data_patient_2 = create_meal_data(insulin_data_patient_2, cgm_data_patient_2)
    meal_data_patient_1 = meal_data_patient_1.iloc[:, 0:30]
    meal_data_patient_2 = meal_data_patient_2.iloc[:, 0:30]

    meal_feature_matrix_1 = get_meal_data_feature_matrix(meal_data_patient_1)
    meal_feature_matrix_2 = get_meal_data_feature_matrix(meal_data_patient_2)
    meal_feature_matrix = pd.concat([meal_feature_matrix_1, meal_feature_matrix_2]).reset_index().drop(columns='index')
    meal_feature_matrix['label'] = 1

    no_meal_data_patient_1 = create_no_meal_data(insulin_data_patient_1, cgm_data_patient_1)
    no_meal_data_patient_2 = create_no_meal_data(insulin_data_patient_2, cgm_data_patient_2)
    no_meal_feature_matrix_1 = get_no_meal_data_feature_matrix(no_meal_data_patient_1)
    no_meal_feature_matrix_2 = get_no_meal_data_feature_matrix(no_meal_data_patient_2)
    no_meal_feature_matrix = pd.concat([no_meal_feature_matrix_1, no_meal_feature_matrix_2]).reset_index().drop(
        columns='index')
    no_meal_feature_matrix['label'] = 0

    meal_and_no_meal_data = pd.concat([meal_feature_matrix, no_meal_feature_matrix]).reset_index().drop(columns='index')
    shuffled_data = shuffle(meal_and_no_meal_data, random_state=1).reset_index().drop(columns='index')
    shuffled_data_without_label = shuffled_data.drop(columns='label')

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    kfold = KFold(n_splits=10)
    model = RandomForestClassifier(n_estimators=50)

    results = cross_validate(estimator=model,
                             X=shuffled_data_without_label,
                             y=shuffled_data['label'],
                             cv=kfold,
                             scoring=scoring)

    print("Mean Accuracy = {}".format(np.mean(results['test_accuracy'])))
    print("Mean Precision = {}".format(np.mean(results['test_precision'])))
    print("Mean Recall = {}".format(np.mean(results['test_recall'])))
    print("Mean F1 score = {}".format(np.mean(results['test_f1_score'])))

    model = RandomForestClassifier(n_estimators=50)
    X, y = shuffled_data_without_label, shuffled_data['label']
    model.fit(X, y)
    dump(model, 'RandomForestClassifier.pickle')


if __name__ == '__main__':
    get_results()
