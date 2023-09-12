import pandas as pd
import numpy as np
from datetime import timedelta
from numpy import diff
from scipy.stats import iqr
from scipy import signal
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder


def get_meal_data_and_carb_data(insulin_data_original, cgm_data_original):
    insulin_data = insulin_data_original.copy()
    insulin_data = insulin_data.set_index('date_time')
    insulin_data = insulin_data.sort_values(by='date_time', ascending=True).dropna().reset_index()
    insulin_data['BWZ Carb Input (grams)'].replace(0.0, np.nan, inplace=True)
    insulin_data = insulin_data.dropna()
    insulin_data = insulin_data.reset_index().drop(columns='index')
    timestamps = insulin_data['date_time'].values
    timestamps_with_2_hrs_diff = []
    carb_inputs = []

    for i in range(len(timestamps) - 1):
        curr_timestamp = insulin_data['date_time'][i]
        next_timestamp = insulin_data['date_time'][i + 1]
        diff_with_next_timestamp = (next_timestamp - curr_timestamp).seconds
        if diff_with_next_timestamp >= 7200:
            timestamps_with_2_hrs_diff.append(curr_timestamp)
            carb_inputs.append(insulin_data['BWZ Carb Input (grams)'][i])

    timestamps_with_2_hrs_diff.append(insulin_data['date_time'][len(timestamps) - 1])
    carb_inputs.append(insulin_data['BWZ Carb Input (grams)'][len(timestamps) - 1])

    meal_data = []
    for idx, timestamp in enumerate(timestamps_with_2_hrs_diff):
        start = pd.to_datetime(timestamp - timedelta(seconds=1800))
        end = pd.to_datetime(timestamp + timedelta(seconds=7199))
        curr_date = timestamp.date()

        meal_data.append(
            cgm_data_original[cgm_data_original['Date'].dt.date == curr_date]
            .set_index('date_time')
            .between_time(start_time=start.time(), end_time=end.time())['Sensor Glucose (mg/dL)']
            .values
            .tolist()
        )

    meal_data_df = pd.DataFrame(meal_data)
    #     print(meal_data_df.info())
    #     print(meal_data_df.head())

    meal_data_df[31] = np.array(carb_inputs)

    index = meal_data_df.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    meal_data_cleaned = meal_data_df.drop(meal_data_df.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned = meal_data_cleaned.interpolate(method='linear', axis=1)
    #     index_to_drop_again = meal_data_cleaned.isna().sum(axis=1).replace(0, np.nan).dropna().index
    #     meal_data_cleaned = meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    meal_data_cleaned = meal_data_cleaned.replace(0, np.nan).dropna().reset_index().drop(columns='index')

    return meal_data_cleaned.iloc[:, 0:31], meal_data_cleaned.iloc[:, [31]]


def get_velocities(single_meal_data):
    dx = -5  # 300 seconds in 5 mins (which is the duration after which we get a reading)
    y = single_meal_data.tolist()
    dydx = diff(y, 1) / dx

    mini = min(dydx)
    maxi = max(dydx)
    mean = np.mean(dydx)

    return [mini, maxi, mean]


def get_accelerations(single_meal_data):
    dx = -5  # 300 seconds in 5 mins (which is the duration after which we get a reading)
    y = single_meal_data.tolist()
    d2ydx = diff(y, 2) / dx

    mini = min(d2ydx)
    maxi = max(d2ydx)
    mean = np.mean(d2ydx)

    return [mini, maxi, mean]


def get_entropy(single_meal_data):
    #     arr = np.array(single_meal_data)
    ent = entropy(single_meal_data, base=2)

    return ent


def get_iqr(single_meal_data):
    iq_rng = iqr(single_meal_data)

    return iq_rng


def get_fft(single_meal_data):
    fft_res = np.fft.fft(single_meal_data)

    #     freq = np.fft.fftfreq(len(single_meal_data))
    res = sorted(np.abs(fft_res), reverse=True)
    return res[1:7]


def get_psd(data):
    f, pxx = signal.periodogram(data)

    psd_1 = np.mean(pxx[0:6])
    psd_2 = np.mean(pxx[5:11])
    psd_3 = np.mean(pxx[10:])

    return [psd_1, psd_2, psd_3]


def get_meal_data_feature_matrix(meal_data_df):
    meal_feature_matrix = []
    for index, single_meal in meal_data_df.iterrows():
        single_meal_features = []
        single_meal_features.extend(get_velocities(single_meal))
        single_meal_features.extend(get_accelerations(single_meal))
        single_meal_features.append(get_entropy(single_meal))
        single_meal_features.append(get_iqr(single_meal))
        single_meal_features.extend(get_fft(single_meal))
        single_meal_features.extend(get_psd(single_meal))

        meal_feature_matrix.append(single_meal_features)
    return pd.DataFrame(meal_feature_matrix)


def get_kmeans_results(meal_feature_matrix_patient_1_normalized, carb_input_bins, number_of_bins):
    kmeans = KMeans(init="random", n_clusters=number_of_bins, n_init=10, max_iter=300, random_state=42)
    kmeans.fit_predict(meal_feature_matrix_patient_1_normalized)
    kmeans_labels = list(kmeans.labels_)

    sse_kmeans = kmeans.inertia_

    # Calculate the frequency of each label
    counts = np.bincount(kmeans_labels)
    # Normalize the frequency to get the probabilities
    probs = counts / len(kmeans_labels)
    entropy_kmeans = entropy(probs)

    cont_matrix = metrics.cluster.contingency_matrix(carb_input_bins, kmeans_labels)
    purity_kmeans = np.sum(np.max(cont_matrix, axis=0)) / np.sum(cont_matrix)

    print("Bin Cluster Matrix K-Means: \n", cont_matrix)

    return sse_kmeans, entropy_kmeans, purity_kmeans


def get_dbscan_results(meal_feature_matrix_patient_1, carb_input_bins, number_of_bins):
    X = meal_feature_matrix_patient_1.values
    dbscan_labels = DBSCAN(eps=1000, min_samples=4).fit_predict(X)

    # count the number of clusters formed by DBSCAN
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    le = LabelEncoder()
    n_clusters = number_of_bins

    # while the number of clusters formed by DBSCAN is less than number_of_bins,
    # perform KMeans clustering on the cluster with the highest SSE
    while n_dbscan_clusters < number_of_bins:

        sse = []
        for i in range(n_dbscan_clusters):
            cluster_points = X[dbscan_labels == i]
            cluster_sse = np.sum((cluster_points - np.mean(cluster_points, axis=0)) ** 2)
            sse.append(cluster_sse)

        largest_cluster_label = np.argmax(sse)
        largest_cluster_mask = dbscan_labels == largest_cluster_label
        largest_cluster_data = X[largest_cluster_mask]
        kmeans = KMeans(n_clusters=2)
        kmeans_labels = kmeans.fit_predict(largest_cluster_data)
        # merge the KMeans labels with the DBSCAN labels to form the final cluster labels
        kmeans_labels[kmeans_labels != -1] += n_dbscan_clusters
        dbscan_labels[largest_cluster_mask] = kmeans_labels

        pos_indices = np.where(dbscan_labels >= 0)[0]
        pos_integers = dbscan_labels[pos_indices]

        scaled_pos_integers = le.fit_transform(pos_integers)
        dbscan_labels[pos_indices] = scaled_pos_integers

        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    cont_matrix = metrics.cluster.contingency_matrix(carb_input_bins, dbscan_labels)
    print("Bin Cluster Matrix DBSCAN (with -1 data): \n", cont_matrix)

    # To Remove the column of data for -1
    cont_matrix = cont_matrix[:, 1:]
    print("Bin Cluster Matrix DBSCAN (without -1 data): \n", cont_matrix)

    # Calculate the SSE of each final cluster
    sse_dbscan_arr = []
    for i in range(n_dbscan_clusters):
        cluster_points = X[dbscan_labels == i]
        cluster_sse = np.sum((cluster_points - np.mean(cluster_points, axis=0)) ** 2)
        sse_dbscan_arr.append(cluster_sse)
    sse_dbscan = np.sum(sse_dbscan_arr)

    # Compute entropy
    entropy_dbscan = 0
    for i in range(cont_matrix.shape[0]):
        pi = np.sum(cont_matrix[i]) / np.sum(cont_matrix)
        if pi > 0:
            entropy_dbscan -= pi * np.log2(pi)

    purity_dbscan = np.sum(np.max(cont_matrix, axis=0)) / np.sum(cont_matrix)

    return sse_dbscan, entropy_dbscan, purity_dbscan


def get_results():
    insulin_data_patient_1 = pd.read_csv('InsulinData.csv', low_memory=False,
                                         usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
    insulin_data_patient_1['Date'] = pd.to_datetime(insulin_data_patient_1['Date'])
    insulin_data_patient_1["date_time"] = pd.to_datetime(
        insulin_data_patient_1['Date'].astype(str) + ' ' + insulin_data_patient_1['Time'])

    cgm_data_patient_1 = pd.read_csv('CGMData.csv', low_memory=False,
                                     usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgm_data_patient_1['Date'] = pd.to_datetime(cgm_data_patient_1['Date'])
    cgm_data_patient_1["date_time"] = pd.to_datetime(
        cgm_data_patient_1['Date'].astype(str) + ' ' + cgm_data_patient_1['Time'])

    meal_data_patient_1, carb_inputs_df = get_meal_data_and_carb_data(insulin_data_patient_1, cgm_data_patient_1)

    carb_inputs = carb_inputs_df[31].values.tolist()
    carb_input_bins = carb_inputs.copy()

    for i in range(len(carb_input_bins)):
        carb_input_bins[i] = math.floor((carb_input_bins[i] - 1) / 20)

    number_of_bins = max(carb_input_bins) - min(carb_input_bins) + 1

    print("Ground Truth: \n", carb_input_bins)

    meal_feature_matrix_patient_1 = get_meal_data_feature_matrix(meal_data_patient_1)
    scaler = StandardScaler().fit(meal_feature_matrix_patient_1)
    meal_feature_matrix_patient_1_normalized = scaler.transform(meal_feature_matrix_patient_1)

    sse_kmeans, entropy_kmeans, purity_kmeans = get_kmeans_results(
        meal_feature_matrix_patient_1_normalized, carb_input_bins, number_of_bins)

    sse_dbscan, entropy_dbscan, purity_dbscan = get_dbscan_results(
        meal_feature_matrix_patient_1, carb_input_bins, number_of_bins)

    final_output = pd.DataFrame(
        [
            [
                sse_kmeans,
                sse_dbscan,
                entropy_kmeans,
                entropy_dbscan,
                purity_kmeans,
                purity_dbscan,
            ]
        ],
        columns=[
            "SSE for KMeans",
            "SSE for DBSCAN",
            "Entropy for KMeans",
            "Entropy for DBSCAN",
            "Purity for KMeans",
            "Purity for DBSCAN",
        ],
    )

    final_output.to_csv("Results.csv", index=False, header=None)


if __name__ == '__main__':
    get_results()
