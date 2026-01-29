import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize


def norm(train, test):
    scaler = StandardScaler()
    # scaler.fit(np.concatenate((train, test), axis=0))
    scaler.fit(train)
    train_data = scaler.transform(train)
    test_data = scaler.transform(test)
    return train_data, test_data

def swat_preprocessing():
    # the following code is adapted from the source code in [Zhihan Li et al. KDD21]
    # preprocess for SWaT. SWaT.A2_Dec2015, version 0
    dataset_folder = os.path.join('../data/', 'swat')

    test_df = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.xlsx'), header=1)

    test_df = test_df.set_index(' Timestamp')
    test_df['label'] = np.where(test_df['Normal/Attack'] == 'Attack', 1, 0)
    # test_df.apply(lambda x: 1 if test_df['Normal/Attack'] == 'Attack' else 0)
    test_df = test_df.drop('Normal/Attack', axis=1)
    assert test_df.shape == (449919, 52)

    train_df = pd.read_excel(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v0.xlsx'), header=1)
    # train_df = train_df.drop(columns=['Unnamed: 0', 'Unnamed: 52'])
    train_df = train_df.set_index(' Timestamp')
    train_df['label'] = np.where(train_df['Normal/Attack'] == 'Attack', 1, 0)
    train_df = train_df.drop('Normal/Attack', axis=1)

    # following [Zhihan Li et al. KDD21] & [Dan Li. ICANN. 2019]
    # fow SWaT data, due to the cold start of the system, starting point is 21600
    train_df = train_df.iloc[21600:]
    assert train_df.shape == (475200, 52)

    output_dir = '../data/swat/'
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'SWaT_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'SWaT_test.csv'))


def wadi_preprocessing():
    # preprocess for WADI. WADI.A2_19Nov2019
    dataset_folder = os.path.join('../data/', 'wadi')

    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days_new.csv'), index_col=0, header=0)
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdataLABLE.csv'), index_col=0, header=1)

    train_df = train_df.iloc[:, 2:]
    test_df = test_df.iloc[:, 2:]

    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    test_df = test_df.fillna(test_df.mean(numeric_only=True))
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # trim column names
    train_df = train_df.rename(columns=lambda x: x.strip())
    test_df = test_df.rename(columns=lambda x: x.strip())

    train_df['label'] = np.zeros(len(train_df))
    test_df['label'] = np.where(test_df['Attack LABLE (1:No Attack, -1:Attack)'] == -1, 1, 0)
    test_df = test_df.drop(columns=['Attack LABLE (1:No Attack, -1:Attack)'])

    output_dir = '../data/wadi/'
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'WADI_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'WADI_test.csv'))

def swat():
    dataset_folder = os.path.join('data/', 'swat')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_train.csv'))
    train_df = np.array(train_df.set_index(' Timestamp'))
    train_data = train_df[:, :51]
    test_df = pd.read_csv(os.path.join(dataset_folder, 'SWaT_test.csv'))
    test_df = np.array(test_df.set_index(' Timestamp'))
    test_data = test_df[:, :51]

    # scaler = StandardScaler()
    # scaler.fit(np.concatenate((train_data, test_data), axis=0))
    # train_data = scaler.transform(train_data)
    # test_data = scaler.transform(test_data)
    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 51]
    test_labels = test_df[:, 51]

    return train_data, test_data, train_labels, test_labels


def wadi():
    dataset_folder = os.path.join('data/', 'wadi')
    train_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_train.csv'))
    train_df = np.array(train_df.set_index('Row'))
    train_data = train_df[:, :127]
    test_df = pd.read_csv(os.path.join(dataset_folder, 'WADI_test.csv'))
    # print(test_df.columns)
    test_df = np.array(test_df.set_index('Row '))
    test_data = test_df[:, :127]

    train_data, test_data = norm(train_data, test_data)

    train_labels = train_df[:, 127]
    test_labels = test_df[:, 127]

    return train_data, test_data, train_labels, test_labels


# Extract datasets processed by salesforce-merlion such as IOpsCompetition, UCR, NAB, SWaT, WADI, SMAP, etc.
def other_datasets(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)
    # salesforce-merlion==1.1.1
    bias, scale = mvn.bias, mvn.scale

    # salesforce-merlion==1.3.0
    # bias, scale = mvn.bias[0], mvn.scale[0]

    # salesforce-merlion==2.0.0
    # bias, scale = mvn.bias, mvn.scale
    # bias, scale = list(bias.values())[0], list(scale.values())[0]





    # # original (simpler) normalization lines preserved below for auditability.
    # # We replace them with more robust handling because `mvn.bias`/`mvn.scale` can be dicts
    # # or Series depending on merlion version which caused a TypeError in practice.
    # # Keeping the original lines commented prevents accidental loss and documents the change.
    # # Original lines:
    # # bring the original back for WADI
    # # make the modification again for other datasets, so comment the next 4 lines and uncomment the blocks after that.
    # train_time_series = train_time_series_ts.to_pd().to_numpy()
    # train_data = (train_time_series - bias) / scale
    # test_time_series = test_time_series_ts.to_pd().to_numpy()
    # test_data = (test_time_series - bias) / scale



    # convert TimeSeries to pandas for reliable column access
    train_df = train_time_series_ts.to_pd()
    test_df = test_time_series_ts.to_pd()
    train_time_series = train_df.to_numpy()
    test_time_series = test_df.to_numpy()

    # mvn.bias / mvn.scale can have different types across merlion versions
    # - sometimes scalars/arrays, sometimes dicts mapping column names -> values
    # Normalize them into numpy arrays aligned with train_df.columns when possible.
    def _to_array(x, columns):
        # x may be dict, pandas Series, numpy array, or scalar
        if isinstance(x, dict):
            # if keys match columns, extract in that order
            try:
                if all(col in x for col in columns):
                    return np.asarray([x[col] for col in columns], dtype=float)
            except Exception:
                pass
            # otherwise try to take the first value (common for some merlion versions)
            vals = list(x.values())
            return np.asarray(vals[0], dtype=float)
        if hasattr(x, 'to_numpy'):
            try:
                return np.asarray(x.to_numpy(), dtype=float)
            except Exception:
                return np.asarray(x, dtype=float)
        return np.asarray(x, dtype=float)

    bias_arr = _to_array(bias, train_df.columns)
    scale_arr = _to_array(scale, train_df.columns)

    # Broadcast bias/scale to (1, n_features) if needed
    if bias_arr.ndim == 0:
        train_data = (train_time_series - float(bias_arr)) / float(scale_arr)
        test_data = (test_time_series - float(bias_arr)) / float(scale_arr)
    else:
        # ensure shapes match number of features
        n_feats = train_time_series.shape[1]
        if bias_arr.shape[0] != n_feats:
            # try to flatten/truncate or repeat as needed
            bias_arr = bias_arr.flatten()
            if bias_arr.shape[0] > n_feats:
                bias_arr = bias_arr[:n_feats]
            elif bias_arr.shape[0] < n_feats:
                # repeat last value to match
                bias_arr = np.pad(bias_arr, (0, n_feats - bias_arr.shape[0]), mode='edge')
        if scale_arr.shape[0] != n_feats:
            scale_arr = scale_arr.flatten()
            if scale_arr.shape[0] > n_feats:
                scale_arr = scale_arr[:n_feats]
            elif scale_arr.shape[0] < n_feats:
                scale_arr = np.pad(scale_arr, (0, n_feats - scale_arr.shape[0]), mode='edge')

        train_data = (train_time_series - bias_arr.reshape(1, -1)) / scale_arr.reshape(1, -1)
        test_data = (test_time_series - bias_arr.reshape(1, -1)) / scale_arr.reshape(1, -1)

    # end of modification block




    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


def other_datasets_no_Normalize(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])

    train_data = train_time_series_ts.to_pd().to_numpy()
    test_data = test_time_series_ts.to_pd().to_numpy()

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels

if __name__ == "__main__":
    # swat_preprocessing()
    wadi_preprocessing()
