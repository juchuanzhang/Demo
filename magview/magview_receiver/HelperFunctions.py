import pandas as pd
import numpy as np

def extract_features(dataframe, featurenum):
    timeseries = dataframe['amplitude']
    timeseries = timeseries - min(timeseries)
    # print('time series in the function', timeseries)
    total_energy = sum(np.square(timeseries))
    window_len = int(len(timeseries) / featurenum)
    features = pd.DataFrame(index=np.arange(1), columns=np.arange(featurenum))

    for i in range(0, featurenum):
        timeseries_mini = timeseries[window_len * i : window_len * (i + 1)]
        # print('i = ', i, timeseries_mini)
        energy = sum(np.square(timeseries_mini)) / total_energy
        # print('energy=', energy)
        features[i][0] = energy

    return features



if __name__ == "__main__":
    i = 0
    indir = 'split_x395_60p_small\\'
    data_read = pd.read_csv(indir + str(i + 1) + '.csv', header=None)
    data_read.rename(columns={0: 'amplitude'}, inplace=True)
    print(data_read)
    results = extract_features(data_read, 10)
    print(results)
    print('sum of features: ', sum(sum(results.values)))
    # data_read()