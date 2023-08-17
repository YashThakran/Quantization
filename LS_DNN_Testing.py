import pickle
import numpy as np
import scipy.io
from keras.models import load_model
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

snr = np.arange(1, 8)
d = np.arange(1, 8)
for t_snr in d:
    # this dataset path is chosen same as that of training to get the mean and std of the training set
    dataset_path = 'D:/project_new/lsdnn/qpsk/Dataset_{}.mat'.format(t_snr)
    mat = h5py.File(dataset_path, 'r')
    X = np.array(mat['Preamble_Error_Correction_Dataset']['X'])
    Y = np.array(mat['Preamble_Error_Correction_Dataset']['Y'])
    # Normalizing Datasets
    scalerx = StandardScaler()
    scalerx.fit(X)
    scalery = StandardScaler()
    scalery.fit(Y)

    a = np.zeros(104)
    b = np.zeros(104)
    for i in range(scalerx.mean_.shape[0]):
        a[i] = scalerx.mean_[i]
    np.savetxt('D:/project_new/lsdnn/qpsk/DNN{}/mean.dat'.format(t_snr), a, fmt='%f', newline=',')
    for i in range(scalerx.mean_.shape[0]):
        b[i] = math.sqrt(scalerx.var_[i])
    np.savetxt('D:/project_new/lsdnn/qpsk/DNN{}/std.dat'.format(t_snr), b, fmt='%f', newline=',')
    for i in range(scalery.mean_.shape[0]):
        a[i] = scalery.mean_[i]
    np.savetxt('D:/project_new/lsdnn/qpsk/DNN{}/mean_o.dat'.format(t_snr), a, fmt='%f', newline=',')
    for i in range(scalery.mean_.shape[0]):
        b[i] = math.sqrt(scalery.var_[i])
    np.savetxt('D:/project_new/lsdnn/qpsk/DNN{}/std_o.dat'.format(t_snr), b, fmt='%f', newline=',')

    for j in snr:
        dataset_path = 'D:/project_new/lsdnn/qpsk/Dataset_{}.mat'.format(j)
        mat = h5py.File(dataset_path, 'r')
        X = np.array(mat['Preamble_Error_Correction_Dataset']['X'])
        Y = np.array(mat['Preamble_Error_Correction_Dataset']['Y'])
        print('Loaded Dataset Inputs: ', X.shape)
        print('Loaded Dataset Outputs: ', Y.shape)
        # Normalizing Datasets
        #scalerx = StandardScaler()
        #scalerx.fit(X)
        #scalery = StandardScaler()
        #scalery.fit(Y)
        XS = scalerx.transform(X)
        YS = scalery.transform(Y)

        # Split Data into train and test sets
        seed = 7
        train_X, test_X, train_Y, test_Y = train_test_split(XS, YS, test_size=0.2, random_state=seed)
        print('Testing samples: ', test_X.shape[0])

        model = load_model('D:/project_new/lsdnn/qpsk/LS_DNN_{}.h5'.format(t_snr))

        # Testing the model
        Y_pred = model.predict(test_X)
        Original_Testing_X = scalerx.inverse_transform(test_X)
        Original_Testing_Y = scalery.inverse_transform(test_Y)
        Prediction_Y = scalery.inverse_transform(Y_pred)

        result_path = 'D:/project_new/lsdnn/qpsk/DNN{}/DNN_Results_{}.pickle'.format(t_snr, j)
        with open(result_path, 'wb') as f:
            pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)


    for j in snr:
        source_name = 'D:/project_new/lsdnn/qpsk/DNN{}/DNN_Results_{}.pickle'.format(t_snr, j)
        dest_name = 'D:/project_new/lsdnn/qpsk/DNN{}/DNN_Results_{}.mat'.format(t_snr, j)
        a = pickle.load(open(source_name, "rb"))
        scipy.io.savemat(dest_name, {
            'test_x_{}'.format(j): a[0],
            'test_y_{}'.format(j): a[1],
            'corrected_y_{}'.format(j): a[2]
        })
        print("Data successfully converted to .mat file ")








