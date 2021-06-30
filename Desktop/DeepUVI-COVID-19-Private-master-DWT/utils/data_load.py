# encoding utf-8
import numpy as np
import pandas as pd
from utils.utils import Z_Score
from utils.utils import generate_dataset
from utils.utils import generate_asist_dataset
import pywt


def Data_load(config, timesteps_input, timesteps_output):
    W_nodes = pd.read_csv(config['W_nodes'], header=None).to_numpy(np.float32)
    # W_nodes=W_nodes[:,0:-1]
    X = pd.read_csv(config['V_nodes'], header=None).to_numpy(np.float32)#*100
    V_confirmed = pd.read_csv(config['V_confirmed'], header=None).to_numpy(np.float32)
    V_cured = pd.read_csv(config['V_cured'], header=None).to_numpy(np.float32)
    V_suspected = pd.read_csv(config['V_suspected'], header=None).to_numpy(np.float32)
    V_dead = pd.read_csv(config['V_dead'], header=None).to_numpy(np.float32)
    # W_nodes = np.load(config['W_nodes']).astype(np.float32)
    # X = np.load(config['V_nodes']).astype(np.float32)
    # Weather = np.load(config['Weather']).astype(np.float32)
    # Weather=Weather[0:96]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    cAX1,(cHX,cDX,cVX)=pywt.dwt2(X,'haar')


    cAX, X_mean, X_std = Z_Score(X)

    cHX,XH_mean,XH_std= Z_Score(cDX)

    V_confirmed = np.reshape(V_confirmed, (V_confirmed.shape[0], V_confirmed.shape[1], 1)).transpose((1, 2, 0))
    V_confirmed, _, _ = Z_Score(V_confirmed)
    V_cured = np.reshape(V_cured, (V_cured.shape[0], V_cured.shape[1], 1)).transpose((1, 2, 0))
    V_cured, _, _ = Z_Score(V_cured)
    V_suspected = np.reshape(V_suspected, (V_suspected.shape[0], V_suspected.shape[1], 1)).transpose((1, 2, 0))
    V_suspected, _, _ = Z_Score(V_suspected)
    V_dead = np.reshape(V_dead, (V_dead.shape[0], V_dead.shape[1], 1)).transpose((1, 2, 0))
    V_dead, _, _ = Z_Score(V_dead)
    V_conbine = np.concatenate((V_confirmed, V_cured, V_suspected, V_dead), axis=1)

    # V_conbine,(CH_V,CD_V,CV_V)=pywt.dwt2(V_conbine,'haar')
    # Weather, _, _ = Z_Score(Weather)

    index_1 = int(cAX.shape[2] * 0.8)
    index_2 = int(cAX.shape[2])

    indexH_1=int(cHX.shape[2]*0.8)
    indexH_2 = int(cHX.shape[2])

    train_original_data = cAX#[:, :, :index_1]
    train_asist = V_conbine#[:, :, :index_1]

    train_original_data_CH=cHX
    train_asist_CH=V_conbine


    val_original_data = cAX[:, :, index_1:index_2]
    val_asist = V_conbine[:, :, index_1:index_2]

    val_original_data_CH = cHX[:, :, indexH_1:indexH_2]
    val_asist_CH = V_conbine[:, :, indexH_1:indexH_2]


    # index_1 = int(Weather.shape[0] * 0.8)
    # index_2 = int(Weather.shape[0])
    # train_weather = Weather[:index_1]
    # val_weather = Weather[index_1:index_2]

    train_input, train_target = generate_dataset(train_original_data,
                                                 num_timesteps_input=timesteps_input,
                                                 num_timesteps_output=timesteps_output,
                                                 step=288)
    train_input_CH, train_target_CH = generate_dataset(train_original_data_CH,
                                                 num_timesteps_input=int(timesteps_input/2),
                                                 num_timesteps_output=int(timesteps_output/2),
                                                       step=144)


    evaluate_input, evaluate_target = generate_dataset(val_original_data,
                                                       num_timesteps_input=timesteps_input,
                                                       num_timesteps_output=timesteps_output,
                                                       step=288)
    evaluate_input_CH, evaluate_target_CH = generate_dataset(val_original_data_CH,
                                                       num_timesteps_input=int(timesteps_input/2),
                                                       num_timesteps_output=int(timesteps_output/2),
                                                             step=144)

    train_asist_dataset= generate_asist_dataset(train_asist, timesteps_input, timesteps_output,step=288)
    train_asist_dataset_CH= generate_asist_dataset(train_asist_CH, int(timesteps_input/2), int(timesteps_output/2),step=144)

    val_asist_dataset= generate_asist_dataset(val_asist, timesteps_input, timesteps_output,step=288)
    val_asist_dataset_CH= generate_asist_dataset(val_asist_CH, int(timesteps_input/2), int(timesteps_output/2),step=144)

    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set[
        'eval_target'], data_set["train_asist"], data_set["eval_asist"], data_set['X_mean'], data_set['X_std'],data_set['train_input_CH'],data_set[
        'train_target_CH'], data_set['evaluate_input_CH'],data_set['evaluate_target_CH'],data_set['train_asist_dataset_CH'],data_set[
        'val_asist_dataset_CH'],data_set['XH_mean'],data_set['XH_std']\
        = train_input, train_target, evaluate_input, evaluate_target, train_asist_dataset, val_asist_dataset, X_mean, X_std,train_input_CH,train_target_CH,evaluate_input_CH,evaluate_target_CH,train_asist_dataset_CH,val_asist_dataset_CH,XH_mean,XH_std#train_weather, val_weather,
#data_set['train_weather'], data_set['eval_weather'],
    return W_nodes, data_set

