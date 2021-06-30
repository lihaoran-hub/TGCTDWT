import numpy as np
import torch


def Z_Score(matrix):
    mean, std = np.mean(matrix), np.std(matrix)
    return (matrix - mean) / (std+0.001), mean, std


def Un_Z_Score(matrix, mean, std):
    return (matrix * std) + mean


def get_normalized_adj(W_nodes):
    W_nodes = W_nodes + np.diag(np.ones(W_nodes.shape[0], dtype=np.float32))
    D = np.array(np.sum(W_nodes, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    W_nodes = np.multiply(np.multiply(diag.reshape((-1, 1)), W_nodes),
                         diag.reshape((1, -1)))
    return torch.from_numpy(W_nodes)


def generate_asist_dataset(X, num_timesteps_input, num_timesteps_output,step):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1-step)]
    all_feature=[]
    features,features2, target = [], [], []
    for i, j in indices:
        a=[X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)),X[:, :, i+step: i + num_timesteps_input+step].transpose(
                (0, 2, 1))]
        features.append(a)
        # features2.append(
        #     X[:, :, i+288: i + num_timesteps_input+288].transpose(
        #         (0, 2, 1))
        # )

        target.append(X[:, 0, i + num_timesteps_input+step: j+step])#target.append(X[:, 0, i + num_timesteps_input+288: j+288])
    # all_feature.append([features,features2])

    return torch.from_numpy(np.array(features))#,torch.from_numpy(np.array(features2))


def generate_dataset(X, num_timesteps_input, num_timesteps_output,step):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1-step)]
    all_feature=[]
    features, features2, target = [], [], []
    for i, j in indices:
        a=[X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)),X[:, :, i+step: i + num_timesteps_input+step].transpose(
                (0, 2, 1))]
        features.append(a)
        # features2.append(
        #     X[:, :, i+288: i + num_timesteps_input+288].transpose(
        #         (0, 2, 1))
        # )

        # target.append(X[:, 0, i + num_timesteps_input+288: j+288])
        target.append(X[:, 0, i + num_timesteps_input +step: j+step])
    # all_feature=[features,features2]
    a=indices[-1]
    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


def RMSE(v, v_):
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))

def MAPE(y_true, y_pred, null_val=np.nan):
    # y_true=y_true
    # y_pred=y_pred
    y_pred=y_pred.cpu().numpy()
    y_true=y_true.cpu().numpy()
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return np.mean(mape)

'''
    y_pred=y_pred.cpu().numpy()
    y_true=y_true.cpu().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),#数组对应位置元素做除法np.divide,np.subtract两个数组相减
                      y_true))
        mape = np.nan_to_num(mask * mape)#np.nan_to_num使用0代替数组x中的nan元素，使用有限的数字代替inf元素
        return np.mean(mape) * 100
'''
