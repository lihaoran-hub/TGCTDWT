# encoding utf-8

import torch
from utils.utils import RMSE, MAE, MAPE
from utils.utils import Un_Z_Score
import numpy as np
import pywt
device=torch.device("cuda:0")

def Cal_eval_index(pred,pred2, loss_meathod, val_target,CH_val_target, time_slice, mean, std,XH_mean,XH_std,epoch):
    val_index = {}
    val_index['MAE'] = []
    val_index['RMSE'] = []
    val_index['MAPE']=[]
    val_loss = []
    pred=torch.squeeze(pred,dim=-1)
    pred2=torch.squeeze(pred2,dim=-1)
    CV=torch.zeros(pred2.shape[1],pred2.shape[0],pred2.shape[2]).to(device=device)
    CD=torch.zeros(pred2.shape[1],pred2.shape[0],pred2.shape[2]).to(device)

    pred_index = pred#[:, :, item - 1]
    val_target_index = val_target#[:, :, item - 1]

    pred2_index=pred2#[:, :, item - 1]
    CH_val_target_index=CH_val_target#[:, :, item - 1]



    pred_index, val_target_index = Un_Z_Score(pred_index, mean, std), Un_Z_Score(val_target_index, mean, std)

    pred2_index, CH_val_target_index = Un_Z_Score(pred2_index, XH_mean, XH_std), Un_Z_Score(CH_val_target_index, XH_mean, XH_std)

    CA,(CH1,CD1,CV1)=pywt.dwt2(pred_index.permute(1,0,2).cpu(),'haar')

    pred=pywt.idwt2((CA,(CV.cpu(),pred2_index.permute(1,0,2).cpu(),CD.cpu())),'haar')

    pred=torch.tensor(pred).permute(1,0,2).to(device)

    # pred=pywt.idwt2((pred_index.permute(1,0,2).cpu(),(CV.cpu(),pred2_index.permute(1,0,2).cpu(),CD.cpu())),'haar')
    # val_target=pywt.idwt2((val_target_index.permute(1,0,2).cpu(),(CV.cpu(),CH_val_target_index.permute(1,0,2).cpu(),CD.cpu())),'haar')
    # pred=torch.tensor(pred).permute(1,0,2).to(device)
    # val_target=torch.tensor(val_target).permute(1,0,2).to(device)

    # val_target=torch.unsqueeze(val_target,dim=-1)
    # CH_val_target=torch.unsqueeze(CH_val_target,dim=-1)

    # pred=pywt.idwt2((pred.permute(1,0,2).cpu(),(pred2.permute(1,0,2).cpu(),CV.cpu(),CD.cpu())),'haar')
    # val_target=pywt.idwt2((val_target.permute(1,0,2).cpu(),(CH_val_target.permute(1,0,2).cpu(),CV.cpu(),CD.cpu())),'haar')

    # pred=torch.tensor(pred).permute(1,0,2).to(device)
    # val_target=torch.tensor(val_target).permute(1,0,2).to(device)

    if torch.cuda.is_available():
        mean = torch.tensor(mean).cuda()
        std = torch.tensor(std).cuda()
        XH_mean,XH_std=torch.tensor(XH_mean).cuda(),torch.tensor(XH_std).cuda()

    val=val_target_index
    for item in time_slice:


        pred_index = pred[0:-1, :, item - 1]
        val_target_index = val[:, :, item - 1]

        if epoch>0 and epoch%50==0:
            np.savetxt("./results/pred_result_" + str(epoch) + ".csv", pred_index.cpu().detach().numpy(), delimiter=',')
            np.savetxt("./results/true_result_" + str(epoch) + ".csv", val_target_index.cpu().detach().numpy(), delimiter=',')

        loss = loss_meathod(pred_index, val_target_index)
        val_loss.append(loss)

        mae = MAE(val_target_index, pred_index)
        val_index['MAE'].append(mae)

        rmse = RMSE(val_target_index, pred_index)
        val_index['RMSE'].append(rmse)

        mape=MAPE(val_target_index, pred_index,0)
        val_index['MAPE'].append(mape)

    return val_loss, val_index


def Evaluate(model,model2, loss_meathod,loss_meathod3, W_nodes, time_slice, data_set,epoch):
    model.eval()
    eval_input = data_set['eval_input']
    eval_target = data_set['eval_target']
    eval_asist = data_set["eval_asist"]

    CH_eval_input = data_set['evaluate_input_CH']
    CH_eval_target = data_set['evaluate_target_CH']
    CH_eval_asist = data_set["val_asist_dataset_CH"]

    # eval_weather = data_set['eval_weather']

    if torch.cuda.is_available():
        eval_input = eval_input.cuda()
        eval_target = eval_target.cuda()
        # eval_weather = torch.tensor(eval_weather).cuda()
        eval_asist = eval_asist.cuda()

        CH_eval_input = CH_eval_input.cuda()
        CH_eval_target = CH_eval_target.cuda()
        # eval_weather = torch.tensor(eval_weather).cuda()
        CH_eval_asist = CH_eval_asist.cuda()

    pred = model(W_nodes, eval_input, eval_asist)#, eval_weather
    pred2=model2(W_nodes,CH_eval_input,CH_eval_asist)



    # pred_X=pywt.idwt2((pred,(pred2,CV,CD)),'haar')

    # val_target=pywt.idwt2((eval_target,(CH_eval_target,CV,CD)),'haar')

    eval_loss, eval_index = Cal_eval_index(pred,pred2, loss_meathod, eval_target,CH_eval_target, time_slice, data_set['X_mean'], data_set['X_std'],data_set['XH_mean'],data_set['XH_std'],epoch)
    # CH_eval_loss, CH_eval_index = Cal_eval_index(pred2, loss_meathod3, CH_val_target, time_slice, data_set['XH_mean'], data_set['XH_std'],epoch)

    return eval_loss, eval_index
