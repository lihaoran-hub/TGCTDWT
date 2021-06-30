# encoding utf-8
import torch

from utils.utils import Un_Z_Score


def Train(model,model2, optimizer, optimizer2, loss_meathod, loss_meathod3, W_nodes, data_set, batch_size):
    permutation = torch.randperm(data_set['train_input'].shape[0])
    permutation1 = torch.randperm(data_set['train_input_CH'].shape[0])
    epoch_training_losses = []
    loss_mean = 0.0
    X=[]
    CHX=[]
    Y=[]
    CHY=[]

    for i in range(0, data_set['train_input'].shape[0], batch_size):
        model.train()
        optimizer.zero_grad()
        # optimizer2.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch, V_asist = data_set['train_input'][indices], data_set['train_target'][indices], data_set['train_asist'][indices]
        # CHX_batch,CHY_batch,CHV_asist=data_set['train_input_CH'][indices], data_set['train_target_CH'][indices], data_set['train_asist_dataset_CH'][indices]

        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            # CHX_batch = CHX_batch.cuda()
            V_asist = V_asist.cuda()
            # CHV_asist = CHV_asist.cuda()
            y_batch = y_batch.cuda()
            # CHY_batch = CHY_batch.cuda()
            # weather_batch = torch.tensor(weather_batch).cuda()
            std = torch.tensor(data_set['X_std']).cuda()
            mean = torch.tensor(data_set['X_mean']).cuda()
            # CH_mean = torch.tensor(data_set['XH_mean']).cuda()
            # CH_std = torch.tensor(data_set['XH_std']).cuda()

        else:
            std = torch.tensor(data_set['X_std'])
            mean = torch.tensor(data_set['X_mean'])
            # CH_mean = torch.tensor(data_set['XH_mean']).cuda()
            # CH_std = torch.tensor(data_set['XH_std']).cuda()
        perd = model(W_nodes, X_batch, V_asist)#, weather_batch
        # perd2 = model2(W_nodes,CHX_batch,CHV_asist)

        perd, y_batch = Un_Z_Score(perd, mean, std), Un_Z_Score(y_batch, mean, std)
        # perd2, CHY_batch = Un_Z_Score(perd2, CH_mean, CH_std), Un_Z_Score(CHY_batch, CH_mean, CH_std)
        perd=torch.squeeze(perd,dim=-1)
        # perd2=torch.squeeze(perd2,dim=-1)
        loss = loss_meathod(perd, y_batch)
        # loss2 = loss_meathod3(perd2,CHY_batch)

        loss.backward()
        # loss2.backward()
        optimizer.step()
        # optimizer2.step()
        p1=loss.detach().cpu().numpy()
        # p2=loss2.detach().cpu().numpy()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
        X.append(perd[:,:,0])
        Y.append(y_batch[:,:,0])
        # CHX.append(perd2[:,:,0])
        # CHY.append(CHY_batch[:,:,0])
    for i in range(0, data_set['train_input_CH'].shape[0], batch_size):
        model2.train()
        # optimizer.zero_grad()
        optimizer2.zero_grad()

        indices = permutation1[i:i + batch_size]
        # X_batch, y_batch, V_asist = data_set['train_input'][indices], data_set['train_target'][indices], data_set['train_asist'][indices]
        CHX_batch,CHY_batch,CHV_asist=data_set['train_input_CH'][indices], data_set['train_target_CH'][indices], data_set['train_asist_dataset_CH'][indices]

        if torch.cuda.is_available():
            # X_batch = X_batch.cuda()
            CHX_batch = CHX_batch.cuda()
            # V_asist = V_asist.cuda()
            CHV_asist = CHV_asist.cuda()
            # y_batch = y_batch.cuda()
            CHY_batch = CHY_batch.cuda()
            # weather_batch = torch.tensor(weather_batch).cuda()
            # std = torch.tensor(data_set['X_std']).cuda()
            # mean = torch.tensor(data_set['X_mean']).cuda()
            CH_mean = torch.tensor(data_set['XH_mean']).cuda()
            CH_std = torch.tensor(data_set['XH_std']).cuda()

        else:
            # std = torch.tensor(data_set['X_std'])
            # mean = torch.tensor(data_set['X_mean'])
            CH_mean = torch.tensor(data_set['XH_mean']).cuda()
            CH_std = torch.tensor(data_set['XH_std']).cuda()
        # perd = model(W_nodes, X_batch, V_asist)#, weather_batch
        perd2 = model2(W_nodes,CHX_batch,CHV_asist)

        # perd, y_batch = Un_Z_Score(perd, mean, std), Un_Z_Score(y_batch, mean, std)
        perd2, CHY_batch = Un_Z_Score(perd2, CH_mean, CH_std), Un_Z_Score(CHY_batch, CH_mean, CH_std)
        # perd=torch.squeeze(perd,dim=-1)
        perd2=torch.squeeze(perd2,dim=-1)
        # loss = loss_meathod(perd, y_batch)
        loss2 = loss_meathod3(perd2,CHY_batch)

        # loss.backward()
        loss2.backward()
        # optimizer.step()
        optimizer2.step()
        # p1=loss.detach().cpu().numpy()
        p2=loss2.detach().cpu().numpy()
        epoch_training_losses.append(loss2.detach().cpu().numpy())
        loss_mean2 = sum(epoch_training_losses)/len(epoch_training_losses)
        # X.append(perd[:,:,0])
        # Y.append(y_batch[:,:,0])
        CHX.append(perd2[:,:,0])
        CHY.append(CHY_batch[:,:,0])

    X=torch.cat(X,dim=0)
    Y=torch.cat(Y,dim=0)

    CHX=torch.cat(CHX,dim=0)
    CHY=torch.cat(CHY,dim=0)

    return loss_mean,loss_mean2,X,Y,CHX,CHY
