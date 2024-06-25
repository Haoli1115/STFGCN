import argparse
import os
import torch.nn as nn
import datetime
import time
import yaml
import sys
import random
import numpy as np
import torch
sys.path.append("..")
from lib.utils import MaskedMAELoss, print_log, CustomJSONEncoder, cheb_polynomial, scaled_Laplacian
from lib.metrics import RMSE_MAE_MAPE
from lib.mul_data_prepare import get_dataloaders_from_index_data
from STFGCN import stfgcn


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, x_backday_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        x_backday_batch = x_backday_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch, x_backday_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())
    return np.mean(batch_loss_list)



@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, x_backday_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        x_backday_batch = x_backday_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch, x_backday_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
        model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for x_batch, x_backday_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        x_backday_batch = x_backday_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch, x_backday_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step(epoch_loss)

    return epoch_loss


def train(model, trainset_loader, valset_loader, optimizer, scheduler, criterion, clip_grad=0, max_epochs=200,
          early_stop=10, verbose=1, plot=True, log=None, save=True,
          ):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log)
        train_loss_list.append(train_loss)
        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch + 1}\n"
    out_str += f"Best at epoch {best_epoch + 1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)
    test_model(model,testset_loader)
    if save:
        torch.save(best_state_dict, save)

    # if plot:
    #     import matplotlib
    #     matplotlib.use('TkAgg')
    #     plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
    #     plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
    #     plt.title("Epoch-Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.show()

    return model


# @torch.no_grad()
def test_model(model, testset_loader, log=None):
    # model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":

    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="PEMS08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed = 42  # 可以使用任何整数作为种子值

    random.seed(seed)  # 设置Python内置随机数生成器的种子
    np.random.seed(seed)  # 设置NumPy的随机数生成器的种子
    torch.manual_seed(seed)  # 设置PyTorch的随机数生成器的种子

    # 如果使用GPU，还需要设置以下种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch的CUDA随机数生成器的种子
    torch.backends.cudnn.deterministic = True  # 设置使用CuDNN时的随机数生成器的种子
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN的自动调优

    DEVICE = torch.device("cuda:2")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = stfgcn.__name__

    with open(f"STFGCN.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model init parameter-------------------------------- #

    model = stfgcn(DEVICE,**cfg["model_args"]).to(DEVICE)




    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print(total_params)
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # ------------------------------- load dataset ------------------------------- #
    filename = "../data/" + str(dataset).upper() + "/data.npz"

    # 修改Val和test位置；使得
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        filename,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        x_back_length=cfg.get("x_back_length"),
    )

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = save_path+"metrla-dk32-K336"+".pt"
    #metrla - dk32 - K335
    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08","SZTAXI"):
        criterion = nn.HuberLoss().to(DEVICE)

    else:
        raise ValueError("Unsupported dataset")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0),
                                 eps=cfg.get("eps", 1e-8), )

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=cfg["milestones"],  # milestones: [ 20,40,55, 65,75 ]
    #                                                  gamma=cfg.get("lr_decay_rate"),
    #                                                  verbose=False,)#lr=lr∗gamma
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.25,
        patience=2,
        threshold=0.005,
        cooldown=3,
        min_lr=0.000001
    )

    # --------------------------- print model structure -------------------------- #

    for k, v in cfg.items():
        print(k, v)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        save=save,
    )
#@