import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
# import torch.nn as nn
from torch.nn import Linear, Sigmoid, Module
# from torch.nn import MSELoss
# from torch.nn import L1Loss
from torch.nn import HuberLoss
# from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
# import torch.nn.functional as F
from torch.optim import Adam
# from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import random_split
# import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import os
from datetime import datetime
from random import choices


class CSVDataset(Dataset):
    def __init__(self, path: str, x_filter: list, y_filter: list):
        x, y, scaler_x, scaler_y, df = grab_data(path, x_filter, y_filter)
        scaled_x = scaler_x.transform(x)
        scaled_y = scaler_y.transform(y)

        self.x = np.array(scaled_x)
        self.y = np.array(scaled_y)

        self.x = self.x.astype('float32')
        self.y = self.y.astype('float32')

        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        print('TRAINING ON DATASET OF SIZE :', len(x))

    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx: int):
        return [self.x[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test: float = 0.33):
        # determine sizes
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements

    def __init__(self, n_inputs: int):
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs, 16)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()

        self.hidden2a = Linear(16, 32)
        xavier_uniform_(self.hidden2a.weight)
        self.act2a = Sigmoid()

        self.hidden2 = Linear(32, 64)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()

        self.hidden3 = Linear(64, 128)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

        self.hidden4 = Linear(128, 256)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()

        self.hidden5 = Linear(256, 128)
        xavier_uniform_(self.hidden5.weight)
        self.act5 = Sigmoid()

        self.hidden6 = Linear(128, 64)
        xavier_uniform_(self.hidden6.weight)
        self.act6 = Sigmoid()

        self.hidden7 = Linear(64, 32)
        xavier_uniform_(self.hidden7.weight)
        self.act7 = Sigmoid()

        self.hidden8 = Linear(32, 16)
        xavier_uniform_(self.hidden8.weight)
        self.act8 = Sigmoid()

        self.hidden9 = Linear(16, 8)
        xavier_uniform_(self.hidden9.weight)
        self.act9 = Sigmoid()

        self.hidden10 = Linear(8, 4)
        xavier_uniform_(self.hidden10.weight)
        self.act10 = Sigmoid()

        self.hidden11 = Linear(4, 1)
        xavier_uniform_(self.hidden11.weight)

    # forward propagate input
    def forward(self, x: Module):
        x = self.hidden1(x)
        x = self.act1(x)

        x = self.hidden2a(x)
        x = self.act2a(x)

        x = self.hidden2(x)
        x = self.act2(x)

        x = self.hidden3(x)
        x = self.act3(x)

        # x = self.hidden4(x)
        # x = self.act4(x)

        # x = self.hidden5(x)
        # x = self.act5(x)

        x = self.hidden6(x)
        x = self.act6(x)

        x = self.hidden7(x)
        x = self.act7(x)

        x = self.hidden8(x)
        x = self.act8(x)

        x = self.hidden9(x)
        x = self.act9(x)

        x = self.hidden10(x)
        x = self.act10(x)

        x = self.hidden11(x)
        return x


def unweight(x: np.ndarray):
    weights = [['l', 1], ['b', 1], ['parallax', 2], ['parallax_error', 2], ['percent_amp', 1], ['skew', 1], ['PM_x', 1],
               ['parallax_x', 2], ['parallax_over_error', 2], ['b_rgeo_x', 1], ['B_rgeo_xa', 1], ['rpgeo', 2],
               ['b_rpgeo_x', 1], ['B_rpgeo_xa', 1]]
    for weight in weights:
        weighted = weight[0]
        weight = weight[1]
        x[weighted] = x[weighted] / weight
    return x


def weighting(x: np.ndarray):
    weights = [['l', 2], ['b', 2], ['parallax', 1], ['parallax_error', 1], ['percent_amp', 1], ['skew', 1], ['PM_x', 1],
               ['parallax_x', 2], ['parallax_over_error', 2], ['b_rgeo_x', 1], ['B_rgeo_xa', 1], ['rpgeo', 2],
               ['b_rpgeo_x', 1], ['B_rpgeo_xa', 1]]

    for weight in weights:
        weighted = weight[0]
        weight = weight[1]
        x[weighted] = x[weighted] * weight
    return x


def grab_data(data_fp: str, x_filter: list, y_filter: list, filter_flag: bool = True):
    snr = 2
    df = pd.read_csv(data_fp, low_memory=False)

    if filter_flag:
        df = df.loc[(abs(df['l']) > 0) & (abs(df['b']) > 0) & (abs(df['parallax']) > 0) & (
                abs(df['parallax_over_error']) > 1) & (abs(df['pmra']) > 0) & (
                            abs(df['pmra_x_over_error']) > snr) & (abs(df['pmdec']) > 0) & (
                            abs(df['pmdec_over_error']) > snr) & (abs(df['rgeo']) > 0) & (
                            abs(df['rgeo']) < 9999999) & (abs(df['b_rgeo_x']) > 0) & (abs(df['B_rgeo_xa']) > 0) & (
                            abs(df['rgeo_diff']) > 0) & (abs(df['rpgeo']) > 0) & (abs(df['b_rpgeo_x']) > 0) & (
                            abs(df['B_rpgeo_xa']) > 0) & (abs(df['rpgeo_diff']) > 0) & (abs(df['Plx']) > 0) & (
                            abs(df['Plx_over_error']) > snr) & (abs(df['PM_x']) > 0) & (abs(df['pmRA_xa']) > 0) & (
                            abs(df['pmra_x_over_error']) > snr) & (abs(df['pmDE']) > 0) & (
                            abs(df['pmde_x_over_error']) > snr)]  # & (abs(df['j-h'])  > 0) & (abs(df['h-k'])  > 0)]
    else:
        df = df.loc[(abs(df['parallax']) > 0) & (abs(df['parallax_over_error']) > 0) & (abs(df['pmra']) > 0) & (
                abs(df['rgeo']) > 0)]
    # df = manual_norm(df)
    # df = weighting(df)

    x = df[x_filter].copy()
    y = np.log10(df[y_filter].copy())

    scaler_x = StandardScaler()
    scaler_x.fit(x)

    scaler_y = StandardScaler()
    scaler_y.fit(y)
    return x, y, scaler_x, scaler_y, df


# prepare the dataset
def prepare_data(path: str, x_filter: list, y_filter: list):
    # load the dataset
    dataset = CSVDataset(path, x_filter, y_filter)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=8192, shuffle=True)
    test_dl = DataLoader(test, batch_size=4096, shuffle=False)
    return train_dl, test_dl, len(test[0][0])


# train the model

def train_model(train_dl: DataLoader, test_dl: DataLoader, model: Module,
                epochs: int, learn_rate: float, output_dir: str, gamma: float):
    # define the optimization
    # criterion = MSELoss()
    criterion = HuberLoss()
    # criterion = L1Loss()
    opt = Adam(model.parameters(), lr=learn_rate)
    # opt = SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    scheduler = ExponentialLR(opt, gamma=gamma)

    # enumerate epochs
    min_delta = 0.0001
    tolerance = 1000
    # prev_loss = 100
    counter = 0
    # epochss, losss = [], []
    train_stats = pd.DataFrame(columns=['epoch', 'learn_rate', 'loss', 'delta_loss', 'counter', 'val_loss'])
    i = 0
    mse = np.nan
    for epoch in tqdm(range(epochs), total=epochs, desc='Training'):

        train_features, train_labels = next(iter(train_dl))
        opt.zero_grad()
        yhat = model(train_features)
        loss = criterion(yhat, train_labels)
        loss.backward()
        opt.step()

        # for i, (inputs, targets) in enumerate(train_dl):
        #    opt.zero_grad()
        #    yhat = model(inputs)    
        #    loss = criterion(yhat, targets)
        #    loss.backward()
        #    opt.step()

        test_features, test_labels = next(iter(test_dl))
        prediction_val = model(test_features)
        val_loss = criterion(prediction_val, test_labels)
        loss_delta = val_loss - loss
        my_lr = scheduler.get_last_lr()
        train_stats.loc[i + (epoch * len(train_dl))] = [epoch, float(my_lr[0]), float(loss.detach().numpy()),
                                                        float(loss_delta.detach().numpy()), counter,
                                                        val_loss.detach().numpy()]

        if not epoch % 5:
            scheduler.step()

            if not epoch % 250:
                print('Epoch :', epoch, ' Learning rate :', float(my_lr[0]), 'Val Loss :', float(val_loss), ' Loss :',
                      float(loss), ' Delta Loss :', float(loss_delta), ' ES counter :', counter)

                if not epoch % 1000 and epoch > 1:
                    mse = evaluate_model(test_dl, model)
                    print('MSE: %.3f, RMSE: %.3f' % (mse, np.sqrt(mse)))
                    plots(train_stats, output_dir, mse)

        # early stopping
        if epoch <= 1000:
            continue
        if loss_delta > min_delta:
            counter += 1
        elif counter > 10:
            counter -= 1
            if counter >= tolerance:
                print('Epoch :', epoch, ' Learning rate :', float(my_lr[0]), 'Val Loss :', float(val_loss),
                      ' Loss :', float(loss), ' Delta Loss :', float(loss_delta), ' ES counter :', counter)
                print("STOPPING at epoch:", epoch)
                plots(train_stats, output_dir, mse)
                break
        # prev_loss = loss
    return


def plots(train_stats: pd.DataFrame, output_dir: str, mse: float):
    try:
        plt.clf()
        fig, ax = plt.subplots()
        fig.suptitle('MSE: ' + str(round(mse, 3)) + '  RMSE: ' + str(round(np.sqrt(mse), 3)))
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['loss'], 'k', alpha=0.9, ms=2, label='Loss')
        ax.plot(train_stats['epoch'], train_stats['val_loss'], 'green', alpha=0.9, ms=2, label='Validation Loss')
        ax.set(xlabel='Epoch', ylabel='Loss')
        plt.legend()
        fig.tight_layout()
        plt.savefig(output_dir + '/loss.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        fig.suptitle('MSE: ' + str(round(mse, 3)) + '  RMSE: ' + str(round(np.sqrt(mse), 3)))
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['loss'], 'k', alpha=0.9, ms=2, label='Loss')
        ax.plot(train_stats['epoch'], train_stats['val_loss'], 'green', alpha=0.9, ms=2, label='Validation Loss')
        ax.set(xlabel='Epoch', ylabel='Loss', xscale='log', yscale='log')
        plt.legend()
        fig.tight_layout()
        plt.savefig(output_dir + '/loss_log.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['learn_rate'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='Learn Rate')
        fig.tight_layout()
        plt.savefig(output_dir + '/lr.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['learn_rate'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='Learn Rate', xscale='log', yscale='log')
        fig.tight_layout()
        plt.savefig(output_dir + '/lr_log.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        fig.suptitle('Mean: ' + str(np.round(np.mean(train_stats['counter']), 3)) + ' Median: ' + str(
            np.round(np.median(train_stats['counter']), 3)) + ' std: ' +
                     str(np.round(np.std(train_stats['counter']), 3)))
        ax.plot(train_stats['epoch'], train_stats['counter'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='counter')
        fig.tight_layout()
        plt.savefig(output_dir + '/counter.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        fig.suptitle('Mean: ' + str(np.round(np.mean(train_stats['delta_loss']), 3)) + ' Median: ' + str(
            np.round(np.median(train_stats['delta_loss']), 3)) + ' std: ' + str(
            np.round(np.std(train_stats['delta_loss']), 3)))
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['delta_loss'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='delta loss')
        fig.tight_layout()
        plt.savefig(output_dir + '/delta_loss.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        fig.suptitle('Mean: ' + str(np.round(np.mean(train_stats['delta_loss']), 3)) + ' Median: ' + str(
            np.round(np.median(train_stats['delta_loss']), 3)) + ' std: ' + str(
            np.round(np.std(train_stats['delta_loss']), 3)))
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['delta_loss'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='delta loss', xscale='log', yscale='log')
        fig.tight_layout()
        plt.savefig(output_dir + '/log_delta_loss.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    return


# evaluate the model
def evaluate_model(test_dl: DataLoader, model: Module):
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse


# make a class prediction for one row of data
def predict(row: list, model: Module):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


def predict_dist(model: Module, data_fp: str, data_verif_fp: str, output_dir: str, x_filter: list, y_filter: list):
    # verification set
    x, y, scaler_x, scaler_y, df_verif = grab_data(data_verif_fp, x_filter, y_filter, filter_flag=False)
    scaled_x = scaler_x.transform(x)
    scaled_y = scaler_y.transform(y)

    preds = []
    querypoints = np.linspace(0, len(scaled_x), 4, dtype=int)
    for i, row in tqdm(enumerate(scaled_x), total=len(scaled_x), desc=data_verif_fp.strip('.csv')):
        yhat = predict(row, model)[0]
        preds.append(yhat)
        if i in querypoints:
            print('% : ', 100 * (round(i / len(scaled_x), 3)), '  \nTrue : ', scaled_y[i][0],
                  '  \nPredicted : ', yhat[0], '\n------------')

    pred_scaled_y = pd.DataFrame(preds, columns=['NN_rgeo'])
    inversed_y = scaler_y.inverse_transform(np.array(pred_scaled_y))
    df_verif['rgeo_NN'] = 10 ** inversed_y

    # training set
    x, y, scaler_x, scaler_y, df_train = grab_data(data_fp, x_filter, y_filter, filter_flag=False)
    scaled_x = scaler_x.transform(x)
    scaled_y = scaler_y.transform(y)

    preds = []
    querypoints = np.linspace(0, len(scaled_x), 4, dtype=int)
    for i, row in tqdm(enumerate(scaled_x), total=len(scaled_x), desc=data_fp.strip('.csv')):
        yhat = predict(row, model)[0]
        preds.append(yhat)
        if i in querypoints:
            print('% : ', 100 * (round(i / len(scaled_x), 3)), '  \nTrue : ', scaled_y[i][0],
                  '  \nPredicted : ', yhat[0], '\n------------')

    pred_scaled_y = pd.DataFrame(preds, columns=['NN_rgeo'])
    inversed_y = scaler_y.inverse_transform(np.array(pred_scaled_y))
    df_train['rgeo_NN'] = 10 ** inversed_y

    print('Creating plots')
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(df_train['rgeo'], df_train['rgeo_NN'], 'bx', alpha=0.1, ms=2)
    ax.plot(df_verif['rgeo'], df_verif['rgeo_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [Parsec]', ylabel='Predicted [Parsec]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx.jpg', bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.loglog(df_train['rgeo'], df_train['rgeo_NN'], 'bx', alpha=0.1, ms=2)
    ax.loglog(df_verif['rgeo'], df_verif['rgeo_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [Parsec]', ylabel='Predicted [Parsec]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx_log.jpg', bbox_inches='tight')
    print('Plotted predicted parallaxes')

    plot_corrs(df_train, output_dir, x_filter, 'trained')
    plot_corrs(df_verif, output_dir, x_filter, 'verif')
    print('Plotted coloured properties')

    df_verif.to_csv(output_dir + '/PLX_VERIF_OUTPUT.csv')
    df_train.to_csv(output_dir + '/PLX_TRAIN_OUTPUT.csv')
    return


def plot_corrs(df: pd.DataFrame, output_dir: str, properties: list, name: str):
    idx = np.unique(choices(np.arange(len(df)), k=1000))
    x = df['rgeo'].values[idx]
    y = df['rgeo_NN'].values[idx]

    f, axs = plt.subplots(4, len(properties) // 4, figsize=(3.75 * len(properties) // 4, 4 * 4))
    for ax, emergent_property in zip(axs.ravel(), properties):
        colours = df.loc[:, emergent_property].array[idx]
        ax: plt.Axes = ax

        if emergent_property == 'parallax':
            colours = np.clip(colours, -4, 4)
        if emergent_property == 'pmra':
            colours = np.clip(colours, -14, 14)
        if emergent_property == 'pmdec':
            colours = np.clip(colours, -14, 14)
        if emergent_property == 'Plx':
            colours = np.clip(colours, -2, 2)
        if emergent_property == 'PM_x':
            colours = np.clip(colours, 0, 20)

        ax.set_title(emergent_property.replace('_', ' '))
        im = ax.scatter(x, y, edgecolors='black', c=colours, cmap="brg")
        f.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx_colour_' + str(name) + '_vars.jpg', bbox_inches='tight')
    plt.close()
    return


def main(input_prefix: str = '', output_prefix: str = ''):
    """
    Main Driver function
    """
    epochs = 20000
    learn_rate = 0.01
    learn_rate_gamma = 0.999
    data_fp = input_prefix + 'GAIA_STANDARDS.csv'
    data_verif_fp = input_prefix + 'GAIA_VARS.csv'

    # x_filter = ['ra', 'dec', 'l', 'b', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
    #             'chisq', 'uwe', 'Cody_M', 'med_BRP', 'MAD', 'stet_k', 'eta', 'mean_var', 'percent_amp', 'AD',
    #             'skew', 'kurt', 'ra_epoch2000', 'dec_epoch2000', 'parallax_x', 'parallax_over_error',
    #             'PM_x', 'pmRA_xa', 'pmra_error_x', 'pmdec_x', 'pmdec_error_x', 'b_rgeo_x', 'B_rgeo_xa',
    #             'rpgeo', 'b_rpgeo_x', 'B_rpgeo_xa']
    # x_filter = ['l', 'b', 'parallax', 'parallax_error', 'percent_amp', 'skew', 'PM_x', 'parallax_x',
    #             'parallax_over_error', 'b_rgeo_x', 'B_rgeo_xa', 'rpgeo', 'b_rpgeo_x', 'B_rpgeo_xa']
    # x_filter = ['parallax', 'parallax_error', 'parallax_x', 'parallax_over_error', 'b_rgeo_x', 'B_rgeo_xa', 'rpgeo',
    #             'b_rpgeo_x', 'B_rpgeo_xa']
    # x_filter = ['l', 'b', 'parallax', 'parallax_x', 'b_rgeo_x', 'B_rgeo_xa', 'rpgeo', 'b_rpgeo_x', 'B_rpgeo_xa']
    x_filter = ['l', 'b', 'parallax', 'pmra', 'pmdec', 'b_rgeo_x', 'B_rgeo_xa', 'rpgeo', 'b_rpgeo_x', 'B_rpgeo_xa',
                'PM_x', 'pmRA_xa', 'pmDE', 'Plx']  # ,'j-h','h-k']
    y_filter = ['rgeo']
    output_dir = output_prefix + str(epochs) + '_' + str(int(1 / learn_rate)) + '_' + str(
        int(10000 * learn_rate_gamma)) + '/'  # +'_'+dt_string+'/'

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dl, test_dl, vector_len = prepare_data(data_fp, x_filter, y_filter)
    print('Data prepared')

    model = MLP(vector_len)
    print('Model compiled')

    if not os.path.exists(output_dir + 'model.pt'):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        train_model(train_dl, test_dl, model, epochs, learn_rate, output_dir, learn_rate_gamma)
        torch.save(model.state_dict(), output_dir + 'model.pt')
        print('Model trained')
    else:
        print('Loading model from here : ', output_dir + 'model.pt')
        model.load_state_dict(torch.load(output_dir + 'model.pt'))

    print('Predicting distances')
    predict_dist(model, data_fp, data_verif_fp, output_dir, x_filter, y_filter)
    return


# YOU CAN USE CEPHIED AND AGB PERIOD LUMINMOSITY RELATIONS TO VERIFY
if __name__ == '__main__':
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    dpi = 666  # 200-300 as per guidelines
    maxpix = 3000  # max pixels of plot
    width = maxpix / dpi  # max allowed with
    rcParams.update({'axes.labelsize': 'small', 'axes.titlesize': 'small',  # the size of labels and title
                     'xtick.labelsize': 'small', 'ytick.labelsize': 'small',  # the size of the axes ticks
                     'legend.fontsize': 'x-small', 'legend.frameon': False,  # legend font size, no frame
                     'legend.facecolor': 'none', 'legend.handletextpad': 0.25,
                     # legend no background colour, separation from label to point
                     'font.serif': ['Computer Modern', 'Helvetica', 'Arial',  # default fonts to try and use
                                    'Tahoma', 'Lucida Grande', 'DejaVu Sans'],
                     'font.family': 'serif',  # use serif fonts
                     'mathtext.fontset': 'cm', 'mathtext.default': 'regular',  # if in math mode, use these
                     'figure.figsize': [width, 0.7 * width], 'figure.dpi': dpi,
                     # the figure size in inches and dots per inch
                     'lines.linewidth': .75,  # width of plotted lines
                     'xtick.top': True, 'ytick.right': True,  # ticks on right and top of plot
                     'xtick.minor.visible': True, 'ytick.minor.visible': True,  # show minor ticks
                     'text.usetex': True})  # process text with LaTeX instead of matplotlib math mode
    main('/beegfs/car/njm/OUTPUT/vars/','.')
