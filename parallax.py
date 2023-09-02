import os  # Operating system related functionalities
from datetime import datetime  # Working with dates and times
from random import choices, random  # Random sampling and number generation

import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation

import torch  # Main PyTorch library
from torch import Tensor  # Tensor class from PyTorch
#from torch.nn import Linear, Sigmoid, Module, ReLU, HuberLoss, BatchNorm1d  # Neural network layers and loss functions
import torch.nn as nn
from torch.nn.init import xavier_uniform_  # Xavier uniform weight initialization
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR  # Learning rate scheduler
from torch.utils.data import Dataset, DataLoader, random_split  # PyTorch data loading utilities
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler  # Data normalization
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # Evaluating model performance
import matplotlib.pyplot as plt  # Plotting
from matplotlib import rcParams  # Customizing plot parameters

from tqdm import tqdm  # Creating progress bars

class MLP(nn.Module):
    def __init__(self, n_inputs: int):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.hidden2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.hidden3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.hidden4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        
        self.hidden5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        
        self.hidden6 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.2)
        
        self.hidden7 = nn.Linear(16, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        x = self.hidden3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        
        x = self.hidden4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.dropout4(x)
        
        x = self.hidden5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.dropout5(x)
        
        x = self.hidden6(x)
        x = self.bn6(x)
        x = self.act6(x)
        x = self.dropout6(x)
        
        x = self.hidden7(x)
        return x

class CSVDataset(Dataset):
    def __init__(self, path: str, x_filter: list, x_importance: list, y_filter: list):
        x, y, scaler_x, scaler_y, df = grab_data(path, x_filter, x_importance, y_filter)
        self.x = torch.from_numpy(scaler_x.transform(x)).float()
        self.y = torch.from_numpy(scaler_y.transform(y)).float()
        print('TRAINING ON DATASET OF SIZE:', len(x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def get_splits(self, n_test: float = 0.33):
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        return random_split(self, [train_size, test_size])

def grab_data(data_fp: str, x_filter: list, x_importance: list, y_filter: list, filter_flag: bool = True):
    snr = 2
    df = pd.read_csv(data_fp, low_memory=False)

    df = df.replace(np.nan, 0)
    df = df.replace(np.inf, 0)

    if filter_flag:
        df = df.loc[
            (abs(df['parallax_corr_over_error']) > 0.2) &
            (abs(df['parallax_over_error_vvv']) > 0.2) &
            (abs(df['ipd_frac_multi_peak']) < 0.1)
        ]
    else:
        df = df

    x = df[x_filter].copy()
    y = df[y_filter].copy()
    y.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    scaler_x = StandardScaler()
    scaler_x.fit(x)

    # Check if any NaN or infinite values exist in the scaled features
    if np.isnan(scaler_x.mean_).any() or np.isnan(scaler_x.scale_).any():
        scaler_x.mean_ = np.nan_to_num(scaler_x.mean_)
        scaler_x.scale_ = np.nan_to_num(scaler_x.scale_)

        # If any infinite values exist, replace them with large finite values
        scaler_x.mean_[np.isinf(scaler_x.mean_)] = 1e9
        scaler_x.scale_[np.isinf(scaler_x.scale_)] = 1e9

    # Apply feature weights
    scaler_x = weigh_features(scaler_x, x_importance)

    scaler_y = StandardScaler()
    scaler_y.fit(y)

    # Check if any NaN or infinite values exist in the scaled targets
    if np.isnan(scaler_y.mean_).any() or np.isnan(scaler_y.scale_).any():
        scaler_y.mean_ = np.nan_to_num(scaler_y.mean_)
        scaler_y.scale_ = np.nan_to_num(scaler_y.scale_)

        # If any infinite values exist, replace them with large finite values
        scaler_y.mean_[np.isinf(scaler_y.mean_)] = 1e9
        scaler_y.scale_[np.isinf(scaler_y.scale_)] = 1e9

    return x, y, scaler_x, scaler_y, df

def prepare_data(path: str, x_filter: list, x_importance: list, y_filter: list):
    dataset = CSVDataset(path, x_filter, x_importance, y_filter)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=16384, shuffle=True)
    test_dl = DataLoader(test, batch_size=4096, shuffle=False)
    return train_dl, test_dl, len(test[0][0])


def weigh_features(scaler, importance):
    # Function to weigh features based on importance
    # Parameters:
    #   scaler: Scaler object (e.g., StandardScaler)
    #   importance (list): Importance weights for each feature
    # Returns:
    #   scaler_weighted: Weighed scaler object

    scaler_weighted = scaler
    scaler_weighted.mean_ *= importance
    scaler_weighted.scale_ *= importance

    return scaler_weighted

def train_model(train_dl: DataLoader, test_dl: DataLoader, model: nn.Module, epochs: int, learn_rate: float, output_dir: str, gamma: float):

    # During training/validation, calculate additional evaluation metrics
    def evaluate_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return r2, mse, mae

    min_delta = 0.0001
    tolerance = 3
    counter = 0
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=tolerance, verbose=True)
    train_stats = pd.DataFrame(columns=['epoch', 'learn_rate', 'loss', 'delta_loss', 'counter', 'val_loss', 'r2_score', 'mse', 'mae'])
    mse = np.nan

    for epoch in range(epochs):
        model.train()
        for train_features, train_labels in train_dl:
            optimizer.zero_grad()
            yhat = model(train_features)
            loss = criterion(yhat, train_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for test_features, test_labels in test_dl:
                prediction_val = model(test_features)
                val_loss = criterion(prediction_val, test_labels)
                loss_delta = val_loss - loss


        if not epoch % 2:
            my_lr = optimizer.param_groups[0]['lr']
            r2,mse,mae = evaluate_metrics(train_labels.detach().numpy() ,yhat.detach().numpy())
            train_stats.loc[epoch] = [epoch, float(my_lr), float(loss.detach().numpy()),
                                      float(loss_delta.detach().numpy()), counter,
                                      float(val_loss.detach().numpy()), r2, mse, mae]
            scheduler.step(val_loss)

        if not epoch % 16:
            print('Epoch:', epoch, 'Learning rate:', float(my_lr), 'Val Loss:', float(val_loss), 'Loss:',
                  float(loss), 'Delta Loss:', float(loss_delta), 'ES counter:', counter)

            if not epoch % 1024 and epoch > 1:
                plots(train_stats, output_dir, mse)

        if epoch > 1024 and loss_delta > min_delta:
            counter += 1
            if counter > 10 and counter >= tolerance:
                print('Epoch:', epoch, 'Learning rate:', float(my_lr), 'Val Loss:', float(val_loss),
                      'Loss:', float(loss), 'Delta Loss:', float(loss_delta), 'ES counter:', counter)
                print("STOPPING at epoch:", epoch)
                plots(train_stats, output_dir, mse)
                return

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
    try:
        plt.clf()
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['mse'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='MSE', xscale='log', yscale='log')
        fig.tight_layout()
        plt.savefig(output_dir + '/log_mse.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['mae'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='MAE', xscale='log', yscale='log')
        fig.tight_layout()
        plt.savefig(output_dir + '/log_mae.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    try:
        plt.clf()
        fig, ax = plt.subplots()
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(train_stats['epoch'], train_stats['r2_score'], 'k', alpha=0.9, ms=2)
        ax.set(xlabel='Epoch', ylabel='R2', xscale='log', yscale='log')
        fig.tight_layout()
        plt.savefig(output_dir + '/log_r2.jpg', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'Did not plot, because {repr(e)}')
    return


# evaluate the model
def evaluate_model(test_dl: DataLoader, model: nn.Module):
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
def predict(row: list, model: nn.Module):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


def predict_dist(model: nn.Module, data_fp: str, data_verif_fp: str, output_dir: str, x_filter: list, x_importance: list, y_filter: list):
    # verification set
    x, y, scaler_x, scaler_y, df_verif = grab_data(data_verif_fp, x_filter, x_importance, y_filter, filter_flag=False)
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

    pred_scaled_y = pd.DataFrame(preds, columns=['NN_parallax_corr'])
    inversed_y = scaler_y.inverse_transform(np.array(pred_scaled_y))
    df_verif['parallax_NN'] = inversed_y

    # training set
    x, y, scaler_x, scaler_y, df_train = grab_data(data_fp, x_filter, x_importance, y_filter, filter_flag=False)
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

    pred_scaled_y = pd.DataFrame(preds, columns=['NN_parallax_corr'])
    inversed_y = scaler_y.inverse_transform(np.array(pred_scaled_y))
    df_train['parallax_NN'] = inversed_y

    print('Creating plots')
    # Filter the data for this on this side of the bulge
    df_train_filtered = df_train[df_train['parallax_corr'] > 1/8]
    df_verif_filtered = df_verif[df_verif['parallax_corr'] > 1/8]

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(df_train['parallax_corr'], df_train['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.plot(df_verif['parallax_corr'], df_verif['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [mas]', ylabel='Predicted [mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx.jpg', bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(1/df_train['parallax_corr'], 1/df_train['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.plot(1/df_verif['parallax_corr'], 1/df_verif['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [1/mas]', ylabel='Predicted [1/mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_dist.jpg', bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(df_train_filtered['parallax_corr'], df_train_filtered['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.plot(df_verif_filtered['parallax_corr'], df_verif_filtered['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [mas]', ylabel='Predicted [mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx_close.jpg', bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(1/df_train_filtered['parallax_corr'], 1/df_train_filtered['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.plot(1/df_verif_filtered['parallax_corr'], 1/df_verif_filtered['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [1/mas]', ylabel='Predicted [1/mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_dist_close.jpg', bbox_inches='tight')

    print('Creating log plots')
    plt.clf()
    fig, ax = plt.subplots()
    ax.loglog(df_train['parallax_corr'], df_train['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.loglog(df_verif['parallax_corr'], df_verif['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [mas]', ylabel='Predicted [mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx.jpg', bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.loglog(1/df_train['parallax_corr'], 1/df_train['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.loglog(1/df_verif['parallax_corr'], 1/df_verif['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [1/mas]', ylabel='Predicted [1/mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_dist_log.jpg', bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.loglog(df_train_filtered['parallax_corr'], df_train_filtered['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.loglog(df_verif_filtered['parallax_corr'], df_verif_filtered['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [mas]', ylabel='Predicted [mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx_close_log.jpg', bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.loglog(1/df_train_filtered['parallax_corr'], 1/df_train_filtered['parallax_NN'], 'bx', alpha=0.1, ms=2)
    ax.loglog(1/df_verif_filtered['parallax_corr'], 1/df_verif_filtered['parallax_NN'], 'gx', alpha=0.1, ms=2)
    ax.axline((0, 0), (1, 1), color='k')
    ax.set(xlabel='True [1/mas]', ylabel='Predicted [1/mas]')
    fig.tight_layout()
    plt.savefig(output_dir + '/pred_true_dist_close_log.jpg', bbox_inches='tight')


    print('Plotted predicted parallaxes')
    print('Creating colour plots')
    plot_corrs(df_train, output_dir, x_filter, 'trained')
    plot_corrs(df_verif, output_dir, x_filter, 'verif')
    print('Plotted coloured properties')

    df_verif.to_csv(output_dir + '/PLX_VERIF_OUTPUT.csv')
    df_train.to_csv(output_dir + '/PLX_TRAIN_OUTPUT.csv')
    return


def plot_corrs(df: pd.DataFrame, output_dir: str, properties: list, name: str):
    idx = np.unique(choices(np.arange(len(df)), k=1000))
    x = df['parallax_corr'].values[idx]
    y = df['parallax_NN'].values[idx]

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


    #log version 
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
        ax.set_xscale('log')
        ax.set_yscale('log')
        im = ax.scatter(x, y, edgecolors='black', c=colours, cmap="brg")
        f.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir + '/pred_true_plx_colour_log_' + str(name) + '_vars.jpg', bbox_inches='tight')
    plt.close()
    return


def main(input_prefix: str = '', output_prefix: str = ''):
    """
    Main Driver function
    """
    epochs = 2048
    learn_rate = 0.1
    learn_rate_gamma = 0.9

    data_fp = input_prefix + 'VVV_GAIA_STANDARDS.csv'
    data_verif_fp = data_fp#input_prefix + 'GAIA_VARS.csv'

    x_filter = ['parallax_corr','phot_bp_rp_excess_factor_corr',
        'ra','dec','l','b','ecl_lon','ecl_lat',
        'parallax','pmra','pmdec',
        'dec_parallax_corr','dec_pmdec_corr','dec_pmra_corr',
        'parallax_pmdec_corr','parallax_pmra_corr',
        'pm','pmra_pmdec_corr','ra_dec_corr','radial_velocity',
        'ra_parallax_corr','ra_pmdec_corr','ra_pmra_corr',
        'ra_vvv','dec_vvv','l_vvv','b_vvv','parallax_vvv',
        'pmra_vvv','pmdec_vvv',
        'bp_g','bp_rp',
        'g_rp','grvs_mag',
        'J-K','H-K','Z-K','Y-K']
        #Z-K and Y-K might have bad data, check weights.

    x_importance = [0.6,0.2, 0.4,0.4,0.4,0.4,0.4,0.4,
	        0.6,0.6,0.6,0.5,0.3,0.3,
	        0.3,0.3,0.8,0.3,0.3,
	        0.7,0.3,0.3,0.3,0.7,
	        0.5,0.5,0.5,1,1,1,0.5,0.5,
	        0.5,0.4,0.7,0.7,0.4,0.4]

    y_filter = ['parallax_corr']
    output_dir = output_prefix + str(epochs) + '_' + str(int(1 / learn_rate)) + '_' + str(int(10000 * learn_rate_gamma)) + '/'

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dl, test_dl, vector_len = prepare_data(data_fp, x_filter, x_importance, y_filter)
    print('Data prepared')

    model = MLP(vector_len)
    print('Model compiled')

    if True:#not os.path.exists(output_dir + 'model.pt'):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        train_model(train_dl, test_dl, model, epochs, learn_rate, output_dir, learn_rate_gamma)
        torch.save(model.state_dict(), output_dir + 'model.pt')
        print('Model trained')
    else:
        print('Loading model from here : ', output_dir + 'model.pt')
        model.load_state_dict(torch.load(output_dir + 'model.pt'))

    print('Predicting distances')
    predict_dist(model, data_fp, data_verif_fp, output_dir, x_filter, x_importance,  y_filter)
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
    main('/beegfs/car/njm/OUTPUT/','/beegfs/car/njm/GAIA_NN_DIST_DATA/')
