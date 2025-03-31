from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import MSPredictor
from layers import KAN
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'MSPredictor': MSPredictor,
            'KAN': KAN
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'Linear' in self.args.model or 'KAN' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                if 'KAN' in self.args.model:
                    reg_loss = self.model.regularization_loss(regularize_activation=0.1, regularize_entropy=0.01)
                    loss += reg_loss
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'Linear' in self.args.model or 'KAN' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                if 'KAN' in self.args.model:
                    reg_loss = self.model.regularization_loss(regularize_activation=0.1, regularize_entropy=0.01)
                    loss += reg_loss
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(
                f"Epoch: {epoch + 1}, Train Loss: {train_loss:.6f}, Vali Loss: {vali_loss:.6f}, Test Loss: {test_loss:.6f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/', setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = os.path.join('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'Linear' in self.args.model or 'KAN' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 截取 ground truth 的最后 pred_len 时间步
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds_array = np.array(preds)
        trues_array = np.array(trues)
        preds = preds_array.reshape(-1, preds_array.shape[-2], preds_array.shape[-1])
        trues = trues_array.reshape(-1, trues_array.shape[-2], trues_array.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr, nd, nrmse = metric(preds, trues)
        print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, RSE: {rse}')

        np.save(os.path.join(folder_path, 'preds.npy'), preds)
        np.save(os.path.join(folder_path, 'trues.npy'), trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x = batch_x.float().to(self.device)
                if 'Linear' in self.args.model or 'KAN' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                preds.append(outputs.detach().cpu().numpy())
        preds_array = np.array(preds)
        preds = preds_array.reshape(-1, preds_array.shape[-2], preds_array.shape[-1])
        np.save(os.path.join('./results/', setting, 'real_prediction.npy'), preds)
        return preds
