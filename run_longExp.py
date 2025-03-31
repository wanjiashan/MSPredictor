import argparse
import time
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='MSPredictor for Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='longitudinal forecast analysis',
                    help='task name, options:[long_term_forecast, mask, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='MSPredictor',
                    help='model name, options: [MSPredictor]')

# data loader
parser.add_argument('--data',type=str,required=True,default='ETTm1',
                    help='Supported datasets: ETTm1, ETTh1, Electricity, Traffic, Weather')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,'
                         ' S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='Directory path to save/load model checkpoints. Supports relative absolute paths.')

# forecasting task
parser.add_argument('--seq_len',type=int,default=96,help='Input sequence length (time steps) for the model')
parser.add_argument('--label_len', type=int, default=48,help='Length of decoder start token sequence')
parser.add_argument('--pred_len', type=int,default=96,help='Prediction horizon length (time steps)')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for weather or Solar-Energy')


parser.add_argument('--top_k',type=int,default=5,help='Number of top scales/frequencies to select in TimesBlock')
parser.add_argument('--num_kernels', type=int,default=6,help='Number of parallel convolution kernels in Inception block')

parser.add_argument('--num_nodes',type=int,default=7,help='Number of nodes in the constructed graph')
parser.add_argument('--subgraph_size',type=int,default=3,help='Number of nearest neighbors for local subgraph construction')
parser.add_argument('--tanhalpha', type=float, default=3)

#GCN
parser.add_argument('--node_dim',type=int,default=10,help='Dimensionality of node embeddings')
parser.add_argument('--gcn_depth', type=int, default=2, help='')
parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
parser.add_argument('--propalpha', type=float, default=0.3, help='')
parser.add_argument('--conv_channel', type=int, default=32, help='')
parser.add_argument('--skip_channel', type=int, default=32, help='')


# DLinear
parser.add_argument('--individual',action='store_true',default=False,help='Use separate linear layers for each channel (DLinear mode)')
# Formers
parser.add_argument('--embed_type',type=int,default=0,choices=[0, 1, 2, 3, 4],help='''Embedding configuration:
                                                                   0: Default
                                                                   1: Value + Temporal + Positional
                                                                   2: Value + Temporal
                                                                   3: Value + Positional
                                                                   4: Value only''')
parser.add_argument('--enc_in', type=int, default=7, help='encoder size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model',type=int,default=512,help='Hidden dimension size of the model')
parser.add_argument('--n_heads',type=int,default=8,help='Number of attention heads in multi-head attention')
parser.add_argument('--e_layers',type=int,default=2,help='Number of encoder layers in the model')
parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers in the model')
parser.add_argument('--d_ff',type=int,default=2048,help='Hidden dimension of position-wise feed-forward network')
parser.add_argument('--moving_avg',type=int,default=25,help='Window size for moving average smoothing')
parser.add_argument('--factor',type=int,default=1,help='Attention scaling factor')
parser.add_argument('--no_distil',action='store_false',dest='distil',default=True,help='Disable distillation in encoder')
parser.add_argument('--dropout',type=float,default=0.05,help='Dropout probability for all layers')
parser.add_argument('--embed',type=str,default='timeF',choices=['timeF', 'fixed', 'learned'],
                    help='Time features encoding method. Options: timeF, fixed, learned')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention',action='store_true',help='Whether to return encoder attention weights')
parser.add_argument('--do_predict',action='store_true',help='Enable forecasting on unseen future data')

# optimization
parser.add_argument('--num_workers',type=int,default=8,help='Number of subprocesses for data loading')
parser.add_argument('--itr',type=int,default=2,help='Number of independent experimental runs')
parser.add_argument('--train_epochs', type=int,default=10,help='Number of complete training passes through the dataset')
parser.add_argument('--batch_size',type=int,default=32,help='Number of samples per training batch')
parser.add_argument('--patience',type=int,default=3,help='Number of epochs to wait before early stopping')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='Initial learning rate for the optimizer')
parser.add_argument('--des',type=str,default='test',help='Experiment description tag for identification')
parser.add_argument('--loss',type=str,default='MSE',choices=['MSE', 'MAE', 'Huber', 'SmoothL1'],help='Loss function for training. Options: MSE, MAE, Huber, SmoothL1')
parser.add_argument('--lradj',type=str,default='type1',choices=['type1', 'type2', 'type3', 'cosine', 'step'],help='Learning rate adjustment strategy. Options: type1, type2, type3, cosine, step')
parser.add_argument('--use_amp',action='store_true',default=False,help='Multi-scale Fusion Prediction')
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu',action='store_true',default=False,help='Enable distributed training across multiple GPUs')

parser.add_argument('--devices',type=str,default='0,1,2,3',help='Comma-separated GPU device IDs (e.g., "0,1" for first two GPUs)')
parser.add_argument('--test_flop',action='store_true',default=False,help='Calculate model FLOPs (floating point operations)')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    start = time.time()
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('*******************start training : {}**************************>'.format(setting))
        exp.train(setting)

        print('*******************testing : {}*******************************'.format(setting))
        exp.test(setting)


        torch.cuda.empty_cache()
    end = time.time()
    used_time = end -start
    print("time:",used_time)
    f = open("result.txt", 'a')
    f.write('time:{}'.format(used_time))
    f.write('\n')
    f.write('\n')
    f.close()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('**********************testing : {}********************'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
