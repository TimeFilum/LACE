import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file automatically

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='LACE forecasting experiments')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task: multiscale_lace_forecast, zero_shot_forecast, or long_term_forecast')
    #parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='LACE_moirai1',
                        help='model registry name, e.g. LACE_moirai1, Moirai1, DLinear, LSTM')

    # attention maps
    parser.add_argument('--save_attn', type=bool, help='save attention maps for visualization', default=False)

    # data loader
    parser.add_argument('--data', type=str, required=True, default='fixed_split',
                        choices=['fixed_split'], help='shared CSV loader for all release datasets')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--data_name', type=str, default='ETTh1', help='data file name without extension')
    parser.add_argument('--data_resolution', type=str, default='5m', help='data resolution')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--test_split_timestamp', type=str, default=None, help='the timestamp for fixed split test set in format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--lstm_hidden', type=int, default=100, help='hidden size of lstm')
    parser.add_argument('--lstm_layers', type=int, default=1, help='number of layers of lstm')


    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging', default=False)

    # Ablation for multiscale models
    parser.add_argument('--ablation_mode', type=str, default=None, help='ablation type for multiscale models, options: [equal, custom, full]')
    parser.add_argument('--ablation_type', type=str, default=None, help='ablation type for evalautions, options: [forward, burst]')

    # #LACE factors
    # parser.add_argument('--lace_factors', type=int, nargs='+', default=[1, 3, 12, 288],
    #                 help='Space-separated list of aggregation factors (e.g., --multiscale_factors 1 2 6 144)')

    # ==============================================================
    # LACE
    # ==============================================================

    parser.add_argument(
        '--lace_factors',
        type=int,
        nargs='+',
        default=[1, 3, 12, 288],
    )

    parser.add_argument(
        '--lace_adapter_dim',
        type=int,
        default=128,
    )

    parser.add_argument(
        '--lace_representation_norm',
        type=str,
        default='batch',
        choices=['batch', 'layer', 'none'],
    )

    parser.add_argument(
        '--lace_reduction',
        type=str,
        default='sum',
        choices=['sum', 'mean'],
    )

    parser.add_argument(
        '--lace_coherence_mode',
        type=str,
        default='none',
        choices=['none', 'structural'],
    )

    parser.add_argument(
        '--lace_lambda',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '--lace_bias_weight',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '--lace_backbone',
        type=str,
        default='Salesforce/moirai-1.0-R-small',
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help=(
            "LACE model/result directory override, e.g. "
            "LACE_moirai2, LACE_timesfm2, LACE_chronos2."
        ),
    )



    args = parser.parse_args()
    if args.use_gpu and args.gpu_type == 'cuda' and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    elif (args.use_gpu and args.gpu_type == 'mps'
          and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        args.device = torch.device('mps')
        print('Using MPS')
    else:
        args.use_gpu = False
        args.device = torch.device('cpu')
        print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'multiscale_lace_forecast':
        from exp.exp_multiscale_lace_forecast import Exp_Multiscale_LACE_Forecast
        Exp = Exp_Multiscale_LACE_Forecast
        print("TASK:", args.task_name)
        print("EXPERIMENT:", Exp.__name__)
        print("MODEL:", args.model)
        print("DESC:", args.des)
        print("PRED_LEN:", args.pred_len)
        print("LACE FACTORS:", args.lace_factors)
    elif args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting_advanced import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        raise ValueError(f'Unsupported task in the LACE release: {args.task_name!r}')

    if args.ablation_type is not None:
        raise ValueError("Exploratory forward/burst ablations are not included in this release.")

    else:
        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                exp = Exp(args)  # set experiments
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
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
                    args.expand,
                    args.d_conv,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                if args.use_gpu:
                    if args.gpu_type == 'mps':
                        torch.backends.mps.empty_cache()
                    elif args.gpu_type == 'cuda':
                        torch.cuda.empty_cache()
        else:
            exp = Exp(args)  # set experiments
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
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
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
