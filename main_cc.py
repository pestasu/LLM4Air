import argparse
import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from easydict import EasyDict as edict
from utils.tools import set_logger, serializable_parts_of_dict, gen_version
from config.config import get_config 
from exp.exp_ha import Exp_HA
from exp.exp_var import Exp_VAR
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
def main():
    fix_seed = [1111,2222,3333,4444,5555,6666,7777,8888,9999]
    random.seed(fix_seed[0])
    torch.manual_seed(fix_seed[0])
    np.random.seed(fix_seed[0])

    parser = argparse.ArgumentParser(description='Air quality Prediction Based on HA and VAR')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--data', type=str, required=True, default='beijing', help='dataset name')
    parser.add_argument('--model', type=str, required=True, default='ha')
    parser.add_argument('--data_root_path', type=str, default='data/')
    parser.add_argument('--data_path', type=str, default='beijing/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=24)

    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers') 
    parser.add_argument('--n_exp', type=int, default=1, help='experiments times')
    parser.add_argument('--version', type=int, default=-1, help='experiments version')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    args = parser.parse_args()

    # save params and loggers
    setting = '{}_{}_{}_{}_bs{}'.format(
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.batch_size,
                )

    if args.is_training:
        args.version = gen_version(args)

    save_floder = os.path.join('saved/'+'/'+args.data+'/'+args.model, f"{setting}_{args.version}")
    data_floder = os.path.join(args.data_root_path, args.data_path)
    pt_dir = Path(save_floder) / "pt"
    log_dir = Path(save_floder) / "logs"
    Path(pt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    args.pt_dir = pt_dir
    args.data_floder = data_floder
    logger = set_logger(log_dir, args.model, args.data, verbose_level=1)
    logger.info(args)
    logger.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>> [ {args.model}-{args.data}({args.version}) ]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    if args.is_training:
        for ii in range(args.n_exp):
            # setting record of experiments
            if args.model == 'ha':
                exp = Exp_HA(args, ii)  # set experiments
            elif args.model == 'var':
                exp = Exp_VAR(args, ii)  # set experiments
            # if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            #     logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>start training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.train(setting)
            exp.test(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = '{}_{}_{}_{}_bs{}'.format(
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.batch_size,
            )
        if args.model == 'ha':
            exp = Exp_HA(args, ii)  # set experiments
        elif args.model == 'var':
            exp = Exp_VAR(args, ii)  # set experiments
        exp.test(setting, is_test=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()