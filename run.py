import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from util.tools import set_logger, serializable_parts_of_dict
from config.config import get_config

def mian()
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Air quality Prediction Based on Pretrained LLM')

    # basic config
    parser.add_argument('--data', type=str, required=True, default='beijing', help='dataset name')
    parser.add_argument('--model', type=str, required=True, default='llmair',
                        help='model name, options: [llmair, airformer, gagnn...]')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers') 
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')

    # pt
    parser.add_argument('--need_pt', type=int, default=0, help='whether continue pretrain')
    parser.add_argument('--pt_model_dir', type=str, default='null', help='the base model dir for need_pt')
    parser.add_argument('--pt_layers', type=str, default='null', help='the layers in llm needed to be trained')
    parser.add_argument('--pt_data', type=str, default='null', help='the dataset used in pretrain, use _ to separate')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    args = parser.parse_args()

    config = get_config(args.model, args.data)
    config.update(vars(args))

    # save params and loggers
    save_floder = os.path.join('saved/'+'/'+config.data+'/'+config.model, f"v_{config.version}")
    pt_dir = Path(save_floder) / "pt"
    log_dir = Path(save_floder) / "logs"
    Path(pt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    config.pt_dir = pt_dir
    logger = set_logger(log_dir, config.model, config.data, verbose_level=1)
    # serializable_dict = serializable_parts_of_dict(config)
    # logger.info(json.dumps(serializable_dict, indent=4))

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "10.3.242.137") # 127.0.0.1
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        logger.info(f"ip:{ip}:{port}, hosts:{hosts}, gpus:{gpus}, rank:{rank}, local_rank:{local_rank}...")
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)


    if config.is_training:
        for ii in range(config.itr):
            # setting record of experiments
            exp = Exp(config)  # set experiments
            setting = '{}_{}_{}_{}_hd{}_bs{}_lr{}_wd{}'.format(
                config.model,
                config.data,
                config.seq_len,
                config.pred_len,
                config.hidden_dim,
                config.batch_size,
                config.learning_rate,
                config.weight_decay
                )

            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = '{}_{}_{}_{}_hd{}_bs{}_lr{}_wd{}'.format(
            config.model,
            config.data,
            config.seq_len,
            config.pred_len,
            config.hidden_dim,
            config.batch_size,
            config.learning_rate,
            config.weight_decay
            )
        exp = Exp(config)  # set experiments
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()