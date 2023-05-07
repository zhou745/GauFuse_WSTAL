from __future__ import print_function
import os
import torch

from utils.utils import write_settings_to_file

def main(training_type = 0):

    if training_type==0:
        import options as options
        from method import Processor
    else:
        raise RuntimeError("unknown runing type")

    args = options.parser.parse_args()
    assert args.run_type in [0, 1]
    # fix random seed for stable results
    torch.manual_seed(args.seed)
    # set visible gpus
    args.gpu_ids = visible_gpu(args.gpus)
    # create folder
    if not os.path.exists('./ckpt/'):
        os.makedirs('./ckpt/')
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')

    out_dir = os.path.join('./ckpt/', args.dataset_name, str(args.model_id))
    log_dir = os.path.join('./logs/', args.dataset_name, str(args.model_id))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.run_type == 0 or args.run_type == 1 and args.pretrained == False:
        write_settings_to_file(args)
    # args.pretrained = True
    processor = Processor(args)
    processor.processing()


def visible_gpu(gpus):
    """
        set visible gpu.
        can be a single id, or a list
        return a list of new gpus ids
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, gpus)))
    return list(range(len(gpus)))


if __name__ == '__main__':
    #0 : pretrain
    #1:  our method

    training_type = 0

    main(training_type=training_type)
