import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--viewmaker_n_head", type=int, default=1)
    parser.add_argument("--viewmaker_n_layers", type=int, default=3)
    parser.add_argument("--viewmaker_activation", type=str, default="relu")
    parser.add_argument("--viewmaker_distortion_budget", type=float, default=0.1)
    parser.add_argument("--viewmaker_hidden_dim", type=int, default=64)
    parser.add_argument("--viewmaker_layer_type", type=str, default="lstm")

    parser.add_argument("--encoder_n_layers", type=int, default=3)
    parser.add_argument("--encoder_hidden_dim", type=float, default=64)
    parser.add_argument("--e_lr", type=float, default=1e-3)

    parser.add_argument("--viewmaker_loss_t", type=float, default=1)
    parser.add_argument("--viewmaker_loss_weight", type=float, default=0.5)
    parser.add_argument("--viewmaker_batch_size", type=int, default=24)
    parser.add_argument("--viewmaker_num_epochs", type=int, default=2)
    parser.add_argument("--v_lr", type=float, default=1e-3)

    parser.add_argument("--post_hoc_n_layers", type=int, default=2)
    parser.add_argument("--post_hoc_h_size", type=int, default=12)
    parser.add_argument("--post_hoc_num_epochs", type=int, default=2)
    parser.add_argument("--post_hoc_save_metric", type=str, default="accuracy")
    parser.add_argument("--post_hoc_batch_size", type=int, default=24)
    parser.add_argument("--post_hoc_lr", type=float, default=1e-3)

    args = parser.parse_args()

    # create a list of the keys of args
    args_keys = list(vars(args).keys())

    return args, args_keys
