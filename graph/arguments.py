import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')

    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, help='mode')
    parser.add_argument('--t2', type=float, help='t2')
    parser.add_argument('--model_name', type=str, help='t2')
    parser.add_argument('--w1', type=float, help='t2')
    parser.add_argument('--w2', type=float, help='t2')
    parser.add_argument('--temp', type=float, help='t2')
    parser.add_argument('--noise_ratio', type=float, help='t2')
    parser.add_argument('--output_dir', type=str, help='t2', default="./log_pos/")

    return parser.parse_args()

