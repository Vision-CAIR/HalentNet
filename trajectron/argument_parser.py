import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conf",
                    help="path to json config file for hyperparameters",
                    type=str,
                    default='../config/config.json')

parser.add_argument("--debug",
                    help="disable all disk writing processes.",
                    action='store_true')

parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=0)


# Model Parameters
parser.add_argument("--offline_scene_graph",
                    help="whether to precompute the scene graphs offline, options are 'no' and 'yes'",
                    type=str,
                    default='yes')

parser.add_argument("--dynamic_edges",
                    help="whether to use dynamic edges or not, options are 'no' and 'yes'",
                    type=str,
                    default='yes')

parser.add_argument("--edge_state_combine_method",
                    help="the method to use for combining edges of the same type",
                    type=str,
                    default='sum')

parser.add_argument("--edge_influence_combine_method",
                    help="the method to use for combining edge influences",
                    type=str,
                    default='attention')

parser.add_argument('--edge_addition_filter',
                    nargs='+',
                    help="what scaling to use for edges as they're created",
                    type=float,
                    default=[0.25, 0.5, 0.75, 1.0]) # We don't automatically pad left with 0.0, if you want a sharp
                                                    # and short edge addition, then you need to have a 0.0 at the
                                                    # beginning, e.g. [0.0, 1.0].

parser.add_argument('--edge_removal_filter',
                    nargs='+',
                    help="what scaling to use for edges as they're removed",
                    type=float,
                    default=[1.0, 0.0])  # We don't automatically pad right with 0.0, if you want a sharp drop off like
                                         # the default, then you need to have a 0.0 at the end.

parser.add_argument('--override_attention_radius',
                    action='append',
                    help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"',
                    default=[])

parser.add_argument('--incl_robot_node',
                    help="whether to include a robot node in the graph or simply model all agents",
                    action='store_true')

parser.add_argument('--map_encoding',
                    help="Whether to use map encoding or not",
                    action='store_true')

parser.add_argument('--augment',
                    help="Whether to augment the scene during training",
                    action='store_true')

parser.add_argument('--node_freq_mult_train',
                    help="Whether to use frequency multiplying of nodes during training",
                    action='store_true')

parser.add_argument('--node_freq_mult_eval',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--scene_freq_mult_train',
                    help="Whether to use frequency multiplying of nodes during training",
                    action='store_true')

parser.add_argument('--scene_freq_mult_eval',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--scene_freq_mult_viz',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--no_edge_encoding',
                    help="Whether to use neighbors edge encoding",
                    action='store_true')

# Data Parameters
parser.add_argument("--data_dir",
                    help="what dir to look in for data",
                    type=str,
                    default='../experiments/processed')

parser.add_argument("--train_data_dict",
                    help="what file to load for training data",
                    type=str,
                    default='train.pkl')

parser.add_argument("--eval_data_dict",
                    help="what file to load for evaluation data",
                    type=str,
                    default='val.pkl')

parser.add_argument("--log_dir",
                    help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str,
                    default='../experiments/logs')

parser.add_argument("--log_tag",
                    help="tag for the log folder",
                    type=str,
                    default='')

parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')

parser.add_argument("--eval_device",
                    help="what device to use during evaluation",
                    type=str,
                    default=None)

# Training Parameters
parser.add_argument("--train_epochs",
                    help="number of iterations to train for",
                    type=int,
                    default=1)

parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=256)

parser.add_argument('--eval_batch_size',
                    help='evaluation batch size',
                    type=int,
                    default=256)

parser.add_argument('--k_eval',
                    help='how many samples to take during evaluation',
                    type=int,
                    default=25)

parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--eval_every',
                    help='how often to evaluate during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--vis_every',
                    help='how often to visualize during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--save_every',
                    help='how often to save during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--load_model',
                    help='path of model for continue training',
                    type=str,
                    default='')
parser.add_argument('--checkpoint',
                    help='the checkpoint to use',
                    type=int,
                    default='12')
parser.add_argument('--g_factor',
                    help='the factor for generator loss',
                    type=int,
                    default=1)
parser.add_argument('--grid_std',
                    help='std of traj grid',
                    type=int,
                    default=5)
parser.add_argument('--grid_max',
                    help='max value of traj grid',
                    type=int,
                    default=9)
parser.add_argument('--pseudo_ratio',
                    help='the ratio of pseudo data for supervised learning loss',
                    type=float,
                    default=0.0)
parser.add_argument('--creative_lambda',
                    help='the coefficient of creative loss',
                    type=float,
                    default=1.0)
parser.add_argument('--creative_t',
                    help='the way to switch z',
                    type=str,
                    default='rand')
parser.add_argument('--creative_loss',
                    help='the way compute creative loss',
                    type=str,
                    default='uni')

args = parser.parse_args()
