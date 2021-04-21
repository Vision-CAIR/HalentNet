import numpy as np
import torch
from torch.utils.data import DataLoader

from argument_parser import args
from environment import derivative_of, Environment, Scene, Node, DoubleHeaderNumpyArray
from environment.data_utils import interpolate_angle
from model.dataset import EnvironmentDataset, collate
from utils import prediction_output_to_trajectories


def calculate_head(vel, vel_norm, hist_head):
    vel_x = vel[..., 0]
    vel_y = vel[..., 1]
    heading_v = np.divide(vel_y, vel_x, out=np.ones_like(vel_x) * np.nan,
                          where=(vel_norm[..., 0] > 0.3))  # skip the compute when the agent is too slow, pad with nan
    heading = np.arctan(heading_v)

    with np.errstate(invalid='ignore'):  # the heading contains nan, '>0' will raise a runtimewarning. ignore it
        heading = heading + (vel_x < 0).astype(int) * ((heading > 0) * 2 - 1).astype(int) * (-1) * np.pi  # extend from [-pi/2, pi/2] to [-pi, pi]
    heading[:len(hist_head)] = hist_head
    heading = interpolate_angle(heading)  # interpolate the time steps when agent is too slow
    return heading


def create_new_node_data(histories_dict, prediction_dict, dt):
    new_data_dict = {}
    for ts, node_dict in histories_dict.items():
        new_data_dict[ts] = {}
        for node, hist_state in node_dict.items():
            pred_state = prediction_dict[ts][node][0, 0]  # current version only works with 1 sample, todo make it compatible with multiple samples
            pos = np.concatenate([hist_state[:, :2], pred_state[:, :2]], axis=0)

            vel = np.gradient(pos, dt, axis=-2)
            vel_norm = np.linalg.norm(vel, axis=-1, keepdims=True)
            acc = np.gradient(vel, dt, axis=-2)
            acc_norm = np.linalg.norm(acc, axis=-1, keepdims=True)
            heading_v = np.divide(vel, vel_norm, out=np.zeros_like(vel), where=(vel_norm > 1.))
            heading = calculate_head(vel, vel_norm, hist_head=hist_state[:, -2])[:, None]
            heading_d = derivative_of(heading, dt, radian=True)

            node_data_new = np.concatenate([pos, vel, acc, heading_v, heading, heading_d, vel_norm, acc_norm], axis=-1)
            new_data_dict[ts][node] = node_data_new
    return new_data_dict


def replace_node_data(new_data_dict, scene, ph):
    new_scenes = list()
    for ts, node_dict in new_data_dict.items():
        for node, new_data in node_dict.items():
            timesteps = new_data.shape[-2]

            first_time_step = ts - (timesteps - ph) + 1  # the first time step of new scene in the orig scene
            last_time_step = first_time_step + timesteps

            new_nodes = list()
            for node_i in scene.nodes:
                if node_i.id != node.id:  # surrounding agent, keep the orig traj and truncate
                    data_first_time_step = first_time_step - node.first_timestep  # the idx of data we want in node.data.data
                    data_last_time_step = last_time_step - node.first_timestep
                    node_i_new_data = node_i.data.data[max(data_first_time_step, 0):data_last_time_step]
                    if len(node_i_new_data) == 0: continue
                    node_i_new_data = DoubleHeaderNumpyArray(node_i_new_data, node_i.data.header)

                    node_i_first_step = 0 if data_first_time_step >= 0 else -data_first_time_step
                    new_node = Node(node_i.type, node_i.id, node_i_new_data,
                                    node_i.length, node_i.width, node_i.height,
                                    node_i_first_step, node_i.is_robot, node_i.description)

                    new_nodes.append(new_node)
                else:
                    node_i_first_step = 0
                    new_data = DoubleHeaderNumpyArray(new_data, node_i.data.header)
                    new_node = Node(node_i.type, node_i.id, new_data,
                                    node_i.length, node_i.width, node_i.height,
                                    node_i_first_step, node_i.is_robot, node_i.description)

                    new_nodes.append(new_node)
                    pseudo_node = new_node
            new_scene = Scene(timesteps, scene.map, scene.dt, scene.name)
            new_scene.nodes = new_nodes
            new_scene.temporal_scene_graph = scene.temporal_scene_graph
            new_scene.pseudo_node = pseudo_node
            new_scenes.append(new_scene)
    return new_scenes


def create_pseudo_scenes(histories_dict, prediction_dict, scene, ph):
    new_data_dict = create_new_node_data(histories_dict, prediction_dict, scene.dt)
    new_scenes_from_prediction = replace_node_data(new_data_dict, scene, ph)
    return new_scenes_from_prediction


def create_pseudo_dataloader(orig_env, new_scenes, hyperparams, args):
    new_env = Environment(orig_env.node_type_list, orig_env.standardization, new_scenes, orig_env.attention_radius, orig_env.robot_type)

    new_dataset = EnvironmentDataset(new_env,
                                     hyperparams['state'],
                                     hyperparams['pred_state'],
                                     scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                     node_freq_mult=hyperparams['node_freq_mult_train'],
                                     hyperparams=hyperparams,
                                     min_history_timesteps=hyperparams['minimum_history_length'],
                                     min_future_timesteps=hyperparams['prediction_horizon'],
                                     return_robot=not args.incl_robot_node,
                                     pseudo=True
                                     )

    vehicle_pseudo_dataloader = DataLoader(new_dataset.node_type_datasets[0],
                                           collate_fn=collate,
                                           pin_memory=False,
                                           batch_size=args.eval_batch_size,
                                           shuffle=False,
                                           num_workers=0)#args.preprocess_workers)

    return vehicle_pseudo_dataloader


def generate_random_pseudo(env, model, hyperparams, n_scene=5, n_t=5):
    selected_scenes = [env.scenes[i] for i in np.random.choice(len(env.scenes), n_scene, replace=False)]
    pseudo_scenes = list()
    with torch.no_grad():
        for i, scene in enumerate(selected_scenes):

            timesteps = np.random.choice(scene.timesteps, n_t, replace=False)
            predictions = model.predict(scene,
                                        timesteps,
                                        hyperparams['prediction_horizon'],
                                        num_samples=1,
                                        min_future_timesteps=8,
                                        z_mode=False,
                                        gmm_mode=False,
                                        full_dist=False)

            if not predictions:
                continue

            fetch_state = {'position': ['x', 'y'],
                           'velocity': ['x', 'y'],
                           'acceleration': ['x', 'y'],
                           'heading': ['°', 'd°']}

            prediction_dict, histories_dict, futures_dict = \
                prediction_output_to_trajectories(predictions,
                                                  scene.dt,
                                                  hyperparams['maximum_history_length'],
                                                  hyperparams['prediction_horizon'],
                                                  prune_ph_to_future=False,
                                                  fetch_state=fetch_state)

            sub_pseudo_scenes = create_pseudo_scenes(histories_dict, prediction_dict, scene, hyperparams['prediction_horizon'])
            pseudo_scenes += sub_pseudo_scenes
    new_data_loader = create_pseudo_dataloader(env, pseudo_scenes, hyperparams, args)
    return new_data_loader