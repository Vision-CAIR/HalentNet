import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.dataset import EnvironmentDataset, collate
from tensorboardX import SummaryWriter
torch.autograd.set_detect_anomaly(False)

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = torch.device('cpu')

# This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def fetch_batch(generator):
    try:
        batch = next(generator)
        not_empty = True
    except StopIteration:
        batch = None
        not_empty = False
    return batch, not_empty


def part2train(model_registrar, part='discriminator'):
    if part == 'discriminator':
        for para in model_registrar.get_all_but_name_match(['discri']).parameters():
            para.requires_grad = False
        for para in model_registrar.get_name_match(['discri']).parameters():
            para.requires_grad = True
    elif part == 'generator':
        for para in model_registrar.get_name_match(['discri']).parameters():
            para.requires_grad = False
        for para in model_registrar.get_all_but_name_match(['discri']).parameters():
            para.requires_grad = True
    else:
        raise KeyError


def main():
    log_writer = None

    with open(os.path.join(args.load_model, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    hyperparams['dynamic']['PEDESTRIAN']['distribution'] = True
    hyperparams['dynamic']['VEHICLE']['distribution'] = True
    hyperparams['pred_state']['VEHICLE'] = {'position': ['x', 'y']}
    hyperparams['pred_state']['PEDESTRIAN'] = {'position': ['x', 'y']}


    # Load training and evaluation environments and scenes
    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    hyperparams_map = hyperparams.copy()
    hyperparams_map['use_map_encoding'] = True

    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams_map,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=not args.incl_robot_node)
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.preprocess_workers)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")


    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams_map,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if args.eval_device is 'cpu' else True,
                                                         batch_size=args.eval_batch_size,
                                                         shuffle=False,
                                                         num_workers=args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_data_path}")

    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")

    # set model
    model_registrar = ModelRegistrar(args.load_model, args.device)
    model_registrar.load_models(args.checkpoint)
    trajectron = Trajectron(model_registrar, hyperparams, None, args.device)
    trajectron.pred_state = {'VEHICLE': trajectron.pred_state['VEHICLE']}
    model_registrar = trajectron.model_registrar
    creative_mode = '_'.join([args.creative_t, args.creative_loss])
    save_path = os.path.join(args.log_dir, '_'.join([args.log_tag, creative_mode, 'lambda', str(args.creative_lambda),
                                                     time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())]))
    model_registrar.model_dir = save_path  # use a different path to save model
    if not args.debug:
        log_writer = SummaryWriter(log_dir=save_path)
        trajectron.set_log_writer(log_writer)
    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()
    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.pred_state = {'VEHICLE': eval_trajectron.pred_state['VEHICLE']}
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')

    optimizer = dict()
    optimizer_d = dict()
    lr_scheduler = dict()
    lr_scheduler_d = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match(['map_encoder', 'discrim']).parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], lr=hyperparams['learning_rate'])
        optimizer_d[node_type] = optim.Adam(model_registrar.get_name_match('discrim').parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
            lr_scheduler_d[node_type] = optim.lr_scheduler.ExponentialLR(optimizer_d[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])
            lr_scheduler_d[node_type] = optim.lr_scheduler.ExponentialLR(optimizer_d[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])


    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('| g_factor: %s' % args.g_factor)
    print('| grid_std: %s' % args.grid_std)
    print('| grid_max: %s' % args.grid_max)
    print('-----------------------')


    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    for epoch in range(args.checkpoint+1, args.train_epochs + 1):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment

        #################################
        #         TRAINING GAN          #
        #################################
        if True: #epoch != args.checkpoint+1:
            node_type = 'VEHICLE'
            data_loader = train_data_loader[node_type]
            curr_iter = curr_iter_node_type[node_type]

            pbar = tqdm(data_loader, ncols=150)
            pbar_gen = iter(pbar)

            d_loss = g_loss = train_loss = torch.zeros(1)
            not_empty = True
            while not_empty:
                trajectron.set_curr_iter(curr_iter)
                trajectron.step_annealers(node_type)

                # ---------- discriminator ----------
                optimizer_d[node_type].zero_grad()
                part2train(model_registrar, 'discriminator')

                n_d_steps = 1
                for _ in range(n_d_steps):
                    batch, not_empty = fetch_batch(pbar_gen)
                    if not not_empty: break

                    d_loss, dc_loss, d_real, d_fake = trajectron.gan_d_loss(batch, node_type, grid_std=args.grid_std, grid_max=args.grid_max)
                    (d_loss + dc_loss).backward()
                    optimizer_d[node_type].step()

                # ---------- generator ----------
                optimizer[node_type].zero_grad()
                part2train(model_registrar, 'generator')

                n_g_steps = 1
                for _ in range(n_g_steps):
                    batch, not_empty = fetch_batch(pbar_gen)
                    if not not_empty: break

                    g_loss, c_loss = trajectron.gan_g_loss(batch, node_type, grid_std=args.grid_std,
                                                           grid_max=args.grid_max)
                    (1/2 * args.g_factor * (g_loss + c_loss)).backward()
                    g_loss, creative_loss = trajectron.gan_g_loss(batch, node_type, grid_std=args.grid_std,
                                                                  grid_max=args.grid_max, creative=creative_mode)
                    (1/2 * args.g_factor * (g_loss + args.creative_lambda * creative_loss)).backward()
                    optimizer[node_type].step()

                # ---------- supervision ----------
                optimizer[node_type].zero_grad()
                part2train(model_registrar, 'generator')

                n_s_steps = 1
                for _ in range(n_s_steps):
                    batch, gen_not_empty = fetch_batch(pbar_gen)
                    if not not_empty: break
                    train_loss = trajectron.train_loss(batch, node_type)
                    train_loss.backward()
                    optimizer[node_type].step()

                f'TL: {train_loss.item():.2f}  DL: {d_loss.item():.2f} GL: {g_loss.item():.2f}'
                pbar.set_description(f"Epoch {epoch} TL: {train_loss.item():.2f}  DL: {d_loss.item():.2f} GL: {g_loss.item():.2f}")

                # Clipping gradients.
                if hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                lr_scheduler[node_type].step()
                lr_scheduler_d[node_type].step()

                if not args.debug:
                    log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                          lr_scheduler[node_type].get_last_lr()[0],
                                          curr_iter)
                    if n_d_steps > 0: log_writer.add_scalar(f"{node_type}/train/gan_d_loss", d_loss, curr_iter)
                    if n_g_steps > 0: log_writer.add_scalar(f"{node_type}/train/gan_g_loss", g_loss, curr_iter)
                    if n_s_steps > 0: log_writer.add_scalar(f"{node_type}/train/sup_loss", train_loss, curr_iter)

                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter

        train_dataset.augment = False
        if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        if args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict random timestep to plot for train data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = trajectron.predict(scene,
                                                 timestep,
                                                 ph,
                                                 z_mode=True,
                                                 gmm_mode=True,
                                                 all_z_sep=False,
                                                 full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/prediction', fig, epoch)

                model_registrar.to(args.eval_device)
                # Predict random timestep to plot for eval data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(eval_scenes, p=eval_scenes_sample_probs)
                else:
                    scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      num_samples=20,
                                                      min_future_timesteps=ph,
                                                      z_mode=False,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction', fig, epoch)

                # Predict random timestep to plot for eval data set
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      min_future_timesteps=ph,
                                                      z_mode=True,
                                                      gmm_mode=True,
                                                      all_z_sep=True,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    if node_type == 'PEDESTRIAN': continue
                    eval_loss = []
                    print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(batch, node_type)
                        pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                        eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
                        del batch

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=50,
                                                          min_future_timesteps=ph,
                                                          full_dist=False)

                    eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                 scene.dt,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 node_type_enum=eval_env.NodeType[0:1],
                                                                                 map=scene.map))

                evaluation.log_batch_errors(eval_batch_errors,
                                            log_writer,
                                            'eval',
                                            epoch,
                                            bar_plot=['kde'],
                                            box_plot=['ade', 'fde'])

                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                eval_batch_errors_ml = []
                for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                    timesteps = scene.sample_timesteps(scene.timesteps)

                    predictions = eval_trajectron.predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=1,
                                                          min_future_timesteps=ph,
                                                          z_mode=True,
                                                          gmm_mode=True,
                                                          full_dist=False)

                    eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                                                                                    scene.dt,
                                                                                    max_hl=max_hl,
                                                                                    ph=ph,
                                                                                    map=scene.map,
                                                                                    node_type_enum=eval_env.NodeType[0:1],
                                                                                    kde=False))

                evaluation.log_batch_errors(eval_batch_errors_ml,
                                            log_writer,
                                            'eval/ml',
                                            epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch)


if __name__ == '__main__':
    main()
