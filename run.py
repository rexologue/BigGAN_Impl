import os
import copy
import yaml
import torch
import functools
from tqdm import tqdm
from argparse import ArgumentParser

# Import my stuff
import utils
from ema import EMA
from fid import FID
import train_utils as tu
from data_utils import get_loader
from biggan import Generator, Discriminator, GAN


def run(path_to_config):
  # Read config
  with open(path_to_config, 'r') as f:
    config = yaml.safe_load(f)

  ######################
  # BASE CONFIGURATION #
  ######################

  device = config['device']
  exp_state_dict_path = os.path.join(config['experiment_root'], 'exp_state_dict.pth')

  if config['mode'] == 'train':
    config['G']['skip_init'] = False
    config['G']['no_optim'] = False
    config['D']['skip_init'] = False
    config['D']['no_optim'] = False
  else:
    config['G']['skip_init'] = True
    config['G']['no_optim'] = False
    config['D']['skip_init'] = True
    config['D']['no_optim'] = False
  
  # Seed RNG
  utils.seed_rng(config['seed'])
  # Prepare root folders if necessary
  config = utils.prepare_root(config)
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Prepare experiment state dict, which holds training stats
  exp_state_dict = {'itr': 0, 'epoch': 0, 'best_FID': 999999, 'saved_ckps': []}
  # Prepare metrics logger (if we fine tune we shouldn't reinitialize)
  logger = utils.MetricsLogger(os.path.join(config['logs_root'], "metrics.json"), reinitialize=(config['mode'] == 'train'))

  ####################
  # BUILDINGS MODELS #
  ####################

  G = Generator(
    resolution=config['resolution'], 
    n_classes=config['n_classes'], 
    **config['G']
  ).to(device)
  
  D = Discriminator(
    resolution=config['resolution'], 
    n_classes=config['n_classes'], 
    **config['D']
  ).to(device)
  
  gan = GAN(G, D)
  
  ####################################
  # EXPONENTIAL MOVING AVERAGE MODEL #
  ####################################
  if config.get('ema'):
    print(f"Preparing EMA for G with decay of {config['ema']['decay']}")

    G_ema_config = copy.deepcopy(config['G'])

    G_ema_config['skip_init'] = True
    G_ema_config['no_optim'] = True

    G_ema = Generator(
      resolution=config['resolution'], 
      n_classes=config['n_classes'], 
      **G_ema_config
    ).to(device)
    
    G_ema.eval()
    
    ema = EMA(G, G_ema, config['ema']['decay'], config['ema']['start_itr'])

  else:
    G_ema = None

  G_params = utils.count_parameters(G)
  D_params = utils.count_parameters(D)
  
  print(f"Number of params in G: {G_params} and D: {D_params}")

  # If loading from a pre-trained model, load weights
  if config['mode'] == 'fine-tune':
    print('Loading weights...')
    utils.load_weights(G, G_ema, D, config['pretrained_path'])

    if os.path.exists(exp_state_dict_path):
      exp_state_dict = torch.load(exp_state_dict_path)

  ###################
  # DATA PROCESSING #
  ###################
  
  loader = get_loader(
    config, 
    batch_size=config['train']['batch_size'],
    start_itr=exp_state_dict['itr']
  )

  # Prepare noise and randomly sampled label generators
  z_, y_ = utils.prepare_z_y(
    batch_size=config['train']['batch_size'], 
    dim_z=G.dim_z, 
    num_classes=config['n_classes'],
    device=device
  )

  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z = z_.sample()[:config['train']['sample_batch_size']]
  fixed_y = y_.sample()[:config['train']['sample_batch_size']]

  ##################
  # FID PREPARINGS #
  ##################
  sample_function = functools.partial(
    utils.sample,
    G=(G_ema if G_ema is not None else G),
    z_=z_, 
    y_=y_
  )

  fid = FID(config, sample_function)

  ##############
  # TRAIN LOOP #
  ##############

  print("Start training")

  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(exp_state_dict['epoch'], config['train']['num_epochs']): 

    pbar = tqdm(loader, total=len(loader), initial=exp_state_dict['itr'])
    for i, (x, y) in enumerate(pbar):
      # Increment the iteration counter
      exp_state_dict['itr'] += 1

      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()

      x, y = x.to(device), y.to(device)

      # Make a train step
      metrics = tu.train(G, D, gan, x, y, z_, y_, config)

      if G_ema is not None:
        ema.update(exp_state_dict['itr'])

      # Every sv_log_interval, log singular values
      if not (exp_state_dict['itr'] % config['train']['sv_log_interval']):
        sv_dict = {**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')}
      else:
        sv_dict = {}

      if not (exp_state_dict['itr'] % config['train']['sample_itr']):
        tu.sample(
          G=(G_ema if G_ema is not None else G),
          fixed_z=fixed_z, 
          fixed_y=fixed_y,
          exp_state_dict=exp_state_dict, 
          config=config
        )

      # Save weights and copies as configured at specified interval
      if not (exp_state_dict['itr'] % config['train']['save_itr']):
        print(f"Making checkpoint at {exp_state_dict['itr']} iteration")

        G.eval()
        D.eval()

        tu.checkpoint(
          G=G, 
          G_ema=G_ema, 
          D=D, 
          exp_state_dict=exp_state_dict, 
          config=config
        )

        torch.save(exp_state_dict, exp_state_dict_path)

      # Test every specified interval
      if not (exp_state_dict['itr'] % config['train']['eval_itr']):
        print(f"Start validation at {exp_state_dict['itr']} iteration")

        G.eval()
        D.eval()

        fid_value = tu.validate(
          G=G, 
          G_ema=G_ema, 
          D=D, 
          exp_state_dict=exp_state_dict, 
          config=config, 
          fid_computer=fid
        )

      else:
        fid_value = None

      logger.log(itr=exp_state_dict['itr'], fid=fid_value, **metrics, **sv_dict)
        
    # Increment epoch counter at end of epoch
    exp_state_dict['epoch'] += 1


def main():
  parser = ArgumentParser()
  parser.add_argument('--cfg_path', type=str,
    help='Path to configuration file')
  args = parser.parse_args()
  run(args.cfg_path)

if __name__ == '__main__':
  main()