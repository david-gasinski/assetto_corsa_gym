import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import shlex

import copy
from omegaconf import OmegaConf
import torch

# Add paths
sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])

# Custom module imports
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
import AssettoCorsaEnv.data_loader as data_loader
from discor.algorithm import SAC, DisCor
from discor.agent import Agent
import common.misc as misc
import common.notifcation as notification
import common.logging_config as logging_config
import common.gui.content_manager as ac_gui
from common.logger import Logger

import time


logger = logging.getLogger(__name__)

load_dotenv()

def launch_ac():
    subprocess.Popen()

def parse_args(hardcode=None):
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument("--config", default="config.yml", type=str, help="Path to configuration file")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the model from (default: None)")
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm type (default: sac)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Any key=value arguments to override config values")
    parser.add_argument("--train-config", default="train-config.yml", type=str, help="Training configuration for assetto corsa, allows the use of multiple tracks")
    
    if hardcode is not None:
        args = parser.parse_args(hardcode.split())
    else:
        args = parser.parse_args()
    args.load_path = os.path.abspath(args.load_path) + os.sep if args.load_path is not None else None
    return args

def main():
    """
        Currently no support for pre training
        This is for testing purposes, will be updated at a later date
        
        To ensure the switching works properly, the current track selected in AC must match 
        the track in config.yml (NOT train-config.yml)
    """
    
    args = parse_args()
    
    config = OmegaConf.load(args.config)
    
    # Apply command line overrides
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(config, cli_conf)
    
    if config.work_dir is not None:
        work_dir = os.path.abspath(args.work_dir) + os.sep + config.track + os.sep + config.car + os.sep
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
        work_dir = os.path.abspath(work_dir) + os.sep
        os.makedirs(work_dir, exist_ok=True)
    config.work_dir = work_dir

    logging_config.create_logging(level=logging.DEBUG, file_name=work_dir + "log.log")
    logging.getLogger().setLevel(logging.INFO)
    
    # enable notifications
    if config.enable_notifications:
        notification_client = notification.NotifcationClient(
            os.getenv("PUSHOVER_APP_KEY"), os.getenv("PUSHOVER_USER_KEY")
        )
        
    # gui manipulation
    gui = ac_gui.ContentManagerGUI()
        
    # log system and git info
    misc.get_system_info()
    misc.get_git_commit_info()

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    logger.info("work_dir: " + work_dir)
    
    # load training config
    train_conf = OmegaConf.load(args.train_config)
    
     # Device to use
    device = torch.device("cuda")
    assert device.type == "cuda", "Only cuda is supported"
    
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)

    if args.algo == 'discor':
        algo = DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            **OmegaConf.to_container(config.SAC), **OmegaConf.to_container(config.DisCor))
    elif args.algo == 'sac':
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=config.seed,
            **OmegaConf.to_container(config.SAC))
    else:
        raise Exception('You need to set algo sac or discor')

    # Update the logger configuration with dynamic values
    config.exp_name = f'{config.AssettoCorsa.car}-{config.AssettoCorsa.track}'
    config.action_dim = env.action_dim
    config.steps = config.Agent.num_steps

    # Initialize wandb logger
    if not config.disable_wandb:
        wandb_logger = Logger(config.copy())
    else:
        wandb_logger = None
        
    agent = Agent(env=env, test_env=env, algo=algo, log_dir=config.work_dir,
                  device=device, seed=config.seed, **config.Agent, wandb_logger=wandb_logger)
    
    if config.enable_notifications: 
            notification_client.send_notifcation("Starting agent training...", "AGENT TRAINING")
        
    for track in train_conf.train_config:
        # update the current config to reflect first training cycle
        config.AssettoCorsa.track = track.track
        
        # initialise the environment
        env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
        # update the agents environment
        agent.change_environment(env)
        
        # change the track in assetto corsa
        gui.change_track(track.ac_track)
        
        # start assetto corsa
        gui.launch_ac()
        time.sleep(10) # wait until the game loads
        
        if config.enable_notifications:
            notification_client.send_notifcation(f"Loaded track {track.track}. Starting training...")
        
        gui.start_game()
        time.sleep(5)    
        
        # start training
        agent.run_without_save() # change
        
        if config.enable_notifications:
            notification_client.send_notifcation(f"Training complete on {track.track}.")
            
        gui.close_ac() 
        time.sleep(5) # wait for all operations to complete       
    
    # once all tracks have been run, save the model
    agent.save_final()
    
        
if __name__ == "__main__":
    main()        
        
        

        
        
     
