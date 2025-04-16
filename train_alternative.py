import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

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

def parse_args(hardcode=None):
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument("--config", default="config.yml", type=str, help="Path to configuration file")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the model from (default: None)")
    parser.add_argument("--resume", type=str, default=False, help="If preloading a dataset, resume training on a specific track.")
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
    
    # create the environment
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
    
    # configure the training config
    # initialise here to make changes if args.resume option 
    training_config = train_conf.train_config
    
    if not args.test and config.load_offline_data:
        data_config_file = os.path.abspath(r"./ac_offline_train_paths.yml")
        logger.info("Loading offline dataset...")
        assert config.dataset_path, "dataset_path not set in config"
        dataset_path = Path(config.dataset_path + os.sep)

        # load data set
        data = data_loader.read_yml(data_config_file)

        for track in data:
            for car in data[track]:
                paths = data[track][car]
                paths = [dataset_path / Path(f"{track}/{car}") / p["id"] / "laps" for p in paths]
                env_load_config = copy.deepcopy(config)
                env_load_config.AssettoCorsa.track = track
                env_load_config.AssettoCorsa.car = car
                env_load = assettoCorsa.make_ac_env(cfg=env_load_config, work_dir=work_dir)
                for laps_path in paths:
                    assert laps_path.exists(), f"{laps_path} not found"
                    agent.load_pre_train_data(laps_path.as_posix(), env_load)

        if config.Agent.use_offline_buffer:
            agent._replay_buffer.online(True)
    
    if config.pre_train:
        agent.pre_train()

    # load an agent
    if args.load_path is not None:
        load_buffer = False if args.test else True
        agent.load(args.load_path, load_buffer=load_buffer)
        
        if args.resume:
            # check if track is within train-config
                        
            for track in training_config:
                track_exists = track.track == args.resume

                if track_exists:
                    logger.info(f"Resuming training on track {args.resume}")
                    # modify training config 
                    # to only include tracks that havent resumed
                    idx = training_config.index(track)
                    training_config = training_config[idx:]
                    
                    # update the number of steps when resuming
                    steps = sum([track.steps for track in training_config])
                    agent.update_steps(steps)
                    break # exit the loop as track found
                    
            if not track_exists:
                logger.info(f"Track {args.resume} does not exist within the training config. Resuming training on default track")

    if config.enable_notifications: 
            notification_client.send_notifcation("Starting agent training...", "AGENT TRAINING")
                            
    for track in training_config:
        # update the current config to reflect first training cycle
        config.AssettoCorsa.track = track.track
        
        # change the track in assetto corsa
        gui.change_track(track.ac_track)
        
        time.sleep(2) # small delay incase of lag
        
        # start assetto corsa
        gui.launch_ac()
        time.sleep(10) # wait until the game loads
        
        if config.enable_notifications:
            notification_client.send_notifcation(f"Loaded track {track.track}. Starting training...", "AGENT TRAINING")
        
        gui.start_game()
        time.sleep(5)    
        
        # initialise the environment
        env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
        # update the agents environment
        agent.change_environment(env)

        time.sleep(2)     
      
        # start training
        agent.run_without_save(track.steps) 
        
        if config.enable_notifications:
            notification_client.send_notifcation(f"Training complete on {track.track}.", "AGENT TRAINING")
            
        gui.close_ac() 
        time.sleep(5) # wait for all operations to complete       
    
    # once all tracks have been run, save the model
    agent.save_final()
    if config.enable_notifications:
        notification_client.send_notifcation(f"Training complete.", "AGENT TRAINING")
            
    
        
if __name__ == "__main__":
    main()        
        
        

        
        
     
