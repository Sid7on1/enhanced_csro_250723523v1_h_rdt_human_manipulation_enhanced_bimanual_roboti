import os
import logging
from typing import Dict, List
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class
class Config(ABC):
    """
    Base configuration class.

    Attributes:
        log_level (int): Logging level.
        device (str): Device to use for tensor operations - 'cpu' or 'cuda'.
        seed (int): Random seed for reproducibility.
        ...

    Methods:
        init_logging(self): Initialize logging with the configured log level.
        to_dict(self): Return the configuration as a dictionary.
        ...

    Subclasses must implement the following methods:
        parse_args(self): Parse command-line arguments and update the config.
        load_config(self, file_path): Load configuration from a file.
        ...

    """

    def __init__(self):
        """Initialize configuration with default values."""
        self.log_level = logging.INFO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        self.dataset_path = ''
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.hidden_size = 256
        self.output_size = 1
        self.num_layers = 2
        self.dropout = 0.2
        self.weight_decay = 0.0001
        self.gradient_clipping = 1.0
        self._parse_args()
        self.init_logging()

    def __str__(self):
        """Return a string representation of the configuration."""
        config_dict = self.to_dict()
        return str(config_dict)

    def init_logging(self):
        """Initialize logging with the configured log level."""
        logging.basicConfig(level=self.log_level)
        logger.info("Logging initialized.")

    def to_dict(self) -> Dict:
        """Return the configuration as a dictionary."""
        config_dict = {
            'log_level': self.log_level,
            'device': self.device,
            'seed': self.seed,
            'dataset_path': self.dataset_path,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            'gradient_clipping': self.gradient_clipping
        }
        return config_dict

    @abstractmethod
    def parse_args(self):
        """Parse command-line arguments and update the configuration."""
        pass

    @abstractmethod
    def load_config(self, file_path: str):
        """Load configuration from a file."""
        pass

# Agent-specific configuration class
class AgentConfig(Config):
    """
    Configuration class for the agent.

    Attributes:
        ...

    Methods:
        ...

    """

    def __init__(self):
        """Initialize agent-specific configuration."""
        super().__init__()
        self.num_agents = 5
        self.agent_learning_rate = 0.0001
        self.policy_update_freq = 5
        self.experience_replay_size = 10000
        self.gamma = 0.99
        self.tau = 0.001
        self.policy_hidden_size = 128

    def to_dict(self) -> Dict:
        """Return the agent configuration as a dictionary."""
        config_dict = super().to_dict()
        config_dict.update({
            'num_agents': self.num_agents,
            'agent_learning_rate': self.agent_learning_rate,
            'policy_update_freq': self.policy_update_freq,
            'experience_replay_size': self.experience_replay_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_hidden_size': self.policy_hidden_size
        })
        return config_dict

    def parse_args(self):
        """
        Parse command-line arguments and update the configuration.

        Example usage:
            python train.py --batch_size 64 --learning_rate 0.0005

        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--learning_rate', type=float, default=self.learning_rate)
        parser.add_argument('--num_epochs', type=int, default=self.num_epochs)
        parser.add_argument('--hidden_size', type=int, default=self.hidden_size)
        parser.add_argument('--output_size', type=int, default=self.output_size)
        parser.add_argument('--num_layers', type=int, default=self.num_layers)
        parser.add_argument('--dropout', type=float, default=self.dropout)
        parser.add_argument('--weight_decay', type=float, default=self.weight_decay)
        parser.add_argument('--gradient_clipping', type=float, default=self.gradient_clipping)
        parser.add_argument('--num_agents', type=int, default=self.num_agents)
        parser.add_argument('--agent_learning_rate', type=float, default=self.agent_learning_rate)
        parser.add_argument('--policy_update_freq', type=int, default=self.policy_update_freq)
        parser.add_argument('--experience_replay_size', type=int, default=self.experience_replay_size)
        parser.add_argument('--gamma', type=float, default=self.gamma)
        parser.add_argument('--tau', type=float, default=self.tau)
        parser.add_argument('--policy_hidden_size', type=int, default=self.policy_hidden_size)
        parser.add_argument('--dataset_path', type=str, default=self.dataset_path)
        parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
        args = parser.parse_args()

        # Update configuration with parsed arguments
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.gradient_clipping = args.gradient_clipping
        self.num_agents = args.num_agents
        self.agent_learning_rate = args.agent_learning_rate
        self.policy_update_freq = args.policy_update_freq
        self.experience_replay_size = args.experience_replay_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_hidden_size = args.policy_hidden_size
        self.dataset_path = args.dataset_path

        # Set logging level
        self.log_level = getattr(logging, args.log_level)
        self.init_logging()

    def load_config(self, file_path: str):
        """Load configuration from a JSON or YAML file."""
        import yaml
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        self._update_config(data)

    def _update_config(self, data: Dict):
        """Update configuration with the provided dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.init_logging()

# Environment-specific configuration class
class EnvironmentConfig(Config):
    """
    Configuration class for the environment.

    Attributes:
        state_size (int): Dimension of the state space.
        action_size (int): Dimension of the action space.
        ...

    Methods:
        ...

    """

    def __init__(self):
        """Initialize environment-specific configuration."""
        super().__init__()
        self.state_size = 8
        self.action_size = 4
        self.max_velocity = 1.0
        self.max_angular_velocity = np.pi
        self.time_step = 0.1

    def to_dict(self) -> Dict:
        """Return the environment configuration as a dictionary."""
        config_dict = super().to_dict()
        config_dict.update({
            'state_size': self.state_size,
            'action_size': self.action_size,
            'max_velocity': self.max_velocity,
            'max_angular_velocity': self.max_angular_velocity,
            'time_step': self.time_step
        })
        return config_dict

    def parse_args(self):
        """
        Parse command-line arguments and update the environment configuration.

        Example usage:
            python train.py --state_size 16 --action_size 6 ...

        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--state_size', type=int, default=self.state_size)
        parser.add_argument('--action_size', type=int, default=self.action_size)
        parser.add_argument('--max_velocity', type=float, default=self.max_velocity)
        parser.add_argument('--max_angular_velocity', type=float, default=self.max_angular_velocity)
        parser.add_argument('--time_step', type=float, default=self.time_step)
        parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
        args = parser.parse_args()

        # Update configuration with parsed arguments
        self.state_size = args.state_size
        self.action_size = args.action_size
        self.max_velocity = args.max_velocity
        self.max_angular_velocity = args.max_angular_velocity
        self.time_step = args.time_step

        # Set logging level
        self.log_level = getattr(logging, args.log_level)
        self.init_logging()

    def load_config(self, file_path: str):
        """Load configuration from a JSON or YAML file."""
        import yaml
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        self._update_config(data)

    def _update_config(self, data: Dict):
        """Update configuration with the provided dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.init_logging()

# Function to get the configuration object
def get_config(config_type: str) -> Config:
    """
    Factory function to get the appropriate configuration object.

    Args:
        config_type (str): Type of configuration - 'agent' or 'environment'.

    Returns:
        Config object of the specified type.

    """
    if config_type == 'agent':
        return AgentConfig()
    elif config_type == 'environment':
        return EnvironmentConfig()
    else:
        raise ValueError(f"Invalid configuration type: {config_type}")

# Example usage
if __name__ == '__main__':
    agent_config = get_config('agent')
    environment_config = get_config('environment')

    # Print configurations
    logger.info("Agent Configuration:")
    logger.info(agent_config)
    logger.info("Environment Configuration:")
    logger.info(environment_config)