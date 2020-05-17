from typing import NamedTuple

import yaml


class TrainConfig(NamedTuple):
    """
    model_type: Model type name in nsp.models ex) "TransformerModel"
    model_configs: Model configs for construct model. ex) {"vocab_size": 32, "d_model":128, ...}
                   This is used as model init arguments.

    data_loader_path: saved DataLoader(Datasets) file path.

    epoch: training epoch number.
    batch_size: batch size.
    val_batch_size: batch size for evaluating.
    learning_rate: learning rate for training.
    learning_rate_min: minimum learning rate.
    optimizer: optimizer name which is in the torch.optim ex) "SGD" or "Adam", etc
    steps_per_log: step number for logging. At every that step number, log current status.
    steps_per_eval: step number for evaluating. At every that step number, evaluating and log result.

    model_save_dir: directory path to save model.
    model_save_prefix: model file name prefix when saving model.

    optimizer_args: optional optimizer parameters (default: None)
    """

    # Configs for model
    model_type: str
    model_configs: dict

    # Configs for data
    data_loader_path: str

    # Configs for training
    epoch: int
    batch_size: int
    val_batch_size: int
    learning_rate: float
    learning_rate_min: float
    optimizer: str
    steps_per_log: int
    steps_per_eval: int

    model_save_dir: str
    model_save_prefix: str

    optimizer_args: dict = {}

    @classmethod
    def load_from_json(cls, config_path, **kwargs):
        """
        Load config from the config file.
        :param config_path: (str) yaml format config file.
        :param **kwargs: parameters to override json setting.
        """
        with open(config_path) as f:
            config = yaml.load(f, yaml.FullLoader)

        config.update(kwargs)
        return cls(**config)


class InferConfig(NamedTuple):
    """
    model_type: Model type name in nsp.models ex) "TransformerModel"
    model_path: The model file path that infer texts.
    vocab_path: Vocab file path to numericalize tokens.
    val_batch_size: batch size for inference.
    """

    model_type: str
    model_path: str
    vocab_path: str
    val_batch_size: int

    @classmethod
    def load_from_json(cls, config_path, **kwargs):
        """
        Load config from the config file.
        :param config_path: (str) yaml format config file.
        :param **kwargs: parameters to override json setting.
        """
        with open(config_path) as f:
            config = yaml.load(f, yaml.FullLoader)

        config.update(kwargs)
        return cls(**config)
