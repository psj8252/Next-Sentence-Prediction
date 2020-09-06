import logging
import os
import os.path as path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchtext.data import Dataset, Example, Iterator

from .config import InferConfig, TrainConfig
from .data_loader import DataLoader, Fields, Vocab
from .models import TransformerModel


class BaseManager:
    def _set_logger(self, file_name):
        """
        Set logger that logging to std.out and file named 'file_name'.
        :param file_name: File path to save log.
        """
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)


class TrainManager(BaseManager):
    def __init__(self, training_config_path, device):
        """
        :param training_config_path: (str) training config file path.
        :param device: (str, torch.device) device for training.
        """
        # Load training Config
        self.config = TrainConfig.load_from_json(training_config_path)

        # Make Output Directory
        self.checkpoint_dir = path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set loogger
        self._set_logger(path.join(self.config.output_dir, "train.log"))
        self.logger.info("Setting logger is complete")

        # Log Configures
        self.logger.info("===== Configures =====")
        for param, value in self.config._asdict().items():
            self.logger.info(f"{param:20} {value}")

        # Set device
        self.device = torch.device(device)
        self.logger.info(f"Setting device:{self.device} is complete")

        # Load Dataset
        self.data_loader = DataLoader.load(self.config.data_loader_path)
        self.train_dataset, self.test_dataset = self.data_loader.train_dataset, self.data_loader.test_dataset
        self.logger.info("Loaded training, test datasets")

        # Create Model
        self.model = TransformerModel(**self.config.model_configs)
        self.model.to(self.device)
        self.logger.info(f"Prepared model type: {type(self.model)}")

    def train(self):
        """
        Start training.
        """
        self.logger.info("------------- Training Start -------------")

        # Set optimizer
        optimizer = getattr(optim, self.config.optimizer)(
            self.model.parameters(), lr=self.config.learning_rate, **self.config.optimizer_args
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epoch * len(self.train_dataset) // self.config.batch_size,
            eta_min=self.config.learning_rate_min,
        )

        for epoch in range(1, self.config.epoch + 1):

            # Shape
            # batch.context, batch.query, batch.reply: (Sequence x BatchSize)
            batches = Iterator(self.train_dataset, batch_size=self.config.batch_size, device=self.device, train=True)

            # Training
            loss_sum = 0.0
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            total_step = int(len(self.train_dataset) / self.config.batch_size + 1)
            for step_num, batch in enumerate(batches):
                self.model.train()
                optimizer.zero_grad()

                # Shape
                # output: (BatchSize)
                output = self.model(batch.context, batch.query, batch.reply)

                loss = self.model.criterion(output)
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_sum += loss.item()

                _TP, _FP, _TN, _FN = self.model.metrics_parts(output)
                TP += _TP
                FP += _FP
                TN += _TN
                FN += _FN

                # logging step
                if (step_num + 1) % self.config.steps_per_log == 0:
                    accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1 = self.get_metrics(
                        TP, FP, TN, FN
                    )
                    self.logger.info(
                        f"{epoch:4} epoches {step_num + 1:6} / {total_step:-6} steps, "
                        f"lr: {optimizer.param_groups[0]['lr']:5.3}, "
                        f"average loss: {loss_sum / (self.config.steps_per_log):8.5}, "
                        + self.metrics_to_log(accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1)
                    )
                    loss_sum = 0.0
                    TP = 0
                    FP = 0
                    TN = 0
                    FN = 0

                # evaluation step
                if (step_num + 1) % self.config.steps_per_eval == 0:
                    # Evaluate
                    eval_loss, accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1 = self._evaluate()
                    self.logger.info(
                        f"{epoch:4} epoches {step_num + 1:6} / {total_step:-6} steps, "
                        f"evlaute loss: {eval_loss:8.5}, "
                        + self.metrics_to_log(accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1)
                    )

                    self.model.save(
                        path.join(
                            self.checkpoint_dir,
                            f"{self.model._get_name()}_{epoch}epoch_{step_num + 1}steps_f1-{f1:.4f}.pth",
                        )
                    )

            # Evaluate
            eval_loss, accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1 = self._evaluate()
            self.logger.info(
                f"{epoch:4} epoches {step_num + 1:6}, "
                f"evaluate loss: {eval_loss:8.5}, "
                + self.metrics_to_log(accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1)
            )

            self.model.save(
                path.join(
                    self.checkpoint_dir, f"{self.model._get_name()}_{epoch}epoch_{step_num + 1}steps_f1-{f1:.4f}.pth",
                )
            )

    def metrics_to_log(self, accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1):
        logstring = (
            f"Acc.: {accuracy:7.4}, "
            f"Prec.: {precision:7.4}, "
            f"Recl.: {recall:7.4}, "
            f"F1: {f1:7.4}, "
            f"NegPrec.: {neg_precision:7.4}, "
            f"NegRecl.: {neg_recall:7.4}, "
            f"NegF1: {neg_f1:7.4}"
        )

        return logstring

    def _evaluate(self):
        """
        Evaluate current model using test dataset.
        """
        self.model.eval()

        with torch.no_grad():
            batches = Iterator(
                self.test_dataset,
                batch_size=self.config.val_batch_size,
                device=self.device,
                sort_key=lambda x: len(x.query),
                train=False,
            )

            loss_sum = 0.0
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for batch in batches:
                output = self.model(batch.context, batch.query, batch.reply)

                # Calculate loss
                loss = self.model.criterion(output)
                loss_sum += loss.item() * len(batch)

                # Calculate metrics
                _TP, _FP, _TN, _FN = self.model.metrics_parts(output)
                TP += _TP
                FP += _FP
                TN += _TN
                FN += _FN

        accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1 = self.get_metrics(TP, FP, TN, FN)

        return loss_sum / len(self.test_dataset), accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1

    def get_metrics(self, TP, FP, TN, FN):
        T = TP + TN
        F = FP + FN
        P = TP + FP
        N = TN + FN
        TOTAL = T + F

        accuracy = T / TOTAL

        precision = TP / P if P else 0.0
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall) if precision or recall else 0.0

        neg_precision = TN / N if N else 0.0
        neg_recall = TN / (TN + FP)
        neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if neg_precision or neg_recall else 0.0

        return (accuracy, precision, recall, f1, neg_precision, neg_recall, neg_f1)


class InferManager(BaseManager):
    def __init__(self, inference_config_path, tokenize=None, device="cpu"):
        """
        :param inference_config_path: (str) inference config file path.
        :param tokenize: (func) tokenizing function. (str) -> (list) of (str) tokens.
        :prarm device: (str, torch.device) device for inference.
        """
        # Set loogger
        self._set_logger(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_infer.log")
        self.logger.info("Setting logger is complete")

        # Set device
        self.device = torch.device(device)
        self.logger.info(f"Setting device:{self.device} is complete")

        # Load training Config
        self.config = InferConfig.load_from_json(inference_config_path)
        self.logger.info(f"Loaded inference config from '{inference_config_path}'")

        # Load Model
        self.model = TransformerModel.load(self.config.model_path)
        self.model.to(self.device)
        self.logger.info(f"Prepared model type: {type(self.model)}")

        # Load vocab
        self.vocab = Vocab.load(self.config.vocab_path)
        self.logger.info(f"Set vocab from '{self.config.vocab_path}'")

        # Set fields
        self.fields = Fields(vocab_path=self.config.vocab_path, tokenize=tokenize)
        self.logger.info(f"Set fields tokenize with '{self.fields.utterance_field.tokenize}'")

    def inference_texts(self, inputs, threshold=None):
        """
        :param inputs: (list of tuple(context, query, reply)) list of (context, query, reply) to inferece.
        :param threshold: (int) if prediction value greater or equal this value, it is considered as true.
        :return: (list) of (int) labels about each text.
        """
        if threshold is None:
            threshold = self.model.cossim_threshold

        self.model.eval()
        with torch.no_grad():
            # Make inference batches
            dataset = self._list_to_dataset(inputs, self.fields.utterance_field)
            batches = Iterator(
                dataset,
                batch_size=self.config.val_batch_size,
                device=self.device,
                train=False,
                shuffle=False,
                sort=False,
            )

            # Predict
            preds = []
            labels = []
            total_step = int(len(dataset) / self.config.val_batch_size + 1)
            for batch in batches:
                output = self.model(batch.context, batch.query, batch.reply)
                label = self.model.to_labels(output)
                preds.extend(output[-1].diagonal().cpu().numpy())
                labels.extend(label)

        return labels, preds

    def _list_to_dataset(self, inputs, utterance_field):
        """
        Make dataset from list of texts.
        :param inputs: (list of tuple(context, query, reply)) list of (context, query, reply) to inferece.
        :param utterance_field: (Field) fields having tokenize function and vocab.
        """
        # Tokenize texts
        tokenized_inputs = [
            [utterance_field.tokenize(context), utterance_field.tokenize(query), utterance_field.tokenize(reply)]
            for context, query, reply in inputs
        ]

        # Make dataset from list
        fields = [
            ("context", self.fields.utterance_field),
            ("query", self.fields.utterance_field),
            ("reply", self.fields.utterance_field),
        ]
        examples = [Example.fromlist(tokenized_input, fields=fields) for tokenized_input in tokenized_inputs]
        dataset = Dataset(examples, fields=fields)

        return dataset
