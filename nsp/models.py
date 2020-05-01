import torch
import torch.nn as nn


class BaseModel(nn.Module):
    save_params = {}

    def save(self, path):
        """
        Save model to file.
        :param path: (str) Path to save model a file.
        'save_params' is dict used as keyword arguments when loading the model.
        """
        torch.save({"save_params": self.save_params, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        """
        Load model from file.
        :param path: (str) Path to load model from the file.
        :return: (torch.nn) Loaded model.
        """
        checkpoint = torch.load(path)
        model = cls(**checkpoint["save_params"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


class TransformerModel(BaseModel):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout, activation):
        """
        :param vocab_size: the number of vocabulary words.
        :param d_model: transformer model dimention.
        :param nhead: the number of transformer head.
        :param dim_feedforward: transformer feedfoward dimention.
        :param num_layers: the number of transformer encoder layers.
        :param dropout: transformer dropout.
        :param activation: transformer activcation
        """
        super(TransformerModel, self).__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.pooler = nn.Linear(d_model, 2)

        # Set parameters for save
        self.save_params = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "num_layers": num_layers,
            "dropout": dropout,
            "activation": activation,
        }

    def forward(self, input_tokens):
        """
        :param input_tokens: (torch.tensor) input tokens. shaped (Sequence x BatchSize x VocabSize)
        :return: (torch.tensor) model result shaped (BatchSize, 2)

        [Shape]
        embedded: (Sequence x BatchSize x d_model)
        encoded: (Sequence x BatchSize x d_model)
        pooled: (BatchSize, 2)
        """
        embedded = self.embed(input_tokens)
        encoded = self.transformer_encoder(embedded)
        pooled = self.pooler(encoded[0])
        return pooled
