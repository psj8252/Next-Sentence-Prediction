import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers, embedding_dim, dropout, activation):
        """
        :param vocab_size: the number of vocabulary words.
        :param d_model: transformer model dimention.
        :param nhead: the number of transformer head.
        :param dim_feedforward: transformer feedfoward dimention.
        :param num_layers: the number of transformer encoder layers.
        :param embedding_dim: the dimention to embed context, query, reply.
        :param dropout: transformer dropout.
        :param activation: transformer activcation
        """
        super(TransformerModel, self).__init__()

        self.word_embed = nn.Embedding(vocab_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.context_ffn = nn.Linear(d_model, embedding_dim)
        self.query_ffn = nn.Linear(d_model, embedding_dim)
        self.reply_ffn = nn.Linear(d_model, embedding_dim)
        self.context_query_mean_ffn = nn.Linear(d_model, embedding_dim)

        # Set parameters for save
        self.save_params = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "num_layers": num_layers,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "activation": activation,
        }

    def forward(self, context_tokens, query_tokens, reply_tokens):
        """
        :param input_tokens: (torch.tensor) input tokens. shaped (Sequence x BatchSize x VocabSize)
        :return: (torch.tensor) model result shaped (BatchSize, 2)

        [Shape]
        word_embed: (Sequence x BatchSize x d_model)
        encoded: (Sequence x BatchSize x d_model)
        output: (ContextQuery-BatchSize, Reply-BatchSize)
        """
        # Word Embed
        context = self.word_embed(context_tokens)
        query = self.word_embed(query_tokens)
        reply = self.word_embed(reply_tokens)

        # Encode
        context = self.transformer_encoder(context)
        query = self.transformer_encoder(query)
        reply = self.transformer_encoder(reply)

        # Feed forward
        context = self.context_ffn(context.sum(dim=0))
        query = self.query_ffn(query.sum(dim=0))
        reply = self.reply_ffn(reply.sum(dim=0))
        context_query = self.context_query_mean_ffn((context + query) / 2)

        # Normalize
        context_query = F.normalize(context_query, p=2, dim=1)
        reply = F.normalize(reply, p=2, dim=1)

        output = torch.matmul(context_query, reply.T)
        return output

    def to_labels(self, output):
        """
        Return to labels (ex [1, 0, 1, 1, 0, ...]) from model output
        """
        return (output.diagonal() > 0.5).cpu().detach().numpy()

    def criterion(self, output):
        positive_score = output.diagonal().sum()
        negative_score = output.exp().fill_diagonal_(0).sum(dim=1).log().sum()
        score = positive_score - negative_score
        return -score

    def metrics_parts(self, output):
        n_positive = output.shape[0]
        answer = torch.zeros_like(output).fill_diagonal_(1)
        is_correct = (output >= 0.5) == answer

        TP = is_correct.diagonal().sum().cpu().detach().item()
        TN = is_correct.sum().cpu().detach().item() - TP
        FP = n_positive * n_positive - n_positive - TN
        FN = n_positive - TP
        return TP, FP, TN, FN
