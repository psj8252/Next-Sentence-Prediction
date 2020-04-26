import sys

from torchtext import vocab
from torchtext.data import Field


class Vocab(vocab.Vocab):
    """
    Vocab object to map word to index.
    """

    def __init__(self, specials=["<pad>", "<sos>", "<eos>", "<sep>"]):
        """
        :param specials: (iterable) Special tokens that considered as words.
        """
        super().__init__({x: 1 for x in ("<unk>", *specials)}, specials=specials)

    def save(self, path):
        """
        Save vocab object to text file. 
        :param path: (str) Path to save. The file format is just lines of words. There is a word in a line.
                      This is for easily looking up and appending words.
        """
        with open(path, "w") as f:
            for word in self.itos:
                f.write(word + "\n")

    @classmethod
    def load(cls, path):
        """
        Load from saved vocab text file.
        :param path: (str) Path to load.
        """
        # Read a files
        vocab = cls(specials=())
        with open(path) as f:
            words = [x.strip() for x in f]

        # Create vocab
        class pseudo_vocab:
            itos = words

        vocab.extend(pseudo_vocab)

        return vocab


class Fields:
    def __init__(self, vocab_path, tokenize=None, preprocessing=None, pad_token="<pad>", **kwargs):
        """
        object containing torchtext utterance and label fields.
        :param vocab_path: (str) vocab file path to create Vocab.
        :param tokenize: (func) function to tokenize utterances (str) -> (list) of tokens format.
                         default tokenizer is Mecab and str.split (if mecab is unavailable)
        :param preprocessing: (func) torchtext preprocessing function.
        :param pad_token: (str) pad token for padding utterance.
        :param kwargs: (dict) keyword arguments for utterance field.
        """
        # Set tokenize
        if not tokenize:
            try:
                from konlpy.tag import Mecab

                mecab = Mecab()
                tokenize = mecab.morphs
            except:
                tokenize = str.split
                print("Error occured! Please install Mecab to use mecab tokenizer!", file=sys.stderr)
                print("Tokenize with 'str.split'", file=sys.stderr)

        # Set fields
        self.utterance_field = Field(tokenize=tokenize, preprocessing=preprocessing, pad_token=pad_token, **kwargs)
        self.label_field = Field(sequential=False, use_vocab=False)

        # Set vocab to utterance field
        vocab = Vocab.load(vocab_path)
        self.utterance_field.vocab = vocab


class DataLoader:
    pass
