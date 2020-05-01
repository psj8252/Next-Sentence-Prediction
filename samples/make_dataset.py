from nsp.data_loader import DataLoader

"""
This is sample showing how to make dataset and vocab from data files.
"""
dl = DataLoader()

# Make dataset and build vocab
dl.make_dataset(train_data_path="data/sample_nsp.tsv", test_data_path="data/sample_nsp.tsv")
vocab = dl.build_vocab()

# Save datasets and vocab
dl.save("NSP_sample.dill")
vocab.save("vocab_nsp_sample.txt")
