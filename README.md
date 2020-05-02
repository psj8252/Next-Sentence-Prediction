# Next-Sentence-Prediction
This repository is for developing next sentence prediction module.

## Install

```bash
pip install https://github.com/psj8252/Next-Sentence-Prediction/archive/v0.0.1.tar.gz

pip install git+https://github.com/psj8252/Next-Sentence-Prediction.git@v0.0.1
```
You can install by typing any of upper commands.

## Usage

### 1. Make Dataset  

You should make dataset and vocab. You can make dataset and vocab using "tsv" data file.  
By default, vocab is made only using training data.
```python
from nsp.data_loader import DataLoader

dl = DataLoader()

# Make dataset and build vocab
dl.make_dataset(train_data_path="train_dataset.tsv", test_data_path="test_dataset.tsv")
vocab = dl.build_vocab()

# Save datasets and vocab
dl.save("Datasets.dill")
vocab.save("vocab.txt")
```

Default train or test tsv file format is alike below.
| ID | Utterance                                                                     | Label |
| -- |:------------------------------------------------------------------------------| -----:|
| 1  | \<SOS\> hi \<SEP\> hello \<EOS\>                                              |   1   |
| 2  | \<SOS\> good morning \<SEP\> good night \<EOS\>                               |   0   |
| 3  | \<SOS\> hi \<CTXSEP\> hello \<CTXEND\> it's rainy \<SEP\> oh, my god \<EOS\>  |   1   |

In utterance field, alway starts with <SOS> and ends with <EOS> and there are three parts, context, query, reply.  
The context is additional and optional conversation text before query. 
The query is main utterance to check that the reply is proper about query.  

The vocab file format is just lines of words. There is a word in a line. Index starts from zero, most upper words.

### 2. Train

You can train model by writing yml config and executing simple python script.
```python3
import nsp

trainer = nsp.manager.TrainManager("train_config.yml", "cpu")
trainer.train()
```
All the information for training is in the config file. You can refer to sample config file which is in "samples/data/train_config.yml".  
```yaml
# Configs for model
model_type: TransformerModel
model_configs:
  vocab_size: 10000
  d_model: 512
  nhead: 4
  dim_feedforward: 1024
  num_layers: 4
  dropout: 0.2
  activation: "relu"

# Configs for data
data_loader_path: "NSP_sample.dill"

# Configs for training
epoch: 10
batch_size: 64
val_batch_size: 512
learning_rate: 1.e-4
optimizer: "SGD"
steps_per_log: 1
steps_per_eval: 1

model_save_dir: "."
model_save_prefix: "models"

optimizer_args: 
  momentum: 0.9
```
You should write config file. You can refer to "nsp/config.py". The each element in config yaml file corresponds to attributes in python training config.

### 3. Infer

You can inference texts using the learned model. You need to write a inference config file.

```yaml
model_type: TransformerModel
model_path: saved_model.pth
vocab_path: vocab.txt
val_batch_size: 128
````

```python
import nsp

inferencer = nsp.manager.InferManager("infer_config.yml", device="cuda:0")
labels = inferencer.inference_texts(["<SOS> hi <SEP> hello <EOS>", "<SOS> hi <SEP> I like apples <EOS>"])
print(labels)
```
