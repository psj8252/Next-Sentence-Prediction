import nsp

inferencer = nsp.manager.InferManager("infer_config.yml", device="cuda:0")
labels = inferencer.inference_texts(["<SOS> hi <SEP> hello <EOS>", "<SOS> hi <SEP> I like apples <EOS>"])
print(labels)
