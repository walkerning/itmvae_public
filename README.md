## RUN
### Setup

* Python 2 is required

* Install some dependencies (Ubuntu 14.04):
```sudo apt install libyaml-dev```

* Create some virutalenv using `conda` or `virtualenv`.

* Clone recursively, with all submodules:
```git clone --recursive <this repo>```

* To run in develop mode:
```python setup.py develop```

### Prepare
#### Prepare dataset

The 20news dataset is given in this repo.

If you want to run on another dataset, read `vae_topicmodel/reader.py` to see which format of file each type of reader expects.

#### Prepare reference corpus

We use the code in ` https://github.com/jhlau/topic_interpretability/` to calculate the NPMI score of the learned topics. It need a reference corpus, see `scripts/dump_corpus_file.py` for an example of dumping 20news back to a reference corpus.


### Run

The entry points is `vae_topic_run` and `vae_topic_runtest`, which will be created in some bin directories after you run `python setup.py develop`. Also you can run `vae_topicmodel/run_test.py` or `vae_topicmodel/run.py`.

The helper bash script `run_cfg.sh` receive a directory that have configurations under it, can be configured using some environment variables, and call `vae_topic_run` using these configs. Finally, it will call `run_npmi.sh` which calculate the NPMI score and print out the NPMI results.

An example of running:

```
VAE_SAVE_MODEL=1 VAE_RANDOM_SEED=12345 VAE_DATASET=20news bash run_cfg.sh configs/20news/sage/ sage_1
```

This command will run training according to the configurations under directory `configs/20news/sage/` on dataset 20news(this is the default value), stored the results under directory `results/20news/sage/sage_1`.
Tensorflow summaries and the final trained model will be saved. (because `VAE_SAVE_MODEL` is set to 1).
During the running, the random seed of `tf.random`, `np.random`, `random` will be set to `12345`.

### Modify the Configurations

See the configuration files under `configs` for examples.
Normally, the configuration file named `model.yaml` configure the model construction, while the file named `train.yaml` configure the training process.

## Cite
If you find this repo helpful, you can cite our paper.
```
@article{ning2020nonparametric,
  title={Nonparametric Topic Modeling with Neural Inference},
  author={Ning, Xuefei and Zheng, Yin and Jiang, Zhuxi and Wang, Yu and Yang, Huazhong and Huang, Junzhou and Zhao, Peilin},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
```
