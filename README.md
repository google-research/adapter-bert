# Adapter-BERT

## Introduction

This repository contains a version of BERT that can be trained using adapters.
Our ICML 2019 paper contains a full description of this technique:
[Parameter-Efficient Transfer Learning for NLP](http://proceedings.mlr.press/v97/houlsby19a.html).

Adapters allow one to train a model to solve new tasks, but adjust only a few
parameters per task. This technique yields compact models that share many
parameters across tasks, whilst performing similarly to fine-tuning the entire
model independently for every task.

The code here is forked from the
[original BERT repo](https://github.com/google-research/bert).
It provides our version of BERT with adapters, and the capability to train it on
the [GLUE tasks](https://gluebenchmark.com/).
For additional details on BERT, and support for additional tasks, see the
original repo.


## Tuning BERT with Adapters

The following command provides an example of tuning with adapters on GLUE.

Fine-tuning may be run on a GPU with at least 12GB of RAM, or a Cloud TPU. The
same constraints apply as for full fine-tuning of BERT. For additional details,
and instructions on downloading a pre-trained checkpoint and the GLUE tasks,
see
[https://github.com/google-research/bert](https://github.com/google-research/bert).


```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=3e-4 \
  --num_train_epochs=5.0 \
  --output_dir=/tmp/adapter_bert_mrpc/
```

You should see an output like this:

```
***** Eval results *****
  eval_accuracy = 0.85784316
  eval_loss = 0.48347527
  global_step = 573
  loss = 0.48347527
```

This means that the Dev set accuracy was 85.78%. Small sets like MRPC have a
high variance in the Dev set accuracy, even when starting from the same
pre-training checkpoint. Therefore results may deviate from this by 2%.

## Citation

Please use the following citation for this work:

```
@inproceedings{houlsby2019parameter,
  title = {Parameter-Efficient Transfer Learning for {NLP}},
  author = {Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Bruna and De Laroussilhe, Quentin and Gesmundo, Andrea and Attariyan, Mona and Gelly, Sylvain},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  year = {2019},
}
```

The paper is uploaded to [ArXiv](https://arxiv.org/abs/1902.00751).

## Disclaimer

This is not an official Google product.

## Contact information

For personal communication, please contact Neil Houlsby
(neilhoulsby@google.com).
