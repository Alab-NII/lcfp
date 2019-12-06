# film-onecommon
A FiLM-based model for onecommon. (Temporally) separated from root repo to avoid conflicts of files. 

https://arxiv.org/abs/1709.07871

## Usage

```shell
cd experiment
python train.py
```

I don't use these arguments now.

--use_attention

--feat_type

--annot_noise

--ctx_view_size


## Model file

models/film_model.py

## Note

Default parameter is one that model achieve the best result.

Invalid memory access error might occur in torch 1.3.0.

Training works in torch 1.3.1.

