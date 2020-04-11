# lcfp
Codes for Language-Conditioned Feature Pyramids for Visual Selection

## Content
- some utilities (e.g. dataset converters)
- experiment codes

These files are common except for the configration.
- attribute_selector_onecommon_exp_n.py
- lcfp_selector_onecommon_exp_n.py
- lcfp_selector_exp_n.py

## Experiment on GuessWhat?! Guesser Sub-task

### Preparation of the dataset

Download related files and locate them in the repository.

#### guesswhat corpus
- data/guesswhat.train.jsonl
- data/guesswhat.valid.jsonl
- data/guesswhat.test.jsonl

#### coco2014 images
- data/img/train2014
- data/img/val2014
- data/img/test2014

### Learning
```:bash
$ python src/lcfp_selector_exp_0.py
```

The model directory will be created in models directory.

Weights and validation history will be stored in the directory.

### Evaluation
```:bash
$ python src/eval_model_guesswhat.py src/lcfp_selector_exp_0.py
```

It choose and evaluate a weight with the best validation based on the history.

It output the result to the standard output.


## Experiment on OneCommon Target Selection Task

### Preparation of the dataset

Download related files and locate them in the repository.

#### onecommon corpus
- onecommon_data/train.txt
- onecommon_data/train_success_only.txt
- onecommon_data/train_uncorrelated.txt
- onecommon_data/valid.txt
- onecommon_data/valid_success_only.txt
- onecommon_data/valid_uncorrelated.txt
- onecommon_data/test.txt
- onecommon_data/test_success_only.txt
- onecommon_data/test_uncorrelated.txt

Run convert_onecommon_data.py script to get converted files and images.

```:bash
$ cd <repository root>
$ python convert_onecommon_data.py -i onecommon_data
```

Converted directory will be created in onecommon_data directory.

### Learning
```:bash
$ python src/lcfp_selector_onecommon_exp_0.py
```

### Evaluation
```:bash
$ python src/eval_model_onecommon.py src/lcfp_selector_onecommon_exp_0.py
```
