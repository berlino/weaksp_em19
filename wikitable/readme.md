## Reprouduce Experiments on WikiTable

1. First, generate the proprocessed file with the following script:

```
bash scripts/gen_processed_pkl.sh
```

2. Evaluate the coverage and generate consistent programs by:

```
python scripts/eval_coverage demo 9 
```

where demo is the experiemnt id and 9 the maximal length of a sketch. 

3. Cache the generated programs with:

```
python scripts/cache_lf.py processed/demo.train.programs.sketch.stat processed/demo.train.programs train processed/train.pkl
python scripts/cache_lf.py processed/demo.test.programs.sketch.stat processed/demo.test.programs test processed/test.pkl
```

Do not want to wait for the preprocessing? Here is my [processed files](https://drive.google.com/file/d/1tNauEhpcl21cWHhl5ciQmIHV6fKOjaar/view).

4. Train the model:

```
python train_seq.py demo
```
where demo is your experiment id.

The configs of the training is in train_config/train_config. Currently, two model types are included:

* seq: seq2seq with abstract programs
* struct: abstract programs with structured alignments

The checkpoints will be available in checkpoints/


