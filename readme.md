# Learning Semantic Parsers from Denotations with Latent Structured Alignments and Abstract Programs

## Setup
Use conda to create a virtual environment and setup this package

    conda create --name weaksp python=3.7
    conda activate weaskp
    pip install --user -e .

The preprocessed data is based on the one provided by [MAPO](https://github.com/crazydonkey200/neural-symbolic-machines). Download it by 

    bash download_data.sh

The default embedding file path is "glove/glove.42B.300d.txt". 

## Reprouduce Experiments on WikiTable

1. First, generate the proprocessed file with the following script:

    bash scripts/gen_processed_pkl.sh

2. Evaluate the coverage and generate consistent programs by:

    python scripts/eval_coverage demo 9 

where demo is the experiemnt id and 9 the maximal length of a sketch. 

3. Cache the generated programs with:

    python scripts/cache_lf.py processed/demo.train.programs.sketch.stat processed/demo.train.programs train processed/train.pkl

    python scripts/cache_lf.py processed/demo.test.programs.sketch.stat processed/demo.test.programs test processed/test.pkl

4. Train the model:

    python train_seq.py demo

where demo is your experiment id.

The configs of the training is in train_config/train_config. Currently, two model types are included:

* seq: seq2seq with abstract programs
* struct: abstract programs with structured alignments

The checkpoints will be available in checkpoints/



