from sempar.context.table_question_context import TableQuestionContext
from sempar.domain_languages.wikitable_abstract_language import WikiTableAbstractLanguage
from allennlp.semparse.domain_languages import ParsingError, ExecutionError
from allennlp.data.tokenizers.token import Token

from wikitable.model.baseline import Programmer
from wikitable.model.seq import SeqProgrammer
from wikitable.model.struct import StructProgrammer
from wikitable.reader.reader import WTReader
from wikitable.reader.util import load_jsonl, load_jsonl_table, load_actions
from wikitable.train_config.train_seq_config import config
from wikitable.trainer.util import get_sketch_prod, filter_sketches, create_opt, clip_model_grad, weight_init, set_seed

import torch
import sys
import pickle
import copy
from tqdm import tqdm
from pathlib import Path
from typing import List
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# hdlr = logging.FileHandler('/tmp/train_baseline.log')
# logger.addHandler(hdlr)

class ReaderUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "wikitable.reader.reader"
        return super().find_class(module, name)


def run_epoch(examples, porgrammer, opt, example_dict):
    counter = 0
    logger.info("%d examples loaded for training", len(examples))
    for example in tqdm(examples):
        # if it does not trigger any programs, then no need to train it
        if (example["id"], example["context"]) not in example_dict:
            logger.info(f"Question not covered: {' '.join(example['tokens'])}")
            continue

        # if the sentence is too long, alignment model will take up too much time
        if len(example["tokens"]) > 30:
            continue
        
        target_lfs = example_dict[(example["id"], example["context"])]
        table_id = example["context"]
        table_lines = tables[table_id]["raw_lines"]

        target_value, target_can = example["answer"] # (targeValue, targetCan)
        tokenized_question = [ Token(token, pos_=pos) for token,pos in  zip(example["tokens"], example["pos_tags"])]
        if len(tokenized_question) == 1: continue # ignore the single-token one

        context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
        context.take_corenlp_entities(example["entities"])
        context.take_features(example["features"], example["prop_features"])
        context.anonymized_tokens = example["tmp_tokens"]

        # if next(programmer.parameters()).is_cuda:  torch.cuda.empty_cache()
        programmer.train()
        opt.zero_grad()

        loss = programmer(context, target_lfs)
        if loss is None:
            # import pdb; pdb.set_trace()
            logger.info("No consistent programs found!")
            logger.info("Question: %s", example["question"])
            logger.info("Table ID: %s", table_id)
        else:
            counter += 1
            logger.info("loss: %f", loss)
            loss.backward()
            clip_model_grad(programmer, config.clip_norm)
            opt.step()
    logger.info("Coverage %f", counter / len(examples))


def test_epoch(examples, programmer, example_dict):
    logger.info("%d examples loaded for testing", len(examples))
    p_counter = 0.0
    s_counter = 0.0
    r_counter = 0.0
    for example in tqdm(examples):
        if (example["id"], example["context"]) not in example_dict:
            continue

        table_id = example["context"]
        table_lines = tables[table_id]["raw_lines"]
        target_lfs = example_dict[(example["id"], example["context"])]

        target_value, target_can = example["answer"] # (targeValue, targetCan)
        tokenized_question = [ Token(token, pos_=pos) for token,pos in  zip(example["tokens"], example["pos_tags"])]
        context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
        context.take_corenlp_entities(example["entities"])
        context.take_features(example["features"], example["prop_features"])
        context.anonymized_tokens = example["tmp_tokens"]

        programmer.eval()
        with torch.no_grad():
            ret_dic = programmer.evaluate(context, target_lfs) 
        if ret_dic['sketch_triggered']: s_counter += 1.0
        if ret_dic['lf_triggered']: p_counter += 1.0
        if ret_dic['is_multi_col']: r_counter += 1.0

        logger.info(f"Question: {context.question_tokens}")
        logger.info(f"Question-ID: {example['id']}")
        logger.info(f"Question Table ID: {example['context']}")
        logger.info(f"Best logical form: {ret_dic['best_program_lf']}")
        logger.info(f"Best score: {ret_dic['best_score']}")
        logger.info(f"Correctness: {ret_dic['lf_triggered']}")
        logger.info(f"MultiCol: {ret_dic['is_multi_col']}")
        logger.info("\n")
        
    p_acc = p_counter / len(examples)
    logger.info("Dev accurary: %f", p_acc)
    s_acc = s_counter / len(examples)
    logger.info("Dev accurary of sketch: %f", s_acc)
    r_percent = r_counter / len(examples)
    logger.info("Dev MulCol percents: %f", r_percent)
    return p_acc 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Please specify a experiment id")
        sys.exit(0)
    
    exp_id = sys.argv[1]
    print(f"Experiment {exp_id}")
    logging.basicConfig(filename=f'log/model_{exp_id}.log', level=logging.INFO)

    print(str(config))
    logger.info(str(config))
    set_seed(config.seed)

    # load raw data
    with open(config.reader_pkl, 'rb') as f:
        unpickler = ReaderUnpickler(f)
        wt_reader = unpickler.load()
    with open(config.sketch_pkl, 'rb') as f:
        example_dict = pickle.load(f)
    with open(config.sketch_test_pkl, 'rb') as f:
        test_example_dict = pickle.load(f)
    sketch_lf_actions = load_actions(config.sketch_action_file)

    # load data
    train_examples = wt_reader.train_examples
    dev_examples = wt_reader.dev_examples
    test_examples = wt_reader.test_examples
    tables = wt_reader.table_dict
    pretrained_embeds = wt_reader.wordvec
    id2prod, prod2id= get_sketch_prod(train_examples, tables)

    # model 
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    if config.model_type == "seq":
        P = SeqProgrammer
    elif config.model_type == "struct":
        P = StructProgrammer
    programmer = P(config.token_embed_size, config.var_token_size, wt_reader.vocab, config.token_rnn_size, 
                            config.token_dropout, config.token_indicator_size,sketch_lf_actions, 
                            config.slot_dropout, wt_reader.pos2id, config.pos_embed_size, 
                            config.prod_embed_size, prod2id, config.prod_rnn_size, config.prod_dropout,
                            config.column_type_embed_size, config.column_indicator_size,
                            config.op_embed_size, config.slot_hidden_score_size, device)
    programmer.to(device)

    # make sure embedding is fixed 
    programmer.load_vector(pretrained_embeds)
    parameters = filter(lambda p: p.requires_grad, programmer.parameters())
    optimizer, scheduler = create_opt(parameters, "Adam", config.lr, config.l2)

    # train it 
    best_model = None
    best_accuracy = 0
    for i in range(15):
        scheduler.step()
        logger.info(f"Epoch {i+1}")
        logger.info(f"Learning rate {optimizer.param_groups[0]['lr']}")
        run_epoch(train_examples, programmer, optimizer, example_dict)
        accuracy = test_epoch(dev_examples, programmer, example_dict)
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(programmer)
            best_accuracy = accuracy
    test_epoch(test_examples, best_model, test_example_dict)

    this_model_path = f'checkpoints/{exp_id}.model'
    print("Dumping model to {0}".format(this_model_path))
    torch.save(best_model.state_dict(), this_model_path)




