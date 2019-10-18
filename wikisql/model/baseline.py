import torch
import random
import logging
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from typing import Union, List, Dict, Any, Set
from collections import defaultdict

from sempar.util import check_multi_col, get_left_side_part
from sempar.action_walker import ActionSpaceWalker
from sempar.context.wikisql_context import WikiSQLContext
from sempar.domain_languages.wikisql_language import WikiSQLLanguage
from model.util import construct_row_selections, construct_junction, construct_same
from module.seq2seq import Seq2Seq

from allennlp.semparse.domain_languages import ParsingError, ExecutionError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def log_sum_exp(score_list: List):
    if isinstance(score_list, list):
        score_v = torch.stack(score_list, 0)
    else:
        score_v = score_list
    ret_v = score_v - F.log_softmax(score_v, dim=0)
    ret_scalar = ret_v.mean(0)
    return ret_scalar

class Programmer(nn.Module):
    """
    Sketch-based programmer
    """
    def __init__(self,
                token_embed_size: int,
                var_token_embed_size: int,
                token_vocab: Dict, 
                rnn_hidden_size: int,
                token_dropout: float,
                token_indicator_size: int,
                sketch_actions_cache: List,
                slot_dropout: int,
                sketch_prod_embed_size: int,
                sketch_prod2id: Dict,
                sketch_prod_rnn_hidden_size: int,
                sketch_prod_dropout: float,
                column_type_embed_size: int,
                column_indicator_size: int,
                op_embd_size: int,
                slot_hidden_score_size: int,
                device) -> None:
        super(Programmer, self).__init__()
        self.device = device
        self.token_embed_size = token_embed_size
        self.rnn_hidden_size = rnn_hidden_size
        self.var_token_embed_size = var_token_embed_size
        self.column_type_embed_size = column_type_embed_size
        self.column_indicator_size = column_indicator_size
        self.slot_hidden_score_size = slot_hidden_score_size
        self.sketch_prod_embed_size = sketch_prod_embed_size
        self.sketch_prod_rnn_hidden_size = sketch_prod_rnn_hidden_size
        self.op_embed_size = op_embd_size

        self.token_embed = nn.Embedding(token_vocab.get_vocab_size(), token_embed_size)
        self.token2var = nn.Linear(token_embed_size, var_token_embed_size) 
        self.token_vocab = token_vocab
        self.token_dropout = nn.Dropout(token_dropout)
        self.token_indicator_embed = nn.Embedding(2, token_indicator_size)

        self.rnn = nn.LSTM(var_token_embed_size + token_indicator_size, 
                    rnn_hidden_size, 1, bidirectional=True, batch_first=True)

        self.sketch_actions_cache = sketch_actions_cache
        self.sketch_dropout = nn.Dropout(sketch_prod_dropout)
        self.sketch_prod_embed = nn.Embedding(len(sketch_prod2id) + 1, 
            sketch_prod_embed_size) # with mask
        self.sketch_prod2id = sketch_prod2id
        self.sketch_prod_rnn = nn.LSTM(self.sketch_prod_embed_size, sketch_prod_rnn_hidden_size, 1, 
            bidirectional=True, batch_first=True)

        self.id2column_types = ["string", "number"]
        self.column_type2id = {v:k for k,v in enumerate(self.id2column_types)}
        self.column_type_embed = nn.Embedding(len(self.id2column_types), 
            column_type_embed_size)
        self.column_indicator_embed = nn.Embedding(2, column_indicator_size)
    
        self.id2op = ["<", ">",  "=", "and"]
        self.op2id = {v:k for k,v in enumerate(self.id2op)}
        self.op_embed = nn.Embedding(len(self.id2op), 
            op_embd_size)

        self.slot_dropout = nn.Dropout(slot_dropout)

        # score slots
        self.rnn2feat_ent = nn.Linear(rnn_hidden_size * 2, sketch_prod_rnn_hidden_size * 2)
        self.rnn2feat_score = nn.Linear(rnn_hidden_size * 2, slot_hidden_score_size)
        self.col2feat = nn.Linear(var_token_embed_size + column_type_embed_size + 
            column_indicator_size, slot_hidden_score_size)
        self.sel2feat = nn.Linear(op_embd_size + var_token_embed_size + column_type_embed_size + 
            column_indicator_size + rnn_hidden_size * 2, slot_hidden_score_size)
        self.all_rows_rep = nn.Parameter(nn.init.normal_(torch.empty(op_embd_size + var_token_embed_size + column_type_embed_size + 
            column_indicator_size + rnn_hidden_size * 2)).to(device))
        
        self.__cur_align_prob_log = None

         # seq2seq
        self.seq2seq = Seq2Seq(self.rnn_hidden_size * 2, self.sketch_prod_embed,
                                self.sketch_prod2id, self.rnn_hidden_size * 2,
                                self.device)

        self.CANDIDATE_ACTION_NUM_BOUND = 128
        self.CONSISTENT_INST_NUM_BOUND = 512
        self.EVAL_NUM_SKETCH_BOUND = 6


    def load_vector(self, word2vec):
        t_v = torch.Tensor(word2vec).to(self.device)
        logger.info("Loading embedding size {}".format( t_v.size()))
        self.token_embed.weight = nn.Parameter(t_v)
        self.token_embed.weight.requires_grad = False
    

    def encode_question(self, tokens:List, token_in_table_feat: List) -> torch.Tensor:
        # tokens = [ token.text for token in  context.annoymized_tokens]
        token_ids = [ self.token_vocab.get_token_index(token.text) for token in tokens]
        token_id_v = torch.LongTensor(token_ids).to(self.device)
        token_vecs = self.token_embed(token_id_v)
        var_token_vecs = self.token2var(token_vecs) 
        drop_var_token_vecs = self.token_dropout(var_token_vecs)

        token_in_table_feat_id = torch.LongTensor(token_in_table_feat).to(self.device)
        token_in_table_feat_v = self.token_indicator_embed(token_in_table_feat_id)

        lstm_input_v = torch.cat([drop_var_token_vecs, token_in_table_feat_v], 1)
        lstm_input_v = lstm_input_v.unsqueeze(0)
        lstm_output, (ht, ct) = self.rnn(lstm_input_v)
        lstm_output = lstm_output.squeeze(0)
        question_rep = torch.cat([ht[0], ht[1]], 1).squeeze(0)
        return lstm_output, question_rep, (ht, ct)


    def encode_sketch(self, sketch_actions:List, encoder_state: torch.Tensor) -> torch.Tensor:
        action_ids = [self.sketch_prod2id[ac] for ac in sketch_actions]
        action_id_v = torch.LongTensor(action_ids).to(self.device)
        action_vecs = self.sketch_prod_embed(action_id_v)
        action_vecs = self.sketch_dropout(action_vecs)

        lstm_input_v = action_vecs.unsqueeze(0)
        # lstm_output, (ht, ct) = self.sketch_prod_rnn(lstm_input_v, encoder_state)
        lstm_output, (ht, ct) = self.sketch_prod_rnn(lstm_input_v)
        lstm_output = lstm_output.squeeze(0)
        sketch_rep = torch.cat([ht[0], ht[1]], 1).squeeze(0)
        return lstm_output, sketch_rep


    def sketch_lf2actions(self, world: WikiSQLLanguage):
        lf2actions = dict()
        for actions in self.sketch_actions_cache:
            lf = world.action_sequence_to_logical_form(actions)
            lf2actions[lf] = actions
        return lf2actions


    def construct_candidates(self, 
                        world:WikiSQLLanguage, 
                        token_encodes:torch.Tensor) -> Dict:
        """
        Get the candidate reps for each type of slot
        """
        candidate_rep_dic = dict()

        # Column
        _column_reps = []
        _column_actions = []
        _string_column_reps = []
        _string_column_actions = []
        _number_column_reps = []
        _number_column_actions = []
        for typed_column_name, column_type in world.table_context.column2types.items():
            column_name = typed_column_name.split(f"{column_type}_column:")[1]
            column_tokens = column_name.split("_")
            token_v = torch.LongTensor([self.token_vocab.get_token_index(t) 
                for t in column_tokens]).to(self.device)
            column_embed = torch.mean(self.token_embed(token_v), 0)
            column_var_v = self.token2var(column_embed)
            column_type_embed = self.column_type_embed.weight[
                    self.column_type2id[column_type]]
            column_indicator_v = self.column_indicator_embed.weight[
                    world.table_context.column_feat[typed_column_name]]
            _column_v = torch.cat([column_var_v, column_type_embed, 
                    column_indicator_v], 0)

            _column_reps.append(_column_v)
            _column_actions.append(f"Column -> {typed_column_name}")

            if column_type == "string":
                _string_column_reps.append(_column_v)
                _string_column_actions.append(
                    f"StringColumn -> {typed_column_name}")

            elif column_type in ["number"]:
                _number_column_reps.append(_column_v)
                _number_column_actions.append(
                    f"NumberColumn -> {typed_column_name}")
            else:
                raise NotImplementedError

        if len(_column_reps) > 0:
            _column_rep = torch.stack(_column_reps, 0)
            candidate_rep_dic["Column"] = (_column_rep, _column_actions)
        if len(_string_column_reps) > 0:
            _string_column_rep = torch.stack(_string_column_reps, 0)
            candidate_rep_dic["StringColumn"] = (_string_column_rep, _string_column_actions)
        if len(_number_column_reps) > 0:
            _number_column_rep = torch.stack(_number_column_reps, 0)
            candidate_rep_dic["NumberColumn"] = (_number_column_rep, _number_column_actions)

        row_selection_reps = []
        row_selection_actions = []
        junction_candidates = []
        junction_flag = True
        # Number
        if len(world.table_context._num2id) > 0 and len(_number_column_reps) > 0:
            for _num in world.table_context._num2id:
                for _num_col_rep, _num_col_ac in zip(_number_column_reps, _number_column_actions):
                    for _op in ["<", ">", "="]:
                        _id = world.table_context._num2id[_num]
                        _num_rep = token_encodes[_id]
                        _op_rep = self.op_embed.weight[self.op2id[_op]]
                        _sel_rep = torch.cat([_op_rep, _num_col_rep, _num_rep], 0)

                        _num_action = f"Number -> {_num}"
                        _action_seq = construct_row_selections("number", _op, _num_col_ac, _num_action)
                        row_selection_reps.append(_sel_rep)
                        row_selection_actions.append(_action_seq)
                    
                        if junction_flag:
                            _op_rep_and = self.op_embed.weight[self.op2id[_op]] + \
                                self.op_embed.weight[self.op2id["and"]] 
                            _sel_rep_and = torch.cat([_op_rep_and, _num_col_rep, _num_rep], 0) 
                            junction_candidates.append(("num", _num, _action_seq, _sel_rep_and))

        # Str
        if len(world.table_context._entity2id) > 0 and len(_string_column_reps) > 0:
            for _entity in world.table_context._entity2id:
                _s, _e = world.table_context._entity2id[_entity]
                assert _e > _s
                for _str_col_rep, _str_col_ac in zip(_string_column_reps, _string_column_actions):
                    str_col_name = _str_col_ac.split(" -> ")[1]
                    # take knowledge graph here
                    if _entity not in world.table_context._knowledge_graph[str_col_name]:
                        continue
                    for _op in ["="]:
                        _ent_rep = torch.mean(token_encodes[_s : _e], 0)
                        # _ent_rep = (token_encodes[_s] + token_encodes[_e-1]) / 2.0
                        _op_rep = self.op_embed.weight[self.op2id[_op]] 
                        _sel_rep = torch.cat([_op_rep, _str_col_rep, _ent_rep,], 0) 
                        
                        _str_action = f"str -> {_entity}"
                        _action_seq = construct_row_selections("string", _op, _str_col_ac, _str_action)
                        row_selection_reps.append(_sel_rep)
                        row_selection_actions.append(_action_seq)

                        if junction_flag:
                            _op_rep_and = self.op_embed.weight[self.op2id[_op]] + \
                                self.op_embed.weight[self.op2id["and"] ] 
                            _sel_rep_and = torch.cat([_op_rep_and, _str_col_rep, _ent_rep], 0) 
                            junction_candidates.append(("str", _entity, _action_seq, _sel_rep_and))

        # sample some candidates
        if self.training and len(junction_candidates) > 12:
            junction_candidates = random.sample(junction_candidates, 12)

        if not junction_flag: assert len(junction_candidates) == 0
        for i in range(len(junction_candidates)):
            ent_1_type, ent_1, action_seq_1, and_rep_1 = junction_candidates[i]
            for j in range(len(junction_candidates)):
                if i == j: continue
                ent_2_type, ent_2, action_seq_2, and_rep_2 = junction_candidates[j]
                if ent_1 == ent_2: continue

                _sel_rep = sum([and_rep_1, and_rep_2]) 
                _action_seq = construct_junction("and", action_seq_1, action_seq_2)
                row_selection_reps.append(_sel_rep)
                row_selection_actions.append(_action_seq) 
        
        # all_rows
        row_selection_reps.append(self.all_rows_rep)
        row_selection_actions.append(["List[Row] -> all_rows"])

        row_rep_v = torch.stack(row_selection_reps, 0)
        candidate_rep_dic["List[Row]"] = (row_rep_v, row_selection_actions)
        logger.info(f"{len(row_selection_actions)} row selections found")
        
        return candidate_rep_dic


    def collect_candidate_scores(self, 
                            world:WikiSQLLanguage, 
                            token_encodes:torch.Tensor, 
                            candidate_rep_dic: Dict,
                            sketch_encodes:torch.Tensor, 
                            slot_dict: Dict):
        """
        Collect candidate score for each slot
        """
        token4ent_encodes = self.rnn2feat_ent(token_encodes)
        token4score_encodes = self.rnn2feat_score(token_encodes)

        ret_score_dict = dict()
        for idx in slot_dict:
            slot_type = slot_dict[idx]
            if slot_type == "List[Row]":
                slot_rep_v = sketch_encodes[idx].unsqueeze(1) # rep_dim * num_slot
                slot_att_scores = torch.mm(token4ent_encodes, slot_rep_v) # num_tokens * num_slot
                att_over_token = F.softmax(slot_att_scores.transpose(0,1), dim=1) # num_slot * num_tokens
                att_token_col_v = torch.mm(att_over_token, token4score_encodes) # num_slot * feat_dim

                candidate_v, candidate_a = candidate_rep_dic[slot_type]
                candidate_feat_v = self.sel2feat(candidate_v)
                att_over_sel = torch.mm(att_token_col_v, candidate_feat_v.transpose(0,1)) # num_slot * num_column
                att_over_sel = F.log_softmax(att_over_sel, dim=1)
                ret_score_dict[idx] = att_over_sel.squeeze()
            else:
                assert "Column" in slot_type
                slot_rep_v = sketch_encodes[idx].unsqueeze(1) # rep_dim * num_slot
                slot_att_scores = torch.mm(token4ent_encodes, slot_rep_v) # num_tokens * num_slot
                att_over_token = F.softmax(slot_att_scores.transpose(0,1), dim=1) # num_slot * num_tokens
                att_token_col_v = torch.mm(att_over_token, token4score_encodes) # num_slot * feat_dim

                candidate_v, candidate_a = candidate_rep_dic[slot_type]
                candidate_feat_v = self.col2feat(candidate_v)
                att_over_col = torch.mm(att_token_col_v, candidate_feat_v.transpose(0,1)) # num_slot * num_column
                att_over_col = F.log_softmax(att_over_col, dim=1)
                ret_score_dict[idx] = att_over_col.squeeze(0)

        return ret_score_dict


    def slot_filling(self, 
                        world:WikiSQLLanguage, 
                        token_encodes:torch.Tensor, 
                        token_state: torch.Tensor,
                        candidate_rep_dic: Dict,
                        sketch_actions: List):
        """
        1) collect scores for each individual slot 2) find all the paths recursively
        """
        slot_dict = world.get_slot_dict(sketch_actions)
        sketch_encodes, sketch_rep = self.encode_sketch(sketch_actions,token_state)
        candidate_score_dic = self.collect_candidate_scores(world, token_encodes, 
                candidate_rep_dic, sketch_encodes, slot_dict)

        possible_paths = []
        path_scores = []
        def recur_compute(prefix, score, i):
            if i == len(sketch_actions):
                possible_paths.append(prefix)
                path_scores.append(score)
                return
            if i in slot_dict:
                _slot_type = slot_dict[i]
                if _slot_type not in candidate_rep_dic:
                    return   # this sketch does not apply here

                slot_rep = sketch_encodes[i] 
                candidate_v, candidiate_actions = candidate_rep_dic[_slot_type]

                if len(candidiate_actions) == 1:
                    action = candidiate_actions[0]
                    new_prefix = prefix[:]
                    if isinstance(action, list):
                        new_prefix += action
                    else:
                        new_prefix.append(action)
                    recur_compute(new_prefix, score, i + 1) 
                    return
                
                if len(candidiate_actions) > self.CANDIDATE_ACTION_NUM_BOUND:
                    _, top_k = torch.topk(candidate_score_dic[i], self.CANDIDATE_ACTION_NUM_BOUND, dim=0)
                    ac_idxs = top_k.cpu().numpy()    
                else:
                    ac_idxs = range(len(candidiate_actions))

                # for ac_ind, action in enumerate(candidiate_actions):
                for ac_ind in ac_idxs:
                    action = candidiate_actions[ac_ind]
                    new_prefix = prefix[:]
                    if score:
                        new_score = score + candidate_score_dic[i][ac_ind]
                    else:
                        new_score = candidate_score_dic[i][ac_ind]
                    if isinstance(action, list):
                        new_prefix += action
                    else:
                        new_prefix.append(action)
                    recur_compute(new_prefix, new_score, i + 1)
            else:
                new_prefix = prefix[:]
                new_prefix.append(sketch_actions[i])
                recur_compute(new_prefix, score, i + 1) 
        
        recur_compute([], None, 0)
        return possible_paths, path_scores


    def forward(self, 
            context: WikiSQLContext,
            sketch2program: Dict) -> torch.Tensor:
        world = WikiSQLLanguage(context)

        # encode questions
        token_in_table_feat = context.question_in_table_feat
        token_encodes, token_reps, last_state = self.encode_question(context.question_tokens, token_in_table_feat)

        sketch_lf2actions = self.sketch_lf2actions(world)
        consistent_scores = []
        candidate_rep_dic = self.construct_candidates(world, token_encodes)
        for sketch_lf in sketch2program:
            if len(sketch2program[sketch_lf]) > self.CONSISTENT_INST_NUM_BOUND:
                continue
            sketch_actions = sketch_lf2actions[sketch_lf]
            seq_log_likeli = self.seq2seq(world, token_reps, token_encodes, sketch_actions)
            _paths, _log_scores = self.slot_filling(world, token_encodes, last_state, 
                    candidate_rep_dic, sketch_actions)
            
            # only one path
            if len(_paths) == 1:
                consistent_scores.append(seq_log_likeli)
                continue

            _gold_scores = []
            for _path, _score in zip(_paths, _log_scores):
                assert _score is not None
                _path_lf = world.action_sequence_to_logical_form(_path)
                # logger.info(_path_lf)
                if _path_lf in sketch2program[sketch_lf]:
                    _gold_scores.append(_score) 
            
            # aggregate consistent instantiations
            if len(_gold_scores) > 0:
                _score = seq_log_likeli + log_sum_exp(_gold_scores) 
                if torch.isnan(_score) == 0:
                    consistent_scores.append(_score)
                else:
                    logger.warning("Nan loss founded!")

        if len(consistent_scores) > 0:
            return -1 * log_sum_exp(consistent_scores)
        else:
            return None

    def filter_program_by_execution(self,
                                world:WikiSQLLanguage, 
                                actions: List):
        try:
            world.execute_action_sequence(actions)
            return True
        except:
            return False

    def evaluate(self, 
            context: WikiSQLContext,
            sketch2program: Dict) -> Dict:
        """
        Return a dictionary for different analysis
        """
        world = WikiSQLLanguage(context)
        ret_dic = defaultdict(int)

        # encode question and offline sketches
        token_in_table_feat = context.question_in_table_feat
        token_encodes, token_reps, last_state = self.encode_question(
                context.question_tokens, token_in_table_feat)

        sketch_actions_and_scores = self.seq2seq.beam_decode(world, 
            token_reps, token_encodes, self.EVAL_NUM_SKETCH_BOUND)
        
        max_score = None
        best_sketch_actions = None
        best_sketch_lf = None
        best_program_actions = None
        best_program_lf = None
        candidate_rep_dic = self.construct_candidates(world, token_encodes)
        for sketch_actions, sketch_log_score in sketch_actions_and_scores:
            sketch_lf = world.action_sequence_to_logical_form(sketch_actions)
            _paths, _log_scores = self.slot_filling(world, token_encodes, last_state, 
                candidate_rep_dic, sketch_actions)

            if self.__class__.__name__ == "ConcreteProgrammer":
                assert self._ConcreteProgrammer__cur_align_prob_log is not None
                align_prob_log = self._ConcreteProgrammer__cur_align_prob_log.squeeze()
                sketch_log_score = sketch_log_score + align_prob_log
                self._ConcreteProgrammer__cur_align_prob_log = None

            # only one path
            if len(_paths) == 1:
                if not self.filter_program_by_execution(world, _paths[0]):
                    continue
                _path_lf = world.action_sequence_to_logical_form(_paths[0])
                _seq_score = sketch_log_score
                if max_score is None or _seq_score > max_score:
                    max_score = _seq_score
                    best_sketch_lf = sketch_lf
                    best_sketch_actions = sketch_actions
                    best_program_lf = _path_lf 
                    best_program_actions = _paths[0]
                continue

            # multiple path
            for _path, _score in zip(_paths, _log_scores):
                if not self.filter_program_by_execution(world, _path):
                    continue
                assert _score is not None
                _path_lf = world.action_sequence_to_logical_form(_path)
                _seq_score = _score + sketch_log_score
                if max_score is None or _seq_score > max_score:
                    max_score = _seq_score
                    best_sketch_lf = sketch_lf
                    best_sketch_actions = sketch_actions
                    best_program_lf = _path_lf 
                    best_program_actions = _path
        
        assert max_score is not None
        ret_dic["best_program_lf"] = best_program_lf
        ret_dic["best_program_actions"] = best_program_actions
        ret_dic["best_sketch_lf"] = best_sketch_lf
        ret_dic["best_sketch_actions"] = best_sketch_actions
        ret_dic["best_score"] = torch.exp(max_score)
        ret_dic["is_multi_col"] = check_multi_col(world, 
                        best_sketch_actions, best_program_actions)

        if best_sketch_lf in sketch2program:
            ret_dic["sketch_triggered"] = True
            if best_program_lf in sketch2program[best_sketch_lf]:
                ret_dic["lf_triggered"] = True
            else:
                ret_dic["lf_triggered"] = False
        else:
            ret_dic["sketch_triggered"] = False
            ret_dic["lf_triggered"] = False

        return ret_dic


    def filter_by_keywords(self, 
                        lf: str,
                        keywords: List) -> bool:
        for w in keywords:
            if w not in lf:
                return False
        return True


    def compute_entropy(self, 
            context: WikiSQLContext,
            sketch2program: Dict,
            keywords: List) -> Dict:
        """
        Return a dictionary for different analysis
        """
        world = WikiSQLLanguage(context)
        ret_dic = defaultdict(int)

        # encode question and offline sketches
        token_in_table_feat = context.question_in_table_feat
        token_encodes, token_reps, last_state = self.encode_question(
                context.question_tokens, token_in_table_feat)

        consist_prob_logs = []
        # consist_sketch_logs = []
        entropy = []

        candidate_rep_dic = self.construct_candidates(world, token_encodes)
        sketch_lf2actions = self.sketch_lf2actions(world)
        for sketch_lf in sketch2program:
            sketch_actions = sketch_lf2actions[sketch_lf]    
            sketch_log_score = self.seq2seq(world, token_reps, token_encodes, sketch_actions)
            _paths, _log_scores = self.slot_filling(world, token_encodes, last_state,
                candidate_rep_dic, sketch_actions)
            
            # sketch entropy
            # entropy.append(-1 * sketch_log_score * torch.exp(sketch_log_score))
            # consist_sketch_logs.append(sketch_log_score)

            # only one path
            if len(_paths) == 1:
                if not self.filter_program_by_execution(world, _paths[0]):
                    continue
                _path_lf = world.action_sequence_to_logical_form(_paths[0])
                _seq_score = sketch_log_score
                if _path_lf in sketch2program[sketch_lf]:
                    if self.filter_by_keywords(_path_lf, keywords):
                        entropy.append(_seq_score)
                    # consist_prob_logs.append(_seq_score)
                    consist_prob_logs.append(_seq_score)
                continue

            # multiple path
            for _path, _score in zip(_paths, _log_scores):
                if not self.filter_program_by_execution(world, _path):
                    continue
                _path_lf = world.action_sequence_to_logical_form(_path)
                _seq_score = _score + sketch_log_score
                if _path_lf in sketch2program[sketch_lf]:
                    if self.filter_by_keywords(_path_lf, keywords):
                        entropy.append(_seq_score)
                    consist_prob_logs.append(_seq_score)

        if len(entropy) > 0:
            # print(f"length {len(entropy)}")
            ret_dic["entropy"] =  log_sum_exp(consist_prob_logs) - log_sum_exp(entropy)
            ret_dic["triggered"] = True
            if max(consist_prob_logs) > max(entropy):
                ret_dic["is_correct"] = False
            else:
                ret_dic["is_correct"] = True
            p = torch.exp(max(entropy)) / sum(torch.exp(l_) for l_ in consist_prob_logs)
            ret_dic["proportion"] = p 
        else:
            ret_dic["triggered"] = False
            # print("Not matched")
        return ret_dic