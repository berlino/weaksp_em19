from pathlib import Path

class Config():
    def __init__(self):
        # data dir
        self.reader_pkl = "processed/wikitable_glove_42B_minfreq_3.pkl"

        # sketch file
        self.sketch_pkl = "processed/train.pkl" 
        self.sketch_test_pkl = "processed/test.pkl"
        self.sketch_action_file = "processed/sketch.actions"

        # config for programmer
        self.token_embed_size = 300
        self.var_token_size  = 256
        self.token_dropout = 0.5
        self.token_rnn_size = 256
        self.token_indicator_size = 16
        self.pos_embed_size = 64
        self.slot_dropout = 0.25
        self.prod_embed_size = 512
        self.prod_rnn_size = 512
        self.prod_dropout = 0.25
        self.op_embed_size = 128

        self.column_type_embed_size = 16
        self.column_indicator_size = 16
        self.slot_hidden_score_size = 512

        self.model_type = "struct"

        # config for training
        self.lr = 5e-4
        self.l2 = 1e-5
        self.clip_norm = 3
        self.gpu_id = 1
        self.seed = 3264

    def __repr__(self):
        return str(vars(self))

config = Config()