import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )
    
def __init__():
    pass

class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.args = args
        self.init_n = None
        self.max_seq_len = self.args.max_seq_len
        self.col_len = len(self.args.num_cols) + len(self.args.cate_cols)
        self.batch_size = self.args.batch_size

        self.layer1 = nn.Linear(self.args.max_seq_len * self.col_len, 1024) 
        self.layer2 = nn.Linear(1024, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 18)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        self.init_n = len(input[0])
        input = torch.stack(input[:-1]) # mask 제외
        batch_size = input[0].size(0)
        x = self.layer1(input.contiguous().view(batch_size, -1))
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.sigmoid(x)

        return x
        

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.projection_dim = self.args.projection_dim
        self.n_layers = self.args.n_layers
        self.lstm_hidden_dim = self.args.lstm_hidden_dim
        self.cate_cols = self.args.cate_cols
        self.num_cols = self.args.num_cols
        # questions 결정
        qnum = [None, 3, 10, 5]
        self.questions = qnum[self.args.level_group]

        # embedding
        for cat_col in self.args.cate_cols:
            setattr(self,
                    'embedding_' + cat_col,
                    nn.Embedding(getattr(self.args, 'input_size_'+cat_col) + 2, self.hidden_dim)       
            )       # 원래 +1 이었는데 unknown value까지 생각해서 +2로 바꿈
        
        
        # embedding combination projection
        self.cat_proj = nn.Linear(self.hidden_dim * len(self.args.cate_cols), 
                                   self.projection_dim // 2)
        
        
        if self.args.cate_cols:
            # layer normalization
            self.layernorm = nn.LayerNorm(self.projection_dim // 2)
            self.num_proj = nn.Linear(len(self.args.num_cols), 
                                  self.projection_dim // 2)
        else: 
            # layer normalization
            self.layernorm = nn.LayerNorm(self.projection_dim)
            self.num_proj = nn.Linear(len(self.args.num_cols), 
                                  self.projection_dim)
        
        # batch normalization
        self.batchnorm = nn.BatchNorm1d(self.projection_dim)

        # comb_proj + len(num_cols) into model
        self.lstm = nn.LSTM(
            self.projection_dim, self.lstm_hidden_dim, self.n_layers, batch_first=True
        )

        

        # Fully connected layer
        # self.fc = nn.Linear(self.lstm_hidden_dim * self.args.max_seq_len, 128) # 맞춰야할 문제수
        self.fc = nn.Linear(self.lstm_hidden_dim * self.args.max_seq_len, self.questions)
        # self.fca = nn.Linear(128, self.questions)
        # self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.8)
    
    def forward(self, input):
        
        # input 형태 : ['elapsed_time', 'level', 'event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
        # a = len(self.args.cate_cols)
        a = len(self.num_cols)
        num_data = input[:a]  # 왜 a 대신 len(self.num_cols)로 하면 안되는건지?
        cate_data = input[a:-1] # mask 제외
        # test, question, tag, _, mask, interaction = input

        # batch_size = interaction.size(0)
        batch_size = input[0].size(0)

        # Embedding
        # embed_event_name = self.embedding_event_name(event_name) 형식 
        
        # categorical embedding
        # concat all categorical + numerical
        num_data = torch.stack(num_data, 2) # list to tensor
        num_X = self.layernorm(self.num_proj(num_data))
        # num_X = self.num_proj(num_data)

        if self.cate_cols:
            cat_embed = torch.cat(
                [getattr(self, 'embedding_' + cat_col)(cate_data[i]) for i, cat_col in enumerate(self.cate_cols)],
                2,
            )
            cat_X = self.layernorm(self.cat_proj(cat_embed))
            # cat_X = self.cat_proj(cat_embed)
            X = torch.cat([num_X, cat_X], -1)
        else: X = num_X
        
        out, _ = self.lstm(X)
        # out = self.dropout(out)
        out = out.contiguous().view(batch_size, -1)
        # out = self.relu(self.fc(out))
        out = self.fc(out)
        # out = self.fca(out)
        # out = self.dropout(out)
        out = self.activation(out)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.projection_dim = self.args.projection_dim
        self.n_layers = self.args.n_layers
        self.lstm_hidden_dim = self.args.lstm_hidden_dim
        self.cate_cols = self.args.cate_cols
        self.num_cols = self.args.num_cols
        self.bert_hidden_dim = 8
        self.inter_size = 8
        
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # questions 결정
        qnum = [None, 3, 10, 5]
        self.questions = qnum[self.args.level_group]

        # embedding
        for cat_col in self.args.cate_cols:
            setattr(self,
                    'embedding_' + cat_col,
                    nn.Embedding(getattr(self.args, 'input_size_'+cat_col) + 2, self.hidden_dim)       
            )       # 원래 +1 이었는데 unknown value까지 생각해서 +2로 바꿈

        # embedding combination projection
        self.cat_proj = nn.Linear(self.hidden_dim * len(self.args.cate_cols), 
                                   self.projection_dim // 2)
        
        
        if self.args.cate_cols:
            # layer normalization
            self.layernorm = nn.LayerNorm(self.projection_dim // 2)
            self.num_proj = nn.Linear(len(self.args.num_cols), 
                                  self.projection_dim // 2)
        else: 
            # layer normalization
            self.layernorm = nn.LayerNorm(self.projection_dim)
            self.num_proj = nn.Linear(len(self.args.num_cols), 
                                  self.projection_dim)
        
        # batch normalization
        self.batchnorm = nn.BatchNorm1d(self.projection_dim)

        # comb_proj + len(num_cols) into model
        self.lstm = nn.LSTM(
            self.projection_dim, self.lstm_hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.bert_hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.inter_size,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.lstm_hidden_dim * self.args.max_seq_len, self.questions)

        self.activation = nn.Sigmoid()

    def forward(self, input):

        a = len(self.num_cols)
        num_data = input[:a]  # 왜 a 대신 len(self.num_cols)로 하면 안되는건지?
        cate_data = input[a:-1] # mask 제외
        mask = input[-1]

        batch_size = input[0].size(0)

        # Embedding
        # embed_event_name = self.embedding_event_name(event_name) 형식 
        
        # categorical embedding
        # concat all categorical + numerical
        num_data = torch.stack(num_data, 2) # list to tensor
        num_X = self.layernorm(self.num_proj(num_data))
        # num_X = self.num_proj(num_data)

        if self.cate_cols:
            cat_embed = torch.cat(
                [getattr(self, 'embedding_' + cat_col)(cate_data[i]) for i, cat_col in enumerate(self.cate_cols)],
                2,
            )
            cat_X = self.layernorm(self.cat_proj(cat_embed))
            # cat_X = self.cat_proj(cat_embed)
            X = torch.cat([num_X, cat_X], -1)
        else: X = num_X
        
        out, _ = self.lstm(X)
        

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        out = sequence_output.contiguous().view(batch_size, -1)
        out = self.fc(out).view(batch_size, -1)
        out = self.activation(out)
        
        return out


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )

        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, mask, interaction = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out).view(batch_size, -1)
        return out
