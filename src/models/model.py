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

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.projection_dim = self.args.projection_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # TODO: 각 feature에 맞는 embedding 구성하기
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        # self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        # self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        # self.embedding_question = nn.Embedding(
        #     self.args.n_questions + 1, self.hidden_dim // 3
        # )
        # self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        
        for cat_col in self.args.cate_cols:
            setattr(self,
                    'embedding_' + cat_col,
                    nn.Embedding(getattr(self.args, 'input_size_'+cat_col) + 1, self.hidden_dim)       
            )
        
        
        # embedding combination projection
        self.comb_proj = nn.Linear(self.hidden_dim * len(self.args.cate_cols), self.projection_dim - len(self.args.num_cols))

        # comb_proj + len(num_cols) into model
        self.lstm = nn.LSTM(
            self.input_dim, self.projection_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.projection_dim, 18) # 맞춰야할 문제수
    
    def forward(self, input):
        
        # input 형태 : ['elapsed_time', 'level', 'event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
        cate_data = input[2:]
        # test, question, tag, _, mask, interaction = input

        # batch_size = interaction.size(0)
        batch_size = input[0].size(0)

        # Embedding
        # embed_event_name = self.embedding_event_name(event_name) 형식 
        # concat all categorical 
        embed = torch.cat(
            [getattr(self, 'embedding_' + cat_col)(cate_data[i]) for i, cat_col in enumerate(self.args.cate_cols)],
            2,
        )

        X = self.comb_proj(embed)
        
        # 연속형 변수를 앞에 concat
        X = torch.cat([input[0].unsqueeze(-1), input[1].unsqueeze(-1), X], -1)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.projection_dim)
        out = self.fc(out)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):

        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
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

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
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
