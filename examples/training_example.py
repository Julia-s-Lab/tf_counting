from tqdm import tqdm
import torch.nn as nn
from random import randint
import torch
import math
from causal_transformer_decoder import (
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
)

import numpy as np
import fire, itertools, os
from matplotlib import pyplot as plt

CHARS = "bac"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CharVocab:
    def __init__(self):

        self.idx_to_char = ["P", "S", "E"]
        self.idx_pad_token = 0
        self.idx_start_token = 1
        self.idx_end_token = 2
        self.idx_to_char += [char for char in CHARS]
        self.idx_to_char += [str(i) for i in range(config['N'])]
        self.char_to_idx = {
            self.idx_to_char[k]: k for k in range(len(self.idx_to_char))
        }

    def __len__(self):
        return len(self.idx_to_char)

    def sent_to_idx(self, sent):
        return [self.char_to_idx[char] for char in sent]

    def sents_to_idx(self, sents):
        return [self.sent_to_idx(sent) for sent in sents]

    def idx_to_sent(self, idx_list):
        return "".join([self.idx_to_char[idx] for idx in idx_list])


def create_random_sent(max_length=20):
    """ Create random sequence containing 1 to {max_length} characters """
    length = randint(1, max_length)
    sent = "".join([CHARS[randint(0, len(CHARS) - 1)] for _ in range(length)])
    return sent

def hypothesis(count):
    value = 0
    for k, v in count.items():
        value += config['coeffs'][k]*(v**config['powers'][k])
    return (value % config['N'])

def generate_xy():
    """ create input/label pair """
    sent_input = create_random_sent()
    count, label = dict(zip(CHARS, [0 for i in range(len(CHARS))])), ""
    for char in sent_input:
        count[char] += 1
        label += str(hypothesis(count))
    return sent_input, label

def batch_generator(vocab):
    while True:
        x, y = list(zip(*[generate_xy() for _ in range(config['batch_size'])]))
        x_idx = vocab.sents_to_idx(x)
        y_idx = vocab.sents_to_idx(y)

        # add end token as it is used for training loss
        y_idx = [elt + [vocab.idx_end_token] for elt in y_idx]

        x_idx = [torch.LongTensor(elt).to(DEVICE) for elt in x_idx]
        y_idx = [torch.LongTensor(elt).to(DEVICE) for elt in y_idx]
        x_padded = nn.utils.rnn.pad_sequence(x_idx, batch_first=True)
        y_padded = nn.utils.rnn.pad_sequence(y_idx, batch_first=True)
        yield x_padded, y_padded


class PositionalEncoding(nn.Module):
    """ Code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask


class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()

        hdim = 256
        nhead = config['num_heads']
        dim_feedforward = hdim * 4
        num_layers = config['num_layers']
        self.vocab = vocab
        vocab_size = len(vocab)

        self.embedding = nn.Embedding(vocab_size, hdim)
        self.positional_encoding = PositionalEncoding(hdim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers,
        ).to(device=DEVICE)

        if config['optimized_decoder']:
            self.decoder = CausalTransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward,
                ),
                num_layers=num_layers,
            ).to(device=DEVICE)
        else:
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=hdim, nhead=nhead, dim_feedforward=dim_feedforward
                ),
                num_layers=num_layers,
            ).to(device=DEVICE)

        self.classification_layer = nn.Linear(hdim, vocab_size)

    def forward(self, inputs, teach_forcing_tokens):
        """This function should only be used for training

        Args:
            inputs (torch.Tensor): bsz, input_len, hdim
            teach_forcing_tokens (torch.Tensor): bsz, output_len, hdim
                Each tensor needs to start with start token.
                Doesn't need to end with end token.

        Returns:
            (torch.Tensor): [description]
        """

        input_embed = self.positional_encoding(
            self.embedding(inputs).permute(1, 0, 2)
        )  # input_len, bsz, hdim

        teach_forcing_embed = self.positional_encoding(
            self.embedding(teach_forcing_tokens).permute(1, 0, 2)
        )  # output_len, bsz, hdim

        memory_mask = inputs == 0  # for source padding masks

        encoded = self.encoder(
            input_embed, src_key_padding_mask=memory_mask
        )  # input_len, bsz, hdim

        if config['optimized_decoder']:
            decoded = self.decoder(
                teach_forcing_embed,
                memory=encoded,
                memory_key_padding_mask=memory_mask,
            )  # output_len, bsz, hdim
        else:
            tgt_mask = generate_square_subsequent_mask(
                teach_forcing_embed.size(0), DEVICE
            )
            decoded = self.decoder(
                teach_forcing_embed,
                encoded,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_mask,
            )  # output_len, bsz, hdim

        logits = self.classification_layer(decoded)  # output_len, bsz, vocab_size
        return logits.permute(1, 0, 2)  # bsz, output_len, vocab_size

    def predict(self, sent: str):
        """ Used for inference """

        sent_idx = self.vocab.sent_to_idx(sent)
        sent_tensor = torch.LongTensor(sent_idx).to(DEVICE).unsqueeze(0)

        input_embed = self.positional_encoding(
            self.embedding(sent_tensor).permute(1, 0, 2)
        )  # input_len, 1, hdim

        memory_mask = sent_tensor == 0  # for source padding masks

        encoded = self.encoder(
            input_embed, src_key_padding_mask=memory_mask
        )  # input_len, 1, hdim

        decoded_tokens = (
            torch.LongTensor([self.vocab.idx_start_token]).to(DEVICE).unsqueeze(1)
        )  # 1, 1

        output_tokens = []
        cache = None
        # generation loop
        while len(output_tokens) < 256:  # max length of generation

            decoded_embedding = self.positional_encoding(self.embedding(decoded_tokens))

            if config['optimized_decoder']:
                decoded, cache = self.decoder(
                    decoded_embedding,
                    encoded,
                    cache,
                    memory_key_padding_mask=memory_mask,
                )
            else:
                tgt_mask = generate_square_subsequent_mask(
                    decoded_tokens.size(0), DEVICE
                )
                decoded = self.decoder(
                    decoded_embedding,
                    encoded,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_mask,
                )

            logits = self.classification_layer(decoded[-1, :, :])  # 1, vocab_size
            new_token = logits.argmax(1).item()
            if new_token == self.vocab.idx_end_token:  # end of generation
                break
            output_tokens.append(new_token)
            decoded_tokens = torch.cat(
                [
                    decoded_tokens,
                    torch.LongTensor([new_token]).unsqueeze(1).to(DEVICE),
                ],
                dim=0,
            )  # current_output_len, 1

        return self.vocab.idx_to_sent(output_tokens)


def run_experiment():

    vocab = CharVocab()
    model = Model(vocab).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(enumerate(batch_generator(vocab)))

    task_coeffs = ''.join(map(str, list(config['coeffs'].values())))
    task_powers = ''.join(map(str, list(config['powers'].values())))
    path = f'./results/{config["N"]}/powers_{task_powers}/coeffs_{task_coeffs}/{config["num_layers"]}/{config["num_heads"]}/{config["lr"]}/{config["batch_size"]}'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path+'/results', exist_ok=True)

    _loss = []
    for num_step, data in pbar:
        model.train()
        inputs, labels = data

        teach_forcing_tokens = labels.clone()
        teach_forcing_tokens = torch.cat(
            [torch.ones_like(teach_forcing_tokens[:, :1]), teach_forcing_tokens], dim=1
        )  # adding start token

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, teach_forcing_tokens)
        loss = criterion(outputs[:, :-1, :].permute(0, 2, 1), labels)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": loss})  # display loss in progress bar

        _loss.append(loss.item())

        if num_step == config['max_steps']:
            break

        if num_step % config['interval_test'] == 0:  # try inference to check that it works well & save checkpoint
            torch.save({
            'step': num_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            }, path+f'/results/model{num_step}.pt')

            test_sent = create_random_sent()
            model.eval()
            with torch.no_grad():
                print(f"Input {test_sent}, output: {model.predict(test_sent)}")
        
    np.save(path+f'/results/loss.npy', _loss)
    plt.title(f"Train Loss")
    plt.plot(_loss, lw=2)
    plt.grid()
    plt.savefig(path+'/results/loss.pdf')
    plt.show()
    plt.close()

def retrieve_config(sweep_step):
    grid = {
        'coeffs': [dict(zip(CHARS, [1 for i in range(len(CHARS))])), dict(zip(CHARS, [2 for i in range(len(CHARS))])), dict(zip(CHARS, [i for i in range(len(CHARS))]))],
        'powers': [dict(zip(CHARS, [1 for i in range(len(CHARS))])), dict(zip(CHARS, [2 for i in range(len(CHARS))])), dict(zip(CHARS, [i for i in range(len(CHARS))]))],
        'N': [3],
        'lr': [5e-5],
        'batch_size': [128],
        'max_steps': [10000],
        'interval_test': [100],
        'num_layers': [2],
        'num_heads': [4],
        'optimized_decoder': [True]
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  # slurm var will start from 1

    config = {
        'sweep_step': sweep_step,
        'coeffs': step_grid['coeffs'],
        'powers': step_grid['powers'],
        'N': step_grid['N'],
        'lr': step_grid['lr'],
        'batch_size': step_grid['batch_size'],
        'max_steps': step_grid['max_steps'],
        'interval_test': step_grid['interval_test'],
        'num_layers': step_grid['num_layers'],
        'num_heads': step_grid['num_heads'],
        'optimized_decoder': step_grid['optimized_decoder']
    }

    return config

def main(sweep_step):
    global config
    config = retrieve_config(sweep_step)
    print(config)
    print('---------.--------')
    run_experiment()

if __name__ == '__main__':
    fire.Fire(main)