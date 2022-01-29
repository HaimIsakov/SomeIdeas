import torch
import torch.nn as nn
import torch.nn.functional as F

class Model2(nn.Module):
    def __init__(self, max_len, embedding_dim, vgenes_dim, v_dict, vocab_size=21,
                 encoding_dim=30):  # check vocab_size, is it for the stop signal?
        super(Model2, self).__init__()
        self.encoding_dim = encoding_dim
        self.max_len = max_len
        self.vgenes_dim = vgenes_dim
        self.embedding_dim = embedding_dim
        self.v_dict = v_dict
        # self.vocab_size = vocab_size
        self.vocab_size = max_len * 20 + 2 + vgenes_dim
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=-1)
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim * (self.max_len + 1), 500),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(500, 200),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        self.mu = nn.Linear(200, self.encoding_dim)
        self.log_sigma = nn.Linear(200, self.encoding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 200),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(200, 500),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(500, self.max_len * 21 + self.vgenes_dim)
        )

    def reparameterize(self, x_mu, x_log_sigma):
        std = torch.exp(0.5 * x_log_sigma)
        eps = torch.randn_like(std)
        return x_mu + eps * std

    def forward(self, padded_input):
        x_emb = self.embedding(padded_input.long())
        x_emb = x_emb.view(-1, (self.max_len + 1) * self.embedding_dim)
        x = self.encoder(x_emb)
        x_mu = self.mu(x)
        x_log_sigma = self.log_sigma(x)
        encoded = self.reparameterize(x_mu, x_log_sigma)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 1, self.max_len * 21 + self.vgenes_dim)
        tcr_chain, v_gene = torch.split(decoded, self.max_len * 21, dim=2)
        v_gene = F.softmax(v_gene.view(-1, self.vgenes_dim), dim=1)
        tcr_chain = F.softmax(tcr_chain.view(-1, self.max_len, 21), dim=2)  # check if needed
        output = torch.cat((tcr_chain.view(-1, self.max_len * 21), v_gene), dim=1)
        return output, x_mu, x_log_sigma
