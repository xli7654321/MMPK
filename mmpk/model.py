import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import RobertaModel
from gnn import GraphEncoder

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class CrossAttention(nn.Module):
    def __init__(self, d_mol, d_sub, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_mol, d_model)
        self.w_k = nn.Linear(d_sub, d_model)
        self.w_v = nn.Linear(d_sub, d_model)

    def forward(self, mol_embedding, sub_embedding, mask=None):
        q = self.w_q(mol_embedding)
        k = self.w_k(sub_embedding)
        v = self.w_v(sub_embedding)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        
        # scores, mask: (batch_size, num_subs)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)

        return output, scores

class MMPKPredictor(nn.Module):
    def __init__(self, args, num_tasks=1):
        super().__init__()
        self.mol_encoder = GraphEncoder(num_layers=args.num_layers_mol,
                                        d_emb=args.d_mol,
                                        JK=args.JK,
                                        dropout=args.dropout_mol,
                                        gnn_type=args.gnn_type_mol,
                                        graph_pooling=args.graph_pooling,
                                        num_heads=args.num_heads_gat)
        self.sub_encoder = GraphEncoder(num_layers=args.num_layers_sub,
                                        d_emb=args.d_sub,
                                        JK=args.JK,
                                        dropout=args.dropout_sub,
                                        gnn_type=args.gnn_type_sub,
                                        graph_pooling=args.graph_pooling,
                                        num_heads=args.num_heads_gat)
        self.cross_attention = CrossAttention(d_mol=args.d_mol,
                                              d_sub=args.d_sub,
                                              d_model=args.d_model)

        self.mol_proj = nn.Linear(args.d_mol, args.d_model)
        self.fusion_proj = nn.Linear(args.d_model * 2, args.d_model)
        
        self.chemberta = RobertaModel.from_pretrained('ChemBERTa/ChemBERTa-77M-MTR')
        
        self.norm = LayerNorm(args.d_model + 384)
        self.output_layer = nn.Linear(args.d_model + 384 + 1, num_tasks)
        
        self.freeze()

    def forward(self, mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose):
        mol_embedding = self.mol_encoder(mol_gdata)  # (batch_size, d_mol)
        sub_embedding = self.sub_encoder(sub_gdata)  # (num_subs, d_sub)
        
        sub_embedding, scores = self.cross_attention(mol_embedding, sub_embedding, sub_mask)  # (batch_size, num_subs)
        mol_embedding = self.mol_proj(mol_embedding)
        
        graph_embedding = torch.cat((sub_embedding, mol_embedding), dim=-1)
        graph_embedding = self.fusion_proj(graph_embedding)
        
        smi_embedding = self.chemberta(input_ids=token_ids, attention_mask=token_mask).last_hidden_state[:, 0, :]
        
        embedding = self.norm(torch.cat((graph_embedding, smi_embedding), dim=-1))
        
        dose = dose.view(-1, 1)
        embedding = torch.cat((embedding, dose), dim=-1)
        
        y_hat = self.output_layer(embedding)

        return y_hat, scores
    
    def freeze(self):
        for param in self.chemberta.parameters():
            param.requires_grad = False

if __name__ == '__main__':
    pass
