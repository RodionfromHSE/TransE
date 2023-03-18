import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np

class TransE(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_rels: int,
        emb_size: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
    ):
        """
        @param n_nodes: number of nodes (entities)
        @param n_rels: number of relations
        @param emb_size: number of hidden channels (embedding dimensions)
        @param p_norm: p-norm to use
        @param margin: margin to use in margin-based ranking loss
        """
        super().__init__()

        self.n_nodes = n_nodes
        self.n_rels = n_rels
        self.emb_size = emb_size
        
        self.node_emb = nn.Embedding(n_nodes, emb_size)
        self.rel_emb = nn.Embedding(n_rels, emb_size)

        self.p_norm = p_norm
        self.margin = margin

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """
        initialize parameters like in the paper from uniform(-6 / sqrt(k), 6 / sqrt(k)), where k is embedding dimension
        """
        b = 6.0 / (self.emb_size ** 0.5)
        self.node_emb.weight.data.uniform_(-b, b)

        self.rel_emb.weight.data.uniform_(-b, b)
        F.normalize(self.rel_emb.weight, p=self.p_norm, dim=-1, out=self.rel_emb.weight)

    @torch.no_grad()
    def random_sample(self, head: Tensor, rel: Tensor, tail: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        @param head: head node index
        @param rel: relation type index
        @param tail: tail node index
        @return: corrupted triple
        """

        batch_size = head.size(0)
        mask = np.random.choice([True, False], size=batch_size)

        new_head = head.clone()
        new_tail = tail.clone()
        new_head[mask] = torch.randint(self.n_nodes, size=(mask.sum(),))
        new_tail[~mask] = torch.randint(self.n_nodes, size=((~mask).sum(),),)
        
        return new_head, rel, new_tail


    def forward(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        """
        @param head_index: head node index
        @param rel_type: relation type index
        @param tail_index: tail node index
        @return: output of the model
        """

        head_emb = self.node_emb(head_index)
        rel_emb = self.rel_emb(rel_type)
        tail_emb = self.node_emb(tail_index)

        head_emb = F.normalize(head_emb, p=self.p_norm, dim=-1)
        tail_emb = F.normalize(tail_emb, p=self.p_norm, dim=-1)

        return F.normalize(head_emb + rel_emb - tail_emb, p=self.p_norm, dim=-1) # Get the distance
    
    def loss(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        """
        @param head_index: head node index
        @param rel_type: relation type index
        @param tail_index: tail node index
        @return: Margin-based ranking loss
        """
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(pos_score, neg_score, target=torch.zeros_like(pos_score), margin=self.margin)
    