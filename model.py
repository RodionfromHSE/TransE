import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn.kge as KGEModel

class TransE(KGEModel):
    """
    Implementation of the TransE model. 
    Base model from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/kge/base.html#KGEModel.loader

    Methods:
    -------
    init: Initialize the model
    forward: Forward pass
    random_sample: Corrupted triple (Either tail or head are substituted but not both)
    loss: Compute the loss (negative TransE norm for maximization)
    loader: Returns a mini-batch loader that samples a subset of triplets.
    test: Evaluates the model quality by computing Mean Rank and Hits @ k across all possible tail entities. Returns tuple of (Mean Rank, Hits @ k)
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
    ):
        """
        @param num_nodes: number of nodes / entities
        @param num_relations: number of relations
        @param hidden_channels: number of hidden channels / embedding dimensions
        @param p_norm: p-norm to use
        @param margin: margin to use in margin-based ranking loss
        @param sparse: whether graph is sparse or not
        """

        self.p_norm = p_norm
        self.margin = margin
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.reset_parameters()

    def reset_parameters(self):
        """
        initialize parameters like in the paper from uniform(-6 / sqrt(k), 6 / sqrt(k)), where k is embedding dimension
        """
        b = 6.0 / (self.hidden_channels ** 0.5)
        nn.init.uniform_(self.node_emb.weight, -b, b)
        nn.init.uniform_(self.rel_emb.weight, -b, b)

        F.normalize(self.rel_emb.weight, p=self.p_norm, dim=-1, out=self.rel_emb.weight)


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

        return F.normalize(head_emb + rel_emb - tail_emb, p=self.p_norm, dim=-1)
    
    def loss(self, head_index: Tensor, rel_type: Tensor, tail_index: Tensor) -> Tensor:
        """
        @param head_index: head node index
        @param rel_type: relation type index
        @param tail_index: tail node index
        @return: output of the model
        """

        head_emb = self.node_emb(head_index)
        rel_emb = self.rel_emb(rel_type)
        tail_emb = self.node_emb(tail_index)

        pos_score = self(head_emb, rel_emb, tail_emb)
        neg_score = self(*self.random_sample(head_emb, rel_emb, tail_emb))

        return F.margin_ranking_loss(pos_score, neg_score, target=torch.zeros_like(pos_score), margin=self.margin)
    