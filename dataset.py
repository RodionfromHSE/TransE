import torch
from torch.utils.data import Dataset



class Triple:
    def __init__(self, head, tail, relation):
        """
        @param head: head node index
        @param tail: tail node index
        @param relation: relation type index
        """
        self.h = head
        self.t = tail
        self.r = relation

    def to_tuple(self):
        return self.h, self.r, self.t
    

class WN18RR(Dataset):
    def __init__(self, triples):
        """
        @param triples: list of triples
        """
        assert len(triples) > 0, 'triples must not be empty'
        assert isinstance(triples[0], Triple), 'triples must be a Triple object'
        self.triples = triples  

    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            return self.triples[idx].to_tuple()
        tuples = [t.to_tuple() for t in self.triples[idx]]
        return tuples