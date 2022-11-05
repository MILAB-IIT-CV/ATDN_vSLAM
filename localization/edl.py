import torch

def distance(V1, V2):
    # SQRT can be potenetially elliminated for preformance purposes
    return ((V1-V2)**2).sum(-1) #.sqrt()

def embedding_distance_loss(points, embeddings):
    P1, P2, P3 = points
    E1, E2, E3 = embeddings
    
    # Point distances
    DP12 = distance(P1, P2)
    DP32 = distance(P3, P2)
    DP31 = distance(P3, P1)
    
    # Embedding distances
    DE12 = distance(E1, E2)
    DE32 = distance(E3, E2)
    DE31 = distance(E3, E1)


    L1 = torch.abs((DE12/DE32)-(DP12/DP32))
    L2 = torch.abs((DE12/DE31)-(DP12/DP31))
    L3 = torch.abs((DE32/DE31)-(DP32/DP31))

    loss = L1 + L2 + L3
    return loss
