import torch 


def calculate_gram_matrix(features):
    features = features.view(features.size(0),-1)
    G = torch.ones((features.shape[0],features.shape[0]), device=features.device)
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            G[i,j] = torch.dot(features[i],features[j])/(torch.norm(features[i])*torch.norm(features[j]))
    return G

def calculate_effective_rank(G):
    U, S, Vh = torch.linalg.svd(G)
    S = S/torch.sum(S)
    sum = 0.0
    for i in range(len(S)):
        sum += S[i]*torch.log(S[i])
    return -1.0*(sum.item())
        

def calculate_effective_rank_from_features(features):
    G = calculate_gram_matrix(features)
    return calculate_effective_rank(G)

