import torch

def unique_topk(k, similarity, labels):
    _, top = torch.sort(similarity, descending = True, dim = 1)
    topk = torch.full((top.size(0), k), -1, dtype = torch.long)  

    for i in range(top.size(0)):
        classes = set()
        row = []
        
        for idx in top[i]:  
            label = str(labels[idx].tolist())
            
            if label not in classes:
                classes.add(label)
                row.append(idx.item())

            if len(row) == k:
                break 

        topk[i, :len(row)] = torch.tensor(row) 

    return topk 
