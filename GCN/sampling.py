import numpy as np
import torch
from params import create_combined_text

def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, train_text_dict,index_to_champion_name,sampling_sift_pos,mode):
    neg_candidates = np.arange(item_num)
    text=[]
    if sampling_sift_pos:
        neg_items = []
        for idx in range(pos_train_data.shape[0]):

            user = pos_train_data[idx][0]
            if mode == 0:
                text.append(create_combined_text(train_text_dict[user.item()]))
            else:
                data = train_text_dict[user.item()]
                champ_data = index_to_champion_name[pos_train_data[idx][1].item()]
                text.append(create_combined_text(data[data['Champion'] == champ_data]))

            probs = np.ones(item_num)
            probs[interacted_items[user]] = 0
            probs /= np.sum(probs)

            u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)

            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(neg_candidates, (pos_train_data.shape[0], neg_ratio), replace=True)

    neg_items = torch.from_numpy(neg_items)

    return pos_train_data[:,0], pos_train_data[:,1], neg_items,text  # users, pos_items, neg_items
