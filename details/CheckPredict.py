import matplotlib.pyplot as plt
import torch
import numpy as np

val_predict = val(model, val_loader)

def flat(l):
    flatten = []
    for sub_list in l:
        for idx in range(len(sub_list)):
            flatten.append((sub_list[idx].type(torch.int)).cpu().item())
    return flatten

# get true validation scores
def get_true_val(dataloader):
    true_score = []
    for batch_idx, data in tqdm(enumerate(dataloader)):
        val_label = torch.tensor(list(map(float, data['target'])))
        for idx in range(len(val_label)):
            true_score.append(val_label[idx].cpu().item())
    print("length: {}".format(len(true_score)))
    return true_score


val_res = flat(val_predict)
val_ture = get_true_val(val_loader)

bins = np.linspace(50, 100, 20)
plt.hist(val_res, bins, alpha=0.5, label='predicted attention score')
plt.hist(val_ture, bins,  alpha=0.5, label='true attention score')
plt.legend(loc='upper right')
plt.show()

plt.scatter(x=val_ture, y=val_res, alpha=0.5)
plt.xlabel('True attention score')
plt.ylabel('Predicted attention score')
plt.legend(loc='upper right')
plt.show()

