from config import opt
import torch
from torch.utils.data import DataLoader


def check_null(dataloader):
    for batch, data in enumerate(dataloader):
        # size: (batch, channel, 190 ,190 ,190)
        # print(batch)
        for idx in range(opt.batch_size):
            # print(idx)
            for z in range(data['img']['data'].shape[-1]):
                # print(data['img']['data'][idx, 0, :, :, z])
                if torch.isnan(data['img']['data'][idx, 0, :, :, z]).any():
                    print("batch {} image {}:  {} missing values in slice {}".format(batch, idx,
                                                                                     sum(torch.isnan(
                                                                                         data['img']['data'][idx, 0, :,
                                                                                         :, z]).view(-1)),
                                                                                     z))

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    img_type = 'FA'
    data_set = make_dataset(img_type, target='score')

    # take a look into the sample returned by the dataset
    # default image has missing values
    sample = data_set[0]
    print(sample)
    fa_map = sample['img']
    print('Image Keys:', fa_map.keys())

    # set up a data-loader that directly fits torch
    num_subjects = len(data_set)
    training_split_ratio = 0.8
    num_training_subjects = int(training_split_ratio * num_subjects)
    subjects = data_set.subjects

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.SubjectsDataset(training_subjects)
    validation_set = torchio.SubjectsDataset(validation_subjects)

    temp = DataLoader(training_set, batch_size=16)
    check_null(temp)

