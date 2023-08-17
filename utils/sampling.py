import numpy as np
from torchvision import datasets, transforms

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users

def noniid(dataset, args):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(args.num_users)}
    
    min_num = 40
    max_num = 260
    print('Train dataset size:', num_dataset)
    random_num_size = np.random.randint(min_num, max_num+1, size=args.num_users)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    for i, rand_num in enumerate(random_num_size):

        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set


    return dict_users

def noniid_v2(dataset, args):
    dict_users = {i: list() for i in range(args.num_users)}
    women = []
    men = []
    num_men = 0
    num_women = 0
    for i in range(len(dataset)):
        if dataset.targets[i] == 0:
            num_men += 1
            men.append(i)
        else:
            num_women += 1
            women.append(i)
    # print("num men:", num_men, "num women:", num_women)

    min_num = 40
    step_men = int(1/49. * (num_men/25 - 80))
    step_women = int(1/49. * (num_women/25 - 80))
    print("step men:", step_men, "step women:", step_women)
    itter_men = min_num
    itter_women = min_num + (args.num_users -1) * step_women
    for i in range(args.num_users):
        selection = men[:itter_men] + women[:itter_women]
        # print(len(men[:itter_men]), len(women[:itter_women]))
        dict_users[i].extend(selection)
        del men[:itter_men]
        del women[:itter_women]

        itter_men += step_men
        itter_women -= step_women

    return dict_users


def iid_v2(dataset, num_users):
    dict_users = {i: list() for i in range(num_users)}
    women = []
    men = []
    num_men = 0
    num_women = 0
    for i in range(len(dataset)):
        if dataset.targets[i] == 0:
            num_men += 1
            men.append(i)
        else:
            num_women += 1
            women.append(i)
    # print("num men:", num_men, "num women:", num_women)

    itter_men = int(num_men/num_users)
    diff_men = num_men - int(num_men/num_users) * num_users
    itter_women = int(num_women/num_users)
    diff_women = num_women - int(num_women/num_users) * num_users
    for i in range(num_users):
        selection = men[:itter_men] + women[:itter_women]
        # print(len(men[:itter_men]), len(women[:itter_women]))
        dict_users[i].extend(selection)
        del men[:itter_men]
        del women[:itter_women]

    return dict_users