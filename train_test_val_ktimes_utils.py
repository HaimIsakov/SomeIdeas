

def get_train_graphs_list(train_data):
    dataset_dict = train_data.dataset.dataset_dict
    indices = train_data.indices
    graphs_dict = {}
    for index in indices:
        graphs_dict[index] = dataset_dict[index]['graph']
    train_graphs_list = list(graphs_dict.values())
    return train_graphs_list


def get_labels_distribution(data_loader):
    try:
        dataset_dict = data_loader.dataset.dataset.dataset_dict
    except:
        dataset_dict = data_loader.dataset.dataset_dict
    indices = data_loader.dataset.indices
    label_dict = {}
    for index in indices:
        label_dict[index] = dataset_dict[index]['label']
    labels_values = list(label_dict.values())
    # labels_distribution = Counter(labels_values)
    labels_distribution = {0: labels_values.count(0), 1: labels_values.count(1)}
    return labels_distribution


def rerun_if_bad_train_result(early_stopping_results, threshold=0.5):
    flag = False
    if early_stopping_results['train_auc'] <= threshold:
        flag = True
    return flag
