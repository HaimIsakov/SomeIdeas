from concat_graph_and_values import ConcatValuesAndGraphStructure
from train_test_val_one_time import TrainTestValOneTime


def start_training_process(trainer_and_tester, train_loader, val_loader, test_loader, RECEIVED_PARAMS, device,
                           train_val_dataset):
    rerun_counter = 0
    early_stopping_results = trainer_and_tester.train()

    # # If the train auc is too low (under 0.5 for example) try to rerun the training process again
    flag = rerun_if_bad_train_result(early_stopping_results)
    while flag and rerun_counter <= 3:
        print(f"Rerun this train-val split again because train auc is:{early_stopping_results['train_auc']:.4f}")
        print(f"Rerun number {rerun_counter}")
        # model = self.get_model().to(self.device)
        model = get_model(train_val_dataset, RECEIVED_PARAMS, device)
        trainer_and_tester = TrainTestValOneTime(model, RECEIVED_PARAMS, train_loader, val_loader,
                                                 test_loader, device)
        early_stopping_results = trainer_and_tester.train()
        flag = rerun_if_bad_train_result(early_stopping_results)
        rerun_counter += 1  # rerun_counter - the number of chances we give the model to converge again
    return early_stopping_results


def get_model(dataset, RECEIVED_PARAMS, device):
    data_size = dataset.get_vector_size()
    nodes_number = dataset.nodes_number()
    model = ConcatValuesAndGraphStructure(nodes_number, data_size, RECEIVED_PARAMS, device)
    model = model.to(device)
    return model


def get_train_graphs_list(train_data):
    dataset_dict = train_data.dataset.dataset_dict
    indices = train_data.indices
    graphs_dict = {}
    for index in indices:
        graphs_dict[index] = dataset_dict[index]['graph']
    train_graphs_list = list(graphs_dict.values())
    return train_graphs_list


def get_labels_distribution(data_loader):
    dataset_dict = data_loader.dataset.dataset.dataset_dict
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
