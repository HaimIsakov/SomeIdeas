import os

# tasks_dict2 = {1: 'just_values', 2: 'just_graph', 3: 'graph_and_values', 5: 'one_head_attention'}
class MyTasks:
    def __init__(self, tasks_dict, dataset):
        self.tasks_dict = tasks_dict
        self.dataset = dataset

    def get_task_files(self, task_name):
        if task_name not in self.tasks_dict:
            print("Task is not in global tasks dictionary")
        return self.tasks_dict[task_name](self)

    # @staticmethod
    def just_values(self):
        directory_name = "JustValues"
        mission = 'just_values'
        # params_file_path = os.path.join(directory_name, 'params', "params_file_1_gcn_just_values", f"{self.dataset}_{mission}_params.json")
        try:
            params_file_path = os.path.join(directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        except:
            print("Use default params file")
            params_file_path = os.path.join(directory_name, 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path

    # @staticmethod
    def just_graph_structure(self):
        directory_name = "JustGraphStructure"
        mission = 'just_graph'
        # params_file_path = os.path.join(directory_name, 'params', "params_file_1_gcn_just_values", f"{self.dataset}_{mission}_params.json")
        params_file_path = os.path.join(directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        return directory_name, mission, params_file_path

    # @staticmethod
    def values_and_graph_structure(self):
        directory_name = "ValuesAndGraphStructure"
        mission = 'graph_and_values'
        try:
            # params_file_path = os.path.join(directory_name, 'params', "params_file_1_gcn_just_values", f"{self.dataset}_{mission}_params.json")
            params_file_path = os.path.join(directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        except:
            print("Use default params file")
            params_file_path = os.path.join(directory_name, 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path

    # @staticmethod
    def double_gcn_layers(self):
        directory_name = "DoubleGcnLayers"
        mission = 'double_gcn_layer'
        try:
            # TODO: Run NNI on double GCN Layer, at the meantime use the parameters from gmic+V
            params_file_path = os.path.join("ValuesAndGraphStructure", 'params', "best_params", f"{self.dataset}_graph_and_values.json")
        except:
            print("Use default params file")
            params_file_path = os.path.join("ValuesAndGraphStructure", 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path

    # @staticmethod
    def pytorch_geometric(self):
        directory_name = "PytorchGeometric"
        mission = 'GraphAttentionModel'
        params_file_path = os.path.join(directory_name, 'Models', "pytorch_geometric_params_file.json")
        return directory_name, mission, params_file_path

    def one_head_attention(self):
        directory_name = "OneHeadAttention"
        mission = 'one_head_attention'
        params_file_path = os.path.join(directory_name, 'Models', "one_head_attention_params_file.json")
        return directory_name, mission, params_file_path

    def yoram_attention(self):
        directory_name = "YoramAttention"
        mission = 'yoram_attention'
        try:
            params_file_path = os.path.join(directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        except:
            print("Use default params file")
            params_file_path = os.path.join(directory_name, 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path
