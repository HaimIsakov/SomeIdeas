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
        params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        if not os.path.isfile(params_file_path):
            print("Use default params file")
            params_file_path = os.path.join("Missions", directory_name, 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path

    # @staticmethod
    def just_graph_structure(self):
        directory_name = "JustGraphStructure"
        mission = 'just_graph'
        # params_file_path = os.path.join(directory_name, 'params', "params_file_1_gcn_just_values", f"{self.dataset}_{mission}_params.json")
        params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        if not os.path.isfile(params_file_path):
            print("Use default params file")
            params_file_path = os.path.join("Missions", directory_name, 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path

    # @staticmethod
    def values_and_graph_structure(self):
        directory_name = "ValuesAndGraphStructure"
        mission = 'graph_and_values'
        params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        if not os.path.isfile(params_file_path):
            print("Use default params file")
            params_file_path = os.path.join("Missions", "ValuesAndGraphStructure", 'Models', f"graph_and_values_params_file.json")
        return directory_name, mission, params_file_path

    # @staticmethod
    def double_gcn_layers(self):
        directory_name = "DoubleGcnLayers"
        mission = 'double_gcn_layer'
        # params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"params_{self.dataset}_{mission}.json")
        # params_file_path = os.path.join("ValuesAndGraphStructure", 'Models', f"{mission}_params_file.json")
        if not os.path.isfile(params_file_path):
            print("Use default params file")
            params_file_path = os.path.join("Missions", "ValuesAndGraphStructure", 'Models', f"graph_and_values_params_file.json")
        print("params_file_path", params_file_path)
        return directory_name, mission, params_file_path

    # @staticmethod
    def concat_graph_and_values(self):
        directory_name = "ConcatGraphAndValues"
        mission = 'concat_graph_and_values'
        # params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"params_{self.dataset}_{mission}.json")
        if not os.path.isfile(params_file_path):
            print("Use default params file")
            params_file_path = os.path.join("Missions", "ValuesAndGraphStructure", 'Models', f"graph_and_values_params_file.json")
        print("params_file_path", params_file_path)
        return directory_name, mission, params_file_path

    # @staticmethod
    def pytorch_geometric(self):
        directory_name = "PytorchGeometric"
        mission = 'GraphAttentionModel'
        params_file_path = os.path.join("Missions", directory_name, 'Models', "pytorch_geometric_params_file.json")
        return directory_name, mission, params_file_path

    def one_head_attention(self):
        directory_name = "OneHeadAttention"
        mission = 'one_head_attention'
        params_file_path = os.path.join("Missions", directory_name, 'Models', "one_head_attention_params_file.json")
        return directory_name, mission, params_file_path

    def yoram_attention(self):
        directory_name = "YoramAttention"
        mission = 'yoram_attention'
        params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        if not os.path.isfile(params_file_path):
            print("Use default params file")
            params_file_path = os.path.join("Missions", directory_name, 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path

    def fiedler_vector(self):
        directory_name = "FiedlerVector"
        mission = 'fiedler_vector'
        params_file_path = os.path.join("Missions", directory_name, 'params', "best_params", f"{self.dataset}_{mission}.json")
        if not os.path.isfile(params_file_path):
            print("Use default params file")
            params_file_path = os.path.join("Missions", directory_name, 'Models', f"{mission}_params_file.json")
        return directory_name, mission, params_file_path
