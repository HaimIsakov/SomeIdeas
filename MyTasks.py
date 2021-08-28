import os


class MyTasks:
    def __init__(self, tasks_dict):
        self.tasks_dict = tasks_dict

    def get_task_files(self, task_name):
        if task_name not in self.tasks_dict:
            print("Task is not in global tasks dictionary")
        return self.tasks_dict[task_name]()

    @staticmethod
    def just_values():
        directory_name = "JustValues"
        mission = 'JustValues'
        params_file_path = os.path.join(directory_name, 'Models', "just_values_on_nodes_params_file.json")
        return directory_name, mission, params_file_path

    @staticmethod
    def just_graph_structure():
        directory_name = "JustGraphStructure"
        mission = 'JustGraphStructure'
        params_file_path = os.path.join(directory_name, 'Models', "graph_structure_params_file.json")
        return directory_name, mission, params_file_path

    @staticmethod
    def values_and_graph_structure():
        directory_name = "ValuesAndGraphStructure"
        mission = 'GraphStructure&Values'
        params_file_path = os.path.join(directory_name, 'Models', "values_and_graph_structure_on_nodes_params_file.json")
        return directory_name, mission, params_file_path

    @staticmethod
    def pytorch_geometric():
        directory_name = "PytorchGeometric"
        mission = 'GraphAttentionModel'
        params_file_path = os.path.join(directory_name, 'Models', "pytorch_geometric_params_file.json")
        return directory_name, mission, params_file_path

