import csv
import os

with open(os.path.join(os.path.dirname(__file__), 'combinations.csv')) as file:
    reader = csv.reader(file)
    combinations = [line[0] for line in reader]

results_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')

def get_fscore(model_name, dataset_name):
    file = os.path.join(results_folder, f'{model_name}-val{"-" + dataset_name if dataset_name else ""}.log')
    with open(file, 'r') as file:
        lines = file.readlines()
        line = next(filter(lambda line: line.startswith('Metrics/Fscore'), lines))
        line = line.strip().split(' : ')[1]
        assert line.startswith('tensor('); line = line.removeprefix('tensor(')
        assert line.endswith(')'); line = line.removesuffix(')')
        value = float(line)
        return value
