import json
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class Tracker():
    def __init__(self, json_save_pth):
        '''
        Initializes the Tracker object.

        Parameters:
        json_save_pth (str): Path to save the tracked metrics JSON file.
        '''
        try:
            self.save_pth = json_save_pth
            self.tracking_metrics = {}
        except Exception as e:
            print(f"error in __init__: {e}")
    
    def log_metrics(self, epoch, metric_names: list, metric_values: list):
        '''
        Logs metrics for a given epoch.

        Parameters:
        epoch (int): The epoch number.
        metric_names (list): List of metric names (e.g. ["loss", "accuracy"]).
        metric_values (list): List of corresponding metric values.
        '''
        try:
            epoch_dict = {}
            for name, value in zip(metric_names, metric_values):
                epoch_dict[name] = value
            self.tracking_metrics[str(epoch)] = epoch_dict
        except Exception as e:
            print(f"error in log_metrics: {e}")
    
    def save_metrics(self, save_path=None):
        '''
        Saves tracked metrics to a JSON file.

        Parameters:
        save_path (str, optional): File path to save the metrics. Defaults to the path provided in initialization.
        '''
        try:
            if not save_path:
                save_path = self.save_pth
            with open(save_path, 'w') as f:
                json.dump(self.tracking_metrics, f)
        except Exception as e:
            print(f"error in save_metrics: {e}")

    def load_metrics(self, load_path):
        '''
        Loads tracked metrics from a JSON file.

        Parameters:
        load_path (str): Path to the metrics JSON file.

        Returns:
        dict: Dictionary containing the loaded metrics.
        '''
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"error in load_metrics: {e}")
            return {}

    def graph_metrics(self, metrics: dict, display=True, save=True, save_path='./'):
        '''
        Generates graphs for each metric over epochs, saves them as images, 
        and optionally displays them.

        Parameters:
        metrics (dict): Dictionary of metrics, typically loaded using load_metrics().
        display (bool): If True, display the graphs. Defaults to True.
        save (bool): If True, save the graphs as images. Defaults to True.
        save_path (str): Path where to save the images. Defaults to the current directory.

        Returns:
        list: List of PIL Image objects for each metric graph.
        '''
        try:
            metric_names = list(next(iter(metrics.values())).keys())
            epochs = sorted([int(e) for e in metrics.keys()])
            str_epochs = [str(e) for e in epochs]
            images = []

            for name in metric_names:
                values = [metrics[epoch][name] for epoch in str_epochs]
                plt.figure()
                plt.plot(epochs, values, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel(name)
                plt.title(f'{name} Over Epochs')
                plt.grid(True)

                if save:
                    image_file = f'{save_path}/{name}.jpg'
                    plt.savefig(image_file, format='jpg')
                    print(f"Saved {name} graph as {image_file}")
                
                if display:
                    plt.show()

                plt.close()
                
            return images
        except Exception as e:
            print(f"error in graph_metrics: {e}")
            return []