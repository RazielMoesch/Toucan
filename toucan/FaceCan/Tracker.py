import json
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class Tracker:
    def __init__(self, json_save_pth):
        self.save_pth = json_save_pth
        self.tracking_metrics = {}

    def log_metrics(self, epoch, metric_names, metric_values):
        try:
            epoch_dict = {}
            for name, value in zip(metric_names, metric_values):
                epoch_dict[name] = value
            self.tracking_metrics[str(epoch)] = epoch_dict
        except Exception as e:
            print(f"Error logging metrics: {e}")

    def save_metrics(self, save_path=None):
        try:
            save_path = save_path or self.save_pth
            with open(save_path, 'w') as f:
                json.dump(self.tracking_metrics, f)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def load_metrics(self, load_path):
        try:
            with open(load_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return {}

    def graph_metrics(self, metrics, display=True, save=True, save_path='./'):
        try:
            images = []
            metric_names = list(next(iter(metrics.values())).keys())
            epochs = sorted([int(e) for e in metrics.keys()])
            
            for name in metric_names:
                values = [metrics[str(e)][name] for e in epochs]
                
                fig = plt.figure(figsize=(8, 5))
                plt.plot(epochs, values, marker='o', linestyle='-')
                plt.title(f"{name} Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel(name)
                plt.grid(True)
                
                if save:
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    images.append(Image.open(buf))
                    plt.savefig(f"{save_path}/{name}.png")
                
                if display:
                    plt.show()
                
                plt.close()
            
            return images
        except Exception as e:
            print(f"Error generating graphs: {e}")
            return []