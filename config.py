import json

class Configuration():
    def __init__(self, args, save_path):
        with open(save_path + "/configuration.json", "w") as f:
            json.dump(args, f, indent=4)
        with open(save_path + "/configuration.json", "r") as f:
            config = json.load(f)
        self.config = config
        self.lr = config['lr']
        self.maxlen = config['maxlen']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.threshold = config['threshold']
        self.sampling_weight = config['sampling_weight']
        self.class_weight = config['class_weight']
        self.device = config['device']
        self.model = config['model']
        self.acc_steps = config['acc_steps']
        self.epochs = config['epochs']

    def configprint(self):
        print("========================================")
        print("The settings of this experiment:")
        for i in self.config.keys():
            print(f'{i} : {self.config[i]}')
        print("========================================")



