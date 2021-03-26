class SupervisedModel:
    def train(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def predict_classes(self, x):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError