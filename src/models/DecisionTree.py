import pickle as pk
import pandas as pd

class DecisionTree:
    def __init__(self):
        self.pipline = pk.load(open('src/models/decision_tree_pipe_spotify.pkl', 'rb'))
        self.decision_tree = pk.load(open('src/models/decision_tree.pkl', 'rb'))
        self.min_max = pk.load(open('src/encoders/encoder_one_hot.pkl', 'rb'))
        self.z_score = pk.load(open('src/encoders/encoder_z_score.pkl', 'rb'))
        self.one_hot = pk.load(open('src/encoders/encoder_one_hot.pkl', 'rb'))

    def predict(self, df):
        return self.pipline.predict(df)

    def feature_engineering(self, df):
        pass

    def predict_dt(self, df):
        df = self.feature_engineering(df)
        return self.decision_tree.predict(df)