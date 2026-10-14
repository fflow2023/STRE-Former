import pickle

with open('./features_extraction/IEMOCAP_features_1.pkl', "rb") as fin:
        features_data = pickle.load(fin)
        print(features_data)