import pickle, os
import xlwings as xw

# Load the model and the vocabulary and create a CountVectorizer
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                             'sentiment.pkl'))

model = pickle.load(open(model_path, 'rb'))

@xw.func
def analyze_text(text):
    score = model.predict_proba([text])[0][1]
    return score
