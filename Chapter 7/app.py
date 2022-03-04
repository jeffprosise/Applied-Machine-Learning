import pickle
from flask import Flask, request
 
app = Flask(__name__)
pipe = pickle.load(open('sentiment.pkl', 'rb'))
 
@app.route('/analyze', methods=['GET'])
def analyze():
    if 'text' in request.args:
        text = request.args.get('text')
    else:
        return 'No string to analyze'
 
    score = pipe.predict_proba([text])[0][1]
    return str(score)
 
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
