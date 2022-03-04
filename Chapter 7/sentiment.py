import pickle, sys
 
# Get the text to analyze
if len(sys.argv) > 1:
    text = sys.argv[1]
else:
    text = input('Text to analyze: ')
 
# Load the pipeline containing the model and the vectorizer
pipe = pickle.load(open('sentiment.pkl', 'rb'))
 
# Pass the input text to the pipeline and print the result
score = pipe.predict_proba([text])[0][1]
print(score)
