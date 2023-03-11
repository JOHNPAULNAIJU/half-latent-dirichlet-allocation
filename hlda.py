import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

#nltk.download('wordnet')

stemmer = SnowballStemmer("english")

title = "Vue js. Developers"
desc = "Here we will talk all about vue js. Keep talk only about vue js."

doc_sample = title+' '+desc

result = []

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

print(preprocess(doc_sample))
