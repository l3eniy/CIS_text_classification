from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem


eng_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([eng_stemmer.stem(w) for w in analyzer(doc)])