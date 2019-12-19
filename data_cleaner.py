# Files for data cleaning
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
# File for  sys modules

tokenizer = RegexpTokenizer(r'\w+')
sw = set(stopwords.words('english'))
ps = PorterStemmer()


def clean_text(text):
    text = text.lower()
    text = text.replace('<br /><br />', ' ')
    word_list = tokenizer.tokenize(text)
    clean_list = [w for w in word_list if w not in sw]
    stemmed_list = [ps.stem(w) for w in clean_list]
    clean_review = ' '.join(stemmed_list)
    return clean_review


def clean_file(input_file, output_file):
    out = open(output_file, 'w', encoding='utf8')
    with open(input_file, encoding='utf8') as file:
        reviews = file.readlines()
        for r in reviews:
            cleaned_reviews = clean_text(r)
            print(cleaned_reviews, file=out)
        out.close()


inputfile = "./dataset/imdb_testX.txt"
outputfile = "./imdb_test_clean.txt"

clean_file(inputfile, outputfile)


