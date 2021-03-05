import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
lemma = WordNetLemmatizer()
print(lemma.lemmatize("are little working models of"))

with open("toy_data/dev_predicates.txt", "r") as in_file, open("toy_data/dev_norm_predicates.txt", "w") as out_file:
    for index, line in enumerate(in_file):
        tokens = nltk.sent_tokenize(line)
        # tokens = [lemma.lemmatize(word.lower())
        #           for word in tokens if not word in stopwords.words("english")]
        tokens = [lemma.lemmatize(word.lower(), pos="v") for word in tokens]
        tokens = [lemma.lemmatize(word.lower(), pos="n") for word in tokens]
        if len(tokens) == 0:
            print(tokens)
        output_string = " ".join(tokens)
        output_string += "\n"
        out_file.write(output_string)
        # if len(tokens) == 0:
        #     print(f"{line} is stopword: {line in stopwords.words('english')}")
