import nltk

# List of corpora to download
corpora = [
    'brown',
    'punkt',
    'stopwords'
    'wordnet'
]

for corpus in corpora:
    nltk.download(corpus)
