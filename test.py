from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster.hierarchical import AgglomerativeClustering
from os.path import basename
import operator

dataset = load_files('words', encoding='utf-8')
claster_number = len(dataset['target_names'])

vectorizer = TfidfVectorizer(max_df=10, min_df=2)
matrix = vectorizer.fit_transform(dataset.data)
print(matrix.shape)
# print(dataset['target_names'])

model = AgglomerativeClustering(n_clusters=claster_number, affinity='euclidean', linkage='complete')
preds = model.fit_predict(matrix.toarray())

res = dict()
for i, p in enumerate(preds):
    # print(p)
    res[basename(dataset['filenames'][i])] = dataset['target_names'][p]
    prev = None
for k, v in sorted(res.items(), key=operator.itemgetter(1)):
    if (prev != v):
        print(v, ':')
        prev = v
    print('\t\t', k)
fnames = vectorizer.get_feature_names()
# print(matrix[0])
# print(fnames[956])

# print()
