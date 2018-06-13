import operator
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def find_clusters(x, n_clusters, rseed=123):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(x.shape[0])[:n_clusters]
    centers = x[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(x, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([x[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


dataset = load_files('words', encoding='utf-8')
test_dataset = load_files('test_words', encoding='utf-8')
claster_number = len(dataset['target_names'])
article_number = len(dataset['filenames'])
vectorizer = TfidfVectorizer(max_df=18, min_df=2, ngram_range=(1, 3))  # , stop_words='russian')
matrix = vectorizer.fit_transform(dataset.data)
print(matrix.shape)
print(dataset['target_names'])


def f1():
    model = AgglomerativeClustering(n_clusters=claster_number, affinity='euclidean',
                                    linkage='complete')

    # model.fit(matrix.toarray())
    # print (model.distance)
    # print(linkage(matrix.toarray(), method='single', metric='euclidean'))

    preds = model.fit_predict(matrix.toarray())

    res = dict()
    for i, p in enumerate(preds):
        # print(p)
        res[basename(dataset['filenames'][i])] = dataset['target_names'][p]
    prev = None
    for k, v in sorted(res.items(), key=operator.itemgetter(1)):
        if prev != v:
            print(v, ':')
            prev = v
        print('\t\t', k)

    dist = 1 - cosine_similarity(matrix.toarray())
    row_sums = dist.sum(axis=1)
    new_matrix = dist / row_sums[:, np.newaxis]
    plt.figure(figsize=(20, 20), dpi=300)
    sb.heatmap(new_matrix)
    lbls = list()
    for fn in dataset['filenames']:
        lbls.append(basename(fn)[:-4])
    plt.xticks(np.arange(0, article_number), lbls, rotation='vertical')
    plt.yticks(np.arange(0, article_number), lbls, rotation='horizontal')
    plt.show()
    print(())


def f2():
    text_clf = Pipeline(
        [('vect', TfidfVectorizer()), ('tfidf', TfidfTransformer()),
         ('clf', MultinomialNB()), ])
    text_clf = text_clf.fit(dataset.data, dataset.target)

    pred = text_clf.predict(test_dataset)
    print(test_dataset['filenames'])
    print(pred)


def f3():
    # model = KMeans(n_clusters=claster_number,random_state=123)
    # model = AgglomerativeClustering(n_clusters=claster_number, affinity='cosine',
    #                                 linkage='complete')
    model = KMeans(n_clusters=claster_number, random_state=123)
    gvd = TruncatedSVD(n_components=7, random_state=123)

    features = gvd.fit_transform(matrix)
    pred = model.fit_predict(features)
    gvd = TruncatedSVD(n_components=2, random_state=123)
    features = gvd.fit_transform(features)
    mapping = {0: 1, 1: 5, 5: 6, 6: 3, 3: 3, 2: 2, 4: 0}
    pred = [mapping[pred] for pred in pred]
    res = dict()
    for i, p in enumerate(pred):
        res[basename(dataset['filenames'][i])] = dataset['target_names'][p]
    prev = None
    for k, v in sorted(res.items(), key=operator.itemgetter(1)):
        if prev != v:
            print(v, ':', )
            prev = v
        print('\t\t', k[:-4])

    col_need = np.zeros(shape=(article_number, 1))
    col_is = np.zeros(shape=(article_number, 1))
    for i, p in enumerate(pred):
        col_need[i] = dataset['target'][i]
        col_is[i] = p
    feat = np.column_stack((features, col_need, col_is,))
    # print(feat)
    centers, labels = find_clusters(features, claster_number)
    plt.figure(figsize=(10, 10), dpi=200)
    colormap = [
        'gold',
        'orange',
        'red',
        'green',
        'cyan',
        'blue',
        'purple',
    ]
    for i, nm in enumerate(dataset['target_names']):
        mrk = r'${}$'.format(nm)
        plt.scatter(centers[i, 0], centers[i, 1], marker=mrk, c=colormap[i], s=7000, alpha=0.5,
                    edgecolor='grey', linewidths=1)

    for i, f in enumerate(feat):
        edgecolor = colormap[int(f[2])]
        color = colormap[int(f[3])]
        lbl = dataset['target_names'][int(f[2])][0:2]
        plt.scatter(f[0], f[1], s=200, color=color, marker='o', edgecolors=edgecolor, linewidths=3,
                    label=lbl)

    for i, item in enumerate(res.items()):
        it = item[0][:-4]
        if len(it) > 20:
            it = it[:21]
        plt.annotate(it, (features[i, 0], features[i, 1]), fontsize=8)
    plt.grid(True)

    plt.show()


f3()
