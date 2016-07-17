import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pattern.en import singularize


def read_word_vectors(filename):
    ''' Read and store word vectors in a text file in the following format:

        $ more filename
        the 0.4 0.2 1 <norm>
        of 0.5 0.9 2 <norm>
        ...

        Args:
            filename (str): path to filename

        Returns:
            DataArray
     '''
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize ndarray to store all word vectors
    words = [None] * len(lines)
    vectors = np.zeros((len(lines[0].split()) - 1, len(lines)))

    # Fill in vectors
    for i, line in enumerate(lines):
        temp = line.split()
        words[i] = temp[0]
        vectors[:, i] = np.array(map(lambda x: float(x), temp[1:]))

    # Create a DataArray object
    w = xr.DataArray(data=vectors, dims=('vector', 'word'),
                     coords={'word': words})
    return w


def isplural(pluralForm):
    singularForm = singularize(pluralForm)
    plural = True if pluralForm is not singularForm else False
    return plural


def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels)
    for i, label in enumerate(labels):
        if len(low_dim_embs[i, :]) == 1:
            x = low_dim_embs[i, :]
            y = 0
        else:
            x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.grid('on')


def plot_tsne(words, w, ndim=2):
    tsne = TSNE(n_components=ndim, perplexity=20, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(w.loc[:, words].values.T)
    plt.clf()
    plot_with_labels(low_dim_embs, w['word'].loc[words].values)
