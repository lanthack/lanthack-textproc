import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import read_word_vectors, isplural, plot_with_labels, plot_tsne

# Read vectors from a text file
w = read_word_vectors('../datasets/glove.6B/glove.6B.50d.txt')

# Plot comparative, superlative words to see if an order is preserved.
# Words are not always in order.
plot_tsne(['small', 'smaller', 'smallest'], w, ndim=1)
plot_tsne(['large', 'larger', 'largest'], w, ndim=1)
plot_tsne(['happy', 'happier', 'happiest'], w, ndim=1)



# x = (w.loc[:, 'larger'] - w.loc[:, 'large']).values
# y = (w.loc[:, 'largest'] - w.loc[:, 'large']).values
# df = pd.DataFrame({'larger': x, 'largest': y})
# print np.linalg.norm(df.iloc[:, 0].values)
# print np.linalg.norm(df.iloc[:, 1].values)

# x = (w.loc[:, 'smaller'] - w.loc[:, 'small']).values
# y = (w.loc[:, 'smallest'] - w.loc[:, 'small']).values
# df = pd.DataFrame({'smaller': x, 'smallest': y})
# print np.linalg.norm(df.iloc[:, 0].values)
# print np.linalg.norm(df.iloc[:, 1].values)

# x = (w.loc[:, 'happier'] - w.loc[:, 'happy']).values
# y = (w.loc[:, 'happiest'] - w.loc[:, 'happy']).values
# df = pd.DataFrame({'happier': x, 'happiest': y})
# print np.linalg.norm(df.iloc[:, 0].values)
# print np.linalg.norm(df.iloc[:, 1].values)

# w = read_word_vectors('../datasets/glove.6B/glove.6B.300d.txt')
# words = np.array([word.strip() for word in w['word'].values
#                   if word.strip().isalpha()])
# labels = np.array([isplural(word) for word in words])


# labels = [textblob.Word(word).pluralize()]
# words_to_plot = ['car', 'cars', 'jar', 'jars', 'bottle', 'bottles']
# tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=5000)
# low_dim_embs = tsne.fit_transform(w.loc[:, words_to_plot].values.T)
# plt.clf()
# plot_with_labels(low_dim_embs, w['word'].loc[words_to_plot].values)
