from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors, Word2Vec

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처

mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.font_manager as fm

import matplotlib as mpl
font_list = [font.name for font in fm.fontManager.ttflist]


mpl.get_configdir()
fm.findfont('Malgun Gothic')

# plt.rc('font', family='NanumGothic')
mpl.rc('font', family='Malgun Gothic')


def show_tsne():
    tsne = TSNE(n_components=2)
    X = tsne.fit_transform(X_show)
    df = pd.DataFrame(X, index=vocab_show, columns=['x', 'y'])
    fig = plt.figure()
    fig.set_size_inches(30, 20)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos, fontsize=10)

    plt.xlabel("t-SNE 특성 0")
    plt.ylabel("t-SNE 특성 1")
    plt.show()


def show_pca():
    # PCA 모델을 생성합니다
    pca = PCA(n_components=2)
    pca.fit(X_show)
    # 처음 두 개의 주성분으로 숫자 데이터를 변환합니다
    x_pca = pca.transform(X_show)
    plt.figure(figsize=(30, 20))
    plt.xlim(x_pca[:, 0].min(), x_pca[:, 0].max())
    plt.ylim(x_pca[:, 1].min(), x_pca[:, 1].max())
    for i in range(len(X_show)):
        plt.text(x_pca[i, 0], x_pca[i, 1], str(vocab_show[i]), fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("첫 번째 주성분")
    plt.ylabel("두 번째 주성분")
    plt.show()


model_name = 'C:/Users/alti1/PycharmProjects/DEC-new/data/review/word2vec_test.bin'
model = KeyedVectors.load('./data/review/word2vec_test_noun_pu2.bin')

vocab = list(model.wv.vocab)
X = model[vocab]

# sz개의 단어에 대해서만 시각화
sz = 800
X_show = X[:sz,:]
vocab_show = vocab[:sz]

show_tsne()
show_pca()

# model = Word2Vec.load('data/review/word2vec_test_nounX2.bin')
# # print(model.wv.most_similar('이다'))
# word_vectors = model.wv
# vocabs = word_vectors.vocab.keys()
# word_vectors_list = [word_vectors[v] for v in vocabs]
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# xys = pca.fit_transform(word_vectors_list)
# xs = xys[:,0]
# ys=xys[:,1]
#
# import matplotlib.pyplot as plt
#
#
# def plot_2d_graph(vocabs, xs, ys):
#     plt.figure(figsize=(150, 100))
#     plt.scatter(xs, ys, marker='o')
#     for i, v in enumerate(vocabs):
#         plt.annotate(v, xy=(xs[i], ys[i]))
#
#
# plot_2d_graph(vocabs, xs, ys)
#
# text=[]
# for i,v in enumerate(vocabs):
#     text.append(v)
#
# import plotly
# import plotly.graph_objects as go
# fig = go.Figure(data=go.Scatter(x=xs,
#                                 y=ys,
#                                 mode='markers+text',
#                                 text=text))
#
# fig.update_layout(title='review Word2Vec du')
# fig.show()
#
# plotly.offline.plot(
# fig, filename='word2vec_test_nounX2.html'
# )