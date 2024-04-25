import numpy as np
from sklearn import datasets
from sklearn import manifold
from matplotlib import pyplot as plt

from . import tsne


model = tsne.TSNE(30., True)
data = datasets.fetch_olivetti_faces()["data"].astype(np.float64) # pyright: ignore
labels = np.hstack([np.arange(40).reshape(-1, 1) for _ in range(10)]).flatten()
y = model.fit(data, 3000, 3, 0.8)
plt.subplot(211)
plt.title("TSNE")
plt.scatter(y[:, 0], y[:, 1], 20, labels)

y = manifold.TSNE(perplexity=30.).fit_transform(data)
plt.subplot(212)
plt.title("TSNE of scikit-learn")
plt.scatter(y[:, 0], y[:, 1], 20, labels)
plt.show()
