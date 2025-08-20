import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

data = pd.DataFrame({
    'time':[1,2,2,4,4,1,3,3],
    'classed':[1,1,3,2,3,1,2,2],
    'reviews':[0,0,3,6,4,7,9,1]
})

model = KMeans(n_clusters=3, random_state=42)
model.fit(data)

data['cluster']=model.labels_

print('data', data)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    data['time'], 
    data['classed'], 
    data['reviews'], 
    c=data['cluster'], 
    cmap="viridis"
)

ax.set_xlabel('час')
ax.set_ylabel('к-ть курсів')
ax.set_zlabel('відгуки')

plt.show()