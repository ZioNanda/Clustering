import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans

model = pickle.load(open('output_heart_failure_dataset.sav', 'rb'))

df=pd.read_excel("output_cluster.xlsx")
features = ['age', 'creatinine', 'platelets']
X = df[features]

st.title('Rate of Heart Failure')

numClusters = st.slider("Select Number of Clusters", min_value=1, max_value=10, value=3)

model = KMeans(n_clusters=numClusters)
clusters = model.fit_predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X['age'], X['creatinine'], X['platelets'], c=clusters, cmap='viridis')
ax.set_xlabel('age')
ax.set_ylabel('creatinine')
ax.set_zlabel('platelets')

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

st.pyplot(fig)