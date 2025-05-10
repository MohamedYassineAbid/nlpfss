import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import subprocess

# Set page config
st.set_page_config(page_title="SVM Visualizer with Kernel Trick", layout="wide")

st.title("🧠 SVM Visualizer with Kernel Trick")
st.markdown("""
Explore Support Vector Machines (SVMs) interactively! Adjust parameters, visualize decision boundaries, and learn the theory behind SVM optimization including **dual formulation**, **convexity**, and **KKT conditions**.
""")

# Sidebar for user inputs
st.sidebar.header("🔧 Model Parameters")

kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly", "sigmoid"))
C = st.sidebar.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
gamma = st.sidebar.slider("Gamma (only for rbf/poly/sigmoid)", 0.01, 5.0, 0.5)
degree = st.sidebar.slider("Degree (only for poly)", 1, 5, 3)

# Load dataset
X, y = datasets.make_classification(n_samples=100, n_features=2,
                                     n_redundant=0, n_informative=2,
                                     n_clusters_per_class=1, class_sep=1.5,
                                     random_state=42)
y = 2 * y - 1  # Convert labels to -1 and 1

# Train SVM
clf = svm.SVC(kernel=kernel, C=C, gamma=gamma if kernel != 'linear' else 'scale', degree=degree)
clf.fit(X, y)

# Plotting
def plot_svm():
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 
               s=100, facecolors='none', edgecolors='yellow', label='Support Vectors')
    ax.set_title("SVM Decision Boundary with Support Vectors")
    st.pyplot(fig)

plot_svm()



st.title("📚 SVM with Kernel Trick – Theory & Derivations")

st.markdown("""
This section provides a full theoretical breakdown of Support Vector Machines (SVM), including:

- Kernel mapping and duality
- Quadratic programming formulation
- Karush-Kuhn-Tucker (KKT) conditions
- Final classifier and implementation algorithm

💡 Click below to download the complete LaTeX document.
""")
with open("svm_theory.pdf", "rb") as f:
    st.download_button(
        label="📄 Download SVM Theory PDF",
        data=f,
        file_name="svm_theory.pdf",
        mime="application/pdf"
    )

