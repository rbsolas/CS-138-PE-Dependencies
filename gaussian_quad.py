from lgvalues_weights import *
from lgvalues_abscissa import *
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def f(t):
    return math.sqrt(7*((3*t + 7) / 2) - ((3*t + 7) / 2)**2 - 10)

def g(t):
    return 1 / math.sqrt(7*((3*t + 7) / 2) - ((3*t + 7) / 2)**2 - 10)


def gaussLegendre(n, c, f):
    # I = summation of w_i * f(t_i)
    I = 0

    for i in range(n):
        I += quadrature_weights[n][i] * f(legendre_roots[n][i])

    return c*I

nArray = np.array([i for i in range(2, 65)])
fApproxArray  = []
gApproxArray = []

for n in nArray:
    fApproxArray.append(gaussLegendre(n, 1.5, f))
    gApproxArray.append(gaussLegendre(n, 1.5, g))

df = pd.DataFrame(dict(
    x = nArray,
    y = fApproxArray
))

df_2 = pd.DataFrame(dict(
    x = nArray,
    y = gApproxArray
))

fig1 = px.line(df, x="x", y="y", labels={"x":"Number of points used", "y":"Approximated value for I"}, title="n-point Gaussian Quadrature") 
fig2 = px.line(df_2, x="x", y="y", labels={"x":"Number of points used", "y":"Approximated value for I"}, title="n-point Gaussian Quadrature") 

fig = go.Figure(data = fig1.data + fig2.data)
fig.add_annotation(x=10, y=3.6,
            text="Item 6A",
            showarrow=False
            )
fig.add_annotation(x=10, y=2.8,
            text="Item 6B",
            showarrow=False
            )
fig.show()