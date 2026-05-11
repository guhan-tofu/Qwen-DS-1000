import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")



prompt = '''
You are a data science coding problem generator. Generate problems in the following format:

Problem: [Clear description of the data science task, including sample input data and expected output]

Setup code:
```python
[imports and variable initialization]
```

result = ... # put solution in this variable
BEGIN SOLUTION
```python
[clean, concise solution]
```

---

Here are three examples:

---

Problem:
I have the following DataFrame:
    Col1  Col2  Col3  Type
0      1     2     3     1
1      4     5     6     1
2      7     8     9     2
3    10    11    12     2
4    13    14    15     3
5    16    17    18     3

I would like to shuffle the order of the DataFrame's rows according to a list.
For example, given a list [2, 4, 0, 3, 1, 5] the desired result should be:
    Col1  Col2  Col3  Type
2      7     8     9     2
4     13    14    15     3
0      1     2     3     1
3    10    11    12     2
1     4     5     6     1
5    16    17    18     3

How can I achieve this?

Setup code:
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'Col1': [1, 4, 7, 10, 13, 16],
                   'Col2': [2, 5, 8, 11, 14, 17],
                   'Col3': [3, 6, 9, 12, 15, 18],
                   'Type': [1, 1, 2, 2, 3, 3]})
List = np.random.permutation(len(df))
```

result = ... # put solution in this variable
BEGIN SOLUTION
```python
result = df.iloc[List]
```

---

Problem:
What is the most efficient way to remove negative elements from a numpy array?
For:
import numpy as np
x = np.array([-2, -1.4, -1.1, 0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2])

I want to end up with:
[0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2]

Do not use a for or while loop.

Setup code:
```python
import numpy as np
x = np.array([-2, -1.4, -1.1, 0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2])
```

result = ... # put solution in this variable
BEGIN SOLUTION
```python
result = x[x >= 0]
```

---

Problem:
Let X be a M x N matrix. Denote xi the i-th column of X. I want to create a 3-dimensional N x M x M array consisting of the outer products xi.dot(xi.T) for each column.

How can I do this most elegantly with numpy, using only matrix operations and no loops?

Setup code:
```python
import numpy as np
X = np.random.randint(2, 10, (5, 6))
```

result = ... # put solution in this variable
BEGIN SOLUTION
```python
result = X.T[:, :, None] * X.T[:, None]
```

---

Now generate a new problem in the same format. The problem should:
- Involve [LIBRARY: pandas / numpy / scipy / etc.]
- Be of [DIFFICULTY: easy / medium / hard]
- Cover the topic of [TOPIC: e.g. groupby aggregation / array broadcasting / string manipulation / etc.]
- Have a concise, elegant solution (preferably a one- or two-liner)
- Not use for or while loops (unless the topic specifically requires iteration)
'''


pandas_prompt = '''
You are a pandas coding problem generator. Generate realistic, Stack Overflow-style pandas problems along with clean solutions.

Each problem must follow this exact format:

Problem:
[Clear description of the pandas task. Include a concrete sample DataFrame or data structure showing the input, and show the desired output. The problem should read like a genuine Stack Overflow question.]

Setup code:
<code>
[imports and DataFrame initialization — enough for the solution to run]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
[concise pandas solution, ideally 1–3 lines]
</code>

---

Here are six examples covering different areas of pandas:

---

Problem:
I have the following DF:
        Date
0    2018-01-01
1    2018-02-08
2    2018-02-08

I want to extract the month name and year in a simple way in the following format:
        Date
0    Jan-2018
1    Feb-2018
2    Feb-2018

I have used df.Date.dt.to_period("M") which returns "2018-01" format, but that's not what I want.

Setup code:
<code>
import pandas as pd

df = pd.DataFrame({'Date': ['2019-01-01', '2019-02-08', '2019-02-08', '2019-03-08']})
df['Date'] = pd.to_datetime(df['Date'])
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
df['Date'] = df['Date'].dt.strftime('%b-%Y')
result = df
</code>

---

Problem:
I have a simple dataframe and I would like to bin it by taking the mean of every 3 rows:

    col1
0      2
1      1
2      3
3      1
4      0

Desired output:
    col1
0    2.0
1    0.5

Setup code:
<code>
import pandas as pd

df = pd.DataFrame({'col1': [2, 1, 3, 1, 0]})
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = df.groupby(df.index // 3).mean()
</code>

---

Problem:
I have a DataFrame where one column contains comma-separated values, and I want to split it so each value gets its own row:

id  var1  var2
1   A     Z,Y
2   B     X
3   C     W,U,V

Desired output:
id  var1  var2
1   A     Z
1   A     Y
2   B     X
3   C     W
3   C     U
3   C     V

Setup code:
<code>
import pandas as pd

df = pd.DataFrame(
    [["A", "Z,Y"], ["B", "X"], ["C", "W,U,V"]],
    index=[1, 2, 3],
    columns=['var1', 'var2']
)
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = df.drop('var2', axis=1).join(
    df.var2.str.split(',', expand=True).stack()
    .reset_index(drop=True, level=1).rename('var2')
)
</code>

---

Problem:
I have the following DataFrame and want the mean and standard deviation of column `b` within each group defined by column `a`:

   a   b
0  1  12
1  1  13
2  1  23
3  2  22
4  2  23
5  2  24
6  3  30
7  3  35
8  3  55

Desired output:
   mean        std
a
1  16.0   6.082763
2  23.0   1.000000
3  40.0  13.228757

Setup code:
<code>
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1,1,1,2,2,2,3,3,3], 'b': [12,13,23,22,23,24,30,35,55]})
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = df.groupby("a")["b"].agg([np.mean, np.std])
</code>

---

Problem:
I have a square correlation matrix in pandas and want to return all pairs where the correlation is above 0.3, excluding the diagonal and lower triangle (to avoid duplicates):

          0         1         2
0  1.000000  0.214119 -0.073414
1  0.214119  1.000000  0.419219
2 -0.073414  0.419219  1.000000

Desired output — a DataFrame with a MultiIndex of (Col1, Col2) and one value column:
           Pearson Correlation Coefficient
Col1 Col2
1    2                           0.419219

Setup code:
<code>
import pandas as pd
import numpy as np

np.random.seed(10)
df = pd.DataFrame(np.random.rand(10, 5))
corr = df.corr()
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
corr_triu = corr.where(~np.tril(np.ones(corr.shape)).astype(bool)).stack()
corr_triu.name = 'Pearson Correlation Coefficient'
corr_triu.index.names = ['Col1', 'Col2']
result = corr_triu[corr_triu > 0.3].to_frame()
</code>

---

Problem:
I have two DataFrames with timestamps and want to join them using a forward asof merge — for each row in df2, attach data from the next available row in df1 at or after that timestamp:

df1:                              df2:
Timestamp            data         Timestamp            stuff
2019/04/02 11:00:01  111          2019/04/02 11:00:14  101
2019/04/02 11:00:15  222          2019/04/02 11:00:15  202
2019/04/02 11:00:29  333          2019/04/02 11:00:16  303
2019/04/02 11:00:30  444          2019/04/02 11:00:30  404
                                  2019/04/02 11:00:31  505

Desired output (df2 with df1's data column appended by forward-looking match):
Timestamp            stuff  data
2019/04/02 11:00:14  101    222
2019/04/02 11:00:15  202    222
...

Setup code:
<code>
import pandas as pd

df1 = pd.DataFrame({
    'Timestamp': ['2019/04/02 11:00:01','2019/04/02 11:00:15','2019/04/02 11:00:29','2019/04/02 11:00:30'],
    'data': [111, 222, 333, 444]
})
df2 = pd.DataFrame({
    'Timestamp': ['2019/04/02 11:00:14','2019/04/02 11:00:15','2019/04/02 11:00:16','2019/04/02 11:00:30','2019/04/02 11:00:31'],
    'stuff': [101, 202, 303, 404, 505]
})
df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])
df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = pd.merge_asof(df2, df1, on='Timestamp', direction='forward')
</code>

---

Now generate a NEW pandas problem following this exact format. The problem should:
- Cover the pandas topic of: [TOPIC — e.g. pivot tables / MultiIndex / string methods / merge / apply / window functions / dtype conversion / etc.]
- Be of difficulty: [DIFFICULTY — easy / medium / hard]
- Show a concrete before/after table in the problem description
- Have a concise pandas solution (preferably under 4 lines)
- Include a complete, runnable setup code block

You are to use the pydantic model provided to you to give structured output.
It has two fields: "question" and "answer".
The "question" field should contain everything up to and including BEGIN SOLUTION\n<code> — do NOT include the solution code inside the question field.
The "answer" field should contain only the solution code.
'''

matplotlib_prompt = '''
You are a matplotlib coding problem generator. Generate realistic, Stack Overflow-style matplotlib problems along with clean solutions.

Each problem must follow this exact format:

Problem:
[Clear description of the matplotlib task. Provide the exact data variables already defined, and describe precisely what the plot should look like — chart type, labels, styling, layout, etc. The problem should read like a genuine Stack Overflow question or a concise plotting instruction.]

Setup code:
<code>
[imports and variable definitions — all data the solution will need, fully initialized]
# [imperative instruction describing exactly what to plot]
# SOLUTION START
</code>

BEGIN SOLUTION
<code>
[concise matplotlib solution, ideally 1–6 lines]
</code>

---

Here are six examples covering different areas of matplotlib:

---

Problem:
I have three lists of data points and I want to make a scatter plot of `a` over `b`, then annotate each point with its corresponding label from `c`. How do I add text annotations to each scatter point?

Setup code:
<code>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = [2.56422, 3.77284, 3.52623]
b = [0.15, 0.3, 0.45]
c = [58, 651, 393]

# make scatter plot of a over b and annotate each data point with the corresponding numbers in c
# SOLUTION START
</code>

BEGIN SOLUTION
<code>
fig, ax = plt.subplots()
plt.scatter(a, b)

for i, txt in enumerate(c):
    ax.annotate(txt, (a[i], b[i]))
</code>

---

Problem:
I have two 1D arrays `x` and `y` and want to plot both as histograms on a single chart so I can compare their distributions. The histograms should overlap, so I need to set the transparency of both to 0.5.

Setup code:
<code>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.rand(10)
y = np.random.rand(10)
bins = np.linspace(-1, 1, 100)

# Plot two histograms of x and y on a single chart
# Set the transparency of the histograms to be 0.5
# SOLUTION START
</code>

BEGIN SOLUTION
<code>
plt.hist(x, bins, alpha=0.5, label="x")
plt.hist(y, bins, alpha=0.5, label="y")
</code>

---

Problem:
I have two 10x10 random matrices `x` and `y` and want to display them side-by-side as heatmaps using `imshow`. Both subplots should share a single colorbar placed to the right of both plots. How do I add a shared colorbar for multiple subplots?

Setup code:
<code>
import matplotlib.pyplot as plt
import numpy as np

x = np.random.random((10, 10))
y = np.random.random((10, 10))

# make two colormaps with x and y and put them into different subplots
# use a single colorbar for these two subplots
# SOLUTION START
</code>

BEGIN SOLUTION
<code>
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(x, vmin=0, vmax=1)
im = axes[1].imshow(y, vmin=0, vmax=1)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
</code>

---

Problem:
I want to make a donut chart (a pie chart with a hole in the middle) using `data` as the values and `l` as the slice labels. The ring width should be 0.4.

Setup code:
<code>
import matplotlib.pyplot as plt

l = ["a", "b", "c"]
data = [225, 90, 50]

# Make a donut plot using `data` and use `l` for the pie labels
# Set the wedge width to be 0.4
# SOLUTION START
</code>

BEGIN SOLUTION
<code>
plt.pie(data, labels=l, wedgeprops=dict(width=0.4))
</code>

---

Problem:
I have two sets of bar heights, `blue_bar` and `orange_bar`, with three values each. I want to plot them as a grouped bar chart — both series side-by-side for each group — so the bars don't overlap. How do I offset the bars horizontally?

Setup code:
<code>
import matplotlib.pyplot as plt
import numpy as np

blue_bar = (23, 25, 17)
orange_bar = (19, 18, 14)

# Plot blue_bar and orange_bar side-by-side in the same bar plot
# Make sure the bars don't overlap with each other
# SOLUTION START
</code>

BEGIN SOLUTION
<code>
ind = np.arange(len(blue_bar))
plt.figure(figsize=(10, 5))
width = 0.3
plt.bar(ind, blue_bar, width, label="Blue bar label")
plt.bar(ind + width, orange_bar, width, label="Orange bar label")
</code>

---

Problem:
I have three 1D arrays of random values `x`, `y`, `z` and want to make a 3D scatter plot. After plotting, I want to adjust the viewing angle to have an azimuth of 100 degrees and elevation of 50 degrees.

Setup code:
<code>
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = np.random.random(10)
y = np.random.random(10)
z = np.random.random(10)

# Make a 3D scatter plot of x, y, z
# Change the view to have azimuth=100 and elevation=50
# SOLUTION START
</code>

BEGIN SOLUTION
<code>
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z)
ax.azim = 100
ax.elev = 50
</code>

---

Now generate a NEW matplotlib problem following this exact format. The problem should:
- Cover the matplotlib topic of: [TOPIC — e.g. twin axes / log scale / custom legends / contour plots / fill_between / error bars / animation / axis formatting / etc.]
- Be of difficulty: [DIFFICULTY — easy / medium / hard]
- Clearly describe the desired visual output in the problem statement
- Provide fully initialized data variables in the setup code
- End the setup code block with a concise imperative comment and # SOLUTION START
- Have a clean matplotlib solution (preferably under 8 lines)

You are to use the pydantic model provided to you to give structured output.
It has two fields: "question" and "answer".
The "question" field should contain everything up to and including BEGIN SOLUTION\n<code> — do NOT include the solution code inside the question field.
The "answer" field should contain only the solution code.
'''

numpy_prompt = '''
You are a numpy coding problem generator. Generate realistic, Stack Overflow-style numpy problems along with clean solutions.

Each problem must follow this exact format:

Problem:
[Clear description of the numpy task. Include a concrete sample array or data structure showing the input, and show the desired output. The problem should read like a genuine Stack Overflow question.]

Setup code:
<code>
[imports and array initialization — enough for the solution to run]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
[concise numpy solution, ideally 1–3 lines. No for or while loops unless the topic specifically requires iteration.]
</code>

---

Here are six examples covering different areas of numpy:

---

Problem:
Let's say I have a 1D numpy positive integer array like this:
a = array([1, 0, 3])

I would like to encode this as a 2D one-hot array:
b = array([[0, 1, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 1]])

The leftmost element corresponds to 0 in `a` (NO MATTER whether 0 appears in `a` or not), and the rightmost vice versa.
Is there a quick way to do this only using numpy, without looping over `a`?

Setup code:
<code>
import numpy as np
a = np.array([1, 0, 3])
</code>

b = ... # put solution in this variable
BEGIN SOLUTION
<code>
b = np.zeros((a.size, a.max()+1))
b[np.arange(a.size), a] = 1
</code>

---

Problem:
I have a pair of 3D numpy arrays `a` and `b`, and I want to sort the entries of `b` by the values of `a`, but only along axis=0. My naive attempt using `b[np.argsort(a, axis=0)]` produces a (3,3,3,3,3) shaped result instead of (3,3,3).

For example:
a = [[[1,1,1],[1,1,1],[1,1,1]],
     [[3,3,3],[3,2,3],[3,3,3]],
     [[2,2,2],[2,3,2],[2,2,2]]]

b = arange(27).reshape(3,3,3)

Desired output: b reordered along axis 0 according to the sorted order of a.

What's the right way to do this?

Setup code:
<code>
import numpy as np
a = np.random.rand(3, 3, 3)
b = np.arange(3*3*3).reshape((3, 3, 3))
</code>

c = ... # put solution in this variable
BEGIN SOLUTION
<code>
sort_indices = np.argsort(a, axis=0)
static_indices = np.indices(a.shape)
c = b[sort_indices, static_indices[1], static_indices[2]]
</code>

---

Problem:
I have integers in the range 0..2**m - 1 and I would like to convert them to binary numpy arrays of length m. For example with m=4:
- 15 → (1,1,1,1)
- 2  → (0,0,1,0)

Given an n-element integer array, I want to produce an (n, m) binary matrix. np.unpackbits always gives 8 bits and isn't what I want here.

Setup code:
<code>
import numpy as np
a = np.array([1, 2, 3, 4, 5])
m = 8
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = (((a[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
</code>

---

Problem:
Given a 2-dimensional array, I would like to normalize each row with the L1 norm (without using a loop). For example:

X = [[ 1, -2,  3,  6],
     [ 4,  5, -6,  5],
     [-1,  2,  5,  5]]

The L1 norm of each row is the sum of absolute values. How do I divide each row by its own L1 norm in a vectorized way?

Setup code:
<code>
from numpy import linalg as LA
import numpy as np
X = np.array([[ 1, -2,  3,  6],
              [ 4,  5, -6,  5],
              [-1,  2,  5,  5],
              [ 4,  5, 10,-25],
              [ 5, -2, 10, 25]])
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
l1 = np.abs(X).sum(axis=1)
result = X / l1.reshape(-1, 1)
</code>

---

Problem:
Let X be an M x N matrix. Denote xi the i-th column of X. I want to create a 3-dimensional N x M x M array consisting of the outer products xi.dot(xi.T) for each column.

How can I do this most elegantly with numpy, using only matrix operations and no loops?

Setup code:
<code>
import numpy as np
X = np.random.randint(2, 10, (5, 6))
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = X.T[:, :, None] * X.T[:, None]
</code>

---

Problem:
I have arrays of different shapes and want to zero-pad each one to match a target shape, padding only to the right and bottom (i.e. appending zeros, not prepending). For example:

a = np.ones((41, 13))
target shape = (93, 13)

How do I pad `a` to shape (93, 13) with zeros along the bottom?

Setup code:
<code>
import numpy as np
a = np.ones((41, 13))
shape = (93, 13)
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant')
</code>

---

Now generate a NEW numpy problem following this exact format. The problem should:
- Cover the numpy topic of: [TOPIC — e.g. broadcasting / fancy indexing / linear algebra / random sampling / array reshaping / masked arrays / FFT / statistical functions / etc.]
- Be of difficulty: [DIFFICULTY — easy / medium / hard]
- Show a concrete before/after array in the problem description
- Have a concise numpy solution (preferably under 4 lines) with no for or while loops
- Include a complete, runnable setup code block

You are to use the pydantic model provided to you to give structured output.
It has two fields: "question" and "answer".
The "question" field should contain everything up to and including BEGIN SOLUTION\n<code> — do NOT include the solution code inside the question field.
The "answer" field should contain only the solution code.
'''

sklearn_prompt = '''
You are a scikit-learn coding problem generator. Generate realistic, Stack Overflow-style scikit-learn problems along with clean solutions.

Each problem must follow this exact format:

Problem:
[Clear description of the scikit-learn task. Describe the data, the goal, and any specific requirements (e.g. parameters to use, output format). The problem should read like a genuine Stack Overflow question.]

Setup code:
<code>
[imports and data initialization — all variables the solution will need, fully defined]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
[concise scikit-learn solution, ideally 1–5 lines]
</code>

---

Here are six examples covering different areas of scikit-learn:

---

Problem:
I have a CSV file without headers. The last column is the target class and the rest are features. I want to split the dataset into a training set (80%) and a test set (20%), and also separate features (X) from the target (y) in each split. How do I do this with scikit-learn? Use random_state=42.

Setup code:
<code>
import numpy as np
import pandas as pd
dataset = load_data()
</code>

x_train, x_test, y_train, y_test = ... # put solution in these variables
BEGIN SOLUTION
<code>
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    dataset.iloc[:, :-1], dataset.iloc[:, -1],
    test_size=0.2, random_state=42
)
</code>

---

Problem:
I have a 2D numpy array and I want to apply MinMaxScaler to normalize the entire array as a single flat range — not column by column (which is the default behaviour). How can I reshape the array, scale it globally, then restore the original shape?

Setup code:
<code>
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
np_array = load_data()
</code>

transformed = ... # put solution in this variable
BEGIN SOLUTION
<code>
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_one_column = np_array.reshape([-1, 1])
result_one_column = scaler.fit_transform(X_one_column)
transformed = result_one_column.reshape(np_array.shape)
</code>

---

Problem:
I want to use GridSearchCV to tune hyperparameters across both a BaggingClassifier and its inner DecisionTreeClassifier simultaneously. For example, searching over max_depth from the inner tree and max_samples from the outer BaggingClassifier. What is the correct param_grid syntax for tuning nested estimator parameters?

Setup code:
<code>
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X_train, y_train = load_data()
assert type(X_train) == np.ndarray
assert type(y_train) == np.ndarray

param_grid = {
    'base_estimator__max_depth': [1, 2, 3, 4, 5],
    'max_samples': [0.05, 0.1, 0.2, 0.5]
}
dt = DecisionTreeClassifier(max_depth=1)
bc = BaggingClassifier(dt, n_estimators=20, max_samples=0.5, max_features=0.5)
</code>

clf = ... # put solution in this variable
BEGIN SOLUTION
<code>
clf = GridSearchCV(bc, param_grid)
clf.fit(X_train, y_train)
</code>

---

Problem:
I performed feature selection using ExtraTreesClassifier and SelectFromModel on a pandas DataFrame. The output of model.transform(X) is a numpy array with no column names. How do I recover the names of the selected columns from the original DataFrame?

Setup code:
<code>
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np

X, y = load_data()
clf = ExtraTreesClassifier(random_state=42)
clf = clf.fit(X, y)
</code>

column_names = ... # put solution in this variable
BEGIN SOLUTION
<code>
model = SelectFromModel(clf, prefit=True)
column_names = X.columns[model.get_support()]
</code>

---

Problem:
I have a list of 3 queries and a list of 5 documents. I want to compute the cosine similarity between each query and all 5 documents using TF-IDF vectors, producing a 3×5 similarity matrix. How do I vectorize the queries using the same vocabulary fitted on the documents and compute cosine similarity?

Setup code:
<code>
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

queries, documents = load_data()
assert type(queries) == list
assert type(documents) == list

tfidf = TfidfVectorizer()
tfidf.fit_transform(documents)
</code>

cosine_similarities_of_queries = ... # put solution in this variable
BEGIN SOLUTION
<code>
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities_of_queries = []
for query in queries:
    query_tfidf = tfidf.transform([query])
    cosine_similarities_of_queries.append(
        cosine_similarity(query_tfidf, tfidf.transform(documents)).flatten()
    )
</code>

---

Problem:
I have a precomputed distance matrix between entities (not raw feature vectors) and want to apply agglomerative hierarchical clustering to group them into 2 clusters. The default sklearn AgglomerativeClustering treats rows as feature vectors, so I need to tell it to use a precomputed distance matrix instead. How do I do this and retrieve the cluster labels as a list?

Setup code:
<code>
import numpy as np
import pandas as pd
import sklearn.cluster

data_matrix = load_data()
</code>

cluster_labels = ... # put solution in this variable
BEGIN SOLUTION
<code>
model = sklearn.cluster.AgglomerativeClustering(
    metric='precomputed', n_clusters=2, linkage='complete'
).fit(data_matrix)
cluster_labels = model.labels_
</code>

---

Now generate a NEW scikit-learn problem following this exact format. The problem should:
- Cover the scikit-learn topic of: [TOPIC — e.g. pipeline construction / cross-validation / label encoding / dimensionality reduction / model persistence / regression / text vectorization / outlier detection / etc.]
- Be of difficulty: [DIFFICULTY — easy / medium / hard]
- Clearly describe the data, the goal, and any required parameters or output format
- Include a complete setup code block with all necessary imports and variables
- Have a concise scikit-learn solution (preferably under 6 lines)

You are to use the pydantic model provided to you to give structured output.
It has two fields: "question" and "answer".
The "question" field should contain everything up to and including BEGIN SOLUTION\n<code> — do NOT include the solution code inside the question field.
The "answer" field should contain only the solution code.
'''

scipy_prompt = '''
You are a scipy coding problem generator. Generate realistic, Stack Overflow-style scipy problems along with clean solutions.

Each problem must follow this exact format:

Problem:
[Clear description of the scipy task. Describe the data, the goal, and any specific requirements (e.g. parameters to use, output format). The problem should read like a genuine Stack Overflow question.]

Setup code:
<code>
[imports and data initialization — all variables the solution will need, fully defined]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
[concise scipy solution, ideally 1–5 lines]
</code>

---

Here are six examples covering different areas of scipy:

---

Problem:
I want to perform a two-sample Kolmogorov-Smirnov test to check whether two arrays `x` and `y` come from the same distribution. I tried passing two arrays directly to `kstest` but got a TypeError because it expects a callable, not an array. What is the correct scipy function for a two-sample KS test, and how do I get the test statistic and p-value?

Setup code:
<code>
from scipy import stats
import numpy as np

np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)
</code>

statistic, p_value = ... # put solution in these variables
BEGIN SOLUTION
<code>
statistic, p_value = stats.ks_2samp(x, y)
</code>

---

Problem:
I have scattered 3D data points in the form (x, y, z) with associated values V (e.g. moisture readings from a finite element grid). The grid is irregular — not a regular meshgrid — so standard grid-based interpolation doesn't apply. I want to interpolate the value V at a new query point (25, 20, -30). How do I do this with scipy?

Setup code:
<code>
import numpy as np
import scipy.interpolate

points = np.array([
    [ 27.827,  18.53 , -30.417], [ 24.002,  17.759, -24.782],
    [ 22.145,  13.687, -33.282], [ 17.627,  18.224, -25.197],
    [ 29.018,  18.841, -38.761], [ 24.834,  20.538, -33.012],
    [ 26.232,  22.327, -27.735], [ 23.017,  23.037, -29.23 ],
    [ 28.761,  21.565, -31.586], [ 26.263,  23.686, -32.766]])
V = np.array([0.205, 0.197, 0.204, 0.197, 0.212,
              0.208, 0.204, 0.205, 0.211, 0.215])
request = np.array([[25, 20, -30]])
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = scipy.interpolate.griddata(points, V, request)
</code>

---

Problem:
Given two equal-sized sets of points in 2D space, I want to find the one-to-one assignment that matches each point in set 1 to a point in set 2 such that the total Euclidean distance between matched pairs is minimised. Each point may only be used once. How do I solve this optimal assignment problem with scipy? The result should be an array of indices into `points2` giving the best match for each point in `points1`.

Setup code:
<code>
import numpy as np
import scipy.spatial
import scipy.optimize

points1 = np.array([(x, y) for x in np.linspace(-1, 1, 7) for y in np.linspace(-1, 1, 7)])
N = points1.shape[0]
points2 = 2 * np.random.rand(N, 2) - 1
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
C = scipy.spatial.distance.cdist(points1, points2)
_, result = scipy.optimize.linear_sum_assignment(C)
</code>

---

Problem:
I have an N×a×b numpy array where each of the N slices is a small a×b matrix, and I want to arrange all N slices into a single block-diagonal matrix. `scipy.linalg.block_diag` works but requires each block to be passed as a separate argument, which is impractical when N is in the hundreds. How do I unpack the array into `block_diag` efficiently?

Setup code:
<code>
import numpy as np
from scipy.linalg import block_diag

np.random.seed(10)
a = np.random.rand(100, 2, 2)
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = block_diag(*a)
</code>

---

Problem:
I want to solve a first-order ODE with a time-varying right-hand side: dy/dt = -100*y + sin(t), starting from y(0) = N0 over the interval [-0.1, 0.1]. The new `solve_ivp` API requires a callable for `fun`. How do I define the time-varying ODE and call `solve_ivp` correctly? Store the solution values in `result = sol.y`.

Setup code:
<code>
import scipy.integrate
import numpy as np

N0 = 10
time_span = [-0.1, 0.1]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
def dN1_dt(t, N1):
    return -100 * N1 + np.sin(t)

sol = scipy.integrate.solve_ivp(fun=dN1_dt, t_span=time_span, y0=[N0,])
result = sol.y
</code>

---

Problem:
I have a Voronoi tessellation built from a set of seed `points`, and a separate list of `extraPoints`. For each extra point I want to find which Voronoi cell (i.e. which seed point) it belongs to — effectively the index of the nearest seed point. How do I do this efficiently with scipy spatial, returning an array of seed-point indices, one per extra point?

Setup code:
<code>
import scipy.spatial

points = [[0, 0], [1, 4], [2, 3], [4, 1], [1, 1], [2, 2], [5, 3]]
vor = scipy.spatial.Voronoi(points)
extraPoints = [[0.5, 0.2], [3, 0], [4, 0], [5, 0], [4, 3]]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
kdtree = scipy.spatial.cKDTree(points)
_, result = kdtree.query(extraPoints)
</code>

---

Now generate a NEW scipy problem following this exact format. The problem should:
- Cover the scipy topic of: [TOPIC — e.g. signal processing / sparse matrices / numerical integration / curve fitting / statistical distributions / root finding / distance metrics / image processing / etc.]
- Be of difficulty: [DIFFICULTY — easy / medium / hard]
- Clearly describe the data, the goal, and any required parameters or output format
- Include a complete setup code block with all necessary imports and variables
- Have a concise scipy solution (preferably under 6 lines)

You are to use the pydantic model provided to you to give structured output.
It has two fields: "question" and "answer".
The "question" field should contain everything up to and including BEGIN SOLUTION\n<code> — do NOT include the solution code inside the question field.
The "answer" field should contain only the solution code.
'''

pytorch_prompt = '''
You are a PyTorch coding problem generator. Generate realistic, Stack Overflow-style PyTorch problems along with clean solutions.

Each problem must follow this exact format:

Problem:
[Clear description of the PyTorch task. Describe the tensors, model, or training setup involved, and explain precisely what the solution should achieve. The problem should read like a genuine Stack Overflow question.]

Setup code:
<code>
[imports and data/model initialization — all variables the solution will need, fully defined]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
[concise PyTorch solution, ideally 1–5 lines]
</code>

---

Here are six examples covering different areas of PyTorch:

---

Problem:
I have a PyTorch optimizer created with an initial learning rate, but partway through training I realize the learning rate is too high and want to change it dynamically — without defining a scheduler upfront. Is there a way to update the learning rate of an existing optimizer in-place?

Setup code:
<code>
import numpy as np
import pandas as pd
import torch

optim = load_data()
</code>

BEGIN SOLUTION
<code>
for param_group in optim.param_groups:
    param_group['lr'] = 0.001
</code>

---

Problem:
I have a pre-trained word2vec model loaded with gensim and I want to use its word vectors to initialise a PyTorch `nn.Embedding` layer, then embed an input tensor of word indices. How do I transfer the gensim weights into an `Embedding` layer using `from_pretrained`?

Setup code:
<code>
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

input_Tensor = load_data()
word2vec = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
</code>

embedded_input = ... # put solution in this variable
BEGIN SOLUTION
<code>
weights = torch.FloatTensor(word2vec.wv.vectors)
embedding = torch.nn.Embedding.from_pretrained(weights)
embedded_input = embedding(input_Tensor)
</code>

---

Problem:
Given a list of sequence lengths `lens`, I want to create a 2D boolean mask tensor where each row has `True` (or 1) for positions up to the corresponding length and `False` (or 0) beyond it. For example:

lens = [3, 5, 4]

Should produce:
mask = [[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]]

How do I do this efficiently without looping, returning a LongTensor?

Setup code:
<code>
import numpy as np
import pandas as pd
import torch

lens = load_data()
</code>

mask = ... # put solution in this variable
BEGIN SOLUTION
<code>
max_len = max(lens)
mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
mask = mask.type(torch.LongTensor)
</code>

---

Problem:
I am working on an image segmentation task with 7 classes. My model outputs a tensor of shape [batch, num_classes, height, width] and my targets are integer class indices of shape [batch, height, width]. I tried implementing a custom cross-entropy loss but kept running into shape errors. How do I correctly compute cross-entropy loss for this segmentation setup using PyTorch's built-in loss? Use the default arguments.

Setup code:
<code>
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn.functional as F

images, labels = load_data()
</code>

loss = ... # put solution in this variable
BEGIN SOLUTION
<code>
loss_func = torch.nn.CrossEntropyLoss()
loss = loss_func(images, labels)
</code>

---

Problem:
I have a trained PyTorch classification model and want to get the predicted class along with a calibrated confidence score in the range (0, 1). Currently `torch.max` on the raw logits gives an unbounded score. How do I apply softmax to convert logits to probabilities and then extract the top confidence score?

Setup code:
<code>
import numpy as np
import pandas as pd
import torch

MyNet = torch.nn.Sequential(
    torch.nn.Linear(4, 15),
    torch.nn.Sigmoid(),
    torch.nn.Linear(15, 3),
)
input = load_data()
assert type(input) == torch.Tensor
</code>

confidence_score = ... # put solution in this variable
BEGIN SOLUTION
<code>
output = MyNet(input)
probs = torch.nn.functional.softmax(output.reshape(1, 3), dim=1)
confidence_score, classes = torch.max(probs, 1)
</code>

---

Problem:
I have a 2D tensor and I want to pad it with a border of zeros on all four sides, turning an (H, W) tensor into an (H+2, W+2) tensor. For example:

Input:
1 2
3 4
5 6
7 8

Desired output:
0 0 0 0
0 1 2 0
0 3 4 0
0 5 6 0
0 7 8 0
0 0 0 0

I tried `torch.stack` and `torch.cat` but couldn't get the shapes to align. What is the correct way to do this in PyTorch?

Setup code:
<code>
import numpy as np
import pandas as pd
import torch

t = load_data()
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
result = torch.nn.functional.pad(t, (1, 1, 1, 1))
</code>

---

Now generate a NEW PyTorch problem following this exact format. The problem should:
- Cover the PyTorch topic of: [TOPIC — e.g. custom Dataset/DataLoader / gradient manipulation / model saving and loading / tensor indexing / batch normalisation / attention mechanisms / weight initialisation / mixed precision training / etc.]
- Be of difficulty: [DIFFICULTY — easy / medium / hard]
- Clearly describe the tensors, model setup, and the expected output
- Include a complete setup code block with all necessary imports and variables
- Have a concise PyTorch solution (preferably under 6 lines)

You are to use the pydantic model provided to you to give structured output.
It has two fields: "question" and "answer".
The "question" field should contain everything up to and including BEGIN SOLUTION\n<code> — do NOT include the solution code inside the question field.
The "answer" field should contain only the solution code.
'''

tensorflow_prompt = '''
You are a TensorFlow coding problem generator. Generate realistic, Stack Overflow-style TensorFlow 2.x problems along with clean solutions.

Each problem must follow this exact format:

Problem:
[Clear description of the TensorFlow task. Describe the tensors, model, or pipeline involved, and explain precisely what the solution should achieve. The problem should read like a genuine Stack Overflow question.]

Setup code:
<code>
[imports and data/model initialization — all variables the solution will need, fully defined]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
[concise TensorFlow solution, ideally 1–8 lines]
</code>

---

Here are six examples covering different areas of TensorFlow:

---

Problem:
I'm using TensorFlow 2.10.0. I have a list of integer class labels and need to convert them into a one-hot encoded tensor with 10 classes (depth=10). For example, given [0, 6, 5, 4, 2] the result should be a 5×10 integer tensor where each row has a 1 at the class index and 0 elsewhere. How do I do this with TensorFlow?

Setup code:
<code>
import tensorflow as tf

labels = [0, 6, 5, 4, 2]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
def g(labels):
    return tf.one_hot(indices=labels, depth=10, on_value=1, off_value=0, axis=-1)

result = g(labels.copy())
</code>

---

Problem:
I'm using TensorFlow 2.10.0. I have a `tf.data.Dataset` built from a list of integers and a map function that returns multiple elements per input (e.g. [i, i+1, i+2] for each i). I want the final dataset to yield each of those elements individually as a flat stream, not as nested arrays. How do I use `flat_map` to expand each input element into multiple output elements?

For example, given input = [10, 20, 30], the desired result is [10, 11, 12, 20, 21, 22, 30, 31, 32].

Setup code:
<code>
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
input = [10, 20, 30]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
def g(input):
    ds = tf.data.Dataset.from_tensor_slices(input)
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices([x, x + 1, x + 2]))
    element = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    result = []
    with tf.compat.v1.Session() as sess:
        for _ in range(9):
            result.append(sess.run(element))
    return result

result = g(input)
</code>

---

Problem:
I'm using TensorFlow 2.10.0. I have a list of sequence lengths, e.g. [4, 3, 5, 2], and I want to produce a 2D mask tensor of shape (4, 8) where each row contains 1s up to the corresponding length and 0s beyond it. For example:

lengths = [4, 3, 5, 2]

Desired output:
[[1,1,1,1,0,0,0,0],
 [1,1,1,0,0,0,0,0],
 [1,1,1,1,1,0,0,0],
 [1,1,0,0,0,0,0,0]]

How do I build this mask without looping?

Setup code:
<code>
import tensorflow as tf

lengths = [4, 3, 5, 2]
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
def g(lengths):
    lengths_transposed = tf.expand_dims(lengths, 1)
    range = tf.range(0, 8, 1)
    range_row = tf.expand_dims(range, 0)
    mask = tf.less(range_row, lengths_transposed)
    result = tf.where(mask, tf.ones([4, 8]), tf.zeros([4, 8]))
    return result

result = g(lengths.copy())
</code>

---

Problem:
I'm using TensorFlow 2.10.0. I have two 2D embedding tensors `a` and `b` of the same shape and I want to compute the row-wise squared Euclidean (L2) distance between them. For example:

a = [[1,1,1], [1,1,1]]
b = [[0,0,0], [1,1,1]]

Desired output: [3, 0]  (sum of squared differences per row)

I tried `tf.reduce_sum` but couldn't reduce by row. What is the correct approach?

Setup code:
<code>
import tensorflow as tf

a = tf.constant([[1, 1, 1], [1, 1, 1]])
b = tf.constant([[0, 0, 0], [1, 1, 1]])
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
def g(a, b):
    return tf.reduce_sum(tf.square(tf.subtract(a, b)), 1)

result = g(a.__copy__(), b.__copy__())
</code>

---

Problem:
I need to migrate TensorFlow 1.x code to TensorFlow 2.x. The original code uses `tf.Session` and `sess.run()` to evaluate `tf.reduce_sum(tf.matmul(A, B))` where A and B are 100×100 random normal matrices with a fixed seed. In TF2, `Session` is removed and eager execution is the default. How do I rewrite this using `@tf.function` for graph-mode performance, keeping the same fixed seed behaviour?

Setup code:
<code>
import tensorflow as tf
</code>

result = ... # put solution in this variable
BEGIN SOLUTION
<code>
tf.random.set_seed(10)

def get_values():
    A = tf.random.normal([100, 100])
    B = tf.random.normal([100, 100])
    return A, B

@tf.function
def compute():
    A, B = get_values()
    return tf.reduce_sum(tf.matmul(A, B))

result = compute()
</code>

---

Problem:
I'm using TensorFlow 2.10.0. I have a compiled and trained Keras Sequential model and want to save it in the SavedModel format (a directory containing `saved_model.pb` and a `variables/` folder) rather than as an HDF5 `.h5` file. Using `model.save("my_model")` without a file extension gives me an HDF5 file instead. How do I explicitly save it in the SavedModel format to a directory called `"export/1"`?

Setup code:
<code>
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, name="Input", input_dim=4, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(8, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(2, name="Output", kernel_initializer='he_normal', activation='relu'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

# Save the model in SavedModel format to "export/1"
</code>

BEGIN SOLUTION
<code>
tms_model = tf.saved_model.save(model, "export/1")
</code>

---

Now generate a NEW TensorFlow problem following this exact format. The problem should:
- Cover the TensorFlow topic of: [TOPIC — e.g. custom training loop / tf.GradientTape / Keras layers / tf.data pipeline / tensor reshaping / callbacks / transfer learning / mixed precision / sparse tensors / etc.]
- Be of difficulty: [DIFFICULTY — easy / medium / hard]
- Clearly describe the tensors, model setup, or pipeline and the expected output
- Include a complete setup code block with all necessary imports and variables
- Have a concise TensorFlow solution (preferably under 8 lines)

You are to use the pydantic model provided to you to give structured output.
It has two fields: "question" and "answer".
The "question" field should contain everything up to and including BEGIN SOLUTION\n<code> — do NOT include the solution code inside the question field.
The "answer" field should contain only the solution code.
'''
