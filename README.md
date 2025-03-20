# Python Cheat Sheet - Data Science Interview Ready with Examples

## Core Python & Data Structures

* **Lists (Mutable, Ordered):**

    * Operations: `append`, `insert`, `remove`, `pop`, `index`, `count`, `sort`, `reverse`, slicing `[start:stop:step]`
    * Comprehensions: `[x**2 for x in range(10) if x % 2 == 0]`
    * Use Cases: Dynamic data storage, ordered collections.

    ```
    # Example
    my_list = [1, 2, 3, 4, 5]
    my_list.append(6)  # [1, 2, 3, 4, 5, 6]
    my_list.insert(0, 0)  # [0, 1, 2, 3, 4, 5, 6]
    my_list.remove(3)  # [0, 1, 2, 4, 5, 6]
    popped_item = my_list.pop(1)  # Returns 1, list is now [0, 2, 4, 5, 6]
    index_of_4 = my_list.index(4)  # Returns 2
    count_of_2 = my_list.count(2)  # Returns 1
    my_list.sort()  # [0, 2, 4, 5, 6]
    my_list.reverse()  # [6, 5, 4, 2, 0]
    sliced_list = my_list[1:4]  # [5, 4, 2]
    even_squares = [x**2 for x in range(10) if x % 2 == 0]  # [0, 4, 16, 36, 64]
    ```
* **List Comprehensions:**

    * A concise way to create lists.
    * Syntax: `[expression for item in iterable if condition]`

    ```
    # Example
    numbers = [1, 2, 3, 4, 5]
    squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]
    even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16]
    ```
* **Tuples (Immutable, Ordered):**

    * Use Cases: Returning multiple values from functions, representing fixed records.

    ```
    # Example
    my_tuple = (1, 2, 3)
    # my_tuple[0] = 4  # This would cause an error
    def get_coordinates():
        return (10, 20)
    x, y = get_coordinates()  # x is 10, y is 20
    ```
* **Sets (Mutable, Unordered, Unique):**

    * Operations: `add`, `remove`, `discard`, `union`, `intersection`, `difference`.
    * Use Cases: Removing duplicates, membership testing.

    ```
    # Example
    my_set = {1, 2, 3, 4, 5}
    my_set.add(6)  # {1, 2, 3, 4, 5, 6}
    my_set.remove(3)  # {1, 2, 4, 5, 6}
    my_set.discard(7)  # No error if 7 is not present
    another_set = {4, 5, 6, 7, 8}
    union_set = my_set.union(another_set)  # {1, 2, 4, 5, 6, 7, 8}
    intersection_set = my_set.intersection(another_set)  # {4, 5, 6}
    difference_set = my_set.difference(another_set)  # {1, 2}
    ```
* **Dictionaries (Mutable, Key-Value Pairs):**

    * Operations: `get`, `keys`, `values`, `items`, `pop`, `update`.
    * Use Cases: Storing and retrieving data efficiently.

    ```
    # Example
    my_dict = {"name": "Alice", "age": 30, "city": "New York"}
    name = my_dict.get("name")  # "Alice"
    age = my_dict["age"]  # 30
    keys = my_dict.keys()  # ["name", "age", "city"]
    values = my_dict.values()  # ["Alice", 30, "New York"]
    items = my_dict.items()  # [("name", "Alice"), ("age", 30), ("city", "New York")]
    popped_age = my_dict.pop("age")  # Returns 30, dict is now {"name": "Alice", "city": "New York"}
    my_dict.update({"age": 31, "country": "USA"})  # {"name": "Alice", "city": "New York", "age": 31, "country": "USA"}
    ```
* **Dictionary Comprehensions:**

    * A concise way to create dictionaries.
    * Syntax: `{key_expression: value_expression for item in iterable if condition}`

    ```
    # Example
    numbers = [1, 2, 3, 4, 5]
    number_squares = {number: number**2 for number in numbers}  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
    even_squares_dict = {number: number**2 for number in numbers if number % 2 == 0}  # {2: 4, 4: 16}
    ```
* **String Manipulation:**

    * f-strings, `.format()`, `.split()`, `.join()`, `.strip()`, regular expressions (`re` module).

    ```
    # Example
    name = "Bob"
    age = 25
    greeting = f"Hello, {name}! You are {age} years old."  # "Hello, Bob! You are 25 years old."
    greeting_format = "Hello, {}! You are {} years old.".format(name, age)
    text = "  This is a string with spaces.  "
    stripped_text = text.strip()  # "This is a string with spaces."
    words = stripped_text.split()  # ["This", "is", "a", "string", "with", "spaces."]
    joined_text = "-".join(words)  # "This-is-a-string-with-spaces."
    import re
    pattern = r"\d+"  # Matches one or more digits
    numbers = re.findall(pattern, "abc123def456")  # ["123", "456"]
    # Common string methods and built-in functions
    text = "Python is an amazing language"
    print(len(text))  # 28 (length of the string)
    print(text.lower())  # "python is an amazing language"
    print(text.upper())  # "PYTHON IS AN AMAZING LANGUAGE"
    print(text.replace("amazing", "awesome"))  # "Python is an awesome language"
    print(text.startswith("Python"))  # True
    print(text.endswith("language"))  # True
    print(text.find("amazing"))  # 11 (starting index of "amazing")
    print(text.count("a"))  # 4 (counts occurrences of "a")
    text = "   Remove spaces   "
    print(text.strip())  # "Remove spaces" (removes leading/trailing spaces)
    print(text.lstrip())  # "Remove spaces   " (removes leading spaces)
    print(text.rstrip())  # "   Remove spaces" (removes trailing spaces)
    text = "Split,the,string,by,comma"
    print(text.split(","))  # ["Split", "the", "string", "by", "comma"]
    words = ["Join", "these", "words"]
    print(" ".join(words))  # "Join these words"
    text = "Python"
    print(text[0])  # "P" (indexing)
    print(text[2:5])  # "tho" (slicing)
    print("Py" in text)  # True (substring check)
    print(text[::-1])  # "nohtyP" (reversing)
    ```
* **Functions:**

    * Define functions with parameters, return values, docstrings.
    * Lambda functions (anonymous functions) for concise operations.

    ```
    # Example
    def greet(name="World"):
        """Greets the person with the given name."""
        print(f"Hello, {name}!")
    greet("Alice")  # Hello, Alice!
    greet()  # Hello, World!
    def add(a, b):
        """Returns the sum of a and b."""
        return a + b
    result = add(5, 3)  # 8
    square = lambda x: x**2
    squared_value = square(4)  # 16
    ```
* **Control Flow:**

    * `if`/`elif`/`else`, `for` loops, `while` loops, `break`, `continue`.
    * Exception handling (`try`/`except`/`finally`).

    ```
    # Example
    x = 10
    if x > 0:
        print("Positive")
    elif x < 0:
        print("Negative")
    else:
        print("Zero")
    for i in range(3):  # 0, 1, 2
        print(i)
    j = 0
    while j < 3:
        print(j)
        j += 1
    for k in range(5):
        if k == 3:
            break  # Exit loop when k is 3
        print(k)  # 0, 1, 2
    for m in range(5):
        if m == 2:
            continue  # Skip when m is 2
        print(m)  # 0, 1, 3, 4
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("Cannot divide by zero")
    finally:
        print("This will always execute")
    ```
* **More Built-in Functions and Methods:**

    * `enumerate()`: Adds a counter to an iterable and returns it as an enumerate object.

    ```
    #Example
    my_list = ['a', 'b', 'c']
    for index, value in enumerate(my_list):
        print(index, value)  # Output: 0 a, 1 b, 2 c
    ```
    * `zip()`: Used to combine several iterables into one.

    ```
    #Example
    list1 = [1, 2, 3]
    list2 = ['x', 'y', 'z']
    for item1, item2 in zip(list1, list2):
        print(item1, item2)  # Output: 1 x, 2 y, 3 z
    ```
    * `all()`: Returns True if all elements of an iterable are true.

    ```
    print(all([True, True, False])) #output: False
    print(all([True, True, True])) #output: True
    ```
    * `any()`: Returns True if any element of an iterable is true.

    ```
    print(any([True, True, False])) #output: True
    print(any([False, False, False])) #output: False
    ```
    * `map()`: Applies a given function to each item of an iterable (list, tuple etc.) and returns a list of the results.

    ```
    def square(n):
      return n * n
    numbers = (1, 2, 3, 4)
    result = map(square, numbers)
    print(list(result)) #output: [1, 4, 9, 16]
    ```
    * `filter()`: Constructs an iterator from elements of an iterable for which a function returns true.

    ```
    def is_even(x):
        return x % 2 == 0
    numbers = [1, 2, 3, 4, 5, 6]
    even_numbers = filter(is_even, numbers)
    print(list(even_numbers))  # Output: [2, 4, 6]
    ```
    * `isinstance()`: Checks if an object is an instance of a class or a type.

    ```
    x = 5
    print(isinstance(x, int))  # Output: True
    print(isinstance(x, str))  # Output: False
    ```
    * `dir()`: Returns a list of names in the current local scope or a list of attributes of a specified object.

    ```
    print(dir())  # Lists names in the current scope
    print(dir("hello"))  # Lists string methods
    ```
    * `help()`: Invokes the built-in help system.

    ```
    help(len)  # Displays help information about the len() function
    ```
    * `sorted()`: Returns a new sorted list from the elements of any iterable.

    ```
    # Example
    numbers = [3, 1, 4, 2]
    sorted_numbers = sorted(numbers)  # Output: [1, 2, 3, 4]
    print(sorted_numbers)
    ```

## NumPy (Numerical Python)

import numpy as np
* **Array Creation:**

    * `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, `np.random.rand()`, `np.random.randn()`.

    ```
    # Example
    arr = np.array([1, 2, 3])
    zeros_arr = np.zeros((2, 3))  # 2x3 array of zeros
    ones_arr = np.ones((3, 2))  # 3x2 array of ones
    range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
    linspace_arr = np.linspace(0, 1, 5)  # [0.  , 0.25, 0.5 , 0.75, 1.  ]
    rand_arr = np.random.rand(2, 2)  # 2x2 array of random floats between 0 and 1
    randn_arr = np.random.randn(2, 2)  # 2x2 array of random floats from standard normal distribution
    ```
* **Array Indexing and Slicing:**

    * Similar to lists, but with multi-dimensional support.

    ```
    # Example
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    first_row = arr[0]  # [1, 2, 3]
    second_column = arr[:, 1]  # [2, 5, 8]
    sub_matrix = arr[0:2, 1:3]  # [[2, 3], [5, 6]]
    ```
* **Array Operations:**

    * Element-wise operations (`+`, `-`, `*`, `/`), dot product (`np.dot()`), matrix multiplication (`np.matmul()` or `@`), transpose (`.T`).
    * Aggregation: `np.sum()`, `np.mean()`, `np.median()`, `np.std()`, `np.max()`, `np.min()`.
    * Broadcasting: Operations on arrays of different shapes.
    * Reshaping and Transposing: `.reshape()`, `.flatten()`, `.transpose()`.
    * Boolean indexing and masking.

    ```
    # Example
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    element_wise_sum = arr1 + arr2  # [[ 6,  8], [10, 12]]
    dot_product = np.dot(arr1, arr2)  # [[19, 22], [43, 50]]
    matrix_product = arr1 @ arr2  # [[19, 22], [43, 50]]
    transposed_arr = arr1.T  # [[1, 3], [2, 4]]
    arr = np.array([1, 2, 3, 4, 5])
    total_sum = np.sum(arr)  # 15
    mean_value = np.mean(arr)  # 3.0
    median_value = np.median(arr)  # 3.0
    std_deviation = np.std(arr)  # 1.41421356
    max_value = np.max(arr)  # 5
    min_value = np.min(arr)  # 1
    # Broadcasting example
    arr1 = np.array([1, 2, 3])
    scalar = 2
    result = arr1 * scalar  # [2, 4, 6]
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    reshaped_arr = arr.reshape((3, 2))  # [[1, 2], [3, 4], [5, 6]]
    flattened_arr = arr.flatten()  # [1, 2, 3, 4, 5, 6]
    arr = np.array([1, 2, 3, 4, 5])
    bool_index = arr > 2  # [False, False,  True,  True,  True]
    masked_arr = arr[bool_index]  # [3, 4, 5]
    ```

## Pandas (Data Analysis)

import pandas as pd
* **Series (1D labeled array):**

    * `pd.Series()`, indexing, slicing, vectorized operations.

    ```
    # Example
    s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    print(s['b'])  # 20
    print(s[1])  # 20
    print(s['a':'c'])
    s2 = s * 2  # pd.Series([20, 40, 60], index=['a', 'b', 'c'])
    ```
* **DataFrame (2D labeled table):**

    * `pd.DataFrame()`, reading data (`pd.read_csv()`, `pd.read_excel()`), writing data (`df.to_csv()`).
    * Indexing and Selection: `.loc[]`, `.iloc[]`, boolean indexing.
    * Data Cleaning: `.dropna()`, `.fillna()`, `.drop_duplicates()`, `.astype()`.
    * Data Transformation: `.apply()`, `.map()`, `.groupby()`, `.pivot_table()`, `.merge()`, `.concat()`.
    * Missing Values: `.isna()`, `.notna()`.
    * Time Series: `DatetimeIndex`, resampling.
    * String Methods: `.str.lower()`, `.str.upper()`, `.str.contains()`.

    ```
    # Example
    data = {'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 28],
            'city': ['New York', 'London', 'Paris']}
    df = pd.DataFrame(data)
    # Reading and writing data
    # df = pd.read_csv("my_data.csv")
    # df = pd.read_excel("my_data.xlsx")
    # df.to_csv("output.csv", index=False)
    # Indexing and selection
    print(df.loc[0])  # First row
    print(df.loc[0:1, ['name', 'age']])  # Rows 0 and 1, columns 'name' and 'age'
    print(df.iloc[0])  # First row
    print(df.iloc[0:2, 0:2])
    print(df[df['age'] > 25])  # Rows where age is greater than 25
    # Data cleaning
    df_dropped_na = df.dropna()  # Drop rows with any NaN values
    df_filled_na = df.fillna(0)  # Fill NaN values with 0
    df_no_duplicates = df.drop_duplicates()  # Drop duplicate rows
    df['age'] = df['age'].astype(int)  # Change 'age' column to integer type
    # Data transformation
    df['age_plus_10'] = df['age'].apply(lambda x: x + 10)  # Apply a function to each element in 'age'
    df['age_times_2'] = df['age'].map(lambda x: x * 2)
    grouped_df = df.groupby('city').mean()  # Group by 'city' and calculate the mean of other columns
    pivot_df = df.pivot_table(index='city', columns='name', values='age')
    merged_df = pd.merge(df, grouped_df, on='city', suffixes=('_original', '_mean'))
    concatenated_df = pd.concat([df, df_dropped_na], ignore_index=True)
    # Missing values
    print(df.isna())
    print(df.notna())
    # Time series
    # df['date'] = pd.to_datetime(df['date'])
    # df.set_index('date', inplace=True)
    # resampled_df = df.resample('D').mean()  # Resample by day
    # String methods
    df['name_lower'] = df['name'].str.lower()
    df['name_upper'] = df['name'].str.upper()
    df['name_contains_A'] = df['name'].str.contains('a', case=False)
    ```
## Scikit-learn (Machine Learning)

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```
### Model Selection:
train_test_split(), cross_val_score(), GridSearchCV().

# Example
```
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation scores: {cv_scores}")

param_grid = {'fit_intercept': [True, False]}
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
```
### Preprocessing:
StandardScaler(), MinMaxScaler(), LabelEncoder(), OneHotEncoder(), ColumnTransformer(), Pipeline().

# Example
```
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
X = np.array([[10, 2], [20, 4], [30, 6]])
y = np.array(['a', 'b', 'a'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

min_max_scaler = MinMaxScaler()
X_minmax_scaled = min_max_scaler.fit_transform(X)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # [0, 1, 0]

onehot_encoder = OneHotEncoder()
y_onehot_encoded = onehot_encoder.fit_transform(y.reshape(-1, 1)).toarray()  # [[1, 0], [0, 1], [1, 0]]

# ColumnTransformer
transformer = ColumnTransformer([
    ('scaler', StandardScaler(), [0]),  # Apply StandardScaler to the first column
    ('onehot', OneHotEncoder(), [1])  # Apply OneHotEncoder to the second column
])
X_transformed = transformer.fit_transform(X)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```
### Supervised Learning:
Regression: LinearRegression, RandomForestRegressor.Classification: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC.
# Example
```
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
import numpy as np

# Assuming X_train, X_test, y_train, y_test are defined

# Regression
model_regression = LinearRegression()
model_regression.fit(X_train, y_train)
predictions_regression = model_regression.predict(X_test)

model_random_forest_regressor = RandomForestRegressor()
model_random_forest_regressor.fit(X_train, y_train)
predictions_forest_regression = model_random_forest_regressor.predict(X_test)

# Classification
model_classification = LogisticRegression()
model_classification.fit(X_train, y_train)
predictions_classification = model_classification.predict(X_test)

model_decision_tree_classifier = DecisionTreeClassifier()
model_decision_tree_classifier.fit(X_train, y_train)
predictions_decision_tree = model_decision_tree_classifier.predict(X_test)

model_random_forest_classifier = RandomForestClassifier()
model_random_forest_classifier.fit(X_train, y_train)
predictions_forest_classification = model_random_forest_classifier.predict(X_test)

model_svc = SVC()
model_svc.fit(X_train, y_train)
predictions_svc = model_svc.predict(X_test)
```

### Unsupervised Learning:
Clustering: KMeans.
# Example
```
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
```

### Evaluation Metrics:
Classification: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix.
Regression: mean_squared_error, r2_score.
# Example
```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, mean_squared_error, r2_score
y_true_classification = [0, 1, 0, 1, 0, 1]
y_pred_classification = [0, 1, 1, 1, 0, 0]
accuracy = accuracy_score(y_true_classification, y_pred_classification)
precision = precision_score(y_true_classification, y_pred_classification)
recall = recall_score(y_true_classification, y_pred_classification)
f1 = f1_score(y_true_classification, y_pred_classification)
# roc_auc = roc_auc_score(y_true_classification, y_pred_classification) # Only for binary classification
report = classification_report(y_true_classification, y_pred_classification)
matrix = confusion_matrix(y_true_classification, y_pred_classification)

y_true_regression = [2.5, 3, 4.5, 6, 7.8]
y_pred_regression = [2.2, 3.1, 4, 5.9, 8]
mse = mean_squared_error(y_true_regression, y_pred_regression)
r2 = r2_score(y_true_regression, y_pred_regression)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
# print(f"ROC AUC Score: {roc_auc}")
print(f"Classification Report: \n{report}")
print(f"Confusion Matrix: \n{matrix}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```
### Pipelines: Combine preprocessing and modeling.
# Example
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np # Import numpy
#Assuming X_train, X_test, y_train, y_test are defined

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```
### Visualization (Matplotlib & Seaborn)
```
import matplotlib.pyplot as plt
import seaborn as sns
```
Matplotlib:
plt.plot(), plt.scatter(), plt.hist(), plt.bar(), plt.xlabel(), plt.ylabel(), plt.title(), plt.show().

# Example
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sine Wave")
plt.show()

plt.scatter(x, y)
plt.show()

data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.show()
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]
plt.bar(categories, values)
plt.show()
```
Seaborn:
sns.scatterplot(), sns.histplot(), sns.boxplot(), sns.heatmap(), sns.pairplot(), sns.countplot().Styling and customizing.
# Example

```
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data
data = {'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 28, 26, 32, 29],
            'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Paris'],
            'value': [10, 20, 15, 12, 25, 18]}
df = pd.DataFrame(data)

sns.scatterplot(x='age', y='value', hue='city', data=df)
plt.show()

sns.histplot(data=df, x='age', kde=True)
plt.show()

sns.boxplot(x='city', y='value', data=df
```
### Key Concepts for Interviews
* Data cleaning and preprocessing.
* Feature engineering and selection.
* Model selection and evaluation.
* Common ML algorithms (linear/logistic regression, decision trees, random forests, etc.).
* Overfitting and underfitting.
* Bias-variance tradeoff.
* Regularization.
* Dimensionality reduction (PCA).
* Handling imbalanced data.
* Model evaluation.


