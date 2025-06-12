```python
### loading the necessary packages for the entire project. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.formula.api import ols 
from scipy.stats import norm

import sklearn 

from sklearn.linear_model import LinearRegression
```


### loading the necessary packages for the entire project. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.formula.api import ols 
from scipy.stats import norm

import sklearn 

from sklearn.linear_model import LinearRegression


```python
Data_sale= pd.read_csv( r"C:\Users\akong\OneDrive\Desktop\doc assemble\data camp files\pet_sales (1).csv"          )
```

Data_sale= pd.read_csv( r"C:\Users\akong\OneDrive\Desktop\doc assemble\data camp files\pet_sales (1).csv"          )


```python
Data_sale

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_category</th>
      <th>sales</th>
      <th>price</th>
      <th>vendor_id</th>
      <th>pet_size</th>
      <th>pet_type</th>
      <th>rating</th>
      <th>re_buy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5040</td>
      <td>Equipment</td>
      <td>$123,000</td>
      <td>94.81</td>
      <td>VC_1605</td>
      <td>small</td>
      <td>fish</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4567</td>
      <td>Toys</td>
      <td>$61,000</td>
      <td>120.95</td>
      <td>VC_1132</td>
      <td>small</td>
      <td>cat</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4237</td>
      <td>Toys</td>
      <td>$218,000</td>
      <td>106.34</td>
      <td>VC_802</td>
      <td>small</td>
      <td>hamster</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4364</td>
      <td>Snack</td>
      <td>$69,000</td>
      <td>241.27</td>
      <td>VC_929</td>
      <td>large</td>
      <td>dog</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4184</td>
      <td>Supplements</td>
      <td>$138,000</td>
      <td>133.68</td>
      <td>VC_749</td>
      <td>large</td>
      <td>dog</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>874</th>
      <td>4999</td>
      <td>Snack</td>
      <td>$27,000</td>
      <td>146.93</td>
      <td>VC_1564</td>
      <td>medium</td>
      <td>bird</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>875</th>
      <td>4243</td>
      <td>Snack</td>
      <td>$76,000</td>
      <td>174.07</td>
      <td>VC_808</td>
      <td>medium</td>
      <td>hamster</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>876</th>
      <td>4783</td>
      <td>Snack</td>
      <td>$162,000</td>
      <td>224.12</td>
      <td>VC_1348</td>
      <td>medium</td>
      <td>cat</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>877</th>
      <td>4664</td>
      <td>Bedding</td>
      <td>$34,000</td>
      <td>199.15</td>
      <td>VC_1229</td>
      <td>large</td>
      <td>dog</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>878</th>
      <td>4850</td>
      <td>Toys</td>
      <td>$54,000</td>
      <td>171.85</td>
      <td>VC_1415</td>
      <td>small</td>
      <td>dog</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>879 rows × 9 columns</p>
</div>



Data_sale



```python
Data_sale.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 879 entries, 0 to 878
    Data columns (total 10 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   product_id        879 non-null    int64  
     1   product_category  879 non-null    object 
     2   sales             879 non-null    object 
     3   price             879 non-null    float64
     4   vendor_id         879 non-null    object 
     5   pet_size          879 non-null    object 
     6   pet_type          879 non-null    object 
     7   rating            879 non-null    int64  
     8   re_buy            879 non-null    int64  
     9   sale1             879 non-null    float64
    dtypes: float64(2), int64(3), object(5)
    memory usage: 68.8+ KB
    


```python
Data_sale['sale1']= Data_sale['sales'].str.strip('$').str.replace(',', '').astype('float')

```


```python
Data_sale.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_category</th>
      <th>sales</th>
      <th>price</th>
      <th>vendor_id</th>
      <th>pet_size</th>
      <th>pet_type</th>
      <th>rating</th>
      <th>re_buy</th>
      <th>sale1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5040</td>
      <td>Equipment</td>
      <td>$123,000</td>
      <td>94.81</td>
      <td>VC_1605</td>
      <td>small</td>
      <td>fish</td>
      <td>7</td>
      <td>1</td>
      <td>123000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4567</td>
      <td>Toys</td>
      <td>$61,000</td>
      <td>120.95</td>
      <td>VC_1132</td>
      <td>small</td>
      <td>cat</td>
      <td>10</td>
      <td>0</td>
      <td>61000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4237</td>
      <td>Toys</td>
      <td>$218,000</td>
      <td>106.34</td>
      <td>VC_802</td>
      <td>small</td>
      <td>hamster</td>
      <td>6</td>
      <td>0</td>
      <td>218000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4364</td>
      <td>Snack</td>
      <td>$69,000</td>
      <td>241.27</td>
      <td>VC_929</td>
      <td>large</td>
      <td>dog</td>
      <td>1</td>
      <td>1</td>
      <td>69000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4184</td>
      <td>Supplements</td>
      <td>$138,000</td>
      <td>133.68</td>
      <td>VC_749</td>
      <td>large</td>
      <td>dog</td>
      <td>10</td>
      <td>0</td>
      <td>138000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# test the hypothesy that the mean amount of sales the mean is 100000
mean_sales= Data_sale['sale1'].mean()
mean_sales
```




    116094.42548350398



# test the hypothesy that the mean amount of sales the mean is 100000
mean_sales= Data_sale['sale1'].mean()
mean_sales



```python
# determine the error terms to be use in the t test. 
erro_dist= []
for i in range(5000):
    erro_dist.append(np.mean(
        Data_sale.sample(frac=1, replace= True)['sale1']
    )
                    )
```


```python
# visualise the distribution
plt.hist(erro_dist, bins=50)
plt.title('hist of error terms')
plt.show()
```


    
![png](output_12_0.png)
    


# visualise the distribution
plt.hist(erro_dist, bins=50)
plt.title('hist of error terms')
plt.show()


```python

```


```python
# get the standard deviation of the erro term
standart= np.std(erro_dist, ddof=1)
standart
```




    2261.6151608796827



get the standard deviation of the erro term


```python
#find the z-score
Z_score= (mean_sales-100000)/standart
Z_score
```




    7.116341348385661



find the z-score and interpretation of results


```python
from scipy.stats import norm
1-norm.cdf(Z_score, loc=0,scale=1)
```




    5.541123115904156e-13




```python

```

# are smaller pets having beteer sales that larher pets
## The anover test

m=Data_sale.groupby("pet_size")['sale1'].mean()
s=Data_sale.groupby('pet_size')['sale1'].std()


```python
# are smaller pets having beteer sales that larher pets

m=Data_sale.groupby("pet_size")['sale1'].mean()
```


```python
s=Data_sale.groupby('pet_size')['sale1'].std()
```


```python
s
```




    pet_size
    extra_large    69433.273184
    extra_small    61803.371059
    large          68805.940515
    medium         66727.761902
    small          65631.285450
    Name: sale1, dtype: float64



Anova test of difference ingroups


```python
# Anova test of difference ingroups
value_c= Data_sale['pet_type'].value_counts()
value_c

```




    pet_type
    cat        347
    dog        347
    fish        70
    bird        69
    hamster     23
    rabbit      23
    Name: count, dtype: int64




```python

```


```python
sns.boxplot(x='pet_type',
          y='sale1',
          data= Data_sale)
plt.show()
```


    
![png](output_29_0.png)
    


Anova test of difference ingroups


```python
Anova test of difference ingroups
import pingouin
pingouin.anova(data=Data_sale,
              dv='sale1',
              between='pet_size')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source</th>
      <th>ddof1</th>
      <th>ddof2</th>
      <th>F</th>
      <th>p-unc</th>
      <th>np2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pet_size</td>
      <td>4</td>
      <td>874</td>
      <td>0.373154</td>
      <td>0.827872</td>
      <td>0.001705</td>
    </tr>
  </tbody>
</table>
</div>



# using Kneighbors


```python
from sklearn.neighbors import KNeighborsClassifier

x= Data_sale[['sale1', 'rating']].values
y= Data_sale['re_buy'].values

print(x.shape, y.shape)
```

    (879, 2) (879,)
    


```python
knn= KNeighborsClassifier(n_neighbors=15)
knn.fit(x,y)
```




    KNeighborsClassifier(n_neighbors=15)




```python
# new data to predict
x_new=np.array([[50000, 5],[70000,9],[100000,8]])
print(x_new)
```

    [[ 50000      5]
     [ 70000      9]
     [100000      8]]
    


```python
# predict the  kk means
predictions= knn.predict(x_new)
print('Predictions: {}'. format(predictions))
```

    Predictions: [1 0 0]
    


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, 
                                                    random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)
print(knn.score(x_test, y_test))
```

    0.5643939393939394
    


```python

```


```python
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(x_train, y_train)
    train_accuracies[neighbor] = knn.score(x_train, y_train)
    test_accuracies[neighbor] = knn.score(x_test, y_test)

```


```python
plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```


    
![png](output_40_0.png)
    


plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

# applying regression on the data set


```python
# regression studies
reg= Data_sale
bx= reg.drop('re_buy', axis=1).values
by=reg['re_buy'].values

```


```python

```


```python
print(type(bx), type(by))
```

    <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    


```python
# predict a single feature of regrati
rating= bx[:,7]
rating= rating.reshape(-1,1)
plt.scatter(rating, by)
plt.show()
```


    
![png](output_46_0.png)
    



```python
from sklearn.linear_model import LinearRegression
regg= LinearRegression()
regg.fit(rating, by)
predictions=regg.predict(rating)
plt.scatter(rating,by)
plt.plot(rating, predictions)

```




    [<matplotlib.lines.Line2D at 0x1df161f1dc0>]




    
![png](output_47_1.png)
    



```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, 
                                                    random_state=21, stratify=y)
regg= LinearRegression()
regg.fit(x_train, y_train)
y_predict= regg.predict(x_test)
regg.score(x_test, y_test)

```




    0.005355711539129948


