#!/usr/bin/env python
# coding: utf-8

# # 通过矩阵乘法进行货币换算 
# 
# 在此 notebook 中，你将使用矩阵乘法和 python 软件包 [NumPy](http://www.numpy.org/) 解决一个货币换算问题。此演示旨在帮助你学习如何使用矩阵乘法解决更复杂的问题。
# 
# ## 货币换算问题 
# 
# 这么多年来，你一共去过八个不同的国家/地区，并且留下了一些当地的货币。你打算再回到这八个国家/地区中的某个国家/地区，但是不确定要回到哪个国家/地区。
# 你打算看看哪个航线的机票最便宜。
# 
# 为了做好行程准备，你需要将你的*所有*当地货币转换为你打算前往的国家/地区的货币。
# 因此，要仔细检查银行对你的货币换算结果，你想要计算八个国家/地区的每个国家/地区的总货币金额。
# 要计算换算结果，你首先需要导入包含每个国家/地区的货币换算率的矩阵。我们将使用的数据来自 [Overview Matrix of Exchange Rates from Bloomberg Cross-Rates _Overall Chart_](https://www.bloomberg.com/markets/currencies/cross-rates)（2018 年 1 月 10 日）。 
# 
# <img src="currencyProbImage.png" height=300 width=750>
# 
# 
# 比如money只有一行 它乘以 矩阵的第一列 即代表 把我剩下所有的货币都转换为USD 即money_total的第一个元素；
# 以此类推 矩阵的第二列 即代表 都转换为EUR。
# 
# 你可以将此问题看做接受一个_**输入**向量_（来自 8 个国家/地区的货币），并向这些输入应用 _**权重**矩阵_（换算率矩阵），以使用矩阵乘法和 NumPy 软件包生成 _**输出**向量_（每个国家/地区的货币总金额）。
# 
# ### 编写货币换算问题代码 
# 首先，你需要创建_**输入**向量_来存储表示八个国家/地区货币的 numpy 向量。首先导入 NumPy 软件包，然后使用该软件包根据列表创建向量。接着，将该向量转换为 pandas dataframe，以便在下面整洁地输出列标签，表示与货币金额相关的国家/地区。

# In[4]:


import numpy as np
import pandas as pd

# Creates numpy vector from a list to represent money (inputs) vector.
money = np.asarray([70, 100, 20, 80, 40, 70, 60, 100])

# Creates pandas dataframe with column labels(currency_label) from the numpy vector for printing.
currency_label = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "HKD"]
# 建立dataframe数据是money向量 索引为currency_label列表
money_df = pd.DataFrame(data=money, index=currency_label, columns=["Amounts"])
print("Inputs Vector:")
# 使用dataframe.T 转置
money_df.T


# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }

# 接着，我们需要通过导入货币换算率矩阵创建_**权重**矩阵_。我们将使用  python 软件包 [Pandas](https://pandas.pydata.org/) 快速读取该矩阵并相应地分配行和列标签。此外，我们定义一个变量 **_path_** 来定义货币换算矩阵的位置。下面的代码会导入该权重矩阵，将 DataFrame 转换为 numpy 矩阵，并显示其内容，以帮助你判断如何使用矩阵乘法解决该问题。

# In[7]:


# Sets path variable to the 'path' of the CSV file that contains the conversion rates(weights) matrix.
path = get_ipython().run_line_magic('pwd', '')

# Imports conversion rates(weights) matrix as a pandas dataframe.
conversion_rates_df = pd.read_csv(path+"/currencyConversionMatrix.csv",header=0,index_col=0)

# Creates numpy matrix from a pandas dataframe to create the conversion rates(weights) matrix.
conversion_rates = conversion_rates_df.values

# Prints conversion rates matrix.
print("Weights Matrix:")
conversion_rates_df


# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }

# 上述_**权重**矩阵_提供了每个国家/地区之间的换算率。例如，在第 1 行第 1 列，值 **1.0000** 表示从美元到美元的换算率。在第 2 行第 1 列，值 **1.1956** 表示 1 欧元等于 **1.1956** 美元。在第 1 行第 2 列，值 **0.8364** 表示 1 美元仅等于  **0.8364** 欧元。
# 
# 下面使用矩阵乘法计算了_**输出**向量_。numpy 软件包提供了将两个矩阵相乘（或向量与矩阵相乘）的[函数 _**matmul**_](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html)。你将在下面找到适用于 AI 的矩阵乘法方程，其中_**输入**向量_($x_{1}...x_{n}$) 乘以_**权重**矩阵_($w_{11}...w_{nm}$) 以计算_**输出**向量_($y_{1}...y_{m}$)。 
# 
# $\hspace{4cm} \begin{bmatrix} x_{1}&x_{2}&...&x_{n}\end{bmatrix} \begin{bmatrix} w_{11}&w_{12}&...&w_{1m}\\ w_{21}&w_{22}&...&w_{2m}\\ ...&...&...&... \\ w_{n1}&w_{n2}&...&w_{nm}\end{bmatrix} = \begin{bmatrix} y_{1}&y_{2}&...&y_{m}\end{bmatrix}$ 
# 
# 下面的示例矩阵乘法在**输入**和**权重**中将 $n$ 设为 4，在**权重**和**输出**中将 $m$ 设为 3。
# 
# $\hspace{4cm} \begin{bmatrix} 10 & 2 & 1 & 5\end{bmatrix} \begin{bmatrix} 1 & 20 & 7\\ 3 & 15 & 6 \\ 2 & 5 & 12 \\ 4 & 25 & 9 \end{bmatrix} = \begin{bmatrix} 38 & 360 & 139 \end{bmatrix}$ 
# 
# 从上述示例可以看出，矩阵乘法生成的矩阵（_**输出**向量_）的行维度与第一个矩阵（_**输入**向量_）的一样，列维度将与第二个矩阵（_**权重**矩阵_）的一样。对于货币示例来说，输入矩阵和权重矩阵的列数一样，但是 AI 并非始终都如此。
# 
# ## TODO：矩阵乘法
# 在下面的空白处使用[函数 _**matmul**_](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html) 将 **money** 和 **conversion_rates** 相乘，以计算向量 **money_totals**。之前我们在导入上述 Numpy 软件包时，使用了别名 _**np**_，因此在下面调用 _**matmul**_ 函数时，确保使用别名 _**np**_。此外，确保选择_“单元格”_ 和_“运行所有”_以检查你在下面插入的代码。

# In[8]:


# DONE 1.: Calculates the money totals(outputs) vector using matrix multiplication in numpy.
money_totals = np.matmul(money, conversion_rates)


# Converts the resulting money totals vector into a dataframe for printing.
money_totals_df = pd.DataFrame(data = money_totals, index = currency_label, columns = ["Money Totals"])
print("Outputs Vector:")
money_totals_df.T


# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }

# ### 通过矩阵乘法进行货币换算的解决方案 
# 上面的输出应该与下面的 **Money Totals** 相符。结果可以解析为：将所有货币换算为美元 (**USD**) 等于 **454.28** 美元，将所有货币换算为欧元 (**EUR**) 等于 **379.96** 欧元，等等。
# 
# <img src="money_totals.png" height=225 width=563>
# 
# ### 通过矩阵乘法进行货币换算的解决方案视频   
# 你可以在**线性映射 Lab 解决方案**部分找到解决方案视频。你可能需要重新打开一个浏览器窗口，以便轻松地在向量 Lab Jupyter Notebook 和此 Lab 的解决方案视频之间轻松切换。
