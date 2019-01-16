
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[1]:


# 各库导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# 数据读取
movie_data = pd.read_csv('./tmdb-movies.csv')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[2]:


print('1.电影数据集的行列数为：',movie_data.shape)


# In[3]:


movie_data.head() # 2.1 电影数据集的前五行


# In[4]:


movie_data.tail() # 2-2 电影数据集的末五行


# In[5]:


movie_data.sample() # 2-3 随机抽取一个电影数据样本


# In[6]:


movie_data.dtypes # 3 获取每列的数据类型


# In[7]:


movie_data.isnull().any() # 4 检查各列是否有NaN值


# In[8]:


movie_data['id'].describe() # 5-1 编号id列（int64）的描述性统计信息


# In[9]:


movie_data['popularity'].describe() # 5-2 知名度popularity列（float64)


# In[10]:


movie_data['budget'].describe() # 5-3 预算budget列 (int64)


# In[11]:


movie_data['revenue'].describe() # 5-4 票房revenue列 (int64)


# In[12]:


movie_data['runtime'].describe() # 5-5 时长runtime列（int64）


# In[13]:


movie_data['vote_count'].describe() # 5-5 投票总数vote_count列（int64）


# In[14]:


movie_data['vote_average'].describe() # 5-6 投票均值vote_average列（float64）


# In[15]:


movie_data['release_year'].describe() # 5-7 发行年份release_year列（int64）


# In[16]:


movie_data['budget_adj'].describe() # 5-8 预算（调整后）budget_adj列（float64)


# In[17]:


movie_data['revenue_adj'].describe() # 5-9 票房（调整后）revenue_adj列（float64）


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[18]:


# 这里采用将NaN值都替换为0 并保存至movie_data_adj中
print("处理前NaN值有：", movie_data.isnull().sum().sum(),"个")
movie_data_adj = movie_data.fillna(0)
print("处理前NaN值有：", movie_data_adj.isnull().sum().sum(),"个")


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[19]:


# 注：参考了笔记和https://blog.csdn.net/u011089523/article/details/60341016

# 2.1.1.读取某列数据 
# 各列分别读取 df[['列名']]来访问
movie_data_id = movie_data[['id']]
movie_data_pop = movie_data[['popularity']]
movie_data_bud = movie_data[['budget']]
movie_data_rt = movie_data[['runtime']]
movie_data_vote_avg = movie_data[['vote_average']]
# 各列一起读取 df[['列名1','列名2'...列名的列表]]来访问
movie_data_sel = movie_data[['id', 'popularity', 'budget', 'runtime', 
                             'vote_average']]

# 2.1.2 读取x行数据
# 读取前20行的两种方法 df.head(n) 或 df[m:n]
movie_data_rows_1to20_1 = movie_data.head(20)
movie_data_rows_1to20_2 = movie_data[0:20]
# 读取48，49行数据 注意索引从0开始 前闭后开
movie_data_rows_48to49 = movie_data[47:49]


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[20]:


# 参考了https://blog.csdn.net/GeekLeee/article/details/75268762
# 1.读取popularity>5的所有数据
movie_data_pop_morethan5 = movie_data.loc[movie_data['popularity']>5]

# 2.读取popularity>5 且 发行年份>1996的所有数据
movie_data_pop5p_rls1996p = movie_data.loc[(movie_data['popularity']>5)&(movie_data['release_year']>1996) ]


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[21]:


data = movie_data
# 按release_year分组 获取revenue均值
revenue_mean_groupby_rlsyear = data.groupby(['release_year'])['revenue'].agg('mean')

# 按director分组 获取popularity均值
popularity_mean_groupby_director = data.groupby(['director'])['popularity'].agg('mean')


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[44]:


base_color = sb.color_palette()[0] # 取第一个颜色
y_count = movie_data_adj['popularity'][:20]
""" 
这块有些搞不懂如何去绘制？ 应该用条形图合适还是用直方图合适 感觉二者都不合适 
饼图不适合20个扇形 直方图和条形图似乎x和y有些问题 这里用条形图勉强绘制出 感觉不合适
另有一个问题即如何在sb.barplot中标注出某个条形图具体的数值 在countplot中可以有办法标注出频率 
我猜测应该可以在barplot标注出数值，可是并没有相关资料或者示例....
有些疑惑，请求解答，谢谢！！！
"""
# 绘图
sb.barplot(x = y_count.index.values+1, y = y_count, color = base_color, orient = "v")

"""
可以从图表中得知：
热度第1（数值达32.98）和热度第2（数值达28.41）的电影其流行程度远超第3以及之后所有的电影，差距达到了一倍以上。
第3到第20的电影其热度相差不大 数值均在5-15范围之内 较为稳定
""";


# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[23]:


# 需要考虑净利润随时间变化的情况 所以选择 折线图 适宜
# 调整分箱边缘和中心点
xbin_edges = np.arange(1960, movie_data_adj['release_year'].max()+2,2)
xbin_centers = (xbin_edges + 0.25/2)[:-1]
# 计算每个分箱中的统计数值
data_xbins = pd.cut(movie_data_adj['release_year'], xbin_edges, right = False, include_lowest = True)
y_means = movie_data_adj['revenue_adj'].groupby(data_xbins).mean()-movie_data_adj['budget_adj'].groupby(data_xbins).mean()
y_sems = movie_data_adj['revenue_adj'].groupby(data_xbins).sem()-movie_data_adj['budget_adj'].groupby(data_xbins).sem()
# 绘图
plt.errorbar(x = xbin_centers, y = y_means, yerr = y_sems)
plt.xlabel('release year');
plt.ylabel('Net profit');

"""
可以从图中看出：
随着年份的变化（这里选取的是电影的发行年份作参考）
净利润本在1960-1970年段先下降后上升再下降，较不稳定；
而后在1970-1980年段达到了一个净利润的峰值，可见当时的电影市场火爆；
而后在1980之后，净利润整体呈逐年下降的趋势，趋于稳定，市场也逐渐成熟。
净利润的波动（即误差线）再1960-1980年间较大，考虑到电影市场刚刚兴起，符合实际；
在后来进入市场成熟期之后，1980年之后，波动较小，更加稳定。
PS：不太清楚如何写分析，应该从哪些角度入手，哪些东西该讲，哪些不用讲....
""";


# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
