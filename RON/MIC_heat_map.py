import xlrd #读取excel的库
from scipy import stats

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimSun']
from matplotlib.font_manager import FontProperties
color = sns.color_palette()
sns.set_style('darkgrid')
resArray=[] #先声明一个空list
avg_arr = []
data = xlrd.open_workbook("train.xlsx") #读取文件
table = data.sheet_by_index(0) #按索引获取工作表，0就是工作表1
for i in range(table.nrows): #table.nrows表示总行数
    line=table.row_values(i) #读取每行数据，保存在line里面，line是list
    resArray.append(line) #将line加入到resArray中，resArray是二维list
resArray=np.array(resArray) #将resArray从二维list变成数组
print(resArray)
X = resArray.T

x = X[0]
# y = X[-1]
mic_value = [[0] for _ in range(23)]

for ind, value1 in tqdm(enumerate(X[1:])):
    for value2 in X[1:]:
        mic_value[ind].append(np.abs(stats.spearmanr(value1, value2)[0]))



# 初始化参数
sns.set()
arr = []
uniform_data = mic_value
for data in uniform_data:
    arr.append(data[1:])
all_arr = np.array(arr)
heatmap = sns.heatmap(all_arr)
plt.show()




# torch.save(mic_value, './heat_map')
# mic_value_sort = sorted(mic_value, reverse=True)
#
# X = []
# y = []
# for mic, ind in mic_value_sort:
#     if ind == 341:
#         continue
#     else:
#         y.append(ind)
#         X.append(mic)
# fig, ax = plt.subplots()
# ax.scatter(x=X, y=y)# 绘制散点图
#
# myfont = FontProperties(fname='simhei.ttf')
# plt.ylabel(u'操作变量编号', fontsize=13,fontproperties=myfont)
# plt.xlabel('MIC', fontsize=13, fontproperties=myfont)
# plt.vlines(0.3, 0, 350, colors = "r", linestyles = "dashed")
# plt.show()
#

