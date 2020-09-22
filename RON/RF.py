import xlrd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xlsxwriter
resArray=[] #先声明一个空list
avg_arr = []
data = xlrd.open_workbook("MIC_sort_change.xlsx") #读取文件
table = data.sheet_by_index(0) #按索引获取工作表，0就是工作表1
for i in range(table.nrows): #table.nrows表示总行数
    line=table.row_values(i) #读取每行数据，保存在line里面，line是list
    resArray.append(line) #将line加入到resArray中，resArray是二维list
resArray=np.array(resArray) #将resArray从二维list变成数组
print(resArray)
Y = resArray[:, 0]
X = resArray[:, 1:]
print(type(X), X.shape)
names = np.arange(380)

rf = RandomForestRegressor()
rf.fit(X, Y)
print("Features sorted by their score:")
xx = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)

workbook = xlsxwriter.Workbook('rf_Worksheet11.xlsx')
# # 创建一个worksheet
worksheet = workbook.add_worksheet()
#
# # 写入excel
# # 参数对应 行, 列, 值
for i, (value, ind) in enumerate(xx):
    worksheet.write(0, ind, value)
    worksheet.write(1, ind, i+1)

# 保存
workbook.close()


