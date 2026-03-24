import pandas as pd

# 读取数据
file_path = r"data_sumitomo\厂界\test8\8.xlsx"
df = pd.read_excel(file_path)

# 将时间列转换为datetime格式
df['时间'] = pd.to_datetime(df['时间'])

# 设置时间列为索引
df.set_index('时间', inplace=True)

# 按小时聚合数据，并计算每小时的平均
df_hourly = df.resample('H').mean()

# 重置索引
df_hourly.reset_index(inplace=True)

# 保存结果到新文件
output_path = r'data_sumitomo\厂界\test8\hourly_averaged_data.xlsx'
df_hourly.to_excel(output_path, index=False)

print(f"数据已成功转换并保存到 {output_path}")
