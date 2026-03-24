import pandas as pd
# 读取Excel文件
excel_file_path = r"data_sumitomo\厂界\test8\wind.xlsx"
df = pd.read_excel(excel_file_path)

# 提取dir列中的数值部分
df['dir'] = df['dir'].str.extract(r'(\d+)').astype(float)

# 保存修改后的结果
df.to_excel(excel_file_path, index=False)