import pandas as pd

# 读取原始的xlsx文件
file_path = '1.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 指定要提取的行范围和列名，例如从索引22511到索引25390的所有行，以及列名为'A'和'B'
rows = range(22511, 25391)  # 这是基于索引的行选择，包括索引22511到25390
columns = ['时间', '总烃 mg/m^3']  # 替换为你需要的列名

# 提取指定的行和列
new_df = df.iloc[rows][columns]

# 保存到新的xlsx文件
output_file_path = '2.xlsx'  # 替换为你希望保存的文件路径
new_df.to_excel(output_file_path, index=False)

print(f"指定的行和列已保存到 {output_file_path}")
