import pandas as pd

# 1. 读取要插入的数据列（从源 Excel 文件中提取连续的行）
source_file_path ="data_sumitomo\厂界\东\历史数据记录2023-11-01 00-16-37--2023-11-30 23-16-37.xlsx"

# 假设要选择从第10行到第20行，选择的列为 '要插入的列名'
start_row = 24480
end_row = 27360
source_df = pd.read_excel(source_file_path)

# 提取第10到第20行的数据
selected_data = source_df.loc[start_row:end_row, '总烃 mg/m^3']

# 2. 读取目标 Excel 文件（将提取的数据插入到目标文件的第三列）
target_file_path = r'data_sumitomo\厂界\test8\8.xlsx'
target_df = pd.read_excel(target_file_path)

# 插入提取的数据到第三列
target_df.insert(loc=2, column='E', value=selected_data.reset_index(drop=True))

# 3. 保存更新后的 Excel 文件
output_file_path = r'data_sumitomo\厂界\test8\8.xlsx'
target_df.to_excel(output_file_path, index=False)

print(f"更新的数据已保存到 {output_file_path}")
