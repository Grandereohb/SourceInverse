import requests
from bs4 import BeautifulSoup
import pandas as pd

# 目标网址
url = 'https://q-weather.info/weather/58352/history/?date=2023-11-18'

# 发送GET请求获取网页内容
response = requests.get(url)
response.raise_for_status()  # 确保请求成功

# 解析网页内容
soup = BeautifulSoup(response.content, 'html.parser')

# 提取表格中的数据
rows = soup.find_all('tr', align='center')
times = []
wind_directions = []
wind_speeds = []

# 遍历每一行，提取时次、风向和风速数据
for row in rows:
    columns = row.find_all('td')
    if len(columns) >= 6:  # 确保列数足够，避免异常
        time_str = columns[0].get_text(strip=True)
        # 截取时间部分，仅保留"2023-08-05 01:00"这种格式
        time_str = time_str.split('+')[0].strip() if '+' in time_str else time_str
        times.append(time_str)  # 添加处理后的时次
        wind_directions.append(columns[4].get_text(strip=True))  # 第5列为瞬时风向
        wind_speeds.append(columns[5].get_text(strip=True))  # 第6列为瞬时风速

# 将数据放入DataFrame
df = pd.DataFrame({
    '时间': times,
    'dir': wind_directions,
    'sp': wind_speeds
})

# 指定Excel文件路径
excel_file_path = r"data_sumitomo\厂界\test8\wind.xlsx"

# 写入Excel文件
try:
    # 如果文件存在，读取原有数据
    existing_df = pd.read_excel(excel_file_path)

    # 追加新数据到现有数据框，不覆盖原有数据
    combined_df = pd.concat([existing_df, df], ignore_index=True)

    # 保存结果，不写入新的列名，只追加数据
    combined_df.to_excel(excel_file_path, index=False)
    print(f"数据已成功追加到 {excel_file_path}")
except FileNotFoundError:
    # 如果文件不存在，创建新的文件
    df.to_excel(excel_file_path, index=False)
    print(f"文件不存在，已创建并写入 {excel_file_path}")

