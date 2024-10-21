import pandas as pd

# 读取表格文件
file_path = '/mnt/data/任务安排表.csv'  # 假设表头正确
df = pd.read_csv(file_path, encoding='ISO-8859-1')


# 清理数据，提取表名、字段名和数据类型
# 这里假设列的结构为: [序号, 表名, 字段名, 数据类型]

def generate_sql_from_df(df):
    tables = {}

    for index, row in df.iterrows():
        table_name = row['表名']
        column_name = row['字段名']
        data_type = row['数据类型']

        if pd.notna(table_name) and pd.notna(column_name) and pd.notna(data_type):
            if table_name not in tables:
                tables[table_name] = []
            tables[table_name].append(f"{column_name} {data_type}")

    return tables


# 生成 SQL 语句
tables = generate_sql_from_df(df)

for table, columns in tables.items():
    sql = f"CREATE TABLE {table} (\n    " + ",\n    ".join(columns) + "\n);"
    print(sql)