import pandas as pd
import os

# 预设的映射字典配置
PRESET_MAPPINGS = {
    'bpic': {'case': 'CaseID', 'activityNameEN': 'Activity', 'completeTime': 'Timestamp', 'resource': 'Resource'},
    'sepsis': {'case': 'CaseID', 'event': 'Activity', 'completeTime': 'Timestamp', 'org:group': 'Resource'},
    'helpdesk': {'Case ID': 'CaseID', 'Activity': 'Activity', 'Complete Timestamp': 'Timestamp',
                 'Resource': 'Resource'},
    'wind': {'工作业务主键': 'CaseID', '活动名称': 'Activity', '工作活动完成时间': 'Timestamp', '签字人': 'Resource'},
    'xes': {'CaseID': 'CaseID', 'Activity': 'Activity', 'Timestamp': 'Timestamp', 'Resource': 'Resource'},
    # 【新增】针对上传的 BPIC2015 系列数据集
    # (2015数据使用 activityNameEN 作为标准英文活动名)
    'bpic2015': {
        'case:concept:name': 'CaseID',
        'activityNameEN': 'Activity',
        'time:timestamp': 'Timestamp',
        'org:resource': 'Resource'
    },

    # 【新增】针对上传的 BPIC2020 系列数据集及大部分标准 XES 导出的 CSV
    'bpic2020': {
        'case:concept:name': 'CaseID',
        'concept:name': 'Activity',
        'time:timestamp': 'Timestamp',
        'org:resource': 'Resource'
    },
    'xes_standard': {
        'case:concept:name': 'CaseID',
        'concept:name': 'Activity',
        'time:timestamp': 'Timestamp',
        'org:resource': 'Resource'
    }
}


def process_and_save_dataset(file_path, mapping_dict, dataset_name, output_dir, max_case_length=None, skiprows=0):
    """
    处理原始日志数据集，提取特征并保存。
    包含处理极端长工单的逻辑。
    """
    print(f"[{dataset_name}] 正在处理文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"[错误] 文件未找到: {file_path}")
        return None

    try:
        # 1. 根据文件类型读取数据
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, skiprows=skiprows, engine='openpyxl')
        else:
            df = pd.read_csv(file_path, skiprows=skiprows)

        # 处理可能存在的重名列
        df = df.loc[:, ~df.columns.duplicated()]

        # 2. 列名清理与重命名
        df.columns = [str(c).strip() for c in df.columns]
        valid_cols = {k: v for k, v in mapping_dict.items() if k in df.columns}
        df = df[list(valid_cols.keys())].copy()
        df.rename(columns=valid_cols, inplace=True)
        df['Dataset_Name'] = dataset_name

        # 3. 时间戳格式化与基础清理
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['CaseID', 'Activity', 'Timestamp'], inplace=True)

        if 'Resource' in df.columns:
            df['Resource'] = df['Resource'].fillna('UNKNOWN').astype(str)

        # 4. 极端长工单处理 (异常值过滤)
        if max_case_length is not None and max_case_length > 0:
            case_lengths = df['CaseID'].value_counts()
            valid_cases = case_lengths[case_lengths <= max_case_length].index
            dropped_cases_count = len(case_lengths) - len(valid_cases)
            if dropped_cases_count > 0:
                print(f" -> [过滤] 发现 {dropped_cases_count} 个超长异常工单(>{max_case_length}个事件)，已剔除。")
            df = df[df['CaseID'].isin(valid_cases)].copy()

        # 5. 严格按工单与时间正序排列
        df.sort_values(by=['CaseID', 'Timestamp'], inplace=True)

        # 6. 构建模型输入特征
        df['Prev_Timestamp'] = df.groupby('CaseID')['Timestamp'].shift(1)
        df['Start_Timestamp'] = df.groupby('CaseID')['Timestamp'].transform('min')
        df['End_Timestamp'] = df.groupby('CaseID')['Timestamp'].transform('max')

        df['TimeSinceLast'] = (df['Timestamp'] - df['Prev_Timestamp']).dt.total_seconds() / 3600.0
        df['TimeSinceLast'] = df['TimeSinceLast'].fillna(0).round(4)

        df['TimeSinceStart'] = (df['Timestamp'] - df['Start_Timestamp']).dt.total_seconds() / 3600.0
        df['TimeSinceStart'] = df['TimeSinceStart'].round(4)

        # 7. 构建多任务预测标签
        df['Next_Activity'] = df.groupby('CaseID')['Activity'].shift(-1).fillna('[END]')
        df['Next_Timestamp'] = df.groupby('CaseID')['Timestamp'].shift(-1)

        df['Next_Event_Time'] = (df['Next_Timestamp'] - df['Timestamp']).dt.total_seconds() / 3600.0
        df['Next_Event_Time'] = df['Next_Event_Time'].fillna(0).round(4)

        df['Remaining_Time'] = (df['End_Timestamp'] - df['Timestamp']).dt.total_seconds() / 3600.0
        df['Remaining_Time'] = df['Remaining_Time'].round(4)

        # 8. 清理并保存
        df.drop(columns=['Prev_Timestamp', 'Start_Timestamp', 'End_Timestamp', 'Next_Timestamp'], inplace=True)

        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"processed_{dataset_name}.csv")
        df.to_csv(output_filename, index=False)

        print(f" -> 成功: 已保存至 {output_filename} (工单数: {df['CaseID'].nunique()}, 事件数: {len(df)})")
        return output_filename

    except Exception as e:
        print(f"[错误] 处理数据集 {dataset_name} 时发生异常: {e}")
        return None


if __name__ == '__main__':
    # ==========================================
    # 超参数与执行配置区域 (可直接在 PyCharm 中修改)
    # ==========================================

    # 统一的输出目录
    OUTPUT_DIRECTORY = './dataset'

    # 单个工单允许的最大事件数，超出的工单将被过滤(防噪)，设为 None 则不限制
    MAX_CASE_LENGTH = 500

    # 配置文件列表格式: (文件路径, 列映射字典, 导出的数据集名称, 跳过行数)
    files_to_process = [
        # --- BPIC 2012 ---
        ('data/BPIC_2012.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2012', 0),

        # --- BPIC 2015 ---
        ('data/BPIC2015_1.csv', PRESET_MAPPINGS['bpic2015'], 'BPIC2015_1', 0),
        ('data/BPIC2015_2.csv', PRESET_MAPPINGS['bpic2015'], 'BPIC2015_2', 0),
        ('data/BPIC2015_3.csv', PRESET_MAPPINGS['bpic2015'], 'BPIC2015_3', 0),
        ('data/BPIC2015_4.csv', PRESET_MAPPINGS['bpic2015'], 'BPIC2015_4', 0),
        ('data/BPIC2015_5.csv', PRESET_MAPPINGS['bpic2015'], 'BPIC2015_5', 0),

        # --- BPIC 2017 ---
        ('data/BPIC_2017.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2017', 0),

        # --- BPIC 2018 ---
        ('data/BPIC_2018.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2018', 0),

        # --- BPIC 2019 ---
        ('data/BPIC_2019.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2019', 0),

        # --- BPIC 2020 ---
        ('data/BPIC2020_Dom.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2020_Dom', 0),
        ('data/BPIC2020_Inter.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2020_Inter', 0),
        ('data/BPIC2020_Per.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2020_Per', 0),
        ('data/BPIC2020_Pre.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2020_Pre', 0),
        ('data/BPIC2020_Req.csv', PRESET_MAPPINGS['xes_standard'], 'BPIC2020_Req', 0),
    ]

    for file_path, mapping, name, skip in files_to_process:
        process_and_save_dataset(
            file_path=file_path,
            mapping_dict=mapping,
            dataset_name=name,
            output_dir=OUTPUT_DIRECTORY,
            max_case_length=MAX_CASE_LENGTH,
            skiprows=skip
        )