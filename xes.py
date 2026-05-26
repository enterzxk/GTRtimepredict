import xml.etree.ElementTree as ET
import pandas as pd
import os
import logging
import gzip

# 配置基础日志输出 (遵守最小化输出原则)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _strip_namespace(tag: str) -> str:
    """
    辅助函数：移除 XML 标签中的命名空间 (Namespace)
    XES 文件通常带有复杂的 xmlns 属性，直接匹配 tag 容易失效，通过此步骤进行稳健处理。
    """
    return tag.split('}')[-1] if '}' in tag else tag


def _parse_attributes(element: ET.Element) -> dict:
    """
    辅助函数：解析 XES 节点（如 trace 或 event）下属的数据属性标签。
    将 <string key="..." value="..."/> 转化为 Python 字典。
    """
    attrs = {}
    valid_value_tags = {'string', 'date', 'float', 'int', 'boolean', 'id'}

    for child in element:
        tag_name = _strip_namespace(child.tag)
        if tag_name in valid_value_tags:
            key = child.attrib.get('key')
            val = child.attrib.get('value')

            if key is not None and val is not None:
                # 基础类型转换，保证数值型特征后续不会在网络中引发类型错误
                if tag_name == 'float':
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                elif tag_name == 'int':
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                # 注：date 类型统一由外部 pandas 向量化处理更高效
                attrs[key] = val
    return attrs


def xes_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    核心解析逻辑：将嵌套的 XES XML 解析为平铺的 Pandas DataFrame。
    支持自动识别和读取 .gz 压缩格式。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    logging.info(f"开始解析 XES 文件: {file_path}")
    records = []

    try:
        # 兼容 .gz 压缩文件读取
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(file_path)

        root = tree.getroot()

        # 1. 遍历所有案例 (Trace)
        for trace in root:
            if _strip_namespace(trace.tag) == 'trace':
                trace_attrs = _parse_attributes(trace)

                # 提取案例级 ID (XES 标准中通常为概念名称 concept:name)
                # 若缺失则提供默认值以保证鲁棒性
                case_id = trace_attrs.get('concept:name', 'UNKNOWN_CASE')

                # 2. 遍历案例内的所有事件 (Event)
                for event in trace:
                    if _strip_namespace(event.tag) == 'event':
                        event_attrs = _parse_attributes(event)

                        # 合并案例级特征与事件级特征
                        # 事件级属性优先 (如果存在重名 key，event_attrs 会覆盖 trace_attrs)
                        record = {**trace_attrs, **event_attrs}
                        record['CaseID'] = case_id

                        records.append(record)

    except ET.ParseError as e:
        raise ValueError(f"XML 解析失败，文件可能已损坏或格式不合法: {e}")
    except OSError as e:
        raise ValueError(f"文件读取失败 (可能是损坏的 gz 文件): {e}")

    df = pd.DataFrame(records)
    logging.info(f"解析完成，成功提取 {len(df)} 条事件记录。")
    return df


def standardize_log_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗与标准化：使其适配此前定义的 dataprocessing.py 的输入格式要求。
    """
    if df.empty:
        return df

    # 定义列映射，将 XES 标准列名转换为此前模型规定的列名
    column_mapping = {
        'concept:name': 'Activity',
        'time:timestamp': 'Timestamp',
        'org:resource': 'Resource',
        'org:role': 'Role'
    }

    # 仅重命名数据集中实际存在的列，防止 KeyError
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # 时序特征处理
    if 'Timestamp' in df.columns:
        # XES 时间通常包含时区 (e.g., +02:00)，设置 utc=True 保证跨时区时间的绝对顺序正确
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
        # 移除时间戳解析失败的脏数据
        df = df.dropna(subset=['Timestamp'])

    # 填充空值
    if 'Resource' in df.columns:
        df['Resource'] = df['Resource'].fillna('UNKNOWN_RESOURCE')
    if 'Activity' in df.columns:
        df['Activity'] = df['Activity'].fillna('UNKNOWN_ACTIVITY')

    # 按 CaseID 和时间戳进行严格正序排序 (关键逻辑，防止未来模型穿越)
    if 'CaseID' in df.columns and 'Timestamp' in df.columns:
        df = df.sort_values(by=['CaseID', 'Timestamp']).reset_index(drop=True)

    return df


if __name__ == '__main__':
    # ==========================================
    # 执行示例
    # ==========================================
    # 更新了输入文件名，指向 .xes.gz
    input_file = "PrepaidTravelCost.xes.gz"

    try:
        # 1. 解析 XML 为 DataFrame
        raw_df = xes_to_dataframe(input_file)

        # 2. 清洗并标准化
        cleaned_df = standardize_log_dataframe(raw_df)

        # 3. 输出并保存为 CSV
        output_file = "data/BPI_Challenge_2020_ Domestic Declarations_1_all.csv"
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"标准化数据已保存至: {output_file}")

    except Exception as err:
        logging.error(f"处理失败: {err}")