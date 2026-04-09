import xml.etree.ElementTree as ET
import csv
import os

def parse_xml(xml_content, filename):
    root = ET.fromstring(xml_content)
    data = {}
    
    # ReportTitle
    report_info = root.find('ReportInfo')
    if report_info is not None:
        for attr_name, attr_value in report_info.attrib.items():
            data[attr_name] = attr_value

    data['filename'] = filename

    return data

def save_to_csv(all_data, output_csv):
    """
    Сохраняет все данные в CSV файл
    """
    if not all_data:
        print("Нет данных для сохранения")
        return False
    
    # Получаем все возможные ключи из всех словарей
    all_keys = set()
    for data in all_data:
        all_keys.update(data.keys())
    
    # Сортируем ключи
    all_keys = sorted(list(all_keys))
    
    # Сохраняем в CSV
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_data)
    
    return True

# Парсинг
if __name__ == "__main__":
    directory_in_str = "xml_info_files"
    directory = os.fsencode(directory_in_str)
    all_data = []
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(f"{directory_in_str}/{filename}", 'r', encoding='utf-8') as f:
            xml_data = f.read()
            result = parse_xml(xml_data, filename)
            all_data.append(result)
    
    output_csv = "parsing/all_data_info.csv"
    save_to_csv(all_data, output_csv)