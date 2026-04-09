import xml.etree.ElementTree as ET
import csv
import os

# NOTE: все столбцы с идентичными данными закоментированны
def parse_xml(xml_content, filename):
    root = ET.fromstring(xml_content)
    
    data = {}
    
    # Info
    # info = root.find('Info')
    # if info is not None:
    #     data['device_type'] = info.get('device_type', '')
    #     data['report_date'] = info.get('report_date', '')
    
    # ReportTitle
    report_title = root.find('ReportTitle')
    if report_title is not None:
        data['report_title'] = report_title.get('name', '')

    # Patient
    patient = root.find('Patient')
    if patient is not None:
        data['patient_id'] = patient.get('id', '')
        data['age'] = patient.get('age', '')
        #data['ethnicity'] = patient.get('ethnicity', '')
    
    # Dataset и Scan
    dataset = root.find('Dataset')
    if dataset is not None:
        data['dataset_id'] = dataset.get('id', '')
        scan = dataset.find('Scan')
        if scan is not None:
            data['eye'] = scan.get('eye', '')
            #data['scan_date'] = scan.get('date', '')
            #data['iqThreshold'] = scan.get('iqThreshold', '')
            #data['iqThreshold2'] = scan.get('iqThreshold2', '')
            data['fastq'] = scan.get('fastq', '')
            #data['eye_type'] = scan.get('type', '')
            #data['skipmode'] = scan.get('skipmode', '')
            data['fixation'] = scan.get('fixation', '')
            #data['analysisVersion'] = scan.get('analysisVersion', '')
            #data['Refmirror_pos'] = scan.get('Refmirror_pos', '')
            #data['EyeMagniValue'] = scan.get('EyeMagniValue', '')
            
            # Size
            # size = scan.find('Size')
            # if size is not None:
            #     data['size_x'] = size.get('x', '')
            #     data['size_y'] = size.get('y', '')
            
            # Length
            # length = scan.find('Length')
            # if length is not None:
            #     data['length_x'] = length.get('x', '')
            #     data['length_y'] = length.get('y', '')
    
    # Analysis
    analysis = dataset.find('Analysis') if dataset is not None else None
    if analysis is not None:
        #data['analysis_name'] = analysis.get('name', '')
        data['analysis_module'] = analysis.get('module', '')
        #data['analysis_desc'] = analysis.get('desc', '')
        #data['analysis_version'] = analysis.get('version', '')
        
        # Attr элементы
        for attr in analysis.findall('Attr'):
            attr_id = attr.get('id', '')
            attr_value = attr.get('value', '')
            data[f'{attr_id}'] = attr_value
        
        # ETDRS
        etdrs = analysis.find('ETDRS')
        if etdrs is not None:
            data['etdrs_center'] = etdrs.get('Center', '')
            data['etdrs_inT'] = etdrs.get('InT', '')
            data['etdrs_inS'] = etdrs.get('InS', '')
            data['etdrs_inN'] = etdrs.get('InN', '')
            data['etdrs_inI'] = etdrs.get('InI', '')
            data['etdrs_outT'] = etdrs.get('OutT', '')
            data['etdrs_outS'] = etdrs.get('OutS', '')
            data['etdrs_outI'] = etdrs.get('OutI', '')
            data['etdrs_outN'] = etdrs.get('OutN', '')

 
    disc_topo = dataset.find('DiscTopo') if dataset is not None else None   
    if disc_topo is not None:
        for attr in disc_topo.findall('Attr'):
            attr_id = attr.get('id', '')
            attr_value = attr.get('value', '')
            data[f'{attr_id}'] = attr_value

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

# Запуск парсинга
if __name__ == "__main__":
    directory_in_str = "xml_files"
    directory = os.fsencode(directory_in_str)
    all_data = []
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(f"{directory_in_str}/{filename}", 'r', encoding='utf-8') as f:
            xml_data = f.read()
            result = parse_xml(xml_data, filename)
            all_data.append(result)
    
    output_csv = "parsing/all_data.csv"
    save_to_csv(all_data, output_csv)