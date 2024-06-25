import os
import re
import json
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pysbd

def clean_text(content):

    content = content.replace('\n', ' ')
    content = re.sub(r'\s{2,}', ' ', content)

    # split the content to extract the part after "FINDINGS:"
    findings_split = re.split(r'(?i)FINDING[S]?:', content)
    if len(findings_split) > 1:
        findings_part = findings_split[1]
        findings_text = re.split(r'(?i)IMPRESSION[S]?:', findings_part)[0].strip()
    else:
        findings_text = ""

    # split the content to extract the part after "IMPRESSION:"
    impression_split = re.split(r'(?i)IMPRESSION[S]?:', content)
    impression_text = impression_split[1].strip() if len(impression_split) > 1 else ''

    # split text into sentences
    seg = pysbd.Segmenter(language="en", clean=False)
    findings_sentences = seg.segment(findings_text)
    impression_sentences = seg.segment(impression_text)

    findings_sentences = [sentence for sentence in findings_sentences if bool(re.search(r'\w', sentence))]
    impression_sentences = [sentence for sentence in impression_sentences if bool(re.search(r'\w', sentence))]

    def remove_leading_numbers(sentence):
        return re.sub(r'^\d+[\.\)]\s*', '', sentence)

    findings_sentences = [remove_leading_numbers(sentence) for sentence in findings_sentences]
    impression_sentences = [remove_leading_numbers(sentence) for sentence in impression_sentences]

    # Prepend sentence numbers
    numbered_sentences = []
    sentence_number = 1

    if findings_text:
        numbered_sentences.append("FINDINGS:\n")
        for sentence in findings_sentences:
            if sentence:
                numbered_sentences.append(f"{sentence_number}. {sentence}\n")
                sentence_number += 1

    if impression_text:
        if findings_text:
            numbered_sentences.append("\n")
        numbered_sentences.append("IMPRESSION:\n")
        for sentence in impression_sentences:
            if sentence:
                numbered_sentences.append(f"{sentence_number}. {sentence}\n")
                sentence_number += 1

    # combined numbered sentences with newline characters
    numbered_text = ''.join(numbered_sentences)

    return numbered_text, len(findings_sentences) + len(impression_sentences)

def main():

    data_path = '../data/'
    split = pd.read_csv(os.path.join(data_path, 'mimic-cxr-2.0.0-split.csv'))

    # dedup by subject_id and study_id
    split = split.drop_duplicates(subset=['study_id', 'subject_id'])

    # ensure same subject_id does not show up in both train and test
    train = split[split['split'] == 'train']
    test = split[split['split'] == 'test']
    print(f"# reports in test set {len(test)}")

    subject_ids_train = [f'p{id}' for id in train['subject_id']]
    subject_ids_test = [f'p{id}' for id in test['subject_id']]
    subject_ids_train = set(subject_ids_train)
    subject_ids_test = set(subject_ids_test)
    print(f"# patients in test set {len(subject_ids_test)}")

    overlap = subject_ids_train.intersection(subject_ids_test)

    assert len(overlap) == 0

    base_path = os.path.join(os.environ['HOME'], 'radedit/data/files/')
    train_reports = []
    test_reports = []

    n_sents = []
    for p in range(10, 20):
        path =os.path.join(base_path, f'p{p}/') 
        print(f"Working on files in {path}")

        for entry in tqdm(os.listdir(path)):
            if os.path.isdir(os.path.join(path, entry)):
                patient_id = entry
                report_files = os.listdir(os.path.join(path, entry))
                for report_file in report_files:
                    with open(os.path.join(path, entry, report_file), 'r') as file:
                        report_text = file.read()

                    cleaned, n_sent = clean_text(report_text)
                    if "FINDINGS:" in cleaned.upper() or "IMPRESSION:" in cleaned.upper():
                        if patient_id in subject_ids_train:
                            train_reports.append({
                                'patient_id': patient_id, 
                                'report_id': report_file,
                                'report_text': cleaned
                            })
                        elif patient_id in subject_ids_test:
                            test_reports.append({
                                'patient_id': patient_id, 
                                'report_id': report_file,
                                'report_text': cleaned
                            })
                            n_sents.append(n_sent)
    
    print(f"total patients in test set after processing {len(set([r['patient_id'] for r in test_reports]))}")
    print(f"total report in test set after processing {len(test_reports)}")

    with open(os.path.join(data_path, 'report_length.txt'), 'w') as file:
        for l in n_sents:
            file.write(f"{l}\n")

    with open(os.path.join(data_path, 'train.json'), 'w') as json_file:
        json.dump(train_reports, json_file, indent=4)

    with open(os.path.join(data_path, 'test.json'), 'w') as json_file:
        json.dump(test_reports, json_file, indent=4)
        
        
if __name__ == '__main__':
    main()