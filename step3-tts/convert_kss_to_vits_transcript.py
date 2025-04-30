import os


def convert_kss_to_vits_format(kss_dir, transcript_file, output_dir):
    # transcript.v.1.4.txt 파일 읽기
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_lines = f.readlines()
    
    # transcript 내용을 딕셔너리로 저장 (파일명 : 텍스트)
    transcript_dict = {}
    for line in transcript_lines:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            filename = os.path.basename(parts[0])
            text = parts[1]
            transcript_dict[filename] = text

    # output_dir 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # transcript.txt 파일 초기화 (덮어쓰기)
    output_file_path = os.path.join(output_dir, "transcript.txt")
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        pass

    # KSS dataset 디렉토리 탐색하여 모든 .wav 파일 경로 수집
    wav_files = []
    for root, dirs, files in os.walk(kss_dir):
        for file in files:
            if file.endswith(".wav"):
                rel_path = os.path.relpath(os.path.join(root, file), kss_dir)
                wav_files.append(rel_path)

    # 파일 경로 정렬
    wav_files.sort()

    # 정렬된 .wav 파일 리스트 순서대로 처리
    for rel_path in wav_files:
        base_name = os.path.basename(rel_path)
        if base_name in transcript_dict:
            text = transcript_dict[base_name]
            normalized_text = text
            duration = str(len(text.split()))
            english_translation = ""

            output_text = f"{base_name}|{text}|{normalized_text}|{duration}|{english_translation}\n"

            with open(output_file_path, 'a', encoding='utf-8') as out_f:
                out_f.write(output_text)

            print(f"Processed: {rel_path}")
        else:
            print(f"Text not found for {base_name}")

def convert_kss_to_vits_format22(kss_dir, transcript_file, output_dir):
    # transcript.v.1.4.txt 파일을 읽어들입니다
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_lines = f.readlines()
        #print(transcript_lines)

    # transcript 내용이 파일마다 "파일명|텍스트" 형식으로 되어있으므로 이를 딕셔너리로 저장
    transcript_dict = {}
    for line in transcript_lines:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            filename = os.path.basename(parts[0])  # 파일명만 추출
            text = parts[1]
            transcript_dict[filename] = text
    
    print(transcript_dict)

    # KSS dataset 디렉토리 탐색
    for root, dirs, files in os.walk(kss_dir):
        for file in files:
            if file.endswith(".wav"):
                # 각 .wav 파일에 대해 처리
                wav_file = os.path.join(root, file)
                base_name = os.path.basename(wav_file)

                #print(wav_file)
                #print(f"base_name: {base_name}")

                # 텍스트가 딕셔너리에서 존재하는지 확인
                if base_name in transcript_dict:
                    text = transcript_dict[base_name]
                    print(f"base_name: {base_name}")
                else:
                    print(f"Text not found for {base_name}")

# KSS 데이터셋 디렉토리와 출력 디렉토리 경로 설정
kss_dir = '/var/step3-tts/kaggle-dataset/kss'
transcript_file = '/var/step3-tts/kaggle-dataset/transcript.v.1.4.txt'  # transcript 파일 경로
output_dir = '/var/step3-tts/output'

# 변환 함수 호출
convert_kss_to_vits_format(kss_dir, transcript_file, output_dir)
