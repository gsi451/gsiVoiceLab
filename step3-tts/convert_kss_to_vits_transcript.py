import os
import shutil

def convert_kss_to_vits_format(kss_dir, transcript_file, output_dir):
    # transcript.v.1.4.txt 읽기
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_lines = f.readlines()

    # 파일명 → 텍스트 딕셔너리 생성
    transcript_dict = {}
    for line in transcript_lines:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            filename = os.path.basename(parts[0])
            text = parts[1]
            transcript_dict[filename] = text

    # output_dir 초기화
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "transcript.txt")
    open(output_file_path, 'w', encoding='utf-8').close()

    # .wav 파일 목록 수집 및 정렬
    wav_files = []
    for root, dirs, files in os.walk(kss_dir):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, kss_dir)
                wav_files.append((file, full_path))  # (파일명, 전체경로)
    wav_files.sort()

    # 각 .wav 파일 처리
    for filename, full_path in wav_files:
        if filename in transcript_dict:
            text = transcript_dict[filename]
            line = f"{filename}|{text}\n"

            # transcript.txt에 기록
            with open(output_file_path, 'a', encoding='utf-8') as out_f:
                out_f.write(line)

            # wav 파일 복사
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(full_path, dst_path)

            print(f"Processed and copied: {filename}")
        else:
            print(f"Text not found for {filename}")

# 실행
kss_dir = '/var/step3-tts/kaggle-dataset/kss'
transcript_file = '/var/step3-tts/kaggle-dataset/transcript.v.1.4.txt'
output_dir = '/var/step3-tts/output'
convert_kss_to_vits_format(kss_dir, transcript_file, output_dir)
