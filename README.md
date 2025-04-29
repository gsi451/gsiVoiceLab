# 🎙️ gsiVoiceLab

gsiVoiceLab는 AI 기반 TTS(Text-to-Speech)와 Voice Conversion 기술을 활용해  
**좋아하는 연예인 스타일의 자연스러운 목소리로 음성을 생성하는 개인 음성 AI 프로젝트**입니다.  

공부 및 샘플 프로젝트로 시작하며, 향후 강화학습과 음성 평가 모델을 통해  
대화형 에이전트 형태로 확장할 예정입니다.

---

## 📌 프로젝트 목표

- 음성을 텍스트로 변환 (STT) → 자연스러운 문장 생성 → TTS로 음성 합성
- AI 음성 생성 후, 연예인/감성 스타일로 Voice Conversion
- 강화학습/머신러닝 기반 자연스러움/감정 전달 품질 개선
- 개인화된 대화형 음성 에이전트 구축

---

## 📂 작업 스텝  

- [x] **Step 1 : STT (음성 → 텍스트)**
  - 마이크 입력으로 음성 수집
  - 음성 인식 모델을 통해 텍스트로 변환
  - 📌 해당 부분은 이미 구현 완료  

- [ ] **Step 2 : 자연어 문장 생성**
  - 입력 텍스트에 적절한 대답 문장 생성  
  (ex: "전등 불꺼" → "네, 전등을 끄겠습니다.")  
  - 추후 대화형 에이전트 기능 추가 예정  

- [ ] **Step 3 : TTS (텍스트 → 음성)**
  - VITS / FastSpeech2 / Tacotron2 모델 활용  
  - 음성 데이터셋 확보 및 학습  
  - AI 목소리로 텍스트 음성 합성  

- [ ] **Step 4 : Voice Conversion (VC)**
  - So-VITS-SVC / FreeVC 기반 스타일 변환  
  - 연예인 음성 스타일 LoRA로 파인튜닝  
  - AI 음성을 좋아하는 목소리 스타일로 변환  

- [ ] **Step 5 : 강화학습/머신러닝 적용**
  - 생성 음성의 자연스러움/감정 전달 평가 모델 구축  
  - 리워드 함수 설계  
  - TTS/VC 또는 응답 생성 에이전트 강화학습 적용  

---

## 📦 사용 예정 기술

- Python / PyTorch
- VITS / FastSpeech2 / Tacotron2
- So-VITS-SVC / FreeVC
- Reinforcement Learning (Stable-Baselines3 or Custom)
- 음성 데이터셋 (Kaggle / Common Voice / 개인 음원 LoRA)

---

## 📌 프로젝트 상태  

| 단계             | 상태       |
|:----------------|:------------|
| STT 구현         | ✅ 완료       |
| 문장 생성         | ⬜ 진행 예정  |
| TTS 모델 구축    | ⬜ 진행 예정  |
| Voice Conversion | ⬜ 진행 예정  |
| 강화학습 적용     | ⬜ 진행 예정  |

---

## 📌 향후 계획

- [ ] 베이스 TTS 모델 구축 및 샘플 음성 생성  
- [ ] 연예인 음성 데이터 LoRA 적용  
- [ ] 음성 자연스러움 평가 모델 개발  
- [ ] End-to-End 음성 대화형 에이전트 완성  

---

## 📄 License  
MIT License (or 개인 사용 목적)

---

## 📬 Contact  
**gsi** : [your-email@example.com]

