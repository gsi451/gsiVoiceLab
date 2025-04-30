https://github.com/coqui-ai/TTS


KSS 데이터셋을 이용한 Coqui TTS 모델 훈련
1. 데이터셋 구조 분석

폴더 구조: kss/1/, kss/2/, kss/3/, kss/4/ 내에 wav 파일 존재
메타데이터: transcript.v.1.4.txt 파일에 다음 형식으로 저장

파일경로|한글텍스트|한글텍스트|한글텍스트|오디오길이|영어번역



2. 데이터셋 전처리
Coqui TTS는 특정 형식의 메타데이터를 필요로 합니다. 다음과 같이 전처리해야 합니다:
```
pythonimport pandas as pd
import os

# 메타데이터 파일 로드
def prepare_kss_metadata(transcript_path, output_path, base_path='kss'):
    metadata = []
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 5:
                wav_path = parts[0]  # 예: 1/1_0000.wav
                text = parts[1]      # 첫 번째 한글 텍스트 사용
                
                # 파일 전체 경로
                full_path = os.path.join(base_path, wav_path)
                
                # Coqui TTS 형식에 맞게 저장 (파일 경로|텍스트)
                metadata.append(f"{full_path}|{text}")
    
    # 새 메타데이터 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in metadata:
            f.write(f"{line}\n")
    
    print(f"메타데이터 파일 생성 완료: {output_path}")
    print(f"총 {len(metadata)}개 샘플")
```

# 실행 예시
prepare_kss_metadata(
    transcript_path="kss/transcript.v.1.4.txt",
    output_path="kss/metadata.csv",
    base_path="kss"
)
```

3. 모델 선택 및 구성 파일 생성
한국어 TTS 모델에 적합한 Tacotron2 + HiFiGAN 또는 VITS 구성을 추천합니다. 다음은 XTTS 모델 대신 Tacotron2 모델 구성 예시입니다:
python# config.json 생성
```
import json

tacotron2_config = {
    "model": "tacotron2",
    "run_name": "kss_tacotron2",
    "run_description": "Korean KSS dataset Tacotron2 model",
    
    # 데이터셋 설정
    "dataset": {
        "name": "kss",
        "path": "kss/",
        "meta_file_train": "metadata.csv",
        "meta_file_val": "",  # 검증 데이터 분리가 필요하면 설정
    },
    
    # 오디오 설정
    "audio": {
        "sample_rate": 22050,
        "win_length": 1024,
        "hop_length": 256,
        "num_mels": 80,
        "preemphasis": 0.98,
        "ref_level_db": 20,
        "min_level_db": -100,
        "mel_fmin": 0,
        "mel_fmax": 8000,
    },
    
    # 모델 매개변수
    "model_params": {
        "r": 2,
        "attention_type": "dynamic_convolution",
        "double_decoder_consistency": true,
        "encoder_in_features": 512,
        "decoder_in_features": 512,
        "encoder_out_features": 512,
        "decoder_out_features": 512,
        "attention_heads": 4,
        "attention_norm": "sigmoid",
    },
    
    # 훈련 설정
    "train_config": {
        "batch_size": 32,
        "eval_batch_size": 16,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 4,
        "run_eval": true,
        "test_delay_epochs": 5,
        "epochs": 1000,
        "lr": 1e-3,
        "lr_scheduler": "NoamLR",
        "warmup_steps": 4000,
        "grad_clip": 5.0,
        "scheduler_after_epoch": false,
        "wd": 1e-6,
    },
    
    # 텍스트 처리 설정
    "text": {
        "cleaner": "korean_cleaners",
        "phoneme_language": "ko",
        "enable_eos_bos_chars": false,
        "characters": {
            "pad": "_",
            "eos": "~",
            "bos": "^",
            "characters": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? 가각간갇갈감갑값갓강갖같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괨괩괫괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길김김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝",
            "punctuations": "!'(),-.:;? ",
            "eos_id": 1,
            "bos_id": 0,
            "pad_id": 0,
        },
    },
}

# 구성 파일 저장
with open('kss_tacotron2_config.json', 'w', encoding='utf-8') as f:
    json.dump(tacotron2_config, f, indent=4, ensure_ascii=False)
```

4. 모델 훈련
전처리와 구성 파일 생성 후 모델 훈련을 시작합니다:
```
bash# 터미널에서 실행
python TTS/bin/train_tts.py --config_path kss_tacotron2_config.json
```

5. 보코더 훈련 (또는 사전 훈련된 보코더 사용)
한국어 데이터로 훈련된 Tacotron2와 함께 사용할 HiFiGAN 보코더를 훈련하는 예시:
```
bashpython TTS/bin/train_vocoder.py --config_path TTS/vocoder/configs/hifigan_config.json \
  --coqpit.dataset.path="kss/" \
  --coqpit.dataset.meta_file="metadata.csv" \
  --coqpit.run_name="kss_hifigan" \
  --coqpit.audio.sample_rate=22050
```
  
6. 음성 스타일 변환을 위한 LoRA 적용 (가수 음성으로 미세 조정)
python# 기본 모델을 먼저 훈련한 후, 가수 음성으로 미세 조정

# Coqui TTS가 기본적으로 LoRA를 지원하지 않기 때문에 YourTTS 또는 XTTS 사용 권장

# 가수 데이터셋으로 미세 조정 명령어
```
python TTS/bin/train_tts.py --config_path kss_yourtts_config.json \
  --restore_path /path/to/pretrained/model.pth \
  --coqpit.fine_tuning_run=true \
  --coqpit.dataset.meta_file_train="singer_metadata.csv" \
  --coqpit.run_name="singer_yourtts_finetune"
```
  
7. 학습 후 모델 사용 예시
```
pythonfrom TTS.api import TTS
import torch
```

# 디바이스 설정
```
device = "cuda" if torch.cuda.is_available() else "cpu"
```

# 학습된 모델 로드
```
tts = TTS(model_path="/path/to/kss_model.pth", 
          config_path="/path/to/kss_config.json",
          vocoder_path="/path/to/kss_vocoder.pth",
          vocoder_config_path="/path/to/kss_vocoder_config.json").to(device)
```

# 한글 텍스트를 음성으로 변환
```
tts.tts_to_file(text="안녕하세요, AI가 생성한 한국어 음성입니다.", file_path="output.wav")
```

# 가수 스타일로 음성 생성 (LoRA 모델 적용 후)
```
tts_singer = TTS(model_path="/path/to/finetune_singer_model.pth", 
                config_path="/path/to/finetune_singer_config.json").to(device)
tts_singer.tts_to_file(text="안녕하세요, 가수 스타일로 생성된 음성입니다.", file_path="singer_output.wav")
```

8. 실용적인 팁

데이터셋 분할: 학습/검증/테스트 데이터로 분할하여 과적합 방지
데이터 확장(Augmentation): 속도, 음높이 변경 등으로 데이터 다양성 확보
단계적 훈련: 먼저 더 작은 데이터셋으로 빠른 프로토타입 구축
한국어 특화: 한국어 특유의 음소 처리를 위한 korean_cleaners 활용
체크포인트 저장: 훈련 중간에 체크포인트를 자주 저장하여 학습 진행 관리

이렇게 하면 KSS 데이터셋을 이용해 한국어 TTS 모델을 훈련하고, 나중에 가수 음성 스타일로 변환하는 작업까지 진행할 수 있습니다. 진행 중 특정 단계에서 문제가 있으시면 언제든지 질문해 주세요!




#Coqui TTS와 폴더 구조
현재 구조
```
kss/
  ├── 1/
  │   ├── 1_0000.wav
  │   ├── 1_0001.wav
  │   └── ...
  ├── 2/
  │   ├── 2_0000.wav
  │   └── ...
  ├── 3/
  │   └── ...
  ├── 4/
  │   └── ...
  └── transcript.v.1.4.txt
```
Coqui TTS의 요구사항
결론부터 말씀드리면: 파일들을 한 폴더로 모을 필요는 없습니다.
Coqui TTS는 메타데이터 파일에 지정된 상대 경로나 절대 경로를 따라 오디오 파일을 찾습니다. 현재 폴더 구조를 유지한 채로 메타데이터 파일만 적절히 수정하면 됩니다.
메타데이터 수정 방법
기존 KSS 트랜스크립트 파일을 Coqui TTS 형식으로 변환할 때, 파일 경로를 정확히 유지하면 됩니다:
```
pythonimport os

def prepare_kss_metadata(transcript_path, output_path, base_path='kss'):
    metadata = []
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 5:
                wav_path = parts[0]  # 예: 1/1_0000.wav
                text = parts[1]      # 첫 번째 한글 텍스트 사용
                
                # 파일 경로 유지 (폴더 구조 변경 없음)
                # 상대 경로 유지
                metadata.append(f"{wav_path}|{text}")
    
    # 새 메타데이터 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in metadata:
            f.write(f"{line}\n")
    
    print(f"메타데이터 파일 생성 완료: {output_path}")
    print(f"총 {len(metadata)}개 샘플")
```

# 실행 예시
```
prepare_kss_metadata(
    transcript_path="kss/transcript.v.1.4.txt",
    output_path="kss/metadata.csv"
)
```
트레이닝 설정
학습 시 데이터 경로를 올바르게 설정하는 것이 중요합니다:
```
json"dataset": {
    "name": "kss",
    "path": "kss/",  // KSS 폴더 경로 (메타데이터 파일이 있는 곳)
    "meta_file_train": "metadata.csv"
}
```
이렇게 설정하면, 메타데이터 파일의 경로(예: 1/1_0000.wav)에 데이터셋 경로(kss/)가 자동으로 앞에 추가되어 kss/1/1_0000.wav로 파일을 찾게 됩니다.
메타데이터 구조
변환된 메타데이터 파일(metadata.csv)의 내용은 다음과 같은 형식이 됩니다:
1/1_0000.wav|그는 괜찮은 척하려고 애쓰는 것 같았다.
1/1_0001.wav|다른 한글 텍스트...
2/2_0000.wav|또 다른 한글 텍스트...
데이터 로더 검증
학습 시작 전에 데이터 로더가 제대로 작동하는지 테스트해보는 것이 좋습니다:
```
pythonfrom TTS.tts.datasets.load import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import json
```

# 구성 파일 로드
```
with open('kss_tacotron2_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
```

# 샘플 로드 테스트
```
samples = load_tts_samples(
    config['dataset']['path'],
    config['dataset']['meta_file_train'],
    config['text']['cleaner']
)
```

# 첫 몇 개 샘플 확인
```
for i, sample in enumerate(samples[:5]):
    print(f"샘플 {i+1}:")
    print(f"  - 텍스트: {sample[0]}")
    print(f"  - 파일 경로: {sample[1]}")
    # 파일 존재 확인
    full_path = os.path.join(config['dataset']['path'], sample[1]) 
    print(f"  - 파일 존재: {os.path.exists(full_path)}")
```
이렇게 하면 현재 폴더 구조를 유지한 채로 Coqui TTS 모델을 훈련할 수 있습니다. 별도로 모든 WAV 파일을 한 곳에 모을 필요는 없습니다.
