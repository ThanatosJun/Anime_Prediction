## yolo處理方法

### 目標
透過yolo (https://github.com/deepghs/imgutils/tree/main) (https://huggingface.co/skytnt/anime-seg) 抓取照片中的動漫物件，並交付給swin transformer 作處理。這個部分需要實作這些finction 來串接原本的內容

### 保留


####　流程

1. 提取照片
2. yolo做objective dection，並回傳結果(frame)
3. Crop frame，並將結果送給swin transformer做推論
