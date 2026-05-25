@echo off
cd /d C:\Users\1003380\Anime_Prediction

echo [2/3] Generating B2 embeddings...
C:\Users\1003380\AppData\Local\anaconda3\Scripts\conda.exe run -n nlp python -m src.text_branch.run_text_embedding_pipeline --finetuned-model-path artifacts/finetuned_encoder_B2 --output-prefix text_embeddings_B2 --remove-marketing false

echo [3/3] Evaluating B2 with Ridge...
C:\Users\1003380\AppData\Local\anaconda3\Scripts\conda.exe run -n nlp python src/text_branch/baseline_model.py --embedding-prefix text_embeddings_B2 --report-name text_branch_metrics_B2.json --experiment-name B2

echo PIPELINE_COMPLETE
