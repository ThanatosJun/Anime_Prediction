@echo off
cd /d C:\Users\1003380\Anime_Prediction

echo [1/3] Training B2 encoder (unfreeze-2 + proj-384)...
C:\Users\1003380\AppData\Local\anaconda3\Scripts\conda.exe run -n nlp python -m src.text_branch.finetune_encoder --unfreeze-layers 2 --projection-dim 384 --run-id B2 --epochs 10 --patience 3
if errorlevel 1 goto :error

echo [2/3] Generating B2 embeddings...
C:\Users\1003380\AppData\Local\anaconda3\Scripts\conda.exe run -n nlp python -m src.text_branch.run_text_embedding_pipeline --finetuned-model-path artifacts/finetuned_encoder_B2 --output-prefix text_embeddings_B2 --remove-marketing false
if errorlevel 1 goto :error

echo [3/3] Evaluating B2 with Ridge...
C:\Users\1003380\AppData\Local\anaconda3\Scripts\conda.exe run -n nlp python src/text_branch/baseline_model.py --embedding-prefix text_embeddings_B2 --report-name text_branch_metrics_B2.json --experiment-name B2
if errorlevel 1 goto :error

echo PIPELINE_COMPLETE
exit /b 0

:error
echo PIPELINE_FAILED at step above
exit /b 1
