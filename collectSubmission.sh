rm -f INF5860_Oblig1.zip
zip -r INF5860_Oblig1.zip . -x "*.git*" "*code/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*code/build/*"
