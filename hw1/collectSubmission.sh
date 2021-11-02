rm -f assignment1.zip 
zip -r assignment1.zip . -x "*.git*" "*cs682/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*"
