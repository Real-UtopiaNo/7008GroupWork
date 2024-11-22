# Run the code   
## 0.Environment Setup  
git clone https://github.com/Real-UtopiaNo/7008GroupWork.git  
cd 7008GroupWork  
conda create -n 7008gw python=3.10  
conda activate 7008gw  
## 1.Install lib  
pip install pandas  
pip install scikit-learn  
pip install nltk  
pip install transformers
## 2.Install torch (CPU version is enough to run the code)   
### If you have GPU:  
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118  
### Else:  
pip install torch==2.0.1  
## 3.Download Pre-trained weights  
download model weight and save it at "finetune_roberta_un/"  
https://drive.google.com/file/d/1f9CdrwgKTSogDXUYZBkG8wMTcrXaAkkz/view?usp=drive_link  
## 4.Run the code
python chatbot.py  

# Code Structure  
7008GroupWork  
├─analysis      #store analysis pic  
├─cache      #store encoded database for deep learning retrieve method  
├─Database      #store processed QA pairs  
├─finetune_roberta_un      #store finetuned roberta classifier  
├─nltk_data      #store nltk file  
├─rawdata_processing      #codes for processing the raw data  
└─__pycache__      #custom library  
─ chatbot.py      #Main code for chatbot  
─ Bayesian_clf.py      #Bayesian classifier  
─ Randomforest_clf.py      #Random Forest classifier  
─ Roberta_clf.py      #Roberta classifier  
─ train_roberta.py      #code for roberta classifier finetuning  
─ machinel_chatbot.py      #Machine learning retrieve method (TF-IDF)  
─ deepl_chatbot.py      #Deep learning retrieve method (Roberta)  
─ test_bayesian.py      #Test the accuracy of bayesian classifier  
─ test_roberta.py      #Test the accuracy of roberta classifier  
─ test_randomforest.py      #Test the accuracy of random forest classifier  
