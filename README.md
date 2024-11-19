# 7008GroupWork   
git clone https://github.com/Real-UtopiaNo/7008GroupWork.git  
cd 7008GroupWork  
conda create -n 7008gw python=3.10  
conda activate 7008gw  
pip install pandas  
pip install scikit-learn  
pip install nltk  
pip install transformers

If you have GPU:  
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118  
Else:  
pip install torch  

python chatbot.py  
