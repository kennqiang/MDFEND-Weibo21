# MDFEND: Multi-domain Fake News Detection
This is an official implementation for [**MDFEND: Multi-domain Fake News Detection**](https://dl.acm.org/doi/abs/10.1145/3459637.3482139) which has been accepted by CIKM2021.
## Dataset
The splited dataset (i.e., train, val, test) are in the `MDFEND-Weibo21/data` folder.

You can have access to the original dataset of Weibo21 only after an ["Application to Use Weibo21 for Fake News Detection"](https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAO__Q4mnQlURFcxUTBYOEZSWEk1SFA2Q1BRRDhaOTRQQi4u) has been submitted. 
## Code
### Requirements
Refer to requirements.txt

You can run `pip install -r requirements.txt` to deploy the environment quickly.
### pretrained_model 
You can download pretrained model (Roberta) from https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing and move all the files in the folder into the path `MDFEND-Weibo21/pretrained_model/chinese_roberta_wwm_base_ext_pytorch`.
### Data Preparation
After you download the **Weibo21** dataset (the way to access is described here), move the `train.pkl`, `val.pkl` and `test.pkl` into the path `MDFEND-Weibo21/data`.
### Run
You can run this model through:
```python
python main.py --model_name mdfend --batchsize 32 --lr 0.0007
```
### Reference
```
Nan Q, Cao J, Zhu Y, et al. MDFEND: Multi-domain Fake News Detection[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021: 3343-3347.
```
or in bibtex style:
```
@inproceedings{nan2021mdfend,
  title={MDFEND: Multi-domain Fake News Detection},
  author={Nan, Qiong and Cao, Juan and Zhu, Yongchun and Wang, Yanyan and Li, Jintao},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3343--3347},
  year={2021}
}
```
