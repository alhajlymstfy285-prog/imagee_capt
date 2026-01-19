# Image Captioning with RNN, LSTM, and Transformers

مشروع تعليمي لتوليد تعليقات نصية للصور باستخدام نماذج التسلسل العصبية (RNN, LSTM, Attention, Transformer).

## 📋 المحتويات

- **Vanilla RNN**: نموذج RNN بسيط
- **LSTM**: Long Short-Term Memory
- **Attention LSTM**: LSTM مع آلية الانتباه
- **Transformer**: معمارية Transformer كاملة

## 🚀 التشغيل على Kaggle

### 1. رفع الكود على GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. استخدام Dataset على Kaggle

استخدم **Flickr Image Dataset** من Kaggle:
- Dataset: [flickr30k_images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

### 3. إنشاء Notebook على Kaggle

```python
# في Kaggle Notebook
# أضف الـ dataset من Add Data
# استنسخ الريبو
!git clone <your-github-repo-url>
%cd <repo-name>

# تثبيت المكتبات
!pip install -r requirements.txt

# تشغيل التدريب
!python training/Vanilla_RNN.py
```

## 📦 المتطلبات

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
seaborn>=0.12.0
opencv-python>=4.7.0
numpy>=1.23.0
PyYAML>=6.0
```

## 🏗️ هيكل المشروع

```
├── models/              # تعريفات النماذج
├── training/            # سكريبتات التدريب
├── configs/             # ملفات الإعدادات (YAML)
├── eecs598/             # مكتبة الأدوات
├── rnn_lstm_captioning.py
├── transformers.py
└── a5_helper.py
```

## 🎯 التدريب

```bash
# تدريب Vanilla RNN
python training/Vanilla_RNN.py

# تدريب LSTM
python training/LSTM.py

# تدريب Attention LSTM
python training/Attention_LSTM.py

# تدريب Transformer
python training/Transformer.py
```

## 📊 النتائج

النتائج تُحفظ في مجلد `results/<model_name>/`:
- `results.json`: خسارة التدريب والتحقق
- `model.pth`: أوزان النموذج المدرب

## 🔧 الإعدادات

عدّل ملفات YAML في مجلد `configs/` لتغيير:
- حجم الـ hidden dimension
- معدل التعلم
- عدد الـ epochs
- حجم الـ batch

## 📝 ملاحظات للـ Kaggle

1. **GPU**: فعّل GPU من Settings → Accelerator → GPU
2. **Internet**: فعّل الإنترنت إذا كنت تستخدم pretrained models
3. **Dataset**: أضف Flickr dataset من Add Data
4. **Memory**: راقب استخدام الذاكرة (16GB limit)

## 🐛 حل المشاكل الشائعة

### Overfitting
- أضف dropout
- استخدم weight decay
- قلل عدد الـ epochs
- استخدم early stopping

### Out of Memory
- قلل batch_size
- قلل hidden_dim
- استخدم gradient accumulation

### Vanishing Gradients
- استخدم LSTM بدلاً من RNN
- قلل sequence length
- استخدم gradient clipping

## 📚 المراجع

- [EECS 598: Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/)
- [Show, Attend and Tell Paper](https://arxiv.org/abs/1502.03044)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
