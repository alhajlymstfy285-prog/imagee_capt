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
- أو ابحث عن "flickr" في Add Data

### 3. إنشاء Notebook على Kaggle

**الإعدادات المطلوبة:**
- Accelerator: **GPU T4 x2** (مجاني)
- Internet: **On** (للـ pretrained models)
- Persistence: **Files only**

**الكود:**
```python
# استنساخ الريبو
!git clone <your-github-repo-url>
%cd <repo-name>

# فحص مسارات Dataset (مهم!)
!python check_kaggle_paths.py

# تثبيت المكتبات
!pip install -r requirements.txt

# تحضير البيانات
!python kaggle_setup.py

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

## 🎯 استخدام GloVe Embeddings

يمكنك استخدام GloVe embeddings المدربة مسبقاً بدلاً من تدريب embeddings من الصفر:

### 1. تحميل GloVe
```bash
# تحميل من Stanford NLP
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

الأحجام المتاحة:
- `glove.6B.50d.txt` (50 dimensions)
- `glove.6B.100d.txt` (100 dimensions)
- `glove.6B.200d.txt` (200 dimensions)
- `glove.6B.300d.txt` (300 dimensions)

### 2. استخدام GloVe في الكود
```python
from rnn_lstm_captioning import CaptioningRNN

model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,  # يجب أن يطابق حجم GloVe
    hidden_dim=512,
    cell_type='lstm',
    glove_path='glove.6B.300d.txt',
    freeze_embeddings=False  # True لتجميد الـ embeddings
)
```

### 3. مثال كامل
راجع ملف `glove_usage_example.py` لمثال تفصيلي.

**فوائد GloVe:**
- تحسين الأداء على الكلمات النادرة
- تقليل وقت التدريب
- embeddings مدربة على corpus ضخم

## 🖼️ اختيار CNN Backbone

يدعم المشروع backbones مختلفة لاستخراج features:

### الخيارات المتاحة
- **ResNet50** (الافتراضي) - أداء ممتاز ومتوازن
- **ResNet101** - أداء أفضل لكن أبطأ
- **RegNet-X 400MF** - خفيف وسريع

### الاستخدام
```python
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='resnet50'  # أو 'resnet101' أو 'regnet_x_400mf'
)
```

أو في config file:
```yaml
model:
  backbone: resnet50
  hidden_dim: 512
  attn_dim: 2048  # ResNet50 output channels
```

**المقارنة:**
| Backbone | Parameters | Speed | Performance | Memory |
|----------|-----------|-------|-------------|--------|
| ResNet50 | 25.6M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | متوسط |
| ResNet101 | 44.5M | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | عالي |
| RegNet | 5.2M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | قليل |

راجع `BACKBONE_GUIDE.md` للتفاصيل الكاملة.

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

1. **GPU**: فعّل GPU من Settings → Accelerator → GPU T4 x2
2. **Internet**: فعّل الإنترنت للـ pretrained models
3. **Dataset**: أضف Flickr dataset من Add Data
4. **Memory**: راقب استخدام الذاكرة (16GB limit)
5. **Check Paths**: شغّل `check_kaggle_paths.py` أولاً للتأكد من المسارات

## 📁 الملفات المساعدة

- `check_kaggle_paths.py`: فحص مسارات Dataset على Kaggle
- `kaggle_setup.py`: تحويل Flickr dataset للصيغة المطلوبة
- `kaggle_notebook.ipynb`: Notebook جاهز للاستخدام
- `KAGGLE_INSTRUCTIONS.md`: تعليمات مفصلة بالعربي

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
