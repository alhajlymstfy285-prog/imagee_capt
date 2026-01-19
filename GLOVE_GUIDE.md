# دليل استخدام GloVe Embeddings

## ما هو GloVe؟

GloVe (Global Vectors for Word Representation) هو نموذج لتمثيل الكلمات كـ vectors مدرب على corpus ضخم من النصوص. استخدام GloVe يحسن أداء النموذج خاصة مع الكلمات النادرة.

## التحميل والإعداد

### 1. تحميل GloVe

```bash
# تحميل من Stanford NLP
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# أو استخدم curl
curl -O http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### 2. الأحجام المتاحة

| الملف | الأبعاد | الحجم |
|------|---------|-------|
| `glove.6B.50d.txt` | 50 | ~171 MB |
| `glove.6B.100d.txt` | 100 | ~347 MB |
| `glove.6B.200d.txt` | 200 | ~693 MB |
| `glove.6B.300d.txt` | 300 | ~1 GB |

**ننصح باستخدام 300d للحصول على أفضل أداء.**

### 3. على Kaggle

```python
# في Kaggle Notebook
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```

## الاستخدام في الكود

### طريقة 1: استخدام مباشر في CaptioningRNN

```python
from rnn_lstm_captioning import CaptioningRNN

model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,  # يجب أن يطابق حجم GloVe
    hidden_dim=512,
    cell_type='lstm',
    glove_path='glove.6B.300d.txt',
    freeze_embeddings=False  # False للسماح بالتدريب
)
```

### طريقة 2: تحميل منفصل

```python
from rnn_lstm_captioning import load_glove_embeddings, WordEmbedding

# تحميل GloVe
glove_embeddings = load_glove_embeddings(
    glove_path='glove.6B.300d.txt',
    word_to_idx=word_to_idx,
    embed_dim=300
)

# إنشاء embedding layer
word_embedd = WordEmbedding(
    vocab_size=len(word_to_idx),
    embed_size=300,
    pretrained_embeddings=glove_embeddings,
    freeze=False
)
```

### طريقة 3: استخدام Config Files

في `configs/LSTM.yaml`:

```yaml
embeddings:
  use_glove: true
  glove_path: "glove.6B.300d.txt"
  freeze: false
```

ثم:

```bash
python training/LSTM.py
```

## المعاملات المهمة

### `wordvec_dim`
- يجب أن يطابق حجم GloVe المستخدم
- 50, 100, 200, أو 300

### `freeze_embeddings`
- `False`: السماح بتحديث embeddings أثناء التدريب (ننصح به)
- `True`: تجميد embeddings (مفيد للـ fine-tuning السريع)

## المقارنة

### بدون GloVe (Random Initialization)
```python
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=128,
    hidden_dim=512,
    cell_type='lstm'
)
```

**المميزات:**
- أسرع في البداية
- لا يحتاج تحميل ملفات إضافية

**العيوب:**
- أداء أقل على الكلمات النادرة
- يحتاج epochs أكثر للتقارب

### مع GloVe
```python
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    glove_path='glove.6B.300d.txt'
)
```

**المميزات:**
- أداء أفضل من البداية
- تعامل أفضل مع الكلمات النادرة
- تقارب أسرع

**العيوب:**
- يحتاج تحميل ملف GloVe (~1GB)
- استهلاك ذاكرة أكبر قليلاً

## نصائح للأداء الأمثل

### 1. Fine-tuning Strategy

```python
# البداية: تجميد embeddings
model = CaptioningRNN(..., freeze_embeddings=True)
# تدريب لـ 10 epochs

# ثم: فك التجميد
for param in model.word_embedd.parameters():
    param.requires_grad = True
# تدريب لـ 20 epochs إضافية
```

### 2. Learning Rate

عند استخدام GloVe، استخدم learning rate أصغر:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005  # بدلاً من 0.001
)
```

### 3. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

## استكشاف الأخطاء

### خطأ: "GloVe file not found"

```python
# تأكد من المسار الصحيح
import os
print(os.path.exists('glove.6B.300d.txt'))

# أو استخدم المسار الكامل
glove_path = '/kaggle/working/glove.6B.300d.txt'
```

### خطأ: "Dimension mismatch"

```python
# تأكد من تطابق wordvec_dim مع GloVe
# إذا استخدمت glove.6B.300d.txt:
wordvec_dim = 300  # يجب أن يكون 300
```

### تحذير: "X/Y words found"

هذا طبيعي. الكلمات غير الموجودة في GloVe تُهيأ عشوائياً:
- `<NULL>`, `<START>`, `<END>` لن تكون في GloVe
- بعض الكلمات النادرة قد لا تكون موجودة

## مثال كامل

```python
import torch
from rnn_lstm_captioning import CaptioningRNN
from a5_helper import load_coco_captions, train_captioner

# تحميل البيانات
data = load_coco_captions()
word_to_idx = data['word_to_idx']

# إنشاء النموذج مع GloVe
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    glove_path='glove.6B.300d.txt',
    freeze_embeddings=False
)

# نقل للـ GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# التدريب
train_captioner(
    model=model,
    data=data,
    optimizer=optimizer,
    num_epochs=30,
    batch_size=128,
    device=device
)
```

## المراجع

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [Stanford NLP Downloads](https://nlp.stanford.edu/data/)

## الخلاصة

| الميزة | بدون GloVe | مع GloVe |
|--------|-----------|----------|
| سهولة الإعداد | ✅ سهل | ⚠️ يحتاج تحميل |
| الأداء | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| سرعة التقارب | بطيء | سريع |
| الكلمات النادرة | ضعيف | ممتاز |
| استهلاك الذاكرة | قليل | متوسط |

**التوصية:** استخدم GloVe 300d للحصول على أفضل النتائج.
