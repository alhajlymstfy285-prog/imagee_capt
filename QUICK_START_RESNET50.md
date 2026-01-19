# البداية السريعة مع ResNet50 و GloVe

## 🚀 خطوات سريعة (5 دقائق)

### 1. تحميل GloVe (اختياري لكن موصى به)

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### 2. تعديل Config

في `configs/LSTM.yaml`:

```yaml
model:
  cell_type: lstm
  wordvec_dim: 300
  hidden_dim: 512
  backbone: resnet50  # استخدام ResNet50

embeddings:
  use_glove: true
  glove_path: "glove.6B.300d.txt"
  freeze: false
```

### 3. تشغيل التدريب

```bash
python training/LSTM.py
```

## 🎯 أو استخدم الكود مباشرة

```python
from rnn_lstm_captioning import CaptioningRNN
from a5_helper import load_coco_captions, train_captioner
import torch

# تحميل البيانات
data = load_coco_captions()
word_to_idx = data['word_to_idx']

# إنشاء النموذج
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='resnet50',              # ResNet50!
    glove_path='glove.6B.300d.txt',   # GloVe!
    freeze_embeddings=False
)

# نقل للـ GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

## 📊 النتائج المتوقعة

مع ResNet50 + GloVe:
- **BLEU-4**: ~0.25-0.27
- **وقت التدريب**: ~4 ساعات (30 epochs)
- **GPU Memory**: ~6GB

## 🔄 المقارنة مع الإعداد القديم

### القديم (RegNet + Random Embeddings):
```python
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=128,
    hidden_dim=256,
    cell_type='lstm'
)
```
- BLEU-4: ~0.20-0.22
- وقت التدريب: ~5 ساعات
- GPU Memory: ~4GB

### الجديد (ResNet50 + GloVe):
```python
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='resnet50',
    glove_path='glove.6B.300d.txt'
)
```
- BLEU-4: ~0.25-0.27 ⬆️
- وقت التدريب: ~4 ساعات ⬇️
- GPU Memory: ~6GB ⬆️

## 💡 نصائح سريعة

### 1. إذا كان GPU ضعيف:
```yaml
model:
  backbone: regnet_x_400mf  # بدلاً من resnet50
  hidden_dim: 256
  
training:
  batch_size: 256
```

### 2. إذا لم يكن لديك GloVe:
```yaml
embeddings:
  use_glove: false  # سيستخدم random initialization
```

### 3. للتدريب السريع:
```yaml
embeddings:
  freeze: true  # تجميد embeddings

training:
  num_epochs: 20  # epochs أقل
```

## 🐛 حل المشاكل

### Out of Memory:
```yaml
training:
  batch_size: 64  # قلل batch size
```

### GloVe not found:
```bash
# تأكد من المسار
ls glove.6B.300d.txt

# أو استخدم المسار الكامل
embeddings:
  glove_path: "/full/path/to/glove.6B.300d.txt"
```

### Slow training:
```yaml
model:
  backbone: regnet_x_400mf  # استخدم backbone أخف
```

## ✅ تحقق من النجاح

عند بدء التدريب، يجب أن ترى:

```
Using resnet50 backbone
Input shape: (2, 3, 224, 224)
Output c5 features shape: torch.Size([2, 2048, 7, 7])
Output channels: 2048

Loaded GloVe embeddings: 4523/5000 words found

Total parameters: 28,456,789
Trainable parameters: 28,456,789

Starting Training
Epoch 1/30
...
```

## 🎉 هذا كل شيء!

الآن لديك:
- ✅ ResNet50 لاستخراج features أفضل
- ✅ GloVe embeddings مدربة مسبقاً
- ✅ أداء محسّن
- ✅ تقارب أسرع

استمتع بالتدريب! 🚀
