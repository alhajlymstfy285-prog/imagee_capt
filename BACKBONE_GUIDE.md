# دليل استخدام CNN Backbones المختلفة

## نظرة عامة

يدعم المشروع الآن استخدام backbones مختلفة لاستخراج features من الصور:
- **ResNet50** (الافتراضي) - أداء ممتاز وسرعة جيدة
- **ResNet101** - أداء أفضل لكن أبطأ
- **RegNet-X 400MF** - خفيف وسريع للأجهزة الضعيفة

## المقارنة

| Backbone | Parameters | Output Channels | Speed | Performance | Memory |
|----------|-----------|-----------------|-------|-------------|--------|
| ResNet50 | 25.6M | 2048 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | متوسط |
| ResNet101 | 44.5M | 2048 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | عالي |
| RegNet-X 400MF | 5.2M | 1280 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | قليل |

## الاستخدام

### طريقة 1: في الكود مباشرة

```python
from rnn_lstm_captioning import CaptioningRNN

# استخدام ResNet50 (الافتراضي)
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='resnet50'
)

# استخدام ResNet101
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='resnet101'
)

# استخدام RegNet (للأجهزة الضعيفة)
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='regnet_x_400mf'
)
```

### طريقة 2: عبر Config Files

في `configs/LSTM.yaml`:

```yaml
model:
  cell_type: lstm
  wordvec_dim: 300
  hidden_dim: 512
  backbone: resnet50  # غير هنا
```

ثم:
```bash
python training/LSTM.py
```

## التفاصيل التقنية

### ResNet50
```python
backbone='resnet50'
```
- **Output Shape**: (N, 2048, 7, 7) للصور 224×224
- **Pretrained**: ImageNet
- **Use Case**: الخيار الأفضل للأداء المتوازن
- **Memory**: ~500MB GPU

### ResNet101
```python
backbone='resnet101'
```
- **Output Shape**: (N, 2048, 7, 7)
- **Pretrained**: ImageNet
- **Use Case**: عندما تريد أفضل أداء ممكن
- **Memory**: ~800MB GPU

### RegNet-X 400MF
```python
backbone='regnet_x_400mf'
```
- **Output Shape**: (N, 1280, 4, 4)
- **Pretrained**: ImageNet
- **Use Case**: للتدريب على GPU ضعيف أو Kaggle
- **Memory**: ~300MB GPU

## ملاحظات مهمة

### 1. تحديث attn_dim للـ Attention LSTM

عند استخدام backbone مختلف، يجب تحديث `attn_dim` في config:

```yaml
# ResNet50/101
model:
  attn_dim: 2048
  backbone: resnet50

# RegNet
model:
  attn_dim: 1280
  backbone: regnet_x_400mf
```

### 2. Batch Size

قد تحتاج لتعديل batch_size حسب الـ backbone:

```yaml
# ResNet50
training:
  batch_size: 128

# ResNet101 (أكبر)
training:
  batch_size: 64

# RegNet (أصغر)
training:
  batch_size: 256
```

### 3. Hidden Dimension

يُنصح بتعديل hidden_dim حسب output channels:

```python
# ResNet50/101 (2048 channels)
hidden_dim = 512  # أو 1024

# RegNet (1280 channels)
hidden_dim = 256  # أو 512
```

## أمثلة عملية

### مثال 1: تدريب سريع على Kaggle

```yaml
# configs/LSTM.yaml
model:
  backbone: regnet_x_400mf
  hidden_dim: 256
  attn_dim: 1280

training:
  batch_size: 256
  num_epochs: 30
```

### مثال 2: أفضل أداء على GPU قوي

```yaml
# configs/LSTM.yaml
model:
  backbone: resnet101
  hidden_dim: 1024
  attn_dim: 2048

training:
  batch_size: 64
  num_epochs: 50
```

### مثال 3: متوازن (موصى به)

```yaml
# configs/LSTM.yaml
model:
  backbone: resnet50
  hidden_dim: 512
  attn_dim: 2048

training:
  batch_size: 128
  num_epochs: 40
```

## استكشاف الأخطاء

### خطأ: "Dimension mismatch"

```python
# تأكد من تطابق attn_dim مع backbone output channels
# ResNet50/101: attn_dim = 2048
# RegNet: attn_dim = 1280
```

### خطأ: "Out of Memory"

```python
# قلل batch_size أو استخدم backbone أصغر
training:
  batch_size: 64  # بدلاً من 128
```

### خطأ: "Unsupported backbone"

```python
# تأكد من كتابة الاسم بشكل صحيح
backbone: 'resnet50'  # صحيح
backbone: 'ResNet50'  # خطأ
```

## النتائج المتوقعة

### على COCO Dataset

| Backbone | BLEU-4 | Training Time | GPU Memory |
|----------|--------|---------------|------------|
| ResNet50 | ~0.25 | 4 hours | 6GB |
| ResNet101 | ~0.27 | 6 hours | 10GB |
| RegNet | ~0.23 | 3 hours | 4GB |

*ملاحظة: النتائج تقريبية وتعتمد على hyperparameters*

## التوصيات

### للمبتدئين
```yaml
backbone: resnet50
batch_size: 128
hidden_dim: 512
```

### للـ Kaggle (GPU محدود)
```yaml
backbone: regnet_x_400mf
batch_size: 256
hidden_dim: 256
```

### للبحث (أفضل أداء)
```yaml
backbone: resnet101
batch_size: 64
hidden_dim: 1024
```

## المراجع

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [RegNet Paper](https://arxiv.org/abs/2003.13678)
- [PyTorch Models](https://pytorch.org/vision/stable/models.html)
