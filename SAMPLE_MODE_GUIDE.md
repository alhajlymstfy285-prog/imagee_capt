# دليل استخدام Sample Mode

## 🎯 ما هو Sample Mode؟

Sample Mode يسمح لك باستخدام عدد محدود من الصور للتجربة السريعة قبل التدريب على كل البيانات.

## 📊 المقارنة

| Mode | عدد الصور | وقت التدريب | الاستخدام |
|------|-----------|-------------|-----------|
| **Sample** | 1000 | ~10 دقائق | ✅ للتجربة والتأكد |
| **Full** | ~30,000 | ~4 ساعات | ✅ للتدريب النهائي |

## 🚀 كيفية الاستخدام

### 1. في Config File

في `configs/Vanilla_RNN.yaml`:

```yaml
# للتجربة السريعة (موصى به أولاً)
data:
  use_sample: true
  sample_size: 1000  # استخدام 1000 صورة فقط

# للتدريب النهائي
data:
  use_sample: false  # استخدام كل البيانات (~30,000 صورة)
```

### 2. تشغيل التدريب

```bash
# نفس الأمر في الحالتين
python training/Vanilla_RNN.py
```

## 📝 سيناريوهات الاستخدام

### السيناريو 1: التجربة الأولى

```yaml
# configs/Vanilla_RNN.yaml
data:
  use_sample: true
  sample_size: 500  # عدد قليل جداً

training:
  num_epochs: 2  # epochs قليلة
  batch_size: 32
```

**الهدف**: التأكد أن الكود يعمل بدون أخطاء (~5 دقائق)

### السيناريو 2: ضبط Hyperparameters

```yaml
data:
  use_sample: true
  sample_size: 2000  # عدد معقول

training:
  num_epochs: 10
  batch_size: 64
```

**الهدف**: تجربة learning rates و hidden dimensions مختلفة (~20 دقيقة)

### السيناريو 3: التدريب النهائي

```yaml
data:
  use_sample: false  # كل البيانات!

training:
  num_epochs: 30
  batch_size: 128
```

**الهدف**: الحصول على أفضل نتائج (~4 ساعات)

## 🔍 ماذا يحدث عند التشغيل؟

### مع Sample Mode (use_sample: true):

```
Loading dataset (memory efficient)...
⚠️  Using SAMPLE mode: 1000 images only (for testing)
Building vocabulary...
   Using 1000 images with 5000 captions
Vocabulary size: 3245
Train samples: 800
Val samples: 200
```

### مع Full Mode (use_sample: false):

```
Loading dataset (memory efficient)...
✅ Using FULL dataset: ~30,000 images
Building vocabulary...
Vocabulary size: 8547
Train samples: 24000
Val samples: 6000
```

## 💡 نصائح مهمة

### 1. ابدأ دائماً بـ Sample Mode

```yaml
# الخطوة 1: تجربة سريعة
data:
  use_sample: true
  sample_size: 500

training:
  num_epochs: 2
```

**تأكد من:**
- ✅ الكود يعمل بدون أخطاء
- ✅ الـ loss ينزل
- ✅ لا توجد مشاكل في الذاكرة

### 2. ثم جرب Sample أكبر

```yaml
# الخطوة 2: ضبط hyperparameters
data:
  use_sample: true
  sample_size: 2000

training:
  num_epochs: 10
```

**جرب:**
- 🔧 Learning rates مختلفة
- 🔧 Hidden dimensions مختلفة
- 🔧 Batch sizes مختلفة

### 3. أخيراً Full Training

```yaml
# الخطوة 3: التدريب النهائي
data:
  use_sample: false

training:
  num_epochs: 30
```

## 📊 أحجام Sample الموصى بها

| الهدف | sample_size | num_epochs | الوقت المتوقع |
|-------|-------------|-----------|---------------|
| اختبار سريع | 500 | 2 | ~5 دقائق |
| تجربة hyperparameters | 2000 | 10 | ~20 دقيقة |
| تجربة متقدمة | 5000 | 15 | ~1 ساعة |
| تدريب نهائي | None (full) | 30 | ~4 ساعات |

## 🎯 مثال عملي كامل

### المرحلة 1: التأكد من عمل الكود

```yaml
# configs/Vanilla_RNN.yaml
model:
  backbone: resnet50
  hidden_dim: 256

data:
  use_sample: true
  sample_size: 500

training:
  num_epochs: 2
  batch_size: 32
```

```bash
python training/Vanilla_RNN.py
```

**النتيجة المتوقعة:**
- ✅ يعمل بدون أخطاء
- ✅ Loss ينزل من ~8 إلى ~6
- ⏱️ ~5 دقائق

### المرحلة 2: ضبط الإعدادات

```yaml
data:
  use_sample: true
  sample_size: 2000

training:
  num_epochs: 10
  batch_size: 64
  learning_rate: 0.001  # جرب 0.0005, 0.002
```

```bash
# جرب learning rates مختلفة
python training/Vanilla_RNN.py
```

**النتيجة المتوقعة:**
- ✅ Loss ينزل إلى ~4-5
- ✅ تعرف أفضل learning rate
- ⏱️ ~20 دقيقة

### المرحلة 3: التدريب النهائي

```yaml
data:
  use_sample: false  # كل البيانات!

training:
  num_epochs: 30
  batch_size: 128
  learning_rate: 0.001  # أفضل قيمة من المرحلة 2
```

```bash
python training/Vanilla_RNN.py
```

**النتيجة المتوقعة:**
- ✅ Loss ينزل إلى ~2-3
- ✅ BLEU-4 ~0.20-0.25
- ⏱️ ~4 ساعات

## ⚠️ تحذيرات

### 1. لا تقارن النتائج مباشرة

```
Sample (1000 images):
  Loss: 4.5
  BLEU: 0.15

Full (30,000 images):
  Loss: 2.8
  BLEU: 0.24
```

**Sample mode للتجربة فقط، ليس للنتائج النهائية!**

### 2. Vocabulary مختلف

```
Sample: vocab_size = 3245
Full:   vocab_size = 8547
```

**الـ vocabulary في sample mode أصغر!**

### 3. Overfitting محتمل

في sample mode، النموذج قد يحفظ البيانات بسرعة.

## 🎓 الخلاصة

### استخدم Sample Mode عندما:
- ✅ تريد التأكد أن الكود يعمل
- ✅ تجرب hyperparameters مختلفة
- ✅ تختبر features جديدة
- ✅ وقتك محدود

### استخدم Full Mode عندما:
- ✅ جاهز للتدريب النهائي
- ✅ تريد أفضل النتائج
- ✅ عندك وقت كافي (~4 ساعات)
- ✅ ضبطت كل الإعدادات

## 🚀 البداية السريعة

```bash
# 1. افتح config
nano configs/Vanilla_RNN.yaml

# 2. فعّل sample mode
data:
  use_sample: true
  sample_size: 1000

# 3. شغّل
python training/Vanilla_RNN.py

# 4. إذا نجح، غيّر إلى full mode
data:
  use_sample: false
```

بالتوفيق! 🎉
