# دليل تحميل GloVe Embeddings

## الطريقة الأولى: على Kaggle (الأسهل)

### استخدم Kaggle Dataset جاهز:

1. **أضف Dataset لـ Notebook:**
   - في Kaggle Notebook، اضغط على "Add Data"
   - ابحث عن: `"glove embeddings"`
   - اختار: **"GloVe Global Vectors for Word Representation"**
   - أو استخدم: `stanford-nlp/glove6b`

2. **المسار بيكون:**
   ```python
   glove_path = "/kaggle/input/glove6b/glove.6B.300d.txt"
   ```

3. **عدّل الـ Config:**
   ```yaml
   embeddings:
     use_glove: true
     glove_path: "/kaggle/input/glove6b/glove.6B.300d.txt"
     freeze: false
   ```

---

## الطريقة الثانية: تحميل مباشر

### استخدم السكريبت الجاهز:

```bash
python download_glove.py
```

**ملاحظة:** التحميل هياخد 5-10 دقائق (حجم الملف ~822 MB)

---

## الطريقة الثالثة: تحميل يدوي

### من موقع Stanford:

1. **رابط التحميل:**
   ```
   https://nlp.stanford.edu/data/glove.6B.zip
   ```

2. **فك الضغط:**
   ```bash
   unzip glove.6B.zip
   ```

3. **الملفات المتاحة:**
   - `glove.6B.50d.txt` - 50 dimensions
   - `glove.6B.100d.txt` - 100 dimensions
   - `glove.6B.200d.txt` - 200 dimensions
   - `glove.6B.300d.txt` - 300 dimensions ✓ (استخدم ده)

---

## التحقق من التحميل:

```python
import os

glove_path = "glove.6B.300d.txt"
# أو على Kaggle:
# glove_path = "/kaggle/input/glove6b/glove.6B.300d.txt"

if os.path.exists(glove_path):
    size_mb = os.path.getsize(glove_path) / (1024 * 1024)
    print(f"✓ GloVe file found: {size_mb:.1f} MB")
    
    # عدد الكلمات
    with open(glove_path, 'r', encoding='utf-8') as f:
        num_words = sum(1 for _ in f)
    print(f"✓ Number of words: {num_words:,}")
else:
    print("✗ GloVe file not found!")
```

**النتيجة المتوقعة:**
```
✓ GloVe file found: 990.1 MB
✓ Number of words: 400,000
```

---

## معلومات عن GloVe:

### ما هو GloVe؟
- **Global Vectors for Word Representation**
- Pretrained word embeddings من Stanford NLP
- متدرب على 6 billion tokens من Wikipedia + Gigaword

### الأبعاد المتاحة:
- 50d, 100d, 200d, **300d** ← (الأفضل للـ image captioning)

### الفوائد:
- ✅ بداية أفضل من random initialization
- ✅ الكلمات المتشابهة قريبة من بعض في الـ embedding space
- ✅ بيقلل وقت التدريب
- ✅ بيحسّن جودة الـ captions

### متى تستخدم `freeze: true`؟
- **freeze: false** ← الموديل يعدّل الـ embeddings أثناء التدريب (أفضل)
- **freeze: true** ← الـ embeddings تفضل ثابتة (أسرع، لكن أقل دقة)

---

## استخدام في الكود:

### في الـ Config:
```yaml
embeddings:
  use_glove: true
  glove_path: "/kaggle/input/glove6b/glove.6B.300d.txt"  # على Kaggle
  # أو
  glove_path: "glove.6B.300d.txt"  # محلي
  freeze: false
```

### في الكود:
```python
from rnn_lstm_captioning import load_glove_embeddings

# تحميل GloVe
glove_embeddings = load_glove_embeddings(
    glove_path="glove.6B.300d.txt",
    word_to_idx=vocab['token_to_idx'],
    embedding_dim=300
)

print(f"Loaded embeddings shape: {glove_embeddings.shape}")
# Output: Loaded embeddings shape: torch.Size([vocab_size, 300])
```

---

## الخلاصة:

**على Kaggle:** استخدم Kaggle Dataset (الأسهل)
```yaml
glove_path: "/kaggle/input/glove6b/glove.6B.300d.txt"
```

**محلياً:** حمّل من Stanford أو استخدم `download_glove.py`
```yaml
glove_path: "glove.6B.300d.txt"
```
