# تعليمات التشغيل على Kaggle

## الخطوات بالتفصيل

### 1️⃣ رفع الكود على GitHub

```bash
# في terminal على جهازك
git init
git add .
git commit -m "Image captioning project"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

### 2️⃣ إنشاء Notebook على Kaggle

1. اذهب إلى [Kaggle](https://www.kaggle.com)
2. اضغط على **Code** → **New Notebook**
3. من Settings:
   - **Accelerator**: اختر **GPU T4 x2** (مجاني)
   - **Internet**: فعّل **Internet On**
   - **Persistence**: اختار **Files only**

### 3️⃣ إضافة Dataset

1. اضغط **Add Data** من الجانب الأيمن
2. ابحث عن: **flickr-image-dataset**
3. اختر: `hsankesara/flickr-image-dataset`
4. اضغط **Add**

### 4️⃣ كود التشغيل في Kaggle Notebook

```python
# Cell 1: استنساخ الريبو
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>

# Cell 2: تثبيت المكتبات
!pip install -q -r requirements.txt

# Cell 3: تحضير البيانات من Flickr
!python kaggle_setup.py

# Cell 4: تدريب النموذج
!python training/Vanilla_RNN.py

# Cell 5: عرض النتائج
import json
with open('results/Vanilla_RNN/results.json', 'r') as f:
    results = json.load(f)
    
print(f"Final Train Loss: {results['final_train_loss']:.4f}")
print(f"Final Val Loss: {results['final_val_loss']:.4f}")
print(f"Best Val Loss: {results['best_val_loss']:.4f}")

# رسم منحنى الخسارة
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(results['train_loss_history'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(results['val_loss_history'])
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
```

### 5️⃣ تجربة النماذج المختلفة

```python
# LSTM
!python training/LSTM.py

# Attention LSTM
!python training/Attention_LSTM.py

# Transformer
!python training/Transformer.py
```

### 6️⃣ حفظ النتائج

```python
# حفظ النتائج كـ output
!mkdir -p /kaggle/working/outputs
!cp -r results/* /kaggle/working/outputs/

# تحميل النموذج المدرب
# سيكون متاح في Output tab بعد الـ commit
```

## ⚙️ تعديل الإعدادات

لتحسين الأداء، عدّل ملف `configs/Vanilla_RNN.yaml`:

```yaml
model:
  wordvec_dim: 256      # زود من 128
  hidden_dim: 512       # زود من 128

training:
  num_epochs: 30        # قلل من 50
  batch_size: 32        # قلل من 64 لو في memory issues
  learning_rate: 0.0005 # قلل من 0.001
  weight_decay: 0.0001  # أضف regularization
```

## 🐛 حل المشاكل

### Out of Memory
```python
# قلل batch size في config
batch_size: 16  # بدلاً من 64
```

### Dataset مش موجود
```python
# تأكد من المسار
import os
print(os.listdir('/kaggle/input'))
print(os.listdir('/kaggle/input/flickr-image-dataset'))
```

### الكود مش شغال
```python
# تأكد من المكتبات
!pip list | grep torch
!python --version
```

## 📊 مقارنة النماذج

بعد تدريب كل النماذج:

```python
import json
import pandas as pd

models = ['Vanilla_RNN', 'LSTM', 'Attention_LSTM', 'Transformer']
results_list = []

for model in models:
    try:
        with open(f'results/{model}/results.json', 'r') as f:
            data = json.load(f)
            results_list.append({
                'Model': model,
                'Train Loss': data['final_train_loss'],
                'Val Loss': data['final_val_loss'],
                'Best Val Loss': data['best_val_loss'],
                'Parameters': data['num_params'],
                'Time (s)': data['total_time']
            })
    except:
        pass

df = pd.DataFrame(results_list)
print(df.to_string(index=False))
```

## 💡 نصائح

1. **ابدأ بـ dataset صغير** (max_samples=1000) للتجربة السريعة
2. **استخدم GPU** دايماً للتدريب
3. **راقب الـ validation loss** لتجنب overfitting
4. **احفظ النتائج** قبل إيقاف الـ notebook
5. **استخدم early stopping** لتوفير الوقت

## 🎯 الخطوات التالية

1. جرب LSTM - المفروض يكون أحسن من RNN
2. جرب Attention - المفروض يكون أحسن من LSTM
3. قارن النتائج
4. عدّل الـ hyperparameters
5. جرب augmentation للصور
