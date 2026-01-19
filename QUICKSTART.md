# 🚀 Quick Start - Kaggle

## خطوات سريعة للتشغيل على Kaggle

### 1️⃣ رفع على GitHub (5 دقائق)

```bash
git init
git add .
git commit -m "Image captioning project"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

### 2️⃣ إعداد Kaggle Notebook (2 دقيقة)

1. اذهب إلى [kaggle.com/code](https://www.kaggle.com/code)
2. اضغط **New Notebook**
3. من Settings (⚙️):
   - **Accelerator**: GPU T4 x2
   - **Internet**: On
   - **Persistence**: Files only

### 3️⃣ إضافة Dataset (1 دقيقة)

1. اضغط **+ Add Data** من الجانب الأيمن
2. ابحث عن: `flickr-image-dataset`
3. اختر: `hsankesara/flickr-image-dataset`
4. اضغط **Add**

### 4️⃣ تشغيل الكود (30-60 دقيقة)

انسخ والصق في Cells:

```python
# Cell 1: Clone repo
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
```

```python
# Cell 2: Check paths
!python check_kaggle_paths.py
```

```python
# Cell 3: Install
!pip install -q -r requirements.txt
```

```python
# Cell 4: Prepare data (5-10 min)
!python kaggle_setup.py
```

```python
# Cell 5: Train (30-50 min)
!python training/Vanilla_RNN.py
```

```python
# Cell 6: View results
import json
with open('results/Vanilla_RNN/results.json') as f:
    r = json.load(f)
print(f"Train Loss: {r['final_train_loss']:.2f}")
print(f"Val Loss: {r['final_val_loss']:.2f}")
```

---

## ✅ تم! 

النتائج موجودة في:
- `results/Vanilla_RNN/results.json` - الأرقام
- `results/Vanilla_RNN/training_curves.png` - الرسوم البيانية
- `results/Vanilla_RNN/model.pt` - النموذج المدرب

---

## 🔄 تجربة نماذج أخرى

```python
# LSTM (أفضل من RNN)
!python training/LSTM.py

# Attention LSTM (أفضل من LSTM)
!python training/Attention_LSTM.py

# Transformer (الأفضل)
!python training/Transformer.py
```

---

## 🐛 مشاكل شائعة

### ❌ Dataset not found
**الحل:** تأكد إنك ضفت الـ dataset من Add Data

### ❌ Out of memory
**الحل:** عدّل `configs/Vanilla_RNN.yaml`:
```yaml
training:
  batch_size: 32  # قلل من 64
```

### ❌ Git clone failed
**الحل:** تأكد إن الريبو public على GitHub

---

## 📊 النتائج المتوقعة

| Model | Train Loss | Val Loss | Time |
|-------|-----------|----------|------|
| Vanilla RNN | ~20 | ~35 | 30 min |
| LSTM | ~15 | ~25 | 40 min |
| Attention | ~12 | ~20 | 50 min |
| Transformer | ~10 | ~18 | 60 min |

**ملاحظة:** Vanilla RNN عنده overfitting كبير (Val >> Train)

---

## 💡 نصائح

1. ✅ ابدأ بـ Vanilla RNN للتجربة السريعة
2. ✅ استخدم GPU دايماً
3. ✅ راقب الـ validation loss
4. ✅ احفظ النتائج قبل إيقاف الـ notebook
5. ✅ جرب LSTM بعدين - هيكون أحسن بكتير

---

## 📚 ملفات مفيدة

- `KAGGLE_INSTRUCTIONS.md` - تعليمات مفصلة
- `README.md` - دليل المشروع الكامل
- `kaggle_notebook.ipynb` - Notebook جاهز
- `check_kaggle_paths.py` - فحص المسارات
