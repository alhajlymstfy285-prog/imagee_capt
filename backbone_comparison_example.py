"""
مثال: مقارنة CNN Backbones المختلفة

يوضح هذا المثال كيفية استخدام backbones مختلفة
ومقارنة الأداء والذاكرة المستخدمة.
"""

import torch
from rnn_lstm_captioning import CaptioningRNN, ImageEncoder

# مثال vocabulary بسيط
word_to_idx = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    'a': 3,
    'dog': 4,
    'cat': 5,
}

print("=" * 60)
print("مقارنة CNN Backbones")
print("=" * 60)

backbones = ['resnet50', 'resnet101', 'regnet_x_400mf']

for backbone in backbones:
    print(f"\n{'='*60}")
    print(f"Backbone: {backbone}")
    print(f"{'='*60}")
    
    # إنشاء encoder فقط للمقارنة
    encoder = ImageEncoder(pretrained=False, backbone=backbone, verbose=True)
    
    # حساب عدد parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Output channels: {encoder.out_channels}")
    
    # اختبار forward pass
    dummy_images = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        features = encoder(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    
    # حساب حجم الذاكرة التقريبي
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"Approximate memory: {memory_mb:.1f} MB")

print("\n" + "="*60)
print("إنشاء نموذج كامل مع ResNet50")
print("="*60)

# إنشاء نموذج كامل
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='resnet50',
    image_encoder_pretrained=False  # False للسرعة في المثال
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# اختبار forward pass
dummy_images = torch.randn(2, 3, 224, 224)
dummy_captions = torch.randint(0, len(word_to_idx), (2, 10))

with torch.no_grad():
    loss = model(dummy_images, dummy_captions)

print(f"Loss: {loss.item():.4f}")

print("\n" + "="*60)
print("التوصيات")
print("="*60)
print("""
1. للمبتدئين: استخدم ResNet50
   - أداء ممتاز
   - سرعة جيدة
   - ذاكرة معقولة

2. للـ Kaggle (GPU محدود): استخدم RegNet
   - أسرع
   - يستهلك ذاكرة أقل
   - يسمح بـ batch size أكبر

3. للبحث (أفضل أداء): استخدم ResNet101
   - أفضل أداء
   - يحتاج GPU قوي
   - وقت تدريب أطول
""")

print("\nمثال استخدام في الكود:")
print("""
# ResNet50 (موصى به)
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,
    hidden_dim=512,
    cell_type='lstm',
    backbone='resnet50'
)

# أو في config file:
# configs/LSTM.yaml
model:
  backbone: resnet50
  hidden_dim: 512
  attn_dim: 2048
""")
