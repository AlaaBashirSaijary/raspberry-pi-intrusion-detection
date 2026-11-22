# raspberry-pi-intrusion-detection

مشروع مادة «البحث العلمي» – جامعة حلب، الهندسة المعلوماتية (ماجستير).
إعداد: ألاء سيجري – العام الدراسي 2026م.

## الفكرة
تطبيق عملي لسبعة أبحاث عن أنظمة كشف التسلل خفيفة الوزن على Raspberry Pi،
**بنفس البيانات ونفس التقسيم ونفس المقاييس**، لتكون المقارنة بينها عادلة.
الفجوات التي يعالجها المشروع: قياس موحد للموارد على جهاز واحد، والتعميم عبر مجموعتي بيانات، والإبلاغ عن FPR.

## قواعد المشروع
- الفصل بين التدريب والاختبار **قبل** أي معالجة، وإزالة المكرر بينهما.
- 5 بذور عشوائية، والنتائج بمتوسط ± انحراف.
- لا تُستخدم قياسات الجهاز نفسه (معالج، شبكة) كميزات.
- الإبلاغ عن Accuracy وF1 وTPR وFPR دائماً.
- لا تُقارَن الأرقام هنا بأرقام الأوراق مباشرة (أجهزة وبيانات مختلفة).
- أي هجوم تجريبي يكون على شبكة معزولة خاصة بي فقط.

## الأبحاث المطبَّقة
| المجلد | البحث | النموذج |
|---|---|---|
| `papers/01_sai_svm` | Sai | SVM + اختيار خصائص |
| `papers/02_diab_lightgbm` | Diab | LightGBM |
| `papers/03_wijethilaka_rf` | Wijethilaka | Random Forest |
| `papers/04_cooper_cnn_tflite` | Cooper | CNN مضغوط (TFLite) |

(تُضاف بقية الأبحاث القابلة للتطبيق بنفس القالب.)

## التشغيل
```bash
pip install -r requirements.txt

# تجربة الخط ببيانات وهمية (للتأكد أن كل شيء يعمل فقط، ليست نتائج)
python papers/03_wijethilaka_rf/train.py --synthetic

# تشغيل حقيقي داخل مجموعة واحدة
python papers/03_wijethilaka_rf/train.py --data data/edge_iiotset.csv --label-col Attack_label

# التعميم: تدريب على مجموعة واختبار على أخرى (بالخصائص المشتركة بالاسم)
python papers/03_wijethilaka_rf/train.py --data data/edge_iiotset.csv --test-data data/netflow.csv --label-col Label

# جدول المقارنة
python compare.py          # يكتب results/comparison.md

# على Raspberry Pi بعد وصوله (انسخي models/ إليه)
python common/measure_pi.py --model models/wijethilaka_rf.joblib --watts 3.4
```
تفاصيل البيانات في `data/download.md`.

## النتائج
بعد التشغيل الحقيقي تُنسخ هنا محتويات `results/comparison.md`.
الحالة الآن: **الهيكل جاهز، بانتظار البيانات الحقيقية وجهاز Raspberry Pi.**

## حدود العمل
(تُكتب بعد التجارب: ما لم يُقَس، ما لم يُتحقَّق منه، الفروق عن الأوراق.)
# raspberry-pi-intrusion-detection
