# Spam Detector AI

Spam-Detector-AI là một package Python để phát hiện và lọc tin nhắn spam sử dụng các mô hình Machine Learning. Package này tích hợp với Django hoặc bất kỳ dự án Python nào và cung cấp nhiều loại classifier: Naive Bayes, Random Forest, Support Vector Machine (SVM), Logistic Regression và XGBClassifier.

## Mục lục

- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
    - [Chuẩn bị dữ liệu NLTK](#chuẩn-bị-dữ-liệu-nltk)
    - [Huấn luyện mô hình](#huấn-luyện-mô-hình)
    - [Chạy test](#chạy-test)
    - [Dự đoán tin nhắn spam](#dự-đoán-tin-nhắn-spam)

## Cài đặt

### Cài đặt từ pip

```sh
pip install spam-detector-ai
```

### Cài đặt các dependencies cần thiết

Đảm bảo bạn đã cài đặt các dependencies sau:

- scikit-learn
- nltk
- pandas
- numpy
- joblib
- xgboost

```sh
pip install scikit-learn nltk pandas numpy joblib xgboost
```

## Hướng dẫn sử dụng

### Chuẩn bị dữ liệu NLTK

Trước khi sử dụng, bạn cần tải dữ liệu NLTK. Chạy Python interpreter và thực hiện:

```python
import nltk

nltk.download('wordnet')
nltk.download('stopwords')
```

### Huấn luyện mô hình

Trước khi sử dụng các classifier, bạn phải huấn luyện các mô hình. Dữ liệu huấn luyện được tải từ file CSV trong thư mục `data`. File CSV phải có 3 cột: `label`, `text` và `label_num`.

- `text`: Nội dung tin nhắn cần phân tích
- `label`: Nhãn `ham` hoặc `spam`  
- `label_num`: Số `0` (không phải spam) hoặc `1` (spam)

Để huấn luyện mô hình, chạy lệnh:

```sh
python3 spam_detector_ai/trainer.py
```

⚠️ **Lưu ý**: Có thể xảy ra lỗi "module not found" ⚠️

Nếu gặp lỗi này, hãy sử dụng IDE để chạy file `trainer.py` hoặc thử:

```sh
python -m spam_detector_ai.trainer
```

Lệnh này sẽ huấn luyện tất cả các mô hình và lưu chúng dưới dạng file `.joblib` trong thư mục models:

- `naive_bayes_model.joblib`
- `random_forest_model.joblib`
- `logistic_regression_model.joblib`

### Chạy test

Để kiểm tra hiệu suất của các mô hình đã huấn luyện, chạy:

```sh
python tests/test.py
```

Hoặc nếu gặp lỗi module:

```sh
python -m spam_detector_ai.tests.test
```

### Dự đoán tin nhắn spam bằng gui

```sh
python gui.py
```





## Cấu trúc dự án

```
spam_detector_ai/
├── classifiers/          # Các classifier 
├── data/                 # Dataset mẫu để huấn luyện
├── loading_and_processing/  # Utilities để load và xử lý dữ liệu
├── models/               # Các mô hình đã huấn luyện và vectorizers
├── prediction/           # Class chính để detect spam
├── tests/                # Scripts để test
├── tuning/               # Scripts để fine-tune classifiers
├── training/             # Scripts để huấn luyện classifiers
└── trainer.py            # Script chính để huấn luyện mô hình
```