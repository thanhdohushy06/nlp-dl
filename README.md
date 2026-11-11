# LAB 08 — Pretrained Transformers and BERT Fine-Tuning
**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP & Applications)  
**Học kỳ:** Học kỳ I - 2026  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Sinh viên thực hiện:** Đỗ Khắc Đức Thành - MSSV: 24001697  
**Chủ đề:** From Training a Transformer to Using a Pretrained Language Model  

---

## 1. Tổng Quan Về LAB 08

Trong **LAB 07**, chúng ta đã tự tay xây dựng kiến trúc Transformer từ các khối nguyên tử: *Self-Attention, Multi-Head Attention, Positional Encoding, Feed-Forward Network, Residual Connection và Layer Normalization*. Tuy nhiên, mô hình Transformer được xây dựng từ đầu (train from scratch) vẫn phải tự học toàn bộ các biểu diễn ngôn ngữ từ tập dữ liệu huấn luyện của chính bài toán đó.

**LAB 08 đặt câu hỏi cốt lõi:**
> *Nếu một Transformer đã được huấn luyện trước trên một lượng văn bản rất lớn thì có thể sử dụng kiến thức đó cho một bài toán mới như thế nào?*

### Mạch kiến thức xuyên suốt học phần:
```text
LAB 07: Transformer trained from scratch
   ↓
LAB 08: Pretraining ──> Masked Language Modeling ──> Pretrained Transformer
        ──> Tokenizer + Contextual Embedding ──> Fine-tuning ──> Text Classification
   ↓
LAB 09: Token Classification ──> Named Entity Recognition (NER) / Information Extraction
```

---

## 2. Cấu Trúc Thư Mục Nộp Bài (Deliverables)

Thư mục bài làm bao gồm đầy đủ các tệp theo quy chuẩn tại **Mục 25** của tài liệu hướng dẫn:

```text
lab08/
├── README.md                     # Tài liệu tổng quan, hướng dẫn, Reproducibility, AI Usage, Learning Check
├── calculations.md               # Bài tập tính toán MLM Loss và phân tích Tokenization (Mục 8, 9, 10)
├── prediction.md                 # 3 Dự đoán lý thuyết trước khi chạy thực nghiệm (Mục 11)
├── tokenization.ipynb            # Notebook minh họa chi tiết WordPiece Tokenizer, Special Tokens, Attention Mask
├── pretrained_classification.py  # Mã nguồn Python chạy toàn bộ pipeline huấn luyện và 5 thực nghiệm
├── fine_tuning.ipynb             # Notebook hoàn chỉnh từ nạp dữ liệu, fine-tuning, đánh giá, trực quan hóa
├── results.csv                   # Bảng số liệu định lượng tổng hợp của tất cả các thực nghiệm
├── data_efficiency.png           # Biểu đồ đường cong hiệu quả dữ liệu (Data-efficiency curve)
├── confusion_matrix.png          # Ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử
├── confidence_analysis.png       # Biểu đồ phân tích phân phối độ tự tin và tính chuẩn xác
├── error_analysis.md             # Báo cáo phân tích lỗi chi tiết (FP, FN, High-conf errors, Q1 - Q6)
└── reflection.md                 # Bài luận phản tư 300-500 từ về lý do dùng Pretrained Model (Mục 29)
```

---

## 3. Tái Lập Thực Nghiệm (Reproducibility — Mục 23)

Để đảm bảo tính tái lập khoa học cho toàn bộ kết quả thực nghiệm trong báo cáo:

- **Model:** `bert-base-uncased` (Pretrained Transformer Encoder, 12 layers, 768 hidden dimensions, 12 attention heads, ~110M parameters)
- **Tokenizer:** `BertTokenizer` (WordPiece Tokenizer, vocabulary size = 30,522)
- **Dataset:** IMDb Large Movie Review Dataset (Bài toán phân loại cảm xúc nhị phân Positive / Negative)
- **Train split:** 80% (tương ứng với các kịch bản 10%, 25%, 50%, 100% trong bài tập Data Efficiency)
- **Validation split:** 10% (cố định xuyên suốt các thí nghiệm)
- **Test split:** 10% (cố định để so sánh công bằng giữa các mô hình)
- **Max sequence length:** $T = 128$ (thử nghiệm mở rộng với $T \in \{64, 128, 256\}$)
- **Batch size:** 16 (phù hợp với môi trường huấn luyện CPU/GPU)
- **Learning rate:**
  - Full Fine-tuning: $2 \times 10^{-5}$ (với linear warmup và cosine decay)
  - Frozen Encoder: $1 \times 10^{-3}$ (chỉ tối ưu Linear Head)
  - From Scratch: $5 \times 10^{-4}$
- **Epochs:** 3 epochs cho fine-tuning; 5 epochs cho from-scratch
- **Optimizer:** `AdamW` (weight decay = 0.01, $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
- **Random seed:** 42 (`torch.manual_seed(42)`, `numpy.random.seed(42)`)
- **Hardware:** Intel CPU / PyTorch 2.15 environment
- **Training time:** Ghi nhận chi tiết trong `results.csv`

---

## 4. Kê Khai Sử Dụng AI (AI Usage Declaration — Mục 24)

Tuân thủ quy định học thuật tại Mục 24 của tài liệu LAB 08:
```text
AI-assisted code:
What was generated:
- Bộ khung mã nguồn PyTorch DataLoader, training loop chuẩn với AdamW và scheduler.
- Các hàm vẽ biểu đồ matplotlib/seaborn cho Data Efficiency curve, Confusion Matrix và Confidence Distribution.
- Script trích xuất tự động các mẫu dự đoán sai (FP, FN, High-confidence errors) từ tập kiểm thử.

What was modified:
- Tinh chỉnh kiến trúc Classification Head để trích xuất đúng vector biểu diễn tại vị trí [CLS].
- Điều chỉnh batch size và sequence length tối ưu cho khả năng chạy thực nghiệm mượt mà trên môi trường máy cá nhân.
- Chuẩn hóa định dạng bảng đầu ra lưu trữ tại results.csv.

What was verified:
- Tự tay kiểm tra từng bước tính toán đạo hàm và hàm mất mát MLM Loss trong calculations.md.
- Đối chiếu số lượng token, subword token và giá trị attention mask trong tokenization.ipynb.
- Trực tiếp suy luận và giải thích cơ chế sâu xa cho từng mẫu lỗi và câu hỏi bắt buộc Q1 - Q6 trong error_analysis.md.
- Đọc duyệt và đối chiếu từng luận điểm trong bài reflection.md với kiến thức lý thuyết đã học từ LAB 07 và LAB 08.
```

---

## 5. Bảng Điểm Tự Đánh Giá (Rubric Checklist — Mục 26)

| Thành phần đánh giá (Component) | Điểm tối đa | Trạng thái hoàn thành & Tệp minh chứng |
|:---|:---:|:---|
| **Theory & calculation** | 10 | Hoàn thành xuất sắc trong [calculations.md](file:///c:/Users/thanh/Desktop/lab08/calculations.md) |
| **Prediction before experiment** | 10 | Hoàn thành đầy đủ 3 dự đoán trong [prediction.md](file:///c:/Users/thanh/Desktop/lab08/prediction.md) |
| **Tokenization & attention mask** | 10 | Minh họa toàn diện trong [tokenization.ipynb](file:///c:/Users/thanh/Desktop/lab08/tokenization.ipynb) |
| **Pretrained model implementation** | 15 | Triển khai chuẩn trong [pretrained_classification.py](file:///c:/Users/thanh/Desktop/lab08/pretrained_classification.py) |
| **From-scratch vs pretrained** | 10 | So sánh định lượng và lý luận trong notebook và báo cáo |
| **Frozen vs fine-tuning** | 10 | Thí nghiệm kiểm chứng trong [fine_tuning.ipynb](file:///c:/Users/thanh/Desktop/lab08/fine_tuning.ipynb) & `results.csv` |
| **Data-efficiency experiment** | 10 | Thử nghiệm 10%, 25%, 50%, 100% kèm biểu đồ `data_efficiency.png` |
| **Quantitative evaluation** | 10 | Đầy đủ Accuracy, Precision, Recall, F1, `confusion_matrix.png` |
| **Error & confidence analysis** | 10 | Phân tích 8 mẫu lỗi và phân phối tự tin trong [error_analysis.md](file:///c:/Users/thanh/Desktop/lab08/error_analysis.md) |
| **Reflection** | 5 | Bài luận 480 từ, đầy đủ 6 khía cạnh bắt buộc trong [reflection.md](file:///c:/Users/thanh/Desktop/lab08/reflection.md) |
| **Tổng điểm (Total)** | **100** | **Đạt tiêu chuẩn tối đa theo yêu cầu đề bài** |

---

## 6. Individual Learning Check (Mục 27 — 12 Câu Hỏi Kiểm Tra Cá Nhân)

Sinh viên trả lời ngắn gọn, khúc chiết bằng chính ngôn ngữ và tư duy của mình:

#### 1. Pretraining khác fine-tuning ở đâu?
- **Pretraining:** Huấn luyện mô hình từ đầu trên kho văn bản khổng lồ không gán nhãn bằng tác vụ tự giám sát (như MLM, NSP) nhằm học các biểu diễn ngữ nghĩa và cấu trúc ngôn ngữ tổng quát.
- **Fine-tuning:** Tiếp tục huấn luyện mô hình đã pretrained trên một tập dữ liệu có nhãn chuyên biệt của bài toán cụ thể (downstream task) với tốc độ học nhỏ nhằm thích nghi trọng số với bài toán đó.

#### 2. Masked Language Modeling (MLM) là gì?
MLM là tác vụ tự giám sát trong đó một tỷ lệ token trong câu (thường là 15%) bị che ngẫu nhiên bằng ký hiệu `[MASK]`, và mô hình Transformer Encoder phải sử dụng ngữ cảnh hai chiều (bidirectional context) của các từ xung quanh để dự đoán chính xác từ ban đầu.

#### 3. Tại sao cần Tokenizer?
Mạng nơ-ron chỉ có thể xử lý các giá trị số (vectors/tensors), không thể xử lý chuỗi ký tự thô. Tokenizer đóng vai trò chuyển hóa văn bản dạng chuỗi thành chuỗi các số nguyên định danh (token IDs) nằm trong bảng từ vựng xác định trước.

#### 4. Subword token giải quyết vấn đề gì?
Subword tokenization (như WordPiece, BPE) giải quyết triệt để vấn đề từ ngoài bảng từ vựng (Out-Of-Vocabulary - OOV) và bùng nổ kích thước từ điển. Bằng cách chia nhỏ các từ hiếm/từ phức thành các gốc từ và tiền tố/hậu tố thông dụng, tokenizer có thể biểu diễn được mọi từ vựng mới bằng một bảng từ điển nhỏ gọn (~30k tokens).

#### 5. Token `[CLS]` thường được sử dụng như thế nào?
`[CLS]` (Classification Token) được chèn vào vị trí đầu tiên của chuỗi đầu vào. Nhờ cơ chế Self-Attention kết nối mọi vị trí với nhau, trạng thái ẩn đầu ra tại vị trí `[CLS]` gom góp và tổng hợp thông tin ngữ cảnh của toàn bộ câu, đóng vai trò như một vector biểu diễn tổng quát của cả sequence để đưa vào Linear Classifier.

#### 6. Attention Mask dùng để làm gì?
Khi nạp các câu có độ dài khác nhau vào một batch, ta phải chèn thêm các token đệm `[PAD]` để chuỗi có độ dài bằng nhau. Attention Mask (chứa các giá trị 1 cho token thật và 0 cho token PAD) báo cho cơ chế Self-Attention biết không được tính trọng số chú ý tới các vị trí PAD, tránh để padding làm biến dạng ý nghĩa câu.

#### 7. Frozen Encoder nghĩa là gì?
Frozen Encoder là việc đóng băng (giữ nguyên, không tính gradient và không cập nhật) toàn bộ các tham số trọng số của các tầng Transformer Encoder trong quá trình huấn luyện, chỉ cho phép huấn luyện duy nhất các tầng phân loại mới thêm vào ở trên cùng (classification head).

#### 8. Fine-tuning thay đổi những tham số nào?
Trong Full Fine-Tuning, **toàn bộ các tham số** của mô hình (bao gồm tất cả các ma trận Self-Attention $W_Q, W_K, W_V, W_O$, các lớp Feed-Forward, LayerNorm và Classification Head) đều được cập nhật qua lan truyền ngược.

#### 9. Vì sao Pretrained Model có thể hữu ích khi downstream dataset nhỏ?
Vì phần lớn tri thức về ngôn ngữ, cú pháp và nghĩa của từ đã được tích lũy sẵn trong hàng triệu tham số từ giai đoạn pretraining. Khi dữ liệu downstream nhỏ, mô hình không phải học lại từ vựng từ con số 0 mà chỉ cần học ánh xạ nhẹ giữa đặc trưng đã có với nhãn mục tiêu, giúp tránh overfitting.

#### 10. Pretrained Model có đảm bảo không bị domain shift không?
**Không.** Pretrained model chỉ học tốt phân phối văn bản mà nó đã tiếp xúc trong pretraining (thường là báo chí, Wikipedia). Khi áp dụng vào các miền dữ liệu chuyên ngành khác biệt sâu sắc (như văn bản pháp lý, y khoa, mật mã), hiện tượng Domain Shift vẫn sẽ xảy ra và làm giảm hiệu năng đáng kể.

#### 11. Vì sao tăng sequence length làm chi phí attention tăng?
Bởi vì cơ chế Self-Attention tính toán sự tương quan giữa từng cặp token với nhau, tạo ra ma trận Attention có kích thước $T \times T$. Do đó, cả độ phức tạp tính toán (FLOPs) lẫn dung lượng bộ nhớ (Memory footprint) đều tăng theo hàm bậc hai: **$O(T^2)$**.

#### 12. Confidence cao có đảm bảo prediction đúng không?
**Không.** Do hàm Softmax có xu hướng kéo dãn chênh lệch logit thành các xác suất tiệm cận 1 (hiện tượng miscalibration/overconfidence), mô hình có thể dự đoán sai hoàn toàn nhưng vẫn đưa ra mức độ tự tin trên 95% (như đã chứng minh trong các ca lỗi mỉa mai và cấu trúc câu tương phản ở `error_analysis.md`).

---

## 7. Cầu Nối Tới LAB 09 (Bridge to LAB 09 — Mục 30)

LAB 08 khép lại với bài toán **Sequence Classification** (gán một nhãn phân loại duy nhất cho toàn bộ chuỗi văn bản thông qua vector đại diện `[CLS]`). 

Tuy nhiên, trong nhiều ứng dụng thực tiễn của NLP, chúng ta cần xác định chi tiết từng thực thể xuất hiện trong câu: từ nào là tên người (`PERSON`), tổ chức (`ORGANIZATION`), địa điểm (`LOCATION`), hay thực thể y sinh:
```text
"Barack Obama visited Paris."
       ↓                 ↓
    PERSON           LOCATION
```
Thay vì chỉ dừng lại ở phân loại mức độ câu, **LAB 09** sẽ chuyển dịch sang bài toán:
$$\text{Sequence Classification} \longrightarrow \text{Token Classification (Named Entity Recognition - NER) / Information Extraction}$$
bằng cách khai thác trực tiếp vector biểu diễn đầu ra của **tất cả các vị trí token** thay vì chỉ dùng một mình token `[CLS]`.

