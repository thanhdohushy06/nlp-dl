# Calculation & Theory Exercises
**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Lab 08:** Pretrained Transformers and BERT Fine-Tuning  
**Sinh viên thực hiện:** Đỗ Khắc Đức Thành - MSSV: 24001697  

---

## 1. Calculation Exercise 1 — Masked Language Modeling (MLM) Loss

### 1.1. Bối cảnh bài toán
Cho câu gốc:
> *"The cat eats fish."*

Sau khi áp dụng cơ chế Masking trong bài toán Masked Language Modeling (MLM):
> *"The cat [MASK] fish."*

Mô hình BERT cần dự đoán phân phối xác suất có điều kiện trên toàn bộ từ vựng $V$:
$$P(w_t \mid w_{\neq t})$$
trong đó $w_t$ là token tại vị trí bị mask (ở đây là vị trí thứ 3 trong chuỗi, tương ứng với từ *"eats"*).

Hàm mất mát Cross-Entropy Loss được định nghĩa như sau:
$$L = - \sum_{i \in M} \log P(y_i \mid x)$$
Trong đó:
- $M$: tập các vị trí token bị che (mask positions). Ở ví dụ này chỉ có 1 vị trí: $|M| = 1$.
- $y_i$: token nhãn thực tế (ground truth token). Ở đây $y_i = \text{"eats"}$.
- $P(y_i \mid x)$: xác suất mô hình dự đoán token đúng $y_i$ dưới điều kiện ngữ cảnh $x$.

### 1.2. Dữ liệu dự đoán của mô hình
Phân phối xác suất dự đoán của mô hình trên tập từ vựng rút gọn:
- $P(\text{"cat"} \mid x) = 0.10$
- $P(\text{"dog"} \mid x) = 0.05$
- $P(\text{"eats"} \mid x) = 0.70$
- $P(\text{"fish"} \mid x) = 0.15$

Token thực tế (Ground truth): **"eats"** ($y = \text{"eats"}$).  
Xác suất tương ứng: $P(y \mid x) = 0.70$.

### 1.3. Tính toán Loss cơ sở (Baseline Loss)
Trong chuẩn tính toán của PyTorch (`torch.nn.CrossEntropyLoss`) và Machine Learning thông thường, $\log$ là logarit tự nhiên (cơ số $e$, ký hiệu $\ln$):

$$L = - \ln(0.70) \approx -(-0.3566749) \approx 0.3567$$

*(Ghi chú thêm: Nếu tính theo logarit cơ số 2: $L_2 = -\log_2(0.70) \approx 0.5146$ bits; nếu tính theo cơ số 10: $L_{10} = -\log_{10}(0.70) \approx 0.1549$).*

---

### 1.4. Trả lời các câu hỏi phân tích

#### Câu hỏi 1: Nếu model dự đoán eats = 0.95, loss thay đổi như thế nào?
- **Tính toán:**
  $$L_{0.95} = - \ln(0.95) \approx -(-0.051293) \approx 0.0513$$
- **Nhận xét sự thay đổi:**
  - Độ chênh lệch: $\Delta L = 0.0513 - 0.3567 = -0.3054$ (Loss giảm khoảng **85.6%** so với ban đầu).
  - Giá trị loss tiệm cận rất gần về $0$.
  - **Ý nghĩa:** Khi mô hình dự đoán từ đúng với độ tự tin rất cao ($95\%$), hàm mất mát ghi nhận rằng mô hình đã "hiểu" ngữ cảnh rất tốt và độ bất định (uncertainty) cực nhỏ. Do đó lượng cập nhật trọng số (gradient) sẽ nhỏ đi, giữ cho mô hình ổn định.

#### Câu hỏi 2: Nếu model dự đoán eats = 0.01, loss thay đổi như thế nào?
- **Tính toán:**
  $$L_{0.01} = - \ln(0.01) \approx -(-4.60517) \approx 4.6052$$
- **Nhận xét sự thay đổi:**
  - Độ chênh lệch: $\Delta L = 4.6052 - 0.3567 = +4.2485$ (Loss tăng vọt gấp **12.91 lần** so với mức ban đầu).
  - **Ý nghĩa:** Vì hàm $-\ln(p) \to +\infty$ khi $p \to 0^+$, Cross-Entropy loss phạt cực kỳ nặng các trường hợp mô hình gán xác suất thấp hoặc bỏ qua từ đúng. Điều này sinh ra một gradient rất lớn, buộc bộ tối ưu hóa (optimizer như AdamW) phải cập nhật mạnh các trọng số để sửa chữa sai lệch nghiêm trọng này.

#### Câu hỏi 3: Vì sao Cross-Entropy loss phù hợp với bài toán Masked Language Modeling (MLM)?
1. **Bản chất phân loại đa lớp (Multi-class classification):**  
   Tại mỗi vị trí `[MASK]`, mô hình phải chọn 1 token đúng trong toàn bộ không gian từ vựng $V$ (thường có kích thước rất lớn, ví dụ $|V| \approx 30,000$ đối với BERT WordPiece). Phân phối đầu ra qua hàm Softmax tạo thành một phân phối xác suất rời rạc $\hat{y} \in \mathbb{R}^{|V|}$ thỏa mãn $\sum_{j=1}^{|V|} \hat{y}_j = 1$. Cross-Entropy chính là khoảng cách Kullback-Leibler (KL divergence) giữa phân phối one-hot nhãn thực $y$ và phân phối dự đoán $\hat{y}$.
2. **Nguyên lý cực đại hợp lý (Maximum Likelihood Estimation - MLE):**  
   Tối thiểu hóa hàm Cross-Entropy tương đương trực tiếp với việc tối đa hóa xác suất hợp lý logarit (Log-Likelihood) của các token đúng dưới điều kiện ngữ cảnh hai chiều xung quanh:
   $$\max_{\theta} \sum_{i \in M} \log P_\theta(y_i \mid x)$$
3. **Đặc tính Gradient tối ưu mượt mà khi kết hợp cùng Softmax:**  
   Khi ghép tầng Softmax với Cross-Entropy loss, đạo hàm riêng theo logit đầu vào $z_k$ có dạng đại số cực kỳ thanh lịch và ổn định số học:
   $$\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$$
   - Khi mô hình dự đoán sai hoàn toàn ($\hat{y}_k \approx 0$ trong khi $y_k = 1$), độ dốc gradient đạt mức lớn nhất ($\approx -1$), thúc đẩy mô hình học rất nhanh và không bị hiện tượng triệt tiêu gradient (vanishing gradient).
   - Khi mô hình dự đoán chính xác ($\hat{y}_k \approx 1, y_k = 1$), gradient tiệm cận 0, tránh làm xáo trộn các trọng số đã học tốt.

---

## 2. Calculation Exercise 2 — Tokenization Analysis

### 2.1. Thiết lập thực nghiệm
- Câu đầu vào: `"The cats are playing."`
- Tokenizer sử dụng: `bert-base-uncased` (Chuẩn WordPiece Tokenizer).

### 2.2. Kết quả Tokenization
Dưới đây là các giá trị thu được khi chạy qua tokenizer chuẩn:

1. **Token sequence:**
   ```python
   ['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]']
   ```
   *(Lưu ý đối với tokenizer WordPiece của bert-base-uncased: Từ "cats" đã có sẵn trong từ điển vocab.txt ở ID 8870; từ "playing" có ID 2652. Cả chuỗi được chuyển thành chữ thường uncased và tách dấu chấm câu '.' riêng biệt).*

2. **Token IDs:**
   ```python
   [101, 1996, 8870, 2024, 2652, 1012, 102]
   ```
   Bảng chi tiết từng token:
   | Vị trí (Index) | Token String | Token ID | Loại Token |
   |:---:|:---:|:---:|:---|
   | 0 | `[CLS]` | 101 | Special Token (Đánh dấu bắt đầu chuỗi) |
   | 1 | `the` | 1996 | Word / Subword |
   | 2 | `cats` | 8870 | Word / Subword |
   | 3 | `are` | 2024 | Word / Subword |
   | 4 | `playing` | 2652 | Word / Subword |
   | 5 | `.` | 1012 | Punctuation |
   | 6 | `[SEP]` | 102 | Special Token (Đánh dấu kết thúc chuỗi) |

3. **Special tokens:**
   - `[CLS]` (ID: 101): Token đặc biệt đặt ở vị trí đầu câu, vector biểu diễn tương ứng của nó thường được sử dụng làm đại diện ngữ cảnh của toàn bộ câu cho các downstream tasks (như phân loại văn bản).
   - `[SEP]` (ID: 102): Token đặc biệt phân tách câu/đoạn văn hoặc báo hiệu kết thúc sequence.

4. **Attention Mask:**
   ```python
   [1, 1, 1, 1, 1, 1, 1]
   ```
   Tất cả 7 vị trí đều là $1$ vì đây là các token hợp lệ thực sự của câu (không có token padding `[PAD]` nào).
   *(Nếu chuỗi được đệm `max_length = 10` với 3 token `[PAD]` (ID 0) ở cuối, attention mask sẽ là `[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]`).*

5. **So sánh số lượng (Quantity Comparison):**
   - **Số lượng từ (Words):** $4$ từ tiếng Anh (`"The"`, `"cats"`, `"are"`, `"playing"`) — hoặc $5$ thành phần từ vựng nếu tính dấu chấm câu `.` là một từ riêng lẻ.
   - **Số lượng tokens (Tokens - không tính special tokens):** $5$ tokens (`'the'`, `'cats'`, `'are'`, `'playing'`, `'.'`).
   - **Số lượng subword tokens (Subword tokens - tính cả special tokens):** $7$ tokens (`'[CLS]'`, `'the'`, `'cats'`, `'are'`, `'playing'`, `'.'`, `'[SEP]'`).

**Nhận xét:**
Subword tokenization (như WordPiece hay BPE) giúp cân bằng hoàn hảo giữa kích thước từ điển (tránh Out-Of-Vocabulary - OOV) và độ dài chuỗi token. Những từ thông dụng giữ nguyên dưới dạng một token, còn những từ hiếm hoặc từ phức sẽ được băm nhỏ thành các tiền tố/hậu tố (ví dụ: `unbelievable` $\to$ `un`, `##belie`, `##vable`).

