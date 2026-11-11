# Prediction Before Experiment
**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Lab 08:** Pretrained Transformers and BERT Fine-Tuning  
**Sinh viên thực hiện:** Đỗ Khắc Đức Thành - MSSV: 24001697  

---

## Prediction 1: Pretrained Transformer vs. Transformer from Scratch

> **Câu hỏi:** Một pretrained Transformer fine-tuned trên IMDb có khả năng đạt kết quả tốt hơn Transformer nhỏ được train from scratch trên cùng dữ liệu không?

- **Prediction:** **CÓ** (Pretrained Transformer fine-tuned sẽ vượt trội hơn rõ rệt so với Transformer nhỏ train from scratch trên cùng tập dữ liệu).
- **Reason (Lý do):**
  1. *Tri thức tiền huấn luyện (Prior linguistic knowledge):* Pretrained Transformer (như BERT/DistilBERT) đã được huấn luyện tự giám sát (self-supervised) trên khối lượng văn bản khổng lồ (Wikipedia, BookCorpus hàng tỷ từ) thông qua nhiệm vụ Masked Language Modeling (MLM). Mô hình đã học được cấu trúc cú pháp, biểu diễn ngữ nghĩa đa tầng và mối quan hệ ngữ cảnh sâu sắc của từ vựng trước khi nhìn thấy dữ liệu IMDb.
  2. *Hạn chế của Transformer from scratch:* Kiến trúc Transformer hoàn toàn dựa trên cơ chế Self-Attention, không có sẵn inductive bias mạnh như tính bất biến tịnh tiến của CNN hay tính tuần tự cục bộ của RNN. Do đó, Transformer train from scratch cực kỳ "đói dữ liệu" (data-hungry). Khi huấn luyện từ đầu trên tập IMDb với trọng số khởi tạo ngẫu nhiên, mô hình phải học đồng thời biểu diễn ngôn ngữ từ con số 0 lẫn bài toán phân loại nhị phân. Điều này rất dễ dẫn đến việc hội tụ ở nghiệm cực tiểu địa phương nghèo nàn và overfitting nghiêm trọng.
  3. *Điểm khởi đầu tối ưu (Warm initialization):* Fine-tuning bắt đầu từ không gian trọng số $\theta_{\text{pretrained}}$ đã tối ưu cho việc biểu diễn ngôn ngữ, giúp quá trình tối ưu hóa nhanh hơn và đạt ranh giới phân lớp tổng quát hóa tốt hơn nhiều.
- **Expected Evidence (Bằng chứng kỳ vọng khi chạy thực nghiệm):**
  - **Accuracy & F1-score:** Mô hình Pretrained fine-tuned dự kiến đạt Accuracy $\ge 88\% - 92\%$, trong khi Transformer from scratch với kích thước nhỏ và tài nguyên giới hạn chỉ đạt khoảng $65\% - 75\%$.
  - **Tốc độ hội tụ (Convergence):** Pretrained model giảm loss nhanh chóng và đạt độ chính xác cao chỉ sau 1-3 epochs; trong khi mô hình from scratch loss giảm chậm, dao động nhiều và sớm bão hòa validation accuracy.

---

## Prediction 2: Frozen Pretrained Encoder vs. Full Fine-Tuning

> **Câu hỏi:** Nếu freeze toàn bộ pretrained encoder và chỉ train classification head:
> ```text
> BERT encoder (frozen) ──> classification head (train) ──> prediction
> ```
> Kết quả sẽ:
> - **A.** Tốt hơn fine-tuning toàn bộ
> - **B.** Tương đương
> - **C.** Kém hơn
> - **D.** Không thể dự đoán

- **Lựa chọn:** **C. Kém hơn**
- **Giải thích trước khi chạy (Reasoning):**
  1. *Tính chất của Feature Extraction (Frozen):* Khi đóng băng toàn bộ Encoder, các biểu diễn vector ngữ cảnh $h_{\text{[CLS]}}$ được cố định hoàn toàn theo phân phối ngôn ngữ chung của giai đoạn MLM. Classification head (Linear layer) chỉ có thể học một siêu phẳng phân cách tuyến tính trên không gian biểu diễn cố định này.
  2. *Lợi thế của Task Adaptation (Fine-tuning):* Trong bài toán phân loại cảm xúc (Sentiment Classification), những từ ngữ thể hiện sắc thái tình cảm rõ rệt (ví dụ: *"terrific", "awful", "masterpiece"*), các cấu trúc đảo ngữ, hoặc phủ định kép (*"not bad at all"*) cần được các attention head tập trung trọng số nhiều hơn so với phân phối ngôn ngữ thông thường. Khi mở khóa (unfreeze) để fine-tune toàn bộ, gradient sai số từ hàm mất mát $L_{\text{task}}$ sẽ lan truyền ngược về toàn bộ các tầng Transformer, tinh chỉnh các ma trận Query, Key, Value để "uốn nắn" (adapt) vector biểu diễn $\text{[CLS]}$ phân tách rõ rệt hai cực tính positive/negative.
  3. Do đó, fine-tuning toàn bộ cho phép mô hình thích nghi chuyên sâu với miền bài toán, đem lại Accuracy và F1-score cao hơn đáng kể so với chế độ Frozen.

---

## Prediction 3: Hiệu Quả Khi Dữ Liệu Huấn Luyện Ít (Data Efficiency - 10% Training Data)

> **Câu hỏi:** Nếu chỉ có 10% training data (so sánh giữa các mốc 10%, 25%, 50%, 100%): Pretrained model hay model train from scratch sẽ giảm performance mạnh hơn?  
> *(Yêu cầu: Không trả lời theo cảm tính, phải đưa ra cơ sở từ vai trò của pretraining).*

- **Dự đoán:** **Model train from scratch sẽ giảm performance mạnh hơn rất nhiều.**
- **Cơ sở lý luận từ vai trò của Pretraining (Theoretical Grounding):**
  1. **Khái niệm Sample Complexity và Inductive Bias:**
     Một mô hình Transformer train from scratch sở hữu không gian giả thuyết (hypothesis space) khổng lồ nhưng thiếu inductive bias định hướng. Để thu hẹp không gian giả thuyết về nghiệm tối ưu tổng quát hóa, mô hình đòi hỏi số lượng mẫu huấn luyện cực lớn ($N_{\text{samples}}$ lớn). Khi cắt giảm dữ liệu huấn luyện xuống chỉ còn 10%, số lượng mẫu quá nghèo nàn sẽ khiến mạng nơ-ron nhanh chóng học vẹt (memorization) các đặc trưng ngẫu nhiên (spurious correlations) của tập train, khiến validation/test accuracy sụt giảm nghiêm trọng (hiện tượng overfitting cực độ, hiệu năng có thể tụt về mức đoán ngẫu nhiên $50\% - 60\%$).
  2. **Hiệu ứng Khởi tạo Thuận lợi của Pretraining (Transfer Learning & Data Efficiency):**
     Giai đoạn Pretraining đã hoàn thành phần lớn công việc nặng nhọc nhất: học cấu trúc từ vựng, ngữ pháp, thành ngữ, và sự tương tác từ xa giữa các token. Vector embedding của các từ có cùng sắc thái cảm xúc đã nằm gần nhau trong không gian biểu diễn đa chiều. Vì vậy, khi chuyển giao sang downstream task, bài toán fine-tuning thực chất chỉ là tìm kiếm một ánh xạ phân lớp đơn giản trên không gian đặc trưng đã được định hình trước.
  3. **Độ dốc của đường cong học tập (Learning Curve Slope):**
     Với 10% dữ liệu, Pretrained model vẫn duy trì được hiệu năng cao (ước tính vẫn giữ được $>80\%$ accuracy). Đường cong suy giảm hiệu năng của Pretrained model theo lượng dữ liệu sẽ có độ dốc thoai thoải (tương đối bằng phẳng), trong khi đường cong của model train from scratch sẽ dốc đứng và sụp đổ nhanh chóng khi lượng dữ liệu giảm dần.

