# Reflection Report: Tại Sao Trong Thực Tế Không Bắt Đầu Bằng Việc Train Transformer Từ Đầu?

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Lab 08:** Pretrained Transformers and BERT Fine-Tuning  
**Sinh viên thực hiện:** Đỗ Khắc Đức Thành - MSSV: 24001697  

---

Sau LAB 07, chúng ta đã tự tay xây dựng hoàn chỉnh kiến trúc Transformer Encoder từ các khối cơ bản: Multi-Head Self-Attention, Positional Encoding, Feed-Forward Network, Residual Connections và Layer Normalization. Tuy nhiên, việc nắm vững cách lập trình một Transformer không đồng nghĩa với việc chúng ta nên huấn luyện nó từ con số 0 (train from scratch) khi giải quyết các bài toán NLP thực tế. Qua thực nghiệm tại LAB 08, sự chuyển dịch từ việc tự train sang tận dụng Pretrained Language Models (như BERT, RoBERTa) xuất phát từ 6 yếu tố cốt lõi sau:

### 1. Nhu cầu về Dữ liệu (Data Hunger & Sample Complexity)
Kiến trúc Transformer không chứa các giả định tiên nghiệm (inductive bias) mạnh mẽ về tính tuần tự hay tính cục bộ không gian như RNN hay CNN. Nó học mối quan hệ giữa các từ hoàn toàn dựa trên cơ chế Self-Attention tự do. Do đó, để một Transformer "học" được ngữ pháp, từ vựng và thế giới quan từ đầu, nó đòi hỏi hàng trăm gigabyte đến hàng terabyte văn bản (hàng tỷ tokens). Trong thực tế, các bài toán ứng dụng (như phân loại đánh giá phim IMDb, phân tích cảm xúc khách hàng, trích xuất thực thể y tế) chỉ sở hữu vài nghìn đến vài chục nghìn mẫu gán nhãn. Nếu train from scratch trên tập dữ liệu nhỏ này, mô hình chắc chắn rơi vào tình trạng quá khớp trầm trọng (severe overfitting) và không thể tổng quát hóa.

### 2. Sức mạnh của Biểu diễn Tiền Huấn luyện (Pretrained Contextual Representations)
Các mô hình Pretrained Transformer được huấn luyện bằng các tác vụ tự giám sát (self-supervised) thông minh như Masked Language Modeling (MLM). Nhờ dự đoán hàng triệu từ bị ẩn trong ngữ cảnh hai chiều phong phú, mô hình đã nén toàn bộ tri thức cú pháp, ngữ nghĩa, từ đồng nghĩa và quan hệ phụ thuộc dài (long-range dependencies) vào các vector không gian ẩn. Biểu diễn của token đặc biệt `[CLS]` hoặc các contextual embeddings đầu ra đã là những đặc trưng vô cùng chất lượng, tách bạch rõ ràng cấu trúc ngữ nghĩa trước khi bước vào bất kỳ bài toán cụ thể nào.

### 3. Chi phí Tính toán và Tài nguyên (Computational Cost & Green AI)
Pre-training một mô hình như BERT-base đòi hỏi cụm hàng chục hoặc hàng trăm GPU/TPU mạnh mẽ (như NVIDIA V100/A100) chạy liên tục trong nhiều ngày hoặc nhiều tuần, tiêu tốn hàng chục nghìn USD tiền điện toán và phát thải lượng carbon đáng kể. Đối với các kỹ sư NLP, viện nghiên cứu hoặc doanh nghiệp vừa và nhỏ, việc bỏ ra chi phí khổng lồ như vậy cho mỗi dự án đơn lẻ là điều bất khả thi và lãng phí tài nguyên xã hội. Việc sử dụng lại checkpoint pretrained giúp dân chủ hóa AI: bất kỳ ai có một GPU phổ thông hoặc môi trường miễn phí như Google Colab cũng có thể xây dựng ứng dụng NLP đỉnh cao.

### 4. Tính Chuyên biệt của Bài toán Ứng dụng (Downstream Tasks)
Trong thực tế, mục tiêu cuối cùng của chúng ta là giải quyết các bài toán downstream cụ thể (text classification, QA, NER, sentiment analysis). Bài toán downstream thường chỉ yêu cầu mô hình phân biệt một số lượng nhãn hữu hạn dựa trên tín hiệu ngữ nghĩa đã có. Tận dụng pretrained model cho phép tách biệt rõ ràng giữa "học ngôn ngữ nói chung" (pretraining) và "học tác vụ cụ thể" (task learning).

### 5. Sự Hiệu Quả và Linh Hoạt của Fine-Tuning
Fine-tuning bắt đầu từ một điểm khởi tạo trọng số lý tưởng $\theta_{\text{pretrained}}$. Toàn bộ quá trình fine-tuning chỉ cần một vài epoch (thường từ 2 đến 4 epochs) với tốc độ học nhỏ ($2\times 10^{-5}$ đến $5\times 10^{-5}$), hội tụ cực nhanh chỉ trong vài phút đến vài chục phút. Hơn nữa, ta có thể linh hoạt chọn giữa:
- **Feature Extraction (Frozen Encoder):** Giữ cố định encoder, chỉ huấn luyện classification head nhẹ, cực kỳ tiết kiệm bộ nhớ và thời gian tính toán.
- **Full Fine-Tuning:** Cập nhật toàn bộ tham số, cho phép các attention heads tinh chỉnh thích nghi hoàn hảo với phân phối của bài toán mục tiêu, đem lại hiệu năng vượt trội.

### 6. Nhận thức về Hạn chế và Dịch chuyển Phân phối Miền (Limitations & Domain Shift)
Mặc dù pretrained Transformer rất mạnh, chúng ta không được ngộ nhận rằng mô hình "hiểu mọi thứ". Nếu downstream task có phong cách văn bản, thuật ngữ chuyên ngành (như y khoa PubMed, luật pháp, tài chính) khác biệt quá xa so với văn bản Wikipedia/tin tức thông thường của pretraining data, hiện tượng **Domain Shift** sẽ làm suy giảm hiệu năng nghiêm trọng. Ngoài ra, mô hình còn chịu giới hạn về độ dài chuỗi tối đa (512 tokens trong BERT tiêu chuẩn do độ phức tạp $O(T^2)$ của Self-Attention), nguy cơ quên tri thức cũ (catastrophic forgetting), và xu hướng quá tự tin (overconfidence) ngay cả khi dự đoán sai. 

### Kết luận
Tóm lại, sự kết hợp giữa **Pretraining trên kho ngữ liệu lớn** và **Fine-tuning trên dữ liệu bài toán cụ thể** đã tạo nên bước ngoặt mang tính cách mạng cho NLP hiện đại: tối ưu hóa chi phí, tận dụng tối đa dữ liệu ít ỏi của downstream task, và đạt độ chính xác vượt bậc so với phương pháp train from scratch truyền thống.

