# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603745
- **Họ và tên:** Nguyễn Nhật Minh
- **SEED** (in ở ô đầu notebook): 8853

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ✅  |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅  |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | ✅  |

## 2. Kết quả chính

| Phương án                       | F1 (toàn bộ test) | F1 (SUB 40) | Độ trễ p95 (ms/tin) | Chi phí (USD/1k tin) |
| ------------------------------- | ----------------- | ----------- | ------------------- | -------------------- |
| 1. BoW + LogReg (tự code)       | 0.912             | 0.889       | 0.07 ms             | $0.000               |
| 2. BoW + MLP (deep learning)    | 0.970             | 0.974       | 0.52 ms             | $0.000               |
| 3a. Encoder (MiniLM) + kNN      | 0.919             | 1.000       | 63.90 ms            | $0.000               |
| 3b. Encoder (MiniLM) + LogReg   | 0.977             | 1.000       | 37.89 ms            | $0.000               |
| 3c. Encoder (MiniLM) + MLP head | 0.971             | 1.000       | 39.01 ms            | $0.000               |
| 4a. LLM nhỏ zero-shot (Gemini)  | 0.687             | 0.687       | 1126.05 ms          | $0.016               |
| 4b. LLM lớn zero-shot (Gemini)  | 0.000             | 0.000       | 0.00 ms             | $0.035               |
| 4c. LLM nhỏ few-shot (Gemini)   | 0.621             | 0.621       | 1862.36 ms          | $0.053               |

## 3. Dự đoán và thực tế

**1. Kết quả trên bộ test khó**

Em dự đoán accuracy của BoW + LogReg đạt 65–75%, BoW + MLP đạt 70–80% và Encoder + LogReg đạt 75–85%. Thực tế, kết quả lần lượt chỉ là **54%, 56% và 58%**.

Em đã đánh giá quá cao khả năng xử lý tin nhắn khó của các mô hình. Những tin cố tình viết sai, thay đổi ký tự hoặc giả mạo khiến mô hình khó nhận biết hơn so với dữ liệu đã học.

**2. So sánh các phiên bản LLM**

Em dự đoán LLM lớn có F1 cao hơn LLM nhỏ khoảng 0,02–0,05 và chi phí gấp khoảng 10 lần. Tuy nhiên, LLM lớn trả về kết quả không hợp lệ ở cả **40/40 lần**, nên F1 bằng **0**; LLM nhỏ đạt **0,687**.

Nguyên nhân có thể liên quan đến cấu hình API hoặc giới hạn số token đầu ra quá thấp. Em nhận ra rằng mô hình mạnh hơn vẫn có thể cho kết quả kém nếu cấu hình chưa phù hợp.

**3. Tác dụng của few-shot**

Em dự đoán thêm 4 ví dụ sẽ giúp F1 của LLM nhỏ tăng 0,01–0,03. Thực tế, F1 lại giảm từ **0,687 xuống 0,621**, còn chi phí tăng từ **0,00062 USD lên 0,00211 USD**.

Có thể các ví dụ được chọn chưa phù hợp và làm mô hình dự đoán kém hơn. Vì vậy, thêm ví dụ không phải lúc nào cũng giúp kết quả tốt hơn.

## 4. Phân tích 50 case khó

**Mô hình giảm nhiều nhất:** BoW + LogReg, với F1 giảm từ **0,889 xuống 0,378**. Hai nhóm khó nhất là **né từ khóa** và **thao túng, lừa đảo**.

**Hai trường hợp dự đoán sai:**

- **Case 30:** Spam bị đoán thành ham vì từ `FREE` được viết cách ra thành `F R E E`.
- **Case 1:** Ham bị đoán thành spam vì có các từ như `WON` và `Call`, dù chỉ là tin nhắn giữa bạn bè.

Em thấy mô hình còn phụ thuộc nhiều vào từ khóa và chưa hiểu rõ ngữ cảnh.

## 5. Thử thách cá nhân

**Đề (SEED = 8853):** Thử nghiệm kNN với các giá trị k khác nhau và so sánh F1.

**Cách làm:**

```python
from sklearn.metrics import f1_score
for k in [1, 5, 25]:
    y_pred = knn_predict(E_tr, y_tr, E_te, k=k)
    f1 = f1_score(y_te, y_pred)
    print("k =", k, "F1 =", round(f1, 4))
```

**Kết quả:**

| k   | F1     |
| --- | ------ |
| 1   | 0.8973 |
| 5   | 0.9189 |
| 25  | 0.8796 |

**Giải thích:** Trong ba giá trị đã thử, k = 5 cho F1 cao nhất.

- **k quá nhỏ:** mô hình chỉ dựa vào ít tin gần nhất nên dễ đoán sai nếu gặp tin nhiễu hoặc khác thường.
- **k quá lớn:** mô hình lấy thêm những tin ít giống hơn để bỏ phiếu.

Với kết quả này, em chọn **k = 5** vì cân bằng tốt hơn giữa hai trường hợp trên.

## 6. Phương án cho sản phẩm thật

**Mô hình được chọn:** Encoder (MiniLM) + LogReg — F1 = **0,977**, p95 = **37,89 ms**

Các phương án BoW + LogReg (0,07 ms) và BoW + MLP (0,52 ms) cũng đáp ứng yêu cầu độ trễ nhưng F1 thấp hơn.

**Đánh đổi:** Nếu dùng LLM cho 10 triệu tin/ngày, theo giá và số token đề bài cho, chi phí ước tính khoảng **3.150 USD/tháng** — quá cao so với các phương án truyền thống có chi phí gần như bằng 0. Vì vậy, Encoder + LogReg là lựa chọn tối ưu giữa độ chính xác, tốc độ và chi phí.

## 7. AI log và điều đã học

**AI log:**

Em hỏi AI về các thuật toán và câu lệnh chưa biết để hiểu code. Sau khi đọc phần giải thích, em tự áp dụng vào bài làm.

**Điều em học được:**

Em hiểu rằng mô hình đạt điểm cao trên tập test thường vẫn có thể đoán sai tin nhắn ngụy trang. Kết hợp mô hình nhỏ với LLM có thể giúp cân bằng tốc độ, chi phí và chất lượng.
