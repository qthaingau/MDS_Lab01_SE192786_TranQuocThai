## Lab 1 — Học có giám sát

Bài thực hành sử dụng bộ dữ liệu California Housing để xây dựng quy trình học máy: tìm hiểu dữ liệu, tiền xử lý, huấn luyện mô hình và đánh giá kết quả. Hai bài toán được thực hiện là dự đoán giá nhà bằng hồi quy và phân loại nhà đắt hoặc giá phải chăng.

Notebook thực hiện bài tập: `lab1_notebook.ipynb`.

### Part 1 — Chuẩn bị môi trường

- Tạo môi trường Python riêng `lab1_env` và cài các thư viện cần thiết.
- Import NumPy, Pandas, Matplotlib, Seaborn và Scikit-learn.
- In phiên bản NumPy, Pandas, Scikit-learn và kiểm tra việc import thư viện thành công.

### Part 2 — Khám phá dữ liệu

- Tải California Housing vào Pandas DataFrame; xem kích thước, kiểu dữ liệu, số giá trị thiếu và thống kê mô tả.
- Trả lời các câu hỏi về số lượng mẫu, số đặc trưng, dữ liệu thiếu và khoảng giá trị của thu nhập trung vị `MedInc`.
- Vẽ bốn biểu đồ: phân phối `MedHouseVal` bằng histogram kết hợp KDE; heatmap tương quan; scatter giữa thu nhập và giá nhà với màu theo tuổi nhà; box plot giá nhà theo ba vùng địa lý quy định trong đề.
- Viết hai câu nhận xét cho mỗi biểu đồ và lưu hình vào `eda_plots.png`.
- Tìm ba đặc trưng có tương quan tuyệt đối cao nhất với giá nhà và giải thích khả năng dự đoán của chúng.

### Part 3 — Tiền xử lý dữ liệu

- Tạo nhóm thu nhập `income_cat`, chia dữ liệu thành 80% huấn luyện và 20% kiểm tra bằng phân tầng theo nhóm thu nhập, với `random_state=42`.
- Xóa cột `income_cat` khỏi hai tập và giải thích lý do dùng cách chia có phân tầng.
- Trình bày ý tưởng tạo đặc trưng, so sánh với gợi ý của đề và bổ sung ba cột: `rooms_per_household`, `bedrooms_ratio`, `population_per_household`.
- Giải thích vai trò dự đoán của các đặc trưng bổ sung.
- Tách biến đầu vào và biến mục tiêu; xây dựng pipeline gồm `SimpleImputer(strategy='median')` và `StandardScaler()`.
- Fit pipeline trên tập huấn luyện, sau đó dùng pipeline đó để biến đổi cả tập huấn luyện và tập kiểm tra.

### Part 4 — Các mô hình hồi quy

- Huấn luyện Linear Regression và SGD Regressor để dự đoán `MedHouseVal`.
- Huấn luyện Polynomial Regression bậc 2; vẽ learning curves của mô hình tuyến tính và đa thức bằng cross-validation 5 phần, rồi phân tích bias và variance.
- Lưu biểu đồ learning curves vào `learning_curves.png`.
- Huấn luyện Ridge và Lasso; dùng `GridSearchCV` với `alpha` trong `[0.01, 0.1, 1, 10, 100]` để tinh chỉnh Ridge.
- Tính RMSE, MAE và R² cho Linear Regression, SGD Regressor, Polynomial Regression, Lasso và Ridge sau tinh chỉnh trên cả hai tập dữ liệu.
- Lập bảng so sánh, nhận xét underfitting hoặc overfitting và giải thích mô hình có kết quả kiểm tra tốt nhất.

### Part 5 — Các mô hình phân loại

- Tạo nhãn nhị phân: nhà đắt khi `MedHouseVal >= 2.5`, nhà giá phải chăng khi `MedHouseVal < 2.5`.
- Phân tích tỷ lệ hai lớp và giải thích vì sao chỉ số Accuracy có thể gây hiểu nhầm khi dữ liệu mất cân bằng.
- Huấn luyện KNN với `k=5` và Gaussian Naive Bayes; đánh giá kết quả phân loại.
- Tìm `k` tốt nhất từ 1 đến 20 bằng cross-validation; vẽ và lưu biểu đồ `knn_accuracy_vs_k.png`.
- Huấn luyện Decision Tree với `max_depth=5`, in cấu trúc cây; huấn luyện Random Forest với 100 cây.
- Vẽ độ quan trọng của các đặc trưng, đối chiếu với EDA và lưu hình vào `feature_importances.png`.
- Huấn luyện SVM với kernel RBF và linear, sau đó so sánh AUC-ROC.
- Lập bảng Accuracy, Precision, Recall và AUC-ROC cho các mô hình; đề xuất mô hình sử dụng thực tế dựa trên hiệu quả dự đoán và chi phí tính toán.

### Part 6 — Báo cáo và giải thích

Viết Executive Summary dài 200–300 từ trong Markdown cell của notebook, gồm năm nội dung:

1. Đặc điểm và thách thức của bộ dữ liệu California Housing.
2. Vai trò của các bước tiền xử lý đối với mô hình.
3. Mô hình hồi quy tốt nhất và ý nghĩa sai số dự đoán theo giá trị tiền tệ.
4. Mô hình phân loại tốt nhất và mức độ phù hợp với bài toán.
5. Hai đề xuất cải thiện: bổ sung dữ liệu mới cùng các đặc trưng liên quan; mở rộng việc tinh chỉnh tham số và thử nghiệm mô hình ensemble bằng cross-validation.

Toàn bộ code, kết quả chạy, câu trả lời và nhận xét của Part 1–6 nằm trong notebook. README này là bản tóm tắt nội dung từng phần.
