# Phân loại ý định người dùng trong miền ngân hàng

## 1. Giới thiệu

Dự án xây dựng hệ thống phân loại ý định người dùng (*Intent Classification*) cho các câu hỏi và yêu cầu thuộc miền dịch vụ ngân hàng. Hệ thống nhận đầu vào là câu tiếng Việt và dự đoán nhãn `intent` phù hợp, hỗ trợ xây dựng thành phần hiểu ngôn ngữ tự nhiên cho chatbot hoặc trợ lý ảo.

Ngoài các intent thuộc miền ngân hàng, dự án có lớp `out_of_scope` / `unknown` để nhận biết những câu nằm ngoài phạm vi ứng dụng, thay vì ép chúng vào một intent ngân hàng bất kỳ.

## 2. Mục tiêu

- Phân loại câu người dùng vào intent tương ứng trong miền ngân hàng.
- Xử lý cách diễn đạt đa dạng, chẳng hạn câu ngắn/dài, lịch sự/thân mật, viết tắt hoặc có lỗi chính tả.
- Nhận biết câu ngoài phạm vi ngân hàng.
- Chuẩn bị và sử dụng dữ liệu tiếng Anh – tiếng Việt để phục vụ huấn luyện và đánh giá mô hình.

## 3. Dataset

Dữ liệu của dự án được xây dựng từ tập **CLINC150**, sau khi chọn ra 20 intent được sử dụng trong phạm vi nghiên cứu và chuẩn bị dữ liệu tiếng Việt tương ứng.

- **Dataset gốc CLINC150:** [Thay bằng đường dẫn nguồn CLINC150 mà nhóm sử dụng](https://github.com/clinc/oos-eval/)
- **Dữ liệu tiếng Anh đã chọn:** `subset20_en.json`
- **Dữ liệu tiếng Việt đã dịch/bản địa hoá:** `subset20_vi.json`


## 4. Cấu trúc thư mục

```text
project/
├── data/
│   ├── raw/                     
│   └── processed/
│       ├── subset20_en.json     # Dữ liệu tiếng Anh đã chọn 20 intent
│       └── subset20_vi.json     # Dữ liệu tiếng Việt cuối cùng
├── src/                         # Mã nguồn xử lý dữ liệu và mô hình 
├── requirements.txt             # Các thư viện cần thiết 
└── README.md
```

Các file trong `batch_vi/` chỉ là dữ liệu trung gian được lưu trong quá trình dịch/copy từng đợt. Nếu dữ liệu cuối cùng đã được tổng hợp đầy đủ trong `subset20_vi.json`, nhóm không cần đưa các file batch này lên GitHub.

## 5. Mô tả dữ liệu

| Tệp | Nội dung |
|---|---|
| `subset20_en.json` | Các ví dụ tiếng Anh thuộc 20 intent được chọn từ CLINC150. |
| `subset20_vi.json` | Bản dịch/bản địa hoá tiếng Việt tương ứng với dữ liệu tiếng Anh. |

Hai tệp nên giữ cấu trúc và quy ước nhãn nhất quán để thuận tiện cho việc đối chiếu và thực nghiệm đa ngôn ngữ. Nếu tệp JSON đã có trường `label`, không cần tạo thêm `label_mapping.json`, miễn là tên nhãn được dùng nhất quán.

## 6. Quy trình chuẩn bị dữ liệu

1. Tải CLINC150 từ nguồn được ghi ở trên.
2. Chọn 20 intent thuộc phạm vi dự án.
3. Trích xuất các ví dụ tiếng Anh thành `subset20_en.json`.
4. Dịch và bản địa hoá các câu sang tiếng Việt.
5. Giữ nhất quán nhãn intent giữa hai ngôn ngữ.
6. Kiểm tra chất lượng bản dịch và bảo đảm dữ liệu không bị thiếu hoặc lệch nhãn.
7. Lưu bản tiếng Việt hoàn chỉnh vào `subset20_vi.json`.

## 7. Xử lý Out of Scope (OOS)

Lớp `out_of_scope` / `unknown` dùng cho câu không thuộc miền ngân hàng. Ví dụ, một câu hỏi về thể thao hoặc giải trí không nên bị gán nhầm vào một intent ngân hàng.

Việc thay tên riêng hoặc bản địa hoá ví dụ cần tuân theo quy tắc thống nhất của nhóm. Không nên thay đổi nội dung theo cách làm biến đổi ý nghĩa hoặc khiến nhãn intent không còn chính xác.

## 8. Định dạng dữ liệu

Dữ liệu được lưu ở định dạng JSON. Tên trường cụ thể cần tuân theo cấu trúc thực tế của các tệp trong dự án.

Ví dụ minh hoạ:

```json
[
  {
    "text": "Tôi muốn kiểm tra số dư tài khoản.",
    "label": "balance"
  }
]
```

Ví dụ trên chỉ minh hoạ cách trình bày; hãy kiểm tra `subset20_en.json` và `subset20_vi.json` để xác nhận tên trường và nhãn thực tế.

## 9. Sử dụng dữ liệu bằng Python

```python
import json

with open(
    "data/processed/subset20_vi.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

print("Số lượng mẫu:", len(data))
print(data[0])
```

## 10. Phạm vi hiện tại

- **Miền ứng dụng:** Ngân hàng.
- **Số intent được chọn:** 20 intent, cộng với lớp OOS/unknown theo cách gán nhãn của nhóm.
- **Ngôn ngữ dữ liệu:** Tiếng Anh và tiếng Việt.
- **Dữ liệu trung gian:** Các file dùng khi dịch không bắt buộc đưa lên repository.

## 11. Ghi chú

README này mô tả cấu trúc dự kiến của repository. Trước khi công bố dự án, hãy cập nhật tên thư mục, tên trường JSON, số lượng mẫu và đường dẫn dataset để khớp với các tệp thực tế.
