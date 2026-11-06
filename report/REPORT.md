# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.12.7 / 2.15.0 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510914 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** .Vì: thứ nhất, trường "task" là "image_classification" thể hiện mô hình đang làm nhiệm vụ phân loại ảnh. Thứ hai, không có thông tin bounding box.
3. Danh sách tên lớp (class list) không phải do model tự nghĩ ra, mà được xác định bởi bộ dữ liệu và hệ thống taxonomy dùng trong quá trình huấn luyện. Danh sách này thường được lưu hoặc liên kết với checkpoint của model.
4. Cần ghi cả ba vì `class_id` là Mã định danh của lớp, `class_name`là tên có ý nghĩa của lớp và `taxonomy_name`là hệ thống phân loại mà lớp thuộc về. Ba trường này giúp đảm bảo dữ liệu AI có thể được định danh, diễn giải và truy xuất đúng hệ thống phân loại.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ bức ảnh (Primary class / Dominant class) để người gán nhãn chọn thống nhất một lớp.
6. Vì score chỉ thể hiện mức độ tin cậy hoặc điểm số dự đoán của model, chứ không chứng minh dự đoán đó là nhãn đúng của ảnh.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.61093 ; `bbox_xyxy`=[0.12 , 263.18 , 61.55 , 310.78] ; `bbox_width`=61.44 ; `bbox_height`=47.59
2. Lớp `person`, score =`0.61093`, góc trên bên trái khoảng pixel (0.12,263.18), góc dưới bên phải khoảng pixel (61.55,310.78), chiều rộng 61.44 pixel, chiều cao 47.59 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Lớp `person`, score =`0.61093`
Ngưỡng 0.20: 1 prediction.
Ngưỡng 0.60: 1 prediction.
4. Khi hạ ngưỡng, model giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán có độ tin cậy thấp.Reviewer cần:
    - Kiểm tra dự đoán sai 
    - Kiểm tra nhãn lớp
    - Kiểm tra box trùng lặp
    - Kiểm tra vị trí box
    - Xử lý đối tượng khó nhận diện.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này): 
    - Box phải bao phủ toàn bộ phần đối tượng nhìn thấy trong ảnh.
    - Box phải sát với đường biên của đối tượng, hạn chế tối đa phần nền không liên quan.
    - Không được để box bị lệch, thiếu phần đối tượng hoặc bao gồm vật thể khác.
    - Nếu đối tượng bị che khuất, chỉ bao quanh phần nhìn thấy, trừ khi guideline của dataset quy định khác.
    - Box phải nằm trong giới hạn kích thước ảnh.
    - Tọa độ phải được lưu thống nhất theo định dạng xyxy: [x1, y1, x2, y2], trong đó góc trên bên trái đứng trước góc dưới bên phải.
    - Mỗi bounding box phải gắn đúng class tương ứng với đối tượng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Trên ảnh `kitchen`có vật thể `oven` bị cắt mép bản đồ gần biên ảnh: tôi chọn **guideline** nếu đã có rule (ví dụ “còn ≥ 50% object trong khung thì vẫn gán”); nếu không chắc object là gì vì che khuất nặng thì **escalation** hỏi Lab Coach, không tự bịa lớp.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl; `score`=0.735743; số điểm polygon=`polygon_point_count`=67; 3–5 điểm đầu của `polygon_xy`=[[53.0,344.0],[52.0,345.0],[50.0,
345.0],[49.0,346.0],[48.0,346.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon bám theo biên thật của object (cong/mép), còn box chỉ là hình chữ nhật bao ngoài nên mất chi tiết hình dạng.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? 
`instance_id` là mã định danh cho một vật thể cụ thể trong ảnh(vd. hai `bowl` khác nhau là `kitchen-001` và `kitchen-002`). Nó không phải mã lớp (`class_id`) và không phải ID theo dõi theo thời gian (tracking).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Người gán nhãn phải vẽ polygon bám sát đường biên thực tế của từng vật thể.
    - Các điểm polygon phải nằm trên hoặc sát đường biên vật thể.
    - Không bao gồm phần nền bên ngoài vật thể.
    - Đặt đủ điểm tại các vị trí cong, góc hoặc thay đổi hình dạng.
    - Không vẽ polygon quá đơn giản khiến mất các chi tiết quan trọng.
    - Mỗi vật thể riêng biệt phải có một mask và instance_id riêng.
    - Đảm bảo polygon khép kín và không tự giao cắt.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trên ảnh `kitchen`, ở trên mặt bàn có các `bowl` giao nhau. Tôi chọn **guideline** nếu đã có rule (ví dụ tách biên theo phần giao nhau giữa hai khe nhìn thấy của object); nếu chưa có rule thì chọn **escalation** để hỏi Lab Coach, không tự bịa lớp.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một lớp cho toàn ảnh| Ảnh traffic có thể chứa nhiều phương tiện. Model dự đoán cab với score 0.510914, chưa chắc là nhãn đúng| Quan sát toàn ảnh, chọn một lớp chính theo guideline. | Kiểm tra nhãn có phản ánh đúng chủ đề chính, không phụ thuộc score. |
| Phát hiện vật thể | Mỗi object có một class_id và bounding box [x1,y1,x2,y2] |Vật thể oven bị cắt ở mép trái ảnh. JSON ghi bbox_xyxy = [0.11, 187.54, 195.96, 292.81]| Gán đúng lớp, vẽ box sát vật thể, xử lý phần bị cắt theo guideline. | Kiểm tra class, vị trí box, vật thể bị bỏ sót, box trùng và box sai.|
| Instance segmentation | Mỗi instance có một instance_id, class và mask/polygon riêng. | Hai vật thể bowl trên mặt bàn có vùng box giao nhau (kitchen-003 và kitchen-006), cần kiểm tra ranh giới giữa hai mask. |Vẽ polygon theo đường biên, tách riêng từng cá thể, không lấy phần nền.|  Kiểm tra biên mask, vùng thừa/thiếu và cách xử lý vật thể bị che khuất.|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?
Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng thao tác và báo Lab Coach / GV phụ trách lab.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
