PHẦN 1. GIỚI THIỆU

Bối cảnh nghiên cứu: Tiêu chuẩn hiện tại và Bài toán Vật lýKính thưa các thầy, trong thiết kế hàng không và năng lượng gió, biên dạng cánh (airfoil) chính là "trái tim" của hệ thống khí động học. Một thay đổi rất nhỏ trên đường cong của cánh cũng làm thay đổi cấu trúc dòng chảy, dẫn đến sự thay đổi của 3 đại lượng cốt lõi:Hệ số lực nâng (CL): Quyết định sức tải của máy bay.Hệ số lực cản (CD): Quyết định mức độ tiêu hao năng lượng và hiệu suất bay.Hệ số mô men (CM): Quyết định xu hướng xoay gật (pitch) và tính ổn định dọc của phương tiện.Để tính toán 3 hệ số này, tiêu chuẩn vàng trong ngành kỹ thuật hiện nay là phương pháp Mô phỏng động lực học chất lưu tính toán (CFD). CFD được dùng phổ biến vì nó giải trực tiếp các phương trình vật lý chi phối dòng chảy, cho ta bức tranh cực kỳ chi tiết về trường vận tốc, áp suất và lớp biên.

Hạn chế của CFD: Điểm nghẽn trong khâu Tối ưu hóaTuy nhiên, dưới góc độ của một kỹ sư thiết kế, CFD có một "điểm nghẽn" rất lớn khi áp dụng vào bài toán tối ưu hóa.Chi phí và thời gian tính toán: Giải phương trình Navier-Stokes cho một cấu hình cánh tại các góc tấn và số Reynolds khác nhau có thể ngốn từ hàng chục phút đến nhiều giờ đồng hồ.Sự bất khả thi khi lặp lại: Khi tối ưu hóa, thuật toán cần khảo sát hàng ngàn, thậm chí hàng chục ngàn biến thể hình học khác nhau. Nếu mỗi biến thể đều phải chạy qua mô phỏng CFD, tổng thời gian sẽ phình to lên mức hàng tháng trời. Điều này thu hẹp không gian thiết kế và khiến ta dễ bỏ lỡ các cấu hình cánh đột phá.

Tại sao Machine Learning lại là mảnh ghép phù hợp?Để phá vỡ giới hạn này, em tiếp cận theo hướng sử dụng Machine Learning (Học máy) để đóng vai trò làm mô hình thay thế (Surrogate Model). Thay vì bắt máy tính giải lại các phương trình vật lý phức tạp từ đầu cho mỗi cái cánh mới, em dùng Machine Learning để "học" quy luật ẩn (ánh xạ phi tuyến) giữa đặc trưng hình học, điều kiện dòng chảy và các hệ số khí động học từ một bộ dữ liệu lớn đã có sẵn. Điểm mạnh cốt lõi là: Khi mô hình ML đã được huấn luyện xong, thời gian để nó dự đoán CL,CD,CMcho một hình dạng cánh mới chỉ mất vài mili-giây, tức là nhanh hơn CFD hàng vạn lần.

Phát biểu bài toán nghiên cứuTừ đó, bài toán nghiên cứu của đề tài được phát biểu rõ ràng: Làm thế nào để xây dựng một hệ thống học máy có thể dự đoán cực nhanh và chính xác các hệ số khí động học từ đặc điểm hình học của cánh 2D, và dùng chính hệ thống dự đoán đó để định hướng cho thuật toán tìm ra biên dạng cánh tối ưu nhất mà không cần phụ thuộc vào CFD trong vòng lặp?

Mục tiêu tổng quátMục tiêu cao nhất của em trong đồ án này là: Đề xuất, xây dựng và kiểm chứng một công cụ dự đoán và tối ưu hóa khí động học đầu cuối (end-to-end) dựa trên dữ liệu lớn, giúp giảm triệt để chi phí tính toán trong thiết kế mà vẫn đảm bảo độ tin cậy về mặt vật lý.

Các mục tiêu cụ thể (Các bước giải quyết vấn đề)Để thực hiện mục tiêu tổng quát trên, em chia cấu trúc dự án thành 4 mục tiêu cụ thể, mang tính nối tiếp nhau:Xây dựng bộ dữ liệu: Tổng hợp một bộ dữ liệu đa dạng các họ cánh (NACA, UIUC) với nhiều dải Reynolds và góc tấn khác nhau. Dữ liệu phải đủ rộng thì ML mới học được tính tổng quát.Khai phá đặc trưng hình học (Feature Engineering): Đây là bước thể hiện tư duy kỹ thuật. Em không ném trực tiếp hàng trăm điểm tọa độ thô vào mạng Neural. Em trích xuất chúng thành các thông số mang ý nghĩa vật lý (độ dày cực đại, độ cong cực đại, bán kính mép trước...) để mô hình ML học đúng bản chất khí động học.Huấn luyện và kiểm định khắt khe: Xây dựng mô hình dự đoán CL,CD,CM. Đặc biệt, em phải đánh giá chéo theo nhóm hình học (GroupKFold) để chứng minh mô hình của em có thể dự đoán chính xác cả những cái cánh mà nó chưa từng nhìn thấy bao giờ.Tối ưu hóa hình dáng: Tích hợp mô hình ML vừa tạo vào thuật toán tiến hóa kết hợp với phương pháp biểu diễn hình học tham số CST. Đích đến cuối cùng là sinh ra được một biên dạng cánh mới có tỷ số lực nâng trên lực cản (L/D) cao hơn cấu hình gốc.

PHẦN 2. ĐỐI TƯỢNG VÀ PHẠM VI NGHIÊN CỨU

Đối tượng nghiên cứu và Nguồn dữ liệuThưa các thầy, đối tượng nghiên cứu trọng tâm của đề tài là các biên dạng cánh khí động học 2 chiều (2D airfoil). Để đảm bảo tính tổng quát và không bị "học vẹt" (overfitting) vào một kiểu dáng cụ thể nào, em đã thu thập dữ liệu từ các họ biên dạng đa dạng như NACA, UIUC, Eppler... thông qua các cơ sở dữ liệu công khai chuẩn mực của quốc tế như UIUC Airfoil Database và AirfoilTools,.Về dữ liệu khí động học (các nhãn CL,CD,CMdùng để dạy cho AI), nếu đem chạy CFD truyền thống cho hàng trăm ngàn điều kiện dòng chảy thì sẽ bất khả thi về mặt thời gian. Do đó, em sử dụng công cụ NeuralFoil (một công cụ học sâu đã được kiểm chứng tính tương thích với XFOIL) để giải quyết bài toán sinh dữ liệu lớn.

Tại sao lại chọn Airfoil 2D mà không xét đến mô hình 3D?Chắc hẳn các thầy sẽ đặt câu hỏi: Máy bay thực tế hoạt động trong không gian 3 chiều, tại sao đồ án chỉ dừng lại ở mặt cắt 2 chiều?Dưới góc độ của một kỹ sư thiết kế, em có hai lý do để đưa ra giới hạn này:Thứ nhất - Tính nền tảng của giai đoạn thiết kế sơ bộ (Preliminary Design): Trong hàng không, mặt cắt 2D chính là "linh hồn" tạo ra lực nâng. Nếu một mặt cắt 2D có tỷ số L/D thấp hoặc dễ bị thất tốc, thì cái cánh 3D xây dựng từ nó không thể nào có hiệu suất tốt được. Đa số các kỹ sư đều dùng dữ liệu 2D làm gốc trước khi thiết kế cánh 3D.Thứ hai - Kiểm soát sự bùng nổ của không gian tính toán: Việc chuyển từ 2D lên 3D đồng nghĩa với việc ta phải giải quyết thêm các hiệu ứng vật lý cực kỳ phức tạp như dòng chảy ngang (spanwise flow), xoáy chóp cánh (wingtip vortices) hay downwash. Điều này sẽ làm bùng nổ số chiều dữ liệu, vượt quá giới hạn tài nguyên tính toán của đề tài, làm phân tán sự tập trung khỏi mục tiêu chính là chứng minh tính khả thi của Machine Learning trên cấu trúc cánh cơ bản.

Tại sao lại giới hạn dải vận tốc Mach < 0.8?Phạm vi nghiên cứu của em được giới hạn khắt khe trong dải vận tốc dưới Mach 0.8. Ý nghĩa vật lý của việc này là gì? Thưa các thầy, khi vận tốc vượt ngưỡng Mach 0.8 (bước vào vùng transonic và siêu âm), tính chất của chất lưu thay đổi hoàn toàn. Lúc này, không khí bị nén ép cục bộ mạnh mẽ, sinh ra các sóng xung kích (shockwaves) trên bề mặt cánh. Lực cản lúc này không chỉ là cản ma sát hay cản áp suất thông thường, mà còn xuất hiện "lực cản sóng" (wave drag).Nếu đưa cả dữ liệu siêu âm vào, mạng Neural sẽ bị nhiễu do phải học cùng lúc hai quy luật vật lý hoàn toàn trái ngược nhau. Việc giới hạn dưới Mach 0.8 giúp hệ thống tập trung hoạt động với độ chính xác cao nhất cho mảng máy bay dân dụng, máy bay không người lái (UAV) cỡ nhỏ đến trung bình và năng lượng gió - đây là những ứng dụng mang tính thực tiễn và phổ biến nhất hiện nay.

Các giả định của nghiên cứuĐể hệ thống hoạt động logic, em thiết lập các giả định nghiên cứu (assumptions) sau:Giả định về dữ liệu Ground-Truth: Toàn bộ hệ thống học máy này thuộc nhóm học có giám sát (supervised learning). Có nghĩa là, AI coi các kết quả trả về từ NeuralFoil/XFoil là "chân lý" để học theo. Đề tài tạm thời giả định các công cụ mô phỏng này đủ chính xác, vì hiện tại em chưa có điều kiện để hiệu chỉnh (calibrate) lại bằng số liệu thực nghiệm từ hầm gió vật lý,.Giả định về môi trường lý tưởng 2D: Kết quả tối ưu mà AI đề xuất là kết quả của mặt cắt 2D lý tưởng có sải cánh dài vô hạn. Khi áp dụng lên nguyên mẫu 3D ngoài đời thực, chắc chắn hiệu suất sẽ bị suy giảm một phần do các tổn thất khí động học 3 chiều,.Phần 3: Cơ sở lý thuyết.

Để xây dựng được một mô hình Học máy có khả năng dự đoán vật lý, bản thân mô hình không thể chỉ "học vẹt" các con số, mà hệ thống đặc trưng đầu vào phải được xây dựng dựa trên những "luật chơi" cốt lõi của khí động học. Sau đây, em xin trình bày các cơ sở lý thuyết trọng tâm làm nền tảng cho đề tài.

1. Góc tấn (Angle of Attack - AoA) và Hiện tượng Thất tốc (Stall)

Định nghĩa và vai trò: Thưa các thầy, Góc tấn (AoA) là góc được tạo bởi đường dây cung hình học của biên dạng cánh và hướng của dòng khí tới. Trong vận hành thực tế, đây là tham số quan trọng nhất quyết định trực tiếp đến sự phân bố áp suất trên bề mặt cánh.

Quan hệ tuyến tính và giới hạn: Theo lý thuyết cánh mỏng, khi dòng chảy còn bám sát bề mặt cánh, hệ số lực nâng (CL) có mối quan hệ gần như tuyến tính với góc tấn theo phương trình CL=CL0+aα. Chỉ cần phi công hoặc hệ thống điều khiển thay đổi một lượng nhỏ góc tấn, lực nâng sẽ thay đổi tương ứng. Tuy nhiên, mối quan hệ đẹp đẽ này không kéo dài mãi. Khi góc tấn vượt qua một ngưỡng tới hạn, lớp biên (boundary layer) không còn đủ động lượng, dẫn đến bong tróc khỏi bề mặt cánh, gây ra hiện tượng thất tốc (stall) làm lực nâng sụt giảm nghiêm trọng,.

2. Nguồn gốc của Lực nâng: Từ Bernoulli đến Kutta–Joukowski

Để giải thích tại sao cánh máy bay lại bay được, và quan trọng hơn là tại sao hình dáng cánh lại quyết định lực nâng, em xin tiếp cận qua hai định lý:

Định lý Bernoulli (Góc độ Áp suất): Theo nguyên lý bảo toàn năng lượng của Bernoulli, dòng chảy tăng tốc sẽ làm giảm áp suất tĩnh. Nhờ thiết kế độ cong của airfoil hoặc góc tấn dương, vận tốc dòng khí ở mặt trên cánh di chuyển nhanh hơn mặt dưới, tạo ra một sự chênh lệch áp suất (mặt dưới áp suất cao, mặt trên áp suất thấp), từ đó sinh ra lực nâng trực tiếp đẩy cánh lên,.

Định lý Kutta–Joukowski (Góc độ Hình học và Tuần hoàn): Tuy nhiên, phương trình Bernoulli chưa đủ để giúp ta thiết kế cánh. Dưới góc độ lý thuyết dòng thế, định lý Kutta-Joukowski cho ta công thức L′=ρV∞Γ. Trong đó, Γ là tuần hoàn khí động học. Tại sao điều này lại quan trọng với Machine Learning? Thưa các thầy, để có được giá trị tuần hoàn duy nhất, dòng chảy phải thỏa mãn điều kiện Kutta (dòng chảy phải rời khỏi mép sau cánh một cách mượt mà),. Điều này chứng minh bằng toán học rằng: Chính hình dạng hình học (độ cong, vị trí độ cong, độ sắc của mép sau) là yếu tố quyết định giá trị Γ, và từ đó quyết định lực nâng. Đây chính là cơ sở vật lý vững chắc nhất để em tự tin dùng Machine Learning dự đoán lực nâng dựa vào việc trích xuất các đặc trưng hình học,.

3. Các hệ số khí động học CL,CD,CM

Trong kỹ thuật, để loại bỏ yếu tố kích thước vật lý và vận tốc bay, ta dùng các hệ số không thứ nguyên. Mô hình AI của em tập trung dự đoán 3 đại lượng này:

Hệ số Lực nâng (CL): Từ công thức L=21ρV2SCL, CL cho biết khả năng sinh lực nâng của cấu hình cánh. Ý nghĩa vật lý của nó là tải trọng mà máy bay có thể mang theo.

Hệ số Lực cản (CD): Từ công thức D=21ρV2SCD, CD phản ánh mức độ cản trở của không khí. Ý nghĩa của nó là sự tiêu hao năng lượng; CD càng lớn, máy bay càng tốn nhiên liệu.

Hệ số Mô men (CM): Từ công thức M=21ρV2ScCM, đây là đại lượng đặc trưng cho xu hướng xoay gật (pitching) của biên dạng quanh điểm tham chiếu,. Một CM biến động mạnh sẽ khiến máy bay mất ổn định và cực kỳ khó điều khiển.

Đích đến của kỹ sư thiết kế là cực đại hóa Tỷ số L/D (Lực nâng / Lực cản), tối ưu hóa hiệu suất năng lượng,.

4. Số Reynolds (Reynolds Number)

Định nghĩa và vai trò: Số Reynolds (Re) là tỷ số giữa lực quán tính và lực nhớt của dòng chảy. Nó đóng vai trò là "người chỉ huy" quyết định cấu trúc của lớp biên (boundary layer) bám trên mặt cánh.

Ảnh hưởng khí động học: Thưa các thầy, cùng một cái cánh, nhưng nếu bay ở số Reynolds thấp, lớp biên sẽ là dòng tầng (laminar), dễ bị tách dòng. Nếu bay ở Reynolds cao, dòng chảy chuyển sang trạng thái rối (turbulent), có nhiều động lượng hơn, giúp bám mặt cánh tốt hơn, chống thất tốc tốt hơn nhưng lại sinh ra lực cản ma sát lớn. Do đó, AI không thể dự đoán đúng nếu chỉ nhìn hình học, em bắt buộc phải đưa Số Reynolds vào làm biến đầu vào quan trọng.

5. Hiện tượng Thất tốc (Stall) và Thách thức cho Machine Learning

Cơ chế tách dòng: Như em đã nhắc ở trên, khi góc tấn quá lớn, dòng chảy đối mặt với một gradient áp suất bất lợi. Không khí cạn kiệt động lượng, không thể bám sát độ cong của cánh nữa, nó "bong" ra, hình thành các vùng xoáy và vùng wake (tuần hoàn) khổng lồ phía sau,. Lúc này, CL tụt dốc không phanh, CD tăng đột biến và CM mất kiểm soát,.

Vì sao vùng này lại là "cơn ác mộng" của Machine Learning? Dạ thưa Hội đồng, các thuật toán AI thông thường rất giỏi nội suy các hàm mượt mà, tuyến tính. Tuy nhiên, vùng gần thất tốc (near-stall) và sau thất tốc (post-stall) lại là một môi trường phi tuyến cực kỳ mạnh mẽ. Sự tương tác hỗn loạn giữa lớp biên, áp suất, sự tái bám dòng tạo ra độ phân tán dữ liệu rất lớn. Việc dự đoán chính xác điểm stall và hành vi post-stall là tiêu chí khắt khe nhất để đánh giá xem mô hình surrogate (mô hình thay thế) của em có thực sự nắm bắt được vật lý, hay chỉ đơn thuần là đang khớp đường cong (curve-fitting) một cách mù quáng.

Phần 4: Biểu diễn hình học Airfoil.

PHẦN 4. BIỂU DIỄN HÌNH HỌC AIRFOIL

1. Biểu diễn bằng tọa độ và Chuẩn hóa dây cung

Theo cách thông thường nhất, bề mặt của một biên dạng cánh 2D được biểu diễn bằng một tập hợp các điểm tọa độ (xi,yi) chạy dọc theo mặt trên và mặt dưới. Tuy nhiên, trong học máy, nếu để nguyên kích thước thực tế của cánh, mô hình sẽ bị nhiễu bởi yếu tố tỷ lệ (scale).

Do đó, bước đầu tiên em thực hiện là chuẩn hóa dây cung (Chord Normalization). Toàn bộ trục x của biên dạng được thu phóng về đoạn , với x=0 là mép trước (Leading Edge) và x=1 là mép sau (Trailing Edge). Ý nghĩa: Việc chuẩn hóa này giúp loại bỏ ảnh hưởng của kích thước vật lý tuyệt đối, cho phép hệ thống đặt các cái cánh khổng lồ và các cái cánh mini lên cùng một hệ quy chiếu để so sánh hình dáng, còn yếu tố kích thước thật sẽ được giao cho Số Reynolds xử lý.

2. Khai phá các đại lượng ẩn: Đường camber, Độ dày và Độ cong

Từ tập tọa độ x,y thô, ta chưa thấy được vật lý. Em đã phân tách chúng thành các thông số mang tính quyết định đến dòng chảy:

Đường cong trung bình (Camber line): Là quỹ tích trung điểm giữa mặt trên và mặt dưới tại cùng một vị trí x. Về mặt khí động học, đường camber thể hiện sự bất đối xứng của cánh. Như em đã giải thích ở định lý Kutta-Joukowski, độ cong camber càng lớn thì tuần hoàn Γ càng mạnh, sinh ra lực nâng lớn ngay cả khi góc tấn bằng 0.

Phân bố độ dày (Thickness): Là khoảng cách giữa mặt trên và mặt dưới. Biên dạng dày giúp cánh trì hoãn hiện tượng thất tốc ở góc tấn cao, nhưng lại phải đánh đổi bằng việc gia tăng lực cản áp suất. Ngược lại, cánh mỏng giảm lực cản nhưng rất dễ bị "stall" đột ngột.

Độ cong hình học cục bộ (Curvature - κ): Phản ánh mức độ thay đổi hướng bẻ cong của bề mặt. Nếu bề mặt cánh có một điểm cong gắt (thay đổi đột ngột), tại đó sẽ sinh ra một gradient áp suất cực kỳ bất lợi, làm dòng khí bị "vấp" và bóc tách ngay lập tức.

3. Nghịch lý của Dữ liệu: Vì sao 240 tọa độ → 480 chiều dữ liệu là quá lớn cho Học máy?

Thưa các thầy, thông thường, một file dữ liệu chuẩn để mô tả trơn tru một cái cánh cần khoảng 240 điểm tọa độ (120 điểm mặt trên, 120 điểm mặt dưới),. Vì mỗi điểm có hai tọa độ (x,y), hệ thống máy tính sẽ nhìn cái cánh này như một vector nằm trong không gian 480 chiều (480-D),.

Dưới góc độ xây dựng AI, nếu ném trực tiếp vector 480 chiều này vào mạng Neural hay mô hình Cây quyết định, ta sẽ vấp phải những rào cản chí mạng sau:

Lời nguyền của số chiều (Curse of Dimensionality) và Overfitting: Với 480 biến đầu vào, không gian toán học phình to khủng khiếp. Mô hình sẽ trở nên quá phức tạp, cần một lượng dữ liệu khổng lồ theo cấp số nhân để hội tụ. Thay vì học luật vật lý, nó sẽ học vẹt (overfit) các điểm nhiễu li ti trên bề mặt,.

Mất đi ý nghĩa Khí động học (Lack of Physical Interpretability): Nếu em cho AI học tọa độ thứ 57 (y57) trên mặt cánh, bản thân con số này không mang một ý nghĩa vật lý nào cả. Khí động học (lực nâng, cản, stall) không được quyết định bởi tọa độ y57, mà quyết định bởi tổng hòa của độ dày, độ cong mép trước, góc mép sau... Việc để AI tự mò mẫm kết nối 480 điểm này lại với nhau là cực kỳ rủi ro và làm giảm khả năng tổng quát hóa của mô hình lên các hình dáng cánh mới,,.

Rủi ro sinh ra "Quái vật" hình học: Trong không gian 480 chiều, phần lớn các tổ hợp tọa độ sinh ra ngẫu nhiên đều là những cái cánh bất hợp lý (bề mặt cắt chéo nhau, độ dày âm, zic-zắc),. Nếu tối ưu hóa trực tiếp trên 480 biến này, AI có thể đề xuất ra những nghiệm tối ưu "ảo" không thể chế tạo được trong thực tế.

PHẦN 5. GEOMETRY MANIFOLD (ĐA TẠP HÌNH HỌC)

Geometry Manifold là gì và Tại sao không phải vector nào trong R480cũng là một airfoil hợp lệ?Thưa các thầy, nếu ta biểu diễn một cái cánh bằng 240 điểm (tương đương với một vector trong không gian 480 chiều R480), hãy thử tưởng tượng ta dùng máy tính sinh ra ngẫu nhiên 480 con số bất kỳ. Liệu ta có thu được một cái cánh máy bay không? Câu trả lời là: Xác suất gần như bằng 0.Hầu hết các vector sinh ra ngẫu nhiên trong không gian khổng lồ R480này sẽ tạo ra những "quái vật hình học" vô nghĩa:Mặt trên và mặt dưới cắt chéo nhau (tự giao).Độ dày mang giá trị âm (t(x)=yu(x)−yl(x)<0).Bề mặt gồ ghề, dích dắc, vi phạm nghiêm trọng tính liên tục và độ mượt (chẳng hạn không đạt chuẩn đạo hàm bậc nhất C1hay bậc hai C2).Mép trước bị gãy khúc, mép sau không thỏa mãn điều kiện Kutta.Điều này chứng minh một thực tế vật lý: Không gian của các biên dạng cánh thực tế không chiếm trọn toàn bộ R480. Các biên dạng khả thi (vừa chế tạo được, vừa bay được) bị ràng buộc chặt chẽ bởi các định luật vật lý và hình học, do đó chúng chỉ nằm co cụm trên một cấu trúc/tập con rất mỏng và có tổ chức trong không gian này. Tập con chứa các nghiệm hợp lý đó, trong hình học vi phân, được gọi là một Geometry Manifold (Đa tạp hình học).

Giả thuyết manifold thấp chiều (Low-Dimensional Manifold Hypothesis) và Ví dụ minh họaTừ việc nhận diện được đa tạp này, em áp dụng Giả thuyết manifold thấp chiều. Giả thuyết này cho rằng: Dù cần tới 480 con số để "vẽ" ra một cái cánh cho mượt, nhưng số bậc tự do thực sự (Effective Degrees of Freedom) quyết định khí động học lại nhỏ hơn rất nhiều.Để Hội đồng dễ hình dung, em xin lấy ví dụ về 3 đại lượng: Độ dày cực đại (Maximum Thickness), Độ cong cực đại (Maximum Camber), và Bán kính mép trước (Leading Edge Radius). Khi em, với tư cách là người thiết kế, muốn tăng Độ dày cực đại lên 2%, em không chỉ kéo một điểm tọa độ duy nhất lên cao. Để cái cánh vẫn mượt mà và bay được, sự thay đổi của một thông số "độ dày" này sẽ bắt buộc hàng chục, thậm chí hàng trăm điểm tọa độ mặt trên và mặt dưới phải dịch chuyển đồng thời theo một quy luật liên tục. Tương tự, thay đổi Maximum Camber hay Leading Edge Radius cũng sẽ định hình lại toàn bộ hệ thống tọa độ.Chính sự "ràng buộc đồng thời" này chứng minh rằng 480 tọa độ kia thực chất có tính dư thừa thông tin rất lớn. Chúng bị chi phối bởi một số ít các biến số cốt lõi.

Ý nghĩa sống còn đối với Học máy (Machine Learning)Thưa các thầy, việc thấu hiểu khái niệm Geometry Manifold mang lại hai ý nghĩa cực kỳ to lớn cho việc thiết kế hệ thống AI của em:Tránh lời nguyền số chiều và chống Overfitting: Thay vì bắt AI phải mò mẫm trong không gian R480với rủi ro học vẹt các điểm nhiễu, em dùng một ánh xạ toán học ϕ để "kéo" cái cánh từ không gian tọa độ khổng lồ xuống một không gian đặc trưng thấp chiều hơn rất nhiều (d≪2N). Các đại lượng như Maximum Thickness, Maximum Camber, LE Radius lúc này đóng vai trò chính là các hệ tọa độ mới để xác định vị trí của airfoil trên mặt Manifold.Duy trì tính hợp lệ trong Tối ưu hóa: Khi thuật toán AI đi tìm cái cánh tối ưu nhất, Manifold đóng vai trò như một "hàng rào bảo vệ". Việc định hướng AI di chuyển dọc theo Manifold (hoặc không gian đặc trưng của nó) đảm bảo rằng AI sẽ chỉ đề xuất ra những thiết kế cánh thực sự tồn tại, chế tạo được và tuân thủ vật lý, thay vì "ảo tưởng" ra một tập hợp 480 con số sinh ra một lực nâng hoàn hảo nhưng lại có độ dày âm.

PHẦN 6. KHAI PHÁ ĐẶC TRƯNG HÌNH HỌC (FEATURE ENGINEERING)

Tại sao Descriptor vật lý lại vượt trội hơn so với việc dùng trực tiếp tọa độ thô?Dưới góc độ của một người làm Machine Learning, nếu em ném trực tiếp 240 điểm tọa độ (tương đương 480 chiều dữ liệu) vào mạng Neural, mô hình sẽ lập tức rơi vào "Lời nguyền số chiều" (Curse of Dimensionality). Nó sẽ trở nên quá phức tạp, dễ dàng học vẹt (overfit) các điểm nhiễu thay vì học được quy luật thực sự.Dưới góc độ khí động học, tọa độ (x10,y10) độc lập không hề mang một ý nghĩa vật lý nào cả. Lực nâng hay lực cản không được quyết định bởi một điểm tọa độ, mà bị chi phối bởi tổng hòa hình dáng như cánh dày hay mỏng, mép trước bo tròn hay sắc nhọn.Do đó, thay vì bắt AI tự mò mẫm trong không gian 480 chiều vô nghĩa, em đã dùng Feature Engineering để "nén" cái cánh đó thành 14-18 đặc trưng (descriptors) cốt lõi. Các đặc trưng này trực tiếp đại diện cho các định luật vật lý khí động học, giúp AI học bản chất vấn đề thay vì học vẹt.

Các Descriptor được trích xuất và Ý nghĩa Khí động họcĐể Hội đồng dễ hình dung cách em "dạy" AI hiểu vật lý, em xin lấy ví dụ về 5 nhóm descriptor quan trọng nhất mà em đã trích xuất:Maximum Thickness (Độ dày cực đại):Ý nghĩa vật lý: Độ dày quyết định trực tiếp đến sự phát triển của lớp biên (boundary layer) và khả năng chống tách dòng.Giải thích logic: Khi AI nhìn thấy thông số "độ dày lớn", nó sẽ được học quy luật rằng: cái cánh này có thể trì hoãn hiện tượng thất tốc (stall) rất tốt ở góc tấn cao, nhưng bù lại, nó sẽ sinh ra lực cản áp suất (pressure drag) lớn hơn so với cánh mỏng.Maximum Camber (Độ cong cực đại):Ý nghĩa vật lý: Đặc trưng này là nền tảng của định lý Kutta-Joukowski về tuần hoàn khí động học.Giải thích logic: Camber chính là "cỗ máy" tạo lực nâng. AI sẽ hiểu rằng thông số Camber càng lớn, mức độ tuần hoàn dòng chảy (Γ) càng mạnh, từ đó cánh có thể sinh ra lực nâng (CL) ngay cả khi góc tấn bằng 0 độ.Leading Edge Radius (Bán kính mép trước):Ý nghĩa vật lý: Mép trước là nơi dòng khí đập vào đầu tiên để hình thành vùng đình trệ (stagnation region).Giải thích logic: Nếu mép trước quá sắc nhọn (bán kính nhỏ), dòng khí khi vòng qua mép sẽ gặp gradient áp suất bất lợi cực gắt, dẫn đến bong tróc (tách dòng) ngay lập tức. Mép trước bo tròn hợp lý sẽ giúp dòng chảy bám bề mặt mượt mà hơn ở góc tấn cao. Đây là tín hiệu để AI dự đoán hành vi thất tốc sớm hay muộn.Curvature Energy (Năng lượng độ cong):Ý nghĩa vật lý: Đại lượng này phản ánh mức độ "mượt mà" của bề mặt cánh.Giải thích logic: Bất kỳ một sự thay đổi độ cong đột ngột nào trên mặt cánh cũng sẽ làm dòng chảy bị "vấp", sinh ra gradient áp suất cực kỳ xấu và làm tăng nguy cơ tách dòng. Thông số này giúp AI trừng phạt (đánh giá CDcao) đối với những cấu hình cánh gấp khúc, lồi lõm thiếu tự nhiên.Trailing Edge Geometry (Hình dạng mép sau - Góc và Độ dày):Ý nghĩa vật lý: Mép sau quyết định việc dòng khí rời khỏi cánh có mượt mà và thỏa mãn điều kiện Kutta hay không.Giải thích logic: Những thay đổi rất nhỏ ở phần đuôi cánh sẽ ảnh hưởng cực kỳ mạnh mẽ đến sự hình thành vùng dòng xoáy (wake) phía sau, từ đó quyết định trực tiếp đến Lực cản (CD) và đặc biệt là hệ số Mô men xoay gật (CM).Kết luận của bước này: Thưa Hội đồng, nhờ việc chuyển đổi hình học thành các descriptor vật lý, mô hình học máy của em không cần "nhớ" tọa độ của từng cái cánh. Nó học được quy luật: Cánh cong thì lực nâng cao, cánh đuôi dày thì mô men biến động mạnh. Điều này mang lại khả năng tổng quát hóa tuyệt vời, giúp hệ thống có thể dự đoán chính xác cả những cái cánh hoàn toàn mới mà nó chưa từng nhìn thấy trong tập huấn luyện.

PHẦN 7. XÂY DỰNG BỘ DỮ LIỆU

Nguồn dữ liệu và Vai trò của NeuralFoilThưa các thầy, một mô hình học máy tốt cần được học từ sự đa dạng. Do đó, dữ liệu hình học của đề tài được tổng hợp từ các cơ sở dữ liệu mở chuẩn mực như UIUC Airfoil Database và AirfoilTools. Quá trình này giúp thu thập hơn 571 hình học biên dạng độc lập thuộc nhiều họ khác nhau như NACA, Selig, Eppler... thay vì chỉ giới hạn ở một vài kiểu dáng cố định.Tuy nhiên, ta mới chỉ có tọa độ hình học. Để dạy AI, ta cần các nhãn là hệ số khí động học (CL,CD,CM). Nếu chạy mô phỏng CFD truyền thống cho hàng trăm nghìn mẫu này, thời gian tính toán sẽ là bất khả thi. Vì vậy, em đã sử dụng NeuralFoil – một công cụ sinh dữ liệu cực kỳ mạnh mẽ và tương thích với chuẩn XFoil – để giải bài toán khí động học và tạo ra bộ nhãn dữ liệu một cách nhanh chóng và đáng tin cậy,.

Thiết lập Miền vận hành (Reynolds và Góc tấn - AoA)Một biên dạng cánh sẽ cư xử hoàn toàn khác nhau ở các môi trường bay khác nhau. Do đó, em đã quét dữ liệu qua một không gian vận hành rất rộng:Góc tấn (AoA): Quét từ −25∘đến +25∘

. Tại sao lại rộng như vậy? Thưa Hội đồng, em muốn mô hình AI không chỉ học được "vùng an toàn" (vùng tuyến tính êm ả), mà bắt buộc nó phải nhìn thấy và học được cách dòng chảy sụp đổ ở vùng gần thất tốc (near-stall) và sau thất tốc (post-stall).Số Reynolds: Quét từ 5×104đến 8×105(biến thiên logarit). Miền giá trị này bao phủ toàn bộ dải vận hành thực tế của các thiết bị bay không người lái (UAV) từ cỡ nhỏ đến máy bay dân dụng hạng nhẹ, nắm bắt được các hiệu ứng phức tạp của lớp biên,.3. Làm sạch dữ liệu (Data Cleaning)Dữ liệu thô thu về chứa tới hơn 720,000 mẫu, nhưng không phải mẫu nào cũng dùng được. Em đã thiết lập một hệ thống rây lọc tự động:Lọc lỗi hình học: Loại bỏ ngay các biên dạng cánh bị tự giao (cắt chéo nhau) hoặc diện tích âm, vì chúng không thể chế tạo trong thực tế (loại khoảng 2.5%),,.Lọc lỗi vật lý: Nếu dòng chảy trả về hệ số lực cản âm (CD<0) hoặc tỷ số CL/CDphi logic, em loại bỏ luôn (chiếm 1.7%) vì vi phạm các định luật vật lý cơ bản,.4. Isolation Forest và Tại sao bắt buộc phải loại bỏ Outlier?Thưa các thầy, ngay cả khi dữ liệu đã qua các màng lọc vật lý cơ bản, em vẫn áp dụng thêm thuật toán máy học Isolation Forest trên không gian đặc trưng đa chiều để truy quét các điểm ngoại lai (Outliers). Bước này loại đi thêm khoảng 24,785 mẫu (3.6%).Câu hỏi đặt ra ở đây là: Dưới góc độ nghiên cứu, tại sao ta không giữ lại toàn bộ dữ liệu mà lại phải thẳng tay loại trừ các Outlier?Thưa Hội đồng, lý do nằm ở Bản chất của hình học cực đoan. Các mẫu bị Isolation Forest bắt giữ thường là những cái cánh có độ dày quá mỏng (dưới 3%), độ cong (camber) quá gắt, hoặc có mép sau (trailing edge) dị dạng. Đối với những cấu hình cực đoan này, ngay cả các bộ giải CFD chuyên nghiệp cũng tính toán thiếu ổn định, dòng chảy bị bóc tách hỗn loạn, sinh ra các nhãn khí động học có độ phân tán (variance) cực lớn.Nếu em ném trực tiếp những dữ liệu nhiễu này vào quá trình huấn luyện, mô hình AI thay vì học các "quy luật vật lý" chuẩn mực, nó sẽ cố gắng "học vẹt" (overfit) vào những điểm lỗi. Việc loại bỏ các Outlier giúp hệ thống làm sạch phổ dữ liệu, ép mạng Neural tập trung học đúng phân phối thực tế của các biên dạng cánh khả thi trong kỹ thuật.Kết luận bước này: Trải qua toàn bộ quy trình sàng lọc khắt khe, từ hơn 720,000 mẫu ban đầu, em đã thu được một "mỏ vàng" dữ liệu cực kỳ sạch và chất lượng gồm 665,215 mẫu. Đây chính là nền tảng vững chắc nhất để em tự tin bước vào huấn luyện các mô hình AI phức tạp, nội dung mà em xin phép trình bày ngay ở Phần 8: Surrogate Model tiếp theo.

PHẦN 8. MÔ HÌNH THAY THẾ (SURROGATE MODEL)

1. Cấu trúc Input và Output của Mô hình

Thay vì bắt máy tính giải trực tiếp phương trình Navier-Stokes như phần mềm CFD truyền thống, em thiết lập bài toán dưới dạng một ánh xạ toán học học máy.

Input (Đầu vào): Bao gồm 3 thành phần cốt lõi:

Geometry Features (Đặc trưng hình học): Chính là 14-18 con số mang ý nghĩa vật lý (như độ dày cực đại, độ cong, bán kính mép trước...) mà em đã trích xuất ở Phần 6.

AoA (Góc tấn): Thể hiện tư thế bay.

Reynolds Number: Thể hiện quy mô dòng chảy và độ nhớt.

Output (Đầu ra): Mô hình sẽ trả về dự đoán cho 3 hệ số khí động học tương ứng là CL (Lực nâng), CD (Lực cản), và CM (Mô men).

2. Chiến lược Huấn luyện: Tại sao phải chia tách 3 mô hình riêng biệt?

Thưa các thầy, ban đầu ta có thể nghĩ đến việc dùng chung một mạng Neural mạng lớn để dự đoán cả 3 hệ số cùng lúc. Tuy nhiên, em đã tách ra làm 3 mô hình hồi quy riêng biệt. Lý do xuất phát từ bản chất vật lý hoàn toàn khác nhau của từng đại lượng:

Mô hình CL: Lực nâng có quan hệ tuyến tính đẹp mắt ở góc tấn thấp nhưng lại gãy gập đột ngột ở vùng thất tốc (stall). Em dùng thuật toán HistGradientBoosting vì nó có khả năng xử lý tốt sự kết hợp giữa tuyến tính và phi tuyến này.

Mô hình CD: Lực cản là đại lượng khó đoán nhất vì nó có phân phối lệch phải cực mạnh (tăng vọt lên theo hình parabol khi tách dòng). Ở đây, em dùng kỹ thuật Stall-weighted corrector (hiệu chỉnh trọng số tập trung vào vùng thất tốc) để ép AI phải chú ý vào các lỗi sai ở góc tấn lớn.

Mô hình CM: Mô men lại cực kỳ nhạy cảm với hình dạng mép sau (trailing edge) và độ cong (camber). Em áp dụng chiến lược Regime-Specific ExtraTrees - tức là chia nhỏ dữ liệu theo từng trạng thái dòng chảy (trước stall, gần stall, sau stall) để huấn luyện các cây quyết định độc lập, giúp mô hình bám sát cơ chế vật lý của từng giai đoạn.

3. Đánh giá mô hình: Tại sao dùng GroupKFold thay vì Train/Test Split thông thường?

Kính thưa Hội đồng, đây là quyết định phương pháp luận mang tính sống còn để xác định xem mô hình AI này là "thiên tài" hay chỉ là kẻ "học vẹt".

Trong các bài toán Machine Learning thông thường, người ta hay dùng Train/Test Split (ví dụ chia ngẫu nhiên 80% dữ liệu để huấn luyện, 20% để kiểm tra). Nhưng trong khí động học, cách làm này sẽ dẫn đến sai lầm nghiêm trọng.

Lý do là: Một chiếc cánh (ví dụ NACA 0012) có thể tạo ra hàng trăm mẫu dữ liệu trong dataset của em (khi quét qua các góc tấn từ -25 đến +25 độ và nhiều số Reynolds khác nhau).

Hậu quả của việc chia ngẫu nhiên: Nếu chia ngẫu nhiên, dữ liệu của cùng cái cánh NACA 0012 đó sẽ nằm ở cả tập Huấn luyện lẫn tập Kiểm tra. Lúc này, AI chỉ cần ghi nhớ (memorize) hình dáng của NACA 0012 là có thể đoán đúng kết quả ở tập Kiểm tra. Hiện tượng này gọi là Rò rỉ dữ liệu (Data Leakage). Khi đưa cho AI một cái cánh mới hoàn toàn, nó sẽ dự đoán sai bét vì nó chỉ biết học vẹt những hình dáng đã thấy.

Giải pháp của em là sử dụng GroupKFold Cross-Validation dựa trên định danh hình học (geom_hash):

Thay vì chia ngẫu nhiên từng dòng dữ liệu, em gom nhóm (group) toàn bộ các mẫu thuộc về cùng một biên dạng cánh thành một khối không thể tách rời.

Khi chia dữ liệu thành 5 Folds, hệ thống đảm bảo rằng: Nếu cái cánh A đã nằm trong tập Huấn luyện, thì nó tuyệt đối không được xuất hiện ở tập Kiểm tra.

Ý nghĩa vật lý và thực tiễn: Cách đánh giá này mang tính khắt khe nhất. Nó ép mô hình AI phải học được các "quy luật vật lý tổng quát" (như: cánh càng mỏng thì thất tốc càng gắt, mép trước bo tròn thì bám dòng tốt hơn...) thay vì học vẹt tọa độ.

Chỉ khi vượt qua được bài kiểm tra GroupKFold này, ta mới chứng minh được mô hình có khả năng Ngoại suy hình học (Geometry Extrapolation) – tức là khả năng dự đoán chính xác khí động học cho những thiết kế cánh máy bay hoàn toàn mới, chưa từng tồn tại trên đời. Và đó mới là giá trị cốt lõi để ta mang mô hình này đi tìm kiếm hình dáng tối ưu.

Dạ thưa các thầy, với chiến lược huấn luyện và đánh giá khắt khe như vậy, các mô hình này đã đạt được độ chính xác ra sao và có vi phạm vật lý hay không? Em xin phép chuyển sang Phần 9: Phân tích Kết quả để minh chứng bằng các biểu đồ cụ thể.

Kính thưa Hội đồng, sau khi đã thiết lập xong kiến trúc cho các mô hình học máy ở Phần 8, câu hỏi lớn nhất đặt ra là: Liệu mô hình này có thực sự hiểu được vật lý khí động học, hay chỉ đơn thuần là đang khớp số liệu? Và quan trọng hơn, nó có giúp ta tìm ra được một cái cánh tốt hơn không?

Để trả lời, em xin phép bước sang Phần 9: Phân tích kết quả. Dưới góc độ của một người làm nghiên cứu, em sẽ không chỉ trình bày các con số, mà sẽ đi sâu phân tích ý nghĩa vật lý đằng sau các biểu đồ trọng tâm của đồ án.

1. Phân tích Chất lượng Dữ liệu (Hình 4.2.1 và 4.2.2)

Trước khi nhìn vào kết quả dự đoán, ta cần nhìn vào "nguyên liệu" đầu vào.

Hình 4.2.1: Top 20 biên dạng có điểm bất thường cao nhất theo Isolation Forest.

Mô tả: Biểu đồ cột này thể hiện 20 biên dạng bị thuật toán Isolation Forest đánh giá là ngoại lai (outliers) với điểm số cao nhất.

Giải thích ý nghĩa: Tại sao chúng ta lại có biểu đồ này? Thưa các thầy, khi thu thập dữ liệu từ nhiều nguồn, có những cái cánh mang hình dáng cực đoan (độ dày dưới 3%, camber quá lớn, mép sau dị dạng). Những cấu hình này khiến dòng chảy bị bóc tách hỗn loạn, làm kết quả CFD/NeuralFoil trả về không nhất quán.

Đánh giá: Việc sử dụng thuật toán này đã giúp em "cắt bỏ" thành công khoảng 7.8% lượng dữ liệu nhiễu. Sự sàng lọc này đủ khắt khe để loại bỏ rác, nhưng không làm mất đi các hình dáng có giá trị.

Liên hệ mục tiêu: Bước này đáp ứng trực tiếp mục tiêu xây dựng một bộ dữ liệu chuẩn mực. Nếu em để AI học từ những cái cánh bị lỗi này, nó sẽ bị nhiễu và mất khả năng tổng quát hóa trên các biên dạng thực tế.

Hình 4.2.2: Bản đồ mật độ mẫu phân bố trong không gian Reynolds x AoA.

Mô tả: Đây là bản đồ nhiệt (heatmap) dùng thang đo logarit, thể hiện mật độ dữ liệu quét qua các góc tấn (AoA) và số Reynolds.

Giải thích ý nghĩa: Vùng màu đỏ sậm tập trung ở dải góc tấn từ −10∘ đến +15∘ tại các số Reynolds trung bình và cao. Điều này cho thấy dữ liệu bao phủ rất dày đặc ở vùng vận hành tuyến tính. Mật độ giảm đi ở các góc tấn cực đoan (>20∘ hoặc <−20∘).

Đánh giá & Liên hệ: Sự suy giảm ở vùng cực đoan không phải là lỗi thiếu dữ liệu, mà phản ánh đúng bản chất vật lý: ở các góc này, dòng chảy tách hoàn toàn, các bộ giải thường không hội tụ hoặc sinh ra dữ liệu phi vật lý (như cản âm) nên đã bị bộ lọc tự động loại bỏ. Biểu đồ này chứng minh bộ dữ liệu của em bao phủ trọn vẹn dải vận hành của UAV và máy bay cỡ nhỏ, sẵn sàng cho việc huấn luyện.

2. Phân tích Hiệu năng Dự đoán Khí động học (Hình 4.4.1, 4.5.1, 4.6.1)

Sau khi huấn luyện, đây là lúc ta "khám sức khỏe" cho 3 mô hình Surrogate.

Hình 4.4.1: Parity plot của mô hình Lực nâng (CL)

Mô tả: Trục hoành là CL thực tế, trục tung là CL do AI dự đoán. Các điểm dữ liệu bám rất sát vào đường chéo lý tưởng y=x.

Giải thích & Đánh giá: Mô hình HistGradientBoosting đạt R2=0.9951. Điểm đáng giá nhất của biểu đồ này dưới góc độ kỹ thuật là sự tập trung của các điểm dữ liệu ở vùng CL cao (>1.5). Đây là vùng cận thất tốc (near-stall), nơi quan hệ khí động học trở nên phi tuyến mạnh mẽ. Việc các điểm bị phân tán nhẹ ở đây phản ánh đúng sự gia tăng bất định tự nhiên của dòng chảy, nhưng mô hình vẫn bám sát được thực tế.

Liên hệ mục tiêu: Điều này chứng minh AI đã học được quan hệ giữa độ cong (camber) và tuần hoàn lực nâng theo định lý Kutta-Joukowski, đáp ứng xuất sắc mục tiêu thay thế CFD trong dự đoán CL.

Hình 4.5.1: Parity plot của mô hình Lực cản (CD)

Mô tả: Tương tự đồ thị CL, nhưng dành cho Lực cản. Đường hồi quy y=1.001x gần như trùng khít với đường lý tưởng.

Giải thích & Đánh giá: Thưa các thầy, CD là đại lượng khó đoán nhất vì nó có phân phối lệch phải (tăng vọt hình parabol khi stall). Đạt được R2=0.9980 và đường hồi quy hoàn hảo thế này là một thành tựu lớn. Sự phân tán đều đặn quanh đường chéo ở vùng CD lớn chứng tỏ mô hình không hề bị thiên lệch (bias) khi máy bay đi vào vùng thất tốc.

Liên hệ mục tiêu: Để có được kết quả này, em đã phải thiết kế một kiến trúc AI riêng biệt: dùng "Stall-weighted corrector" (bù sai số có trọng số vùng stall). Nó chứng minh hệ thống feature engineering của em đã cung cấp đúng các đặc trưng vật lý để AI hiểu được sự bóc tách của lớp biên.

Hình 4.6.1: Parity plot của mô hình Mô men (CM)

Mô tả: Đồ thị của CM có hình thái rất đặc biệt: xuất hiện các "vân sọc" nằm ngang.

Giải thích & Đánh giá: Khi mới nhìn, ta có thể tưởng mô hình bị lỗi. Nhưng dưới góc độ nghiên cứu, các vân sọc này lại phản ánh đúng bản chất vật lý tuyệt đẹp của khí động học. Một biên dạng cánh luôn có một giá trị mô men tại lực nâng bằng 0 (CM0) không đổi trong suốt vùng dòng chảy bám. Các vân sọc này chính là các tập dữ liệu quét qua góc tấn của cùng một cái cánh. Tuy R2 chỉ đạt 0.9398 (thấp hơn CL,CD) do CM cực kỳ nhạy cảm với hình dạng mép sau, nhưng mô hình Regime-Specific ExtraTrees đã hoàn thành tốt nhiệm vụ.

Liên hệ mục tiêu: Dù có sai số nhỉnh hơn một chút, kết quả này hoàn toàn đủ độ tin cậy để đóng vai trò làm ràng buộc ổn định dọc trong bài toán tối ưu hóa sắp tới.

3. Phân tích Kết quả Tối ưu hóa (Hình 4.8.1 và 4.8.2)

Đích đến cuối cùng của kỹ sư không phải là dự đoán, mà là sáng tạo ra cái mới. Em đã đưa 3 mô hình trên vào thuật toán tiến hóa kết hợp biểu diễn CST để tìm ra biên dạng tối ưu.

Hình 4.8.1: So sánh hình học giữa Baseline (gốc) và Ứng viên (mới)

Mô tả: Hình ảnh chồng chập mặt cắt giữa cái cánh nguyên bản (đường đứt nét) và cái cánh do AI tự thiết kế ra (đường nét liền màu xanh).

Giải thích & Đánh giá: Các thầy có thể thấy AI không hề tạo ra một hình dáng kỳ dị, vô lý. Nhờ cơ chế kiểm soát của Geometry Manifold, cái cánh mới rất mượt mà, khả thi để chế tạo. Trọng tâm thay đổi nằm ở việc AI đã tự động phân bố lại độ dày và điều chỉnh camber cục bộ để tối ưu hóa trường áp suất.

Hình 4.8.2: So sánh polar khí động học của nhóm ứng viên tối ưu.

Mô tả: 4 biểu đồ thể hiện sự thay đổi của CL,CD,CM và tỷ số L/D theo góc tấn giữa cánh cũ (đen) và cánh mới do AI tạo ra (xanh).

Giải thích & Đánh giá: Nhìn vào biểu đồ, sự thay đổi hình dáng ở Hình 4.8.1 đã mang lại một đột phá về khí động học. Cụ thể theo phân tích trong nghiên cứu, ứng viên tốt nhất đã đẩy lực nâng cực đại (CLmax) tăng khoảng 9.5%. Tuyệt vời hơn nữa, đỉnh của đồ thị CL dịch chuyển sang phải, tức là góc thất tốc (stall angle) được mở rộng thêm 2.04 độ.

Liên hệ mục tiêu tổng quát: Thưa Hội đồng, đây chính là lời giải hoàn hảo cho mục tiêu của đề tài. Nhờ việc đánh giá bằng AI thay vì CFD, hệ thống đã duyệt qua hàng vạn cấu hình trong thời gian cực ngắn để tìm ra một cái cánh vượt trội hơn nguyên bản: mang được tải trọng nặng hơn (CLmax cao) và bay an toàn hơn (Stall trễ hơn) mà không vi phạm bất kỳ định luật vật lý nào.

Dạ thưa các thầy, qua các biểu đồ trên, em đã chứng minh được toàn vẹn tính logic từ dữ liệu, đến khả năng dự đoán vật lý của mô hình, và cuối cùng là hiệu quả của quy trình tối ưu. Tiếp theo, em xin phép chuyển sang Phần 10 để giải thích cơ chế Toán học đằng sau việc AI biến đổi hình dáng cánh (Tối ưu hóa CST).

PHẦN 10. TỐI ƯU HÓA CST

1. CST là gì?

Thưa các thầy, CST viết tắt của Class-Shape Transformation. Đây là một phương pháp toán học dùng để biểu diễn tham số hóa hình học biên dạng cánh. Theo phương pháp này, đường cong của cái cánh được định hình bởi tích của hai hàm số:

Hàm lớp (Class Function - C(x)): Quyết định hình dáng cơ bản (ví dụ: mép trước bo tròn, mép sau sắc nhọn).

Hàm hình dạng (Shape Function - S(x)): Sử dụng các đa thức Bernstein để "nắn" và tinh chỉnh đường cong cục bộ dọc theo thân cánh.

2. Tại sao lại bắt buộc phải dùng CST trong Tối ưu hóa?

Chắc hẳn Hội đồng sẽ đặt câu hỏi: Tại sao không để thuật toán tối ưu tự động thay đổi trực tiếp 240 tọa độ điểm của cánh?

Dưới góc độ kỹ thuật, tối ưu hóa trực tiếp trên tọa độ thô là một thảm họa vì hai lý do:

Lời nguyền số chiều (Curse of Dimensionality): Không gian 480 biến (của 240 điểm x,y) là quá khổng lồ. Thuật toán tìm kiếm sẽ bị lạc lối và không thể hội tụ.

Nguy cơ sinh ra "Quái vật hình học": Việc dịch chuyển tự do các tọa độ điểm rất dễ tạo ra những bề mặt gồ ghề, cắt chéo nhau, hoặc không thể chế tạo thực tế.

Giải pháp: Việc sử dụng CST giúp ta nén toàn bộ hình dáng cái cánh vào một số lượng rất nhỏ các biến thiết kế (chính là các hệ số Ai của đa thức Bernstein). Bằng cách chỉ cho phép thuật toán thay đổi các hệ số CST này, em đảm bảo rằng mọi cái cánh do máy tính sinh ra đều trơn tru, mượt mà và có tính khả thi cao trong chế tạo.

3. Giải thích Logic Quy trình Tối ưu hóa (The Optimization Pipeline)

Để tìm ra biên dạng xuất sắc nhất, em thiết lập một vòng lặp tiến hóa (sử dụng thuật toán Differential Evolution) đi qua 5 bước logic khép kín như sau:

Bước 1: CST Parameters → Candidate Airfoil (Sinh ứng viên)

Logic: Thuật toán tiến hóa sẽ lai ghép và đột biến để đề xuất ra một bộ "Hệ số CST" mới. Từ các con số toán học này, hệ thống sẽ vẽ ra bề mặt hình học thực tế của một chiếc cánh mới (Candidate Airfoil). Ngay lập tức, chiếc cánh này được đưa qua bộ Feature Engineering để trích xuất ra các thông số vật lý cốt lõi (như độ dày cực đại, camber cực đại, bán kính mép trước...).

Bước 2: Candidate Airfoil → Surrogate Model (Chấm điểm qua AI)

Logic: Thay vì mất hàng chục phút đem cái cánh mới này vào phần mềm CFD mô phỏng, em đưa bộ đặc trưng vật lý của nó thẳng vào các mô hình học máy (Surrogate Model) đã huấn luyện ở các phần trước,. Việc dự đoán này diễn ra cực nhanh, chỉ mất chưa tới 1 mili-giây cho mỗi trường hợp.

Bước 3: Surrogate Model → CL/CD (Đánh giá Khí động học)

Logic: Mô hình AI sẽ trả về dự đoán cho 3 hệ số: CL,CD,CM. Mục tiêu thiết kế của em là tối đa hóa hiệu suất bay, do đó hàm mục tiêu chính là cực đại hóa Tỷ số Lực nâng trên Lực cản (L/D) trung bình trên toàn bộ các điều kiện góc tấn,.

Bước 4: Evaluate (Đánh giá Ràng buộc và Vùng tin cậy - Trust Region)

Logic: Đây là bước tư duy kỹ sư quan trọng nhất. Em không mù quáng chấp nhận một cái cánh chỉ vì AI báo điểm L/D cao. Ứng viên phải vượt qua "lưới lọc" khắt khe:

Ràng buộc vật lý & hình học: Độ dày phải dương (t>0), diện tích phải dương (A>0) để đảm bảo chế tạo được. Lực nâng cực đại (CLmax) không được suy giảm so với cánh nguyên bản.

Vùng tin cậy (Manifold Distance): Nếu thuật toán sinh ra một cái cánh có hình dáng "quá kỳ dị", nằm quá xa so với phân phối dữ liệu (Geometry Manifold) mà AI đã từng học, hệ thống sẽ tự động trừ điểm hoặc loại bỏ ứng viên đó. Cơ chế này ngăn chặn tuyệt đối việc AI "ảo tưởng" (extrapolation rủi ro) ở những vùng không có dữ liệu.

Bước 5: Best Airfoil (Xác nhận Biên dạng Tối ưu)

Logic: Vòng lặp từ Bước 1 đến Bước 4 lặp lại hàng vạn lần cho đến khi tìm ra "Nhà vô địch" (Best Airfoil). Cuối cùng, để khẳng định AI không hề "nói dối", nghiệm tối ưu này sẽ được đem đi chạy mô phỏng vật lý độc lập (NeuralFoil/CFD) để xác nhận lại các cải thiện khí động học là hoàn toàn thực chất.

Dạ thưa Hội đồng, nhờ sự kết hợp giữa toán học CST, tốc độ của Machine Learning và sự kiểm soát của Geometry Manifold, hệ thống đã tự động duyệt qua hàng vạn bản vẽ thiết kế để tìm ra phương án tối ưu. Tiếp theo, em xin phép chuyển sang Phần 11 để trình bày cụ thể: Chiếc cánh tối ưu mà hệ thống tạo ra trông như thế nào và vượt trội ra sao so với cánh ban đầu.

nguyên mẫu dưới góc nhìn của một kỹ sư khí động học.

PHẦN 11. KẾT QUẢ TỐI ƯU

1. Phân tích sự thay đổi Hình học (Dựa trên Hình 4.8.1)

Thưa các thầy, Hình 4.8.1 thể hiện sự chồng chập mặt cắt giữa chiếc cánh gốc (đường đứt nét màu đen) và ứng viên xuất sắc nhất do AI tạo ra - Champion Airfoil (đường nét liền màu xanh).

Nhìn vào đồ thị, ta thấy AI không hề tạo ra một hình dáng méo mó hay góc cạnh vô lý. Sự biến đổi hình học diễn ra vô cùng tinh tế:

Phân bố lại độ dày và đường Camber: Thay vì thay đổi ngẫu nhiên, AI đã tự động "nắn" lại đường cong trung bình (camber) và vuốt lại phân bố độ dày dọc theo dây cung.

Sự mượt mà tuyệt đối: Nhờ việc sử dụng hàm toán học CST để biểu diễn biến thiết kế, kết hợp với cơ chế kiểm soát vùng tin cậy (Manifold Distance), chiếc cánh mới duy trì được tính liên tục hình học (đạo hàm bậc cao) cực kỳ mượt mà,. Nó chứng minh đây là một thiết kế hoàn toàn khả thi để chế tạo trong thực tế, chứ không chỉ đẹp trên máy tính.

2. Phân tích sự thay đổi Khí động học (CL,CD và Tỷ số L/D)

Sự tinh chỉnh hình học ở trên đã tạo ra một phản ứng dây chuyền tuyệt vời về mặt khí động học, được minh chứng qua các biểu đồ Polar ở Hình 4.8.2 và số liệu tại Bảng 4.8.1.

Sự thay đổi của Lực nâng (CL): Nhìn vào biểu đồ CL, đường màu xanh (cánh tối ưu) bám sát cánh gốc ở góc tấn thấp, nhưng khi tiến về các góc tấn cao, nó vọt lên mạnh mẽ. Dữ liệu từ Bảng 4.8.1 chỉ ra rằng: Lực nâng cực đại (CLmax) đã được cải thiện tới 9.5% (đạt mức 1.26),. Tuyệt vời hơn nữa, đỉnh của đồ thị này dịch chuyển sang phải, tức là góc thất tốc (Stall AoA) được mở rộng thêm 2.04 độ,.

Sự thay đổi của Lực cản (CD): Dù lực nâng tăng mạnh, nhưng biểu đồ CD cho thấy lực cản của cánh mới gần như tương đương, thậm chí giảm nhẹ ở một số góc tấn so với cánh gốc. Điều này cực kỳ quan trọng, vì thông thường trong thiết kế, muốn tăng sức tải ta phải chấp nhận hy sinh bằng việc tăng lực cản.

Cải thiện Tỷ số L/D (Hiệu suất bay): Tỷ số L/D chính là "thước đo sức khỏe" của cánh. Trong cấu hình nghiên cứu tập trung vào sức tải, chiếc cánh mang lại độ an toàn và sức nâng vượt trội. Còn khi em cấu hình bài toán tối ưu hóa hướng trực tiếp đến hiệu suất hành trình trung bình (như kết quả chạy thử ứng dụng ở Chương 5), hệ thống đã tìm ra ứng viên cải thiện tỷ số L/D trung bình lên tới 41.24% (tăng từ 4.59 lên 6.48),.

3. Giải thích Nguyên nhân Khí động học (Tư duy Kỹ sư)

Tại sao chỉ một vài thay đổi nhỏ trên đường cong (như ở Hình 4.8.1) lại có thể đẩy lùi điểm thất tốc và tăng lực nâng mạnh mẽ đến vậy? Thưa Hội đồng, có 2 cơ chế vật lý cốt lõi đã được AI nắm bắt:

Tối ưu hóa tuần hoàn khí động học (Theo Kutta-Joukowski): Việc AI điều chỉnh độ cong cực đại (maximum camber) và vị trí của nó đã làm thay đổi trực tiếp tuần hoàn trường vận tốc (Γ) xung quanh cánh. Theo định lý Kutta-Joukowski, tuần hoàn lớn hơn sẽ trực tiếp sinh ra hệ số lực nâng (CL) cao hơn ở cùng một điều kiện bay,.

Kiểm soát Gradient áp suất để trì hoãn Thất tốc (Stall Delay): Lý do cánh mới bay được thêm 2.04 độ góc tấn mà không bị "rơi" (thất tốc) là nhờ sự phân bố lại độ dày cục bộ và bán kính mép trước. Bề mặt cánh được nắn mượt hơn giúp giảm độ gắt của gradient áp suất bất lợi (adverse pressure gradient) ở mặt trên của cánh. Nhờ đó, lớp biên (boundary layer) của dòng không khí duy trì được động lượng lâu hơn, bám sát bề mặt cánh ở góc tấn lớn hơn thay vì bị bóc tách (separation) sớm,.

Tiểu kết phần 11: Kính thưa các thầy, kết quả này chứng minh rằng: Hệ thống AI của đề tài không chỉ là một cỗ máy "khớp số" vô tri. Nó thực sự đã học được định lý Kutta-Joukowski và cơ chế bóc tách lớp biên thông qua các đặc trưng hình học. Nhờ đó, nó đủ thông minh để tự động thiết kế ra một nguyên mẫu cánh máy bay vừa mang được tải trọng nặng hơn (CLmax tăng 9.5%), vừa bay an toàn hơn (Stall trễ 2.04 độ) với hiệu suất năng lượng (L/D) vượt trội.

Đề tài đã giải quyết được vấn đề gì?Thưa các thầy, vấn đề lớn nhất trong thiết kế khí động học hiện nay là "nút thắt cổ chai" về thời gian tính toán của CFD,. Khi muốn tối ưu hóa một cái cánh, ta phải thử hàng vạn biến thể. Nếu dùng CFD, quá trình này mất từ vài tuần đến vài tháng, làm bó hẹp sự sáng tạo của kỹ sư,.Đề tài của em đã giải quyết triệt để nút thắt này bằng cách xây dựng một hệ thống tự động hóa đầu cuối (end-to-end):Thay thế bộ giải CFD nặng nề bằng các mô hình học máy (Surrogate Models).Hệ thống có khả năng dự đoán các hệ số CL,CD,CMchỉ trong dưới 1 mili-giây cho mỗi trường hợp, nhanh hơn XFoil từ 10.000 đến 30.000 lần.Quan trọng nhất, em đã ghép nối thành công "bộ não" dự đoán này với thuật toán tiến hóa (Differential Evolution), tạo ra cỗ máy tự động thiết kế cánh. Nó đã tự tìm ra được những ứng viên cánh máy bay xuất sắc: cải thiện sức nâng cực đại (CLmax) thêm 9.5%hoặc tối ưu hóa hiệu suất hành trình (Tỷ số L/D) tăng tới 41.24% so với nguyên bản.

Điểm mới (Sự khác biệt) của nghiên cứu nằm ở đâu?Điểm mới của đề tài không nằm ở việc áp dụng thuật toán AI phức tạp nhất, mà nằm ở tư duy đưa "Vật lý" vào "Học máy". Cụ thể có 3 điểm đột phá:Không học vẹt tọa độ (Feature Engineering & Geometry Manifold): Thay vì ném 480 con số tọa độ khô khan vào mạng Neural (dễ gây nhiễu và ảo tưởng), em đã ép AI học theo các đặc trưng mang tính định luật vật lý như: độ dày, độ cong, bán kính mép trước,. Hệ thống hiểu rằng tăng camber sẽ tăng tuần hoàn lực nâng, chứ không chỉ khớp số liệu mù quáng.Chia để trị theo Bản chất Vật lý: Em không dùng 1 mô hình cho tất cả. Lực nâng CLđược học bằng Gradient Boosting; Lực cản CDdùng cơ chế bù trọng số vùng thất tốc (Stall-weighted); còn Mô men CMdùng Regime-Specific ExtraTrees chia theo từng trạng thái dòng chảy,,. Cách làm này tôn trọng sự khác biệt vật lý của từng hệ số.Đánh giá chống gian lận (Geometry-GroupKFold) và Tối ưu an toàn (Trust Region): Em đã dùng GroupKFold để ép AI phải dự đoán những cái cánh nó chưa từng thấy bao giờ, chứng minh nó thực sự hiểu luật khí động học,. Khi tối ưu, em dùng phương pháp biểu diễn CST kết hợp "Vùng tin cậy" (Manifold Distance) để kiểm soát, đảm bảo AI sinh ra những chiếc cánh mượt mà, chế tạo được, tuyệt đối không vi phạm vật lý (như độ dày âm hay cản âm),.

Hệ thống có thể ứng dụng thực tế như thế nào?Thưa Hội đồng, hệ thống này không sinh ra để loại bỏ hoàn toàn CFD hay hầm gió. Giá trị thực tiễn lớn nhất của nó là trở thành "Công cụ sàng lọc siêu tốc" trong giai đoạn Thiết kế Sơ bộ (Preliminary Design),.Lĩnh vực áp dụng: Công cụ cực kỳ phù hợp để thiết kế cánh cho Máy bay không người lái (UAV) cỡ nhỏ đến trung bình, máy bay dân dụng hạng nhẹ, và đặc biệt là cánh tuabin điện gió – những thiết bị hoạt động ở dải vận tốc dưới Mach 0.8,,.Cách thức sử dụng: Một kỹ sư thay vì vẽ tay và chạy mô phỏng từng cái cánh, giờ đây chỉ cần nhập biên dạng gốc (baseline) và yêu cầu nhiệm vụ (VD: tối đa hóa tải trọng, hoặc tối đa hóa L/D để bay xa hơn). Hệ thống sẽ tự động rà soát hàng vạn thiết kế và nhả ra top 5 ứng viên tốt nhất chỉ trong vài phút,.Định hướng triển khai: Đề tài đã được đóng gói thành phần mềm (API). Trong tương lai, hệ thống hoàn toàn có thể trở thành một nền tảng Web, nơi các công ty kỹ thuật nạp dữ liệu vào, AI rà soát không gian thiết kế, thu hẹp lại vài phương án khả thi nhất, sau đó kỹ sư mới mang những phương án xuất sắc đó đi chạy mô phỏng CFD chi tiết hoặc thổi hầm gió để chốt thiết kế cuối cùng,. Quá trình này sẽ tiết kiệm cho các công ty hàng tháng trời nghiên cứu.
