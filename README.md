# ai5d
AI5D

A Hierarchical 5D AI Matrix Library with a C++20 Core and Python Interface

AI5D is an experimental hierarchical AI architecture designed around a multi-level matrix-processing system.

The project combines:

- Hierarchical 1D → 2D → 3D → 4D → 5D processing
- Parallel computation
- Contiguous memory
- C++20 core
- Python interface through pybind11
- Adaptive compute routing
- Candidate generation, comparison, filtering, and final selection

«WARNING — BETA / EXPERIMENTAL SOFTWARE

AI5D is currently a Beta experimental project.

Bugs, crashes, incomplete implementations, unstable APIs, incorrect results, performance problems, and compatibility issues are expected.

The architecture and API may change without notice.

If you download, build, or use this version, errors are your responsibility.

This project is intended for experimentation, research, development, and testing.»

---

1. Architecture

AI5D is organized as a hierarchical processing system rather than a conventional single neural-network pipeline.

The basic structure is:

1D Workers
    ↓
2D Groups
    ↓
3D Processing Blocks
    ↓
3D → 3D Communication / Hierarchical Reduction
    ↓
4D Filtering
    ↓
Top-K Candidates
    ↓
5D Final Selection

The five levels have different responsibilities.

1D — Worker Layer

1D is the lowest processing level.

Each 1D unit operates on basic pieces of information.

Conceptually:

Input
 ↓
1D Worker
 ↓
Processed information

1D units do not make the final decision.

They produce information that can be grouped and processed by 2D.

---

2D — Group / Leader Layer

2D combines the results of multiple 1D workers.

Conceptually:

1D + 1D + 1D + ...
          ↓
         2D

A 2D unit organizes information into matrix-like structures and evaluates intermediate results.

2D is responsible for turning many low-level worker results into more meaningful groups.

---

3D — P2P Processing Layer

3D contains multiple 2D groups.

Multiple 3D blocks can process information in parallel.

3D blocks can communicate with other 3D blocks and compare their results.

Conceptually:

        ┌── 3D ──┐
        │        │
Input → ├── 3D ──┤ → Reduction
        │        │
        └── 3D ──┘

The purpose is not simply to force every 3D block to agree.

Different blocks can produce different candidates.

The system can then compare and reduce these candidates hierarchically.

For example:

Many 3D
  ↓
Fewer 3D
  ↓
Even fewer 3D
  ↓
4D

This allows the compute structure to scale with the problem.

---

4D — Processing and Filtering Layer

4D receives information from the 3D system.

A 4D unit has its own internal 2D workers.

These internal 2D workers are separate from the original lower-level 2D groups.

Conceptually:

             4D
        ┌────┼────┐
       2D   2D   2D
        │    │    │
        └────┼────┘
             ↓
        Processed results
             ↓
          Ranking
             ↓
           Top-K

The internal 2D workers process the information assigned by 4D.

4D then evaluates the returned results and keeps the strongest candidates.

The default design currently uses:

Top 3

before sending the candidates to 5D.

---

5D — Executive / Final Selection Layer

5D receives the best candidates selected by 4D.

Its job is to make the final decision.

Conceptually:

Candidate A ─┐
Candidate B ─┼─→ 5D → Final result
Candidate C ─┘

5D does not necessarily select the candidate with the highest raw numerical score.

It can consider:

- Context
- Compatibility
- Candidate quality
- Confidence
- Requirements
- Agreement between processing paths
- Other evaluation signals

The architecture also allows 5D to reject all candidates when none of them is acceptable.

This can trigger another processing or generation cycle instead of forcing a bad answer.

---

2. Adaptive Compute

AI5D is designed to support adaptive computation.

The final 5D decision layer remains part of the pipeline, while the amount of computation before reaching 5D can change according to task difficulty.

Example:

Very light
    ↓
~10% compute budget

Normal
    ↓
~25–50%

Hard
    ↓
~50–100%

Very hard
    ↓
Full compute

These percentages represent compute budgets rather than fixed percentages of nodes.

The router decides how much computation should be activated.

The principle is:

«Compute can scale. Final decision cannot.»

---

3. Candidate Selection

AI5D is designed around multiple candidate results instead of immediately forcing a single result.

A simplified example:

3D A → Candidate A
3D B → Candidate B
3D C → Candidate C
3D D → Candidate D
      ↓
     4D
      ↓
Top 3
      ↓
     5D
      ↓
Final candidate

The Top-K value is configurable.

The current default is:

Top-K = 3

However, Top-1, Top-3, Top-5, Top-8, Top-16, or other values may be tested in future benchmarks.

---

4. Programming and Verification

AI5D can be used for tasks where multiple possible solutions need to be compared.

For programming-related tasks, a conceptual pipeline could be:

Problem
   ↓
Multiple candidate solutions
   ↓
3D comparison
   ↓
4D filtering
   ↓
Top candidates
   ↓
5D selection
   ↓
Best solution

The final selection can consider more than raw probability.

For example:

Candidate A
High score
But incomplete error handling

Candidate B
Lower score
Meets all requirements

Candidate C
Medium score
Contains an edge-case bug

          ↓

         5D
          ↓

      Candidate B

Architecture alone does not guarantee programming ability.

The quality of the result depends on the underlying model, training data, objectives, evaluation methods, and verification mechanisms.

Automated tests can be integrated into future versions.

---

5. Tensor and Memory Design

AI5D uses a contiguous memory approach for tensor storage.

The basic Tensor abstraction provides:

- Data storage
- Shape information
- Element access
- Reshaping
- Raw data access
- Vector access

The intended memory layout is designed to make CPU-side processing easier.

Potential optimizations include:

- CPU cache locality
- SIMD
- FMA where mathematically appropriate
- Parallel processing
- Reduced allocation overhead

Contiguous memory does not automatically guarantee high performance.

Actual performance must be measured through benchmarks.

---

6. Mathematical Components

AI5D contains reusable mathematical components under:

include/ai5d/math/

Current planned components include:

activation.hpp
linear.hpp
normalization.hpp
attention.hpp
entropy.hpp
topk.hpp

Possible mathematical mechanisms include:

- GELU or other activation functions
- Linear transformations
- Layer normalization
- Attention
- Entropy-based uncertainty estimation
- Top-K selection

Not every mechanism is mandatory for every task.

The implementation should select the simplest mechanism that performs well for the intended workload.

---

7. 3D Geometry

AI5D may support SE(3)-aware processing when the data actually represents 3D geometry, positions, orientations, or related spatial information.

SE(3) should not automatically be applied to ordinary text embeddings.

For non-geometric data, conventional vector or graph-based processing can be used instead.

---

8. Neural ODE

Neural ODE processing is considered an optional deeper-processing mechanism.

A numerical solver is required for an actual Neural ODE implementation.

For example:

Normal mode
    ↓
Fast residual / transformation

Deep mode
    ↓
Optional ODE processing

Neural ODE should not automatically run on every lightweight query.

---

9. Project Structure

ai5d/
│
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── LICENSE
│
├── include/
│   └── ai5d/
│       │
│       ├── brain.hpp
│       ├── tensor.hpp
│       ├── config.hpp
│       │
│       ├── logging/
│       │   └── logger.hpp
│       │
│       ├── layers/
│       │   ├── layer1d.hpp
│       │   ├── layer2d.hpp
│       │   ├── layer3d.hpp
│       │   ├── layer4d.hpp
│       │   └── layer5d.hpp
│       │
│       ├── math/
│       │   ├── activation.hpp
│       │   ├── linear.hpp
│       │   ├── normalization.hpp
│       │   ├── attention.hpp
│       │   ├── entropy.hpp
│       │   └── topk.hpp
│       │
│       └── routing/
│           └── compute_router.hpp
│
├── src/
│   └── ai5d/
│       │
│       ├── brain.cpp
│       ├── tensor.cpp
│       │
│       ├── logging/
│       │   └── logger.cpp
│       │
│       ├── layers/
│       │   ├── layer1d.cpp
│       │   ├── layer2d.cpp
│       │   ├── layer3d.cpp
│       │   ├── layer4d.cpp
│       │   └── layer5d.cpp
│       │
│       ├── math/
│       │   ├── activation.cpp
│       │   ├── linear.cpp
│       │   ├── normalization.cpp
│       │   ├── attention.cpp
│       │   ├── entropy.cpp
│       │   └── topk.cpp
│       │
│       └── routing/
│           └── compute_router.cpp
│
├── python/
│   └── bindings.cpp
│
├── ai5d/
│   ├── __init__.py
│   └── _version.py
│
├── tests/
│   ├── test_tensor.cpp
│   ├── test_math.cpp
│   ├── test_layers.cpp
│   ├── test_router.cpp
│   └── test_python.py
│
└── examples/
    ├── basic.py
    └── basic.cpp

---

10. C++ Core

The main computational core is written in:

C++20

C++ is responsible for:

- Tensor operations
- Layer implementations
- Mathematical operations
- Parallel processing
- Routing
- Memory management
- Runtime performance

The project is designed around modern C++ rather than implementing the computational core entirely in Python.

---

11. Python Interface

Python is the high-level interface.

The project uses:

pybind11

as the bridge between Python and C++.

The intended goal is to keep the public Python API extremely simple.

Example:

import ai5d

result = (
    ai5d.Brain()
    .in_1d([1.0, 2.0, 3.0, 4.0])
    .to_2d()
    .talk_3d()
    .filter_4d()
    .get_5d()
    .result()
)

The complexity should remain inside the library.

The user should not need to manually control every internal layer for common operations.

---

12. Clang / LLVM

Clang/LLVM is a required toolchain for the project.

Clang is the compiler/toolchain component.

pybind11 is the Python-to-C++ bridge.

They have different responsibilities:

Clang / LLVM
    ↓
Compile C++20

pybind11
    ↓
Connect Python ↔ C++

AI5D does not currently require a separate "clang/" directory in its source tree.

Toolchain configuration belongs in the build and packaging configuration.

---

13. Logging

AI5D provides a lightweight logging system.

Example:

ai5d::log("Brain started");

The logger can provide different levels:

DEBUG
INFO
WARNING
ERROR

Logging is useful during development, testing, and debugging.

---

14. Testing Philosophy

AI5D follows a practical:

«Check → Fix → Test Again»

development cycle.

The first mobile experiments should use a small number of 3D blocks.

Example scaling:

3
↓
6
↓
12
↓
24
↓
48
↓
...

The purpose is to measure:

- RAM usage
- CPU usage
- Execution time
- Stability
- Memory behavior
- Scaling behavior
- Bugs

The system should not attempt to start at maximum scale on limited hardware.

---

15. Benchmarking

Performance claims must be verified experimentally.

Important measurements include:

- Latency
- Throughput
- Tokens/second where applicable
- Memory consumption
- CPU utilization
- Parallel scaling
- Candidate quality
- Accuracy
- Error rate

A mathematically more sophisticated algorithm is not automatically faster or more accurate.

AI5D should therefore rely on benchmarks rather than assumptions.

---

16. Current Development Status

AI5D is currently an experimental Beta project.

Expected status:

Architecture        Experimental
API                 Unstable
Performance         Not guaranteed
Compatibility       Not guaranteed
Algorithms          In development
Documentation       In development
Testing             In development

Do not treat the current version as production-ready software.

---

17. License

AI5D is released under the MIT License.

See:

LICENSE

for the complete license text.

---

18. Project Philosophy

AI5D follows several principles:

Keep the public API simple

Complexity belongs inside the library whenever possible.

Use computation where it matters

Not every task requires maximum compute.

Generate multiple candidates when useful

A single path can fail.

Multiple candidates provide opportunities for comparison and verification.

Let higher levels make decisions

Lower layers process information.

Higher layers evaluate and select.

Do not force a bad answer

If the system cannot find an acceptable candidate, rejection and retry should be possible.

Measure instead of assuming

Architecture, mathematics, and hardware choices must ultimately be validated by experiments.

---

AI5D — Vietnamese

AI5D

Thư viện AI Ma trận Phân cấp 5D với lõi C++20 và giao diện Python

AI5D là một kiến trúc AI thử nghiệm được thiết kế dựa trên hệ thống xử lý ma trận phân cấp nhiều tầng.

Dự án kết hợp:

- Xử lý phân cấp 1D → 2D → 3D → 4D → 5D
- Tính toán song song
- Bộ nhớ liên tục
- Lõi C++20
- Giao diện Python thông qua pybind11
- Adaptive Compute Router
- Sinh, so sánh, lọc và lựa chọn candidate

«CẢNH BÁO — PHIÊN BẢN BETA / THỬ NGHIỆM

AI5D hiện là dự án Beta mang tính thử nghiệm.

Có thể tồn tại bug, crash, chức năng chưa hoàn thiện, API không ổn định, kết quả sai, vấn đề hiệu năng và vấn đề tương thích.

Kiến trúc và API có thể thay đổi mà không báo trước.

Nếu bạn tải, build hoặc sử dụng phiên bản này, việc gặp lỗi là điều có thể xảy ra.

Phiên bản này dành cho thử nghiệm, nghiên cứu, phát triển và kiểm tra.»

---

1. Kiến trúc

AI5D được tổ chức thành một hệ thống xử lý phân cấp thay vì một pipeline mạng neural đơn lẻ.

Cấu trúc cơ bản:

Worker 1D
    ↓
Nhóm 2D
    ↓
Các khối xử lý 3D
    ↓
3D ↔ 3D / giảm cấp phân cấp
    ↓
4D lọc
    ↓
Các candidate Top-K
    ↓
5D lựa chọn cuối

Mỗi tầng có một nhiệm vụ khác nhau.

1D — Tầng Worker

1D là tầng xử lý thấp nhất.

Mỗi worker 1D xử lý những phần thông tin cơ bản.

Input
 ↓
Worker 1D
 ↓
Thông tin đã xử lý

1D không quyết định kết quả cuối cùng.

Nó tạo dữ liệu để 2D tiếp tục tổ chức và xử lý.

---

2D — Tầng Nhóm / Đội trưởng

2D kết hợp kết quả từ nhiều worker 1D.

1D + 1D + 1D + ...
          ↓
         2D

2D tổ chức thông tin thành cấu trúc dạng ma trận và đánh giá các kết quả trung gian.

Có thể hiểu đơn giản:

1D làm việc → 2D gom và quản lý thành quả.

---

3D — Tầng Xử lý P2P

3D chứa nhiều nhóm 2D.

Nhiều khối 3D có thể xử lý song song.

Các khối 3D có thể trao đổi và so sánh kết quả với nhau.

        ┌── 3D ──┐
        │        │
Input → ├── 3D ──┤ → Giảm cấp
        │        │
        └── 3D ──┘

Mục tiêu không phải bắt tất cả 3D phải đồng ý với nhau.

Mỗi 3D có thể đưa ra candidate khác nhau.

Sau đó hệ thống giảm số lượng candidate theo từng cấp:

Nhiều 3D
  ↓
Ít 3D hơn
  ↓
Ít hơn nữa
  ↓
4D

---

4D — Tầng Xử lý và Lọc

4D nhận thông tin từ hệ thống 3D.

Mỗi 4D có nhóm 2D nội bộ riêng.

Những 2D này không phải chính các 2D ban đầu của tầng thấp hơn.

             4D
        ┌────┼────┐
       2D   2D   2D
        │    │    │
        └────┼────┘
             ↓
      Kết quả xử lý
             ↓
           Xếp hạng
             ↓
           Top-K

4D giao nhiệm vụ cho các 2D nội bộ.

Các 2D xử lý rồi trả kết quả về cho 4D.

4D đánh giá và giữ lại những candidate tốt nhất.

Thiết kế mặc định hiện tại:

Top 3

Sau đó 3 candidate được đưa lên 5D.

---

5D — Tầng Điều hành / Lựa chọn cuối

5D nhận những candidate tốt nhất từ 4D.

Nhiệm vụ của 5D là đưa ra quyết định cuối.

Candidate A ─┐
Candidate B ─┼─→ 5D → Kết quả cuối
Candidate C ─┘

5D không nhất thiết phải chọn candidate có điểm số thô cao nhất.

Nó có thể xem xét:

- Ngữ cảnh
- Mức độ phù hợp
- Chất lượng candidate
- Độ tin cậy
- Yêu cầu đầu vào
- Mức độ đồng thuận giữa các nhánh xử lý
- Các tín hiệu đánh giá khác

5D cũng có thể từ chối toàn bộ candidate nếu không có kết quả đủ tốt.

Khi đó hệ thống có thể xử lý hoặc sinh lại thay vì bắt buộc chọn một kết quả tệ.

---

2. Adaptive Compute

AI5D hỗ trợ ý tưởng tính toán thích nghi.

Tầng 5D cuối cùng vẫn được sử dụng để lựa chọn, nhưng lượng compute phía trước có thể thay đổi theo độ khó của nhiệm vụ.

Ví dụ:

Rất nhẹ
    ↓
~10% compute budget

Bình thường
    ↓
~25–50%

Khó
    ↓
~50–100%

Rất khó
    ↓
Full compute

Các phần trăm này là ngân sách tính toán, không nhất thiết có nghĩa là cố định đúng từng phần trăm node.

Router sẽ quyết định lượng tài nguyên cần kích hoạt.

Nguyên tắc:

«Compute có thể co giãn. Quyết định cuối thì không.»

---

3. Lựa chọn Candidate

AI5D ưu tiên việc tạo nhiều candidate khi cần thay vì ép hệ thống chọn ngay một kết quả.

3D A → Candidate A
3D B → Candidate B
3D C → Candidate C
3D D → Candidate D
      ↓
     4D
      ↓
Top 3
      ↓
     5D
      ↓
Kết quả cuối

Top-K có thể cấu hình.

Giá trị mặc định hiện tại:

Top-K = 3

Sau này có thể benchmark Top-1, Top-3, Top-5, Top-8, Top-16 hoặc các giá trị khác.

---

4. Lập trình và kiểm chứng

AI5D có thể phù hợp với những nhiệm vụ cần tạo nhiều phương án rồi so sánh.

Ví dụ với lập trình:

Bài toán
   ↓
Nhiều phương án code
   ↓
3D so sánh
   ↓
4D lọc
   ↓
Các candidate tốt nhất
   ↓
5D lựa chọn
   ↓
Phương án cuối

Việc lựa chọn không nhất thiết dựa hoàn toàn vào xác suất thô.

Ví dụ:

Candidate A
Điểm cao
Nhưng thiếu xử lý lỗi

Candidate B
Điểm thấp hơn
Đáp ứng đầy đủ yêu cầu

Candidate C
Điểm trung bình
Có bug ở edge case

          ↓

         5D
          ↓

      Candidate B

Tuy nhiên, kiến trúc AI5D không tự động biến model thành một lập trình viên giỏi.

Chất lượng còn phụ thuộc vào model, dataset, mục tiêu training, phương pháp đánh giá và cơ chế kiểm chứng.

---

5. Tensor và bộ nhớ

AI5D sử dụng hướng thiết kế contiguous memory cho dữ liệu Tensor.

Tensor cơ bản hỗ trợ:

- Lưu dữ liệu
- Lưu shape
- Truy cập phần tử
- Reshape
- Truy cập dữ liệu thô
- Truy cập vector

Mục tiêu của thiết kế bộ nhớ:

- Tận dụng CPU cache
- SIMD
- FMA khi phù hợp về mặt toán học
- Xử lý song song
- Giảm overhead cấp phát bộ nhớ

Contiguous memory không tự động đảm bảo hiệu năng cao.

Hiệu năng thực tế phải được benchmark.

---

6. Các thành phần toán học

Các module toán học dùng chung nằm tại:

include/ai5d/math/

Hiện tại dự kiến gồm:

activation.hpp
linear.hpp
normalization.hpp
attention.hpp
entropy.hpp
topk.hpp

Các cơ chế có thể bao gồm:

- GELU hoặc activation khác
- Linear transformation
- LayerNorm
- Attention
- Đánh giá entropy/uncertainty
- Top-K selection

Không phải cơ chế nào cũng phải chạy trong mọi nhiệm vụ.

Nguyên tắc là dùng cơ chế đơn giản nhất nhưng vẫn đạt hiệu quả cần thiết.

---

7. Hình học 3D

AI5D có thể hỗ trợ xử lý dựa trên SE(3) khi dữ liệu thực sự biểu diễn:

- Hình học 3D
- Vị trí
- Orientation
- Pose
- Thông tin không gian tương tự

SE(3) không nên được áp dụng mặc định cho embedding văn bản thông thường.

Với dữ liệu không mang ý nghĩa hình học, có thể sử dụng xử lý vector hoặc graph attention thông thường.

---

8. Neural ODE

Neural ODE được xem là một cơ chế xử lý sâu tùy chọn.

Một Neural ODE thực tế cần numerical solver.

Ví dụ:

Normal
   ↓
Residual / transformation nhanh

Deep
   ↓
Neural ODE tùy chọn

Neural ODE không nên tự động chạy cho mọi câu hỏi nhẹ.

---

9. Cấu trúc project

ai5d/
│
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── LICENSE
│
├── include/
│   └── ai5d/
│       ├── brain.hpp
│       ├── tensor.hpp
│       ├── config.hpp
│       │
│       ├── logging/
│       │   └── logger.hpp
│       │
│       ├── layers/
│       │   ├── layer1d.hpp
│       │   ├── layer2d.hpp
│       │   ├── layer3d.hpp
│       │   ├── layer4d.hpp
│       │   └── layer5d.hpp
│       │
│       ├── math/
│       │   ├── activation.hpp
│       │   ├── linear.hpp
│       │   ├── normalization.hpp
│       │   ├── attention.hpp
│       │   ├── entropy.hpp
│       │   └── topk.hpp
│       │
│       └── routing/
│           └── compute_router.hpp
│
├── src/
│   └── ai5d/
│       ├── brain.cpp
│       ├── tensor.cpp
│       │
│       ├── logging/
│       │   └── logger.cpp
│       │
│       ├── layers/
│       │   ├── layer1d.cpp
│       │   ├── layer2d.cpp
│       │   ├── layer3d.cpp
│       │   ├── layer4d.cpp
│       │   └── layer5d.cpp
│       │
│       ├── math/
│       │   ├── activation.cpp
│       │   ├── linear.cpp
│       │   ├── normalization.cpp
│       │   ├── attention.cpp
│       │   ├── entropy.cpp
│       │   └── topk.cpp
│       │
│       └── routing/
│           └── compute_router.cpp
│
├── python/
│   └── bindings.cpp
│
├── ai5d/
│   ├── __init__.py
│   └── _version.py
│
├── tests/
│   ├── test_tensor.cpp
│   ├── test_math.cpp
│   ├── test_layers.cpp
│   ├── test_router.cpp
│   └── test_python.py
│
└── examples/
    ├── basic.py
    └── basic.cpp

---

10. Lõi C++

Phần tính toán chính sử dụng:

C++20

C++ phụ trách:

- Tensor
- Các layer
- Toán học
- Xử lý song song
- Routing
- Quản lý bộ nhớ
- Hiệu năng runtime

---

11. Giao diện Python

Python là giao diện cấp cao.

AI5D sử dụng:

pybind11

để kết nối Python với C++.

Mục tiêu là API Python cực kỳ đơn giản.

Ví dụ:

import ai5d

result = (
    ai5d.Brain()
    .in_1d([1.0, 2.0, 3.0, 4.0])
    .to_2d()
    .talk_3d()
    .filter_4d()
    .get_5d()
    .result()
)

Người dùng thông thường không cần tự quản lý từng layer bên trong.

---

12. Clang / LLVM

Clang/LLVM là toolchain bắt buộc của project.

Clang/LLVM và pybind11 có nhiệm vụ khác nhau:

Clang / LLVM
    ↓
Compile C++20

pybind11
    ↓
Python ↔ C++

Hiện tại không cần tạo riêng thư mục "clang/" trong source tree.

Cấu hình toolchain thuộc về hệ thống build và packaging.

---

13. Logging

AI5D có hệ thống logging nhẹ.

Ví dụ:

ai5d::log("Brain started");

Các level:

DEBUG
INFO
WARNING
ERROR

Logging phục vụ phát triển, debug và kiểm thử.

---

14. Triết lý kiểm thử

AI5D sử dụng chu trình:

«Kiểm tra → Sửa → Kiểm tra lại»

Khi thử nghiệm trên thiết bị yếu, không nên lập tức chạy số lượng 3D quá lớn.

Có thể tăng dần:

3
↓
6
↓
12
↓
24
↓
48
↓
...

Theo dõi:

- RAM
- CPU
- Thời gian chạy
- Độ ổn định
- Memory behavior
- Khả năng scale
- Bug

Mục tiêu ban đầu là xem hệ thống sống được đến đâu, tìm lỗi, sửa lỗi rồi tăng tải.

---

15. Benchmark

Mọi tuyên bố về hiệu năng phải được kiểm chứng bằng benchmark.

Các chỉ số quan trọng:

- Latency
- Throughput
- Tokens/second nếu phù hợp
- RAM
- CPU
- Parallel scaling
- Chất lượng candidate
- Accuracy
- Error rate

Toán học phức tạp hơn không có nghĩa là tự động nhanh hơn hoặc chính xác hơn.

Đo thực tế quan trọng hơn giả định.

---

16. Trạng thái hiện tại

AI5D hiện là project Beta thử nghiệm.

Kiến trúc        Experimental
API              Chưa ổn định
Hiệu năng        Không đảm bảo
Tương thích      Không đảm bảo
Thuật toán       Đang phát triển
Documentation    Đang phát triển
Testing          Đang phát triển

Không nên xem phiên bản hiện tại là production-ready.

---

17. License

AI5D được phát hành theo:

MIT License

Xem file:

LICENSE

để đọc đầy đủ điều khoản.

---

18. Triết lý của AI5D

API bên ngoài phải đơn giản

Sự phức tạp nên được giấu bên trong thư viện khi có thể.

Dùng compute khi nó thực sự cần

Không phải nhiệm vụ nào cũng cần chạy toàn bộ hệ thống.

Tạo nhiều candidate khi có lợi

Một đường xử lý duy nhất có thể sai.

Nhiều candidate cho phép hệ thống so sánh và kiểm chứng.

Tầng trên chịu trách nhiệm lựa chọn

Tầng dưới xử lý thông tin.

Tầng trên đánh giá và lựa chọn.

Không ép chọn một kết quả tệ

Nếu không có candidate đạt yêu cầu, hệ thống nên có khả năng từ chối và thử lại.

Không đoán hiệu năng

Kiến trúc, toán học và phần cứng phải được kiểm chứng bằng thực nghiệm.
