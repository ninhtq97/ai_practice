# Hướng dẫn Build OpenCV với OPENCV_ENABLE_NONFREE=ON

## Tổng quan

OPENCV_ENABLE_NONFREE=ON kích hoạt các thuật toán có bản quyền (SURF, SIFT cũ, ...) trong OpenCV. Điều này cần thiết nếu bạn muốn sử dụng SURF hoặc các thuật toán nonfree khác.

## Yêu cầu

- macOS (10.12+)
- Homebrew: https://brew.sh
- Python 3.6+
- ~2-4 GB dung lượng ổ cứng
- 30-60 phút để build

## Các bước chi tiết

### Bước 1: Cài đặt dependencies

```bash
# Cài build tools
brew install cmake pkg-config wget

# Cài thư viện hỗ trợ
brew install jpeg libpng libtiff openexr
brew install eigen tbb hdf5
```

### Bước 2: Download source code

```bash
# Tạo thư mục làm việc
mkdir -p ~/opencv_build
cd ~/opencv_build

# Clone OpenCV repository (phiên bản 4.x stable)
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x
cd ..

# Clone opencv_contrib (chứa xfeatures2d và các module khác)
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.x
cd ..

# Kiểm tra cấu trúc thư mục
ls -la ~/opencv_build/
# Sẽ thấy: opencv, opencv_contrib
```

### Bước 3: Cấu hình CMake

```bash
# Tạo thư mục build
mkdir -p ~/opencv_build/opencv/build
cd ~/opencv_build/opencv/build

# Lấy thông tin Python
PYTHON_EXECUTABLE=$(which python3)
PYTHON_INCLUDE=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
PYTHON_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "Python path: $PYTHON_EXECUTABLE"
echo "Python include: $PYTHON_INCLUDE"
echo "Python packages: $PYTHON_PACKAGES"

# Chạy CMake với NONFREE enabled
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D PYTHON3_EXECUTABLE=$PYTHON_EXECUTABLE \
  -D PYTHON3_INCLUDE_DIR=$PYTHON_INCLUDE \
  -D PYTHON3_PACKAGES_PATH=$PYTHON_PACKAGES \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_python2=OFF \
  -D INSTALL_PYTHON_EXAMPLES=OFF \
  -D INSTALL_C_EXAMPLES=OFF \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D BUILD_EXAMPLES=OFF \
  -D WITH_TBB=ON \
  -D WITH_EIGEN=ON \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  ..
```

**Tham số quan trọng:**

- `OPENCV_ENABLE_NONFREE=ON`: Bật thuật toán nonfree (SURF, SIFT cũ, ...)
- `OPENCV_EXTRA_MODULES_PATH`: Đường dẫn tới opencv_contrib/modules
- `PYTHON3_EXECUTABLE`: Đảm bảo build cho đúng Python version
- `WITH_TBB=ON`: Tăng tốc xử lý đa luồng

### Bước 4: Build OpenCV

```bash
# Build (sử dụng tất cả CPU cores)
# Lưu ý: Có thể mất 30-60 phút
NCPU=$(sysctl -n hw.ncpu)
echo "Building với $NCPU cores..."

make -j$NCPU
```

**Nếu build bị lỗi:**

```bash
# Xóa build và cố gắng lại
rm -rf ~/opencv_build/opencv/build
mkdir ~/opencv_build/opencv/build
cd ~/opencv_build/opencv/build
# ... chạy lại CMake + make
```

### Bước 5: Cài đặt OpenCV

```bash
# Cài đặt (cần quyền sudo)
cd ~/opencv_build/opencv/build
sudo make install

# Có thể mất vài phút
```

### Bước 6: Kiểm tra cài đặt

```bash
# Kiểm tra version
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Kiểm tra SURF có khả dụng không
python3 -c "import cv2; print('SURF available:', hasattr(cv2.xfeatures2d, 'SURF_create'))"

# Test SURF hoạt động
python3 << 'EOF'
import cv2
try:
    surf = cv2.xfeatures2d.SURF_create()
    print("✓ SURF initialized successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
EOF
```

**Output mong đợi:**

```
OpenCV version: 4.x.x
SURF available: True
✓ SURF initialized successfully!
```

### Bước 7: Dọn dẹp (tùy chọn)

```bash
# Xóa thư mục build để tiết kiệm dung lượng
# (OpenCV đã được cài vào /usr/local)
rm -rf ~/opencv_build
```

## Xử lý sự cố

### CMake không tìm thấy Python

```bash
# Chỉ định rõ Python path
cmake ... \
  -D PYTHON3_EXECUTABLE=$(which python3) \
  -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  ...
```

### Build bị lỗi thiếu dependencies

```bash
# Cài thêm dependencies
brew install cmake pkg-config jpeg libpng libtiff openexr eigen tbb hdf5 jasper

# Cố gắng build lại
rm -rf ~/opencv_build/opencv/build
mkdir ~/opencv_build/opencv/build
cd ~/opencv_build/opencv/build
# ... chạy lại CMake
```

### OpenCV cũ bị conflict

```bash
# Gỡ OpenCV cũ
pip uninstall opencv-python opencv-contrib-python -y
brew uninstall opencv opencv3 -y  # nếu cài qua brew

# Build mới và cài đặt
cd ~/opencv_build/opencv/build
sudo make install
```

### CMakeLists.txt có lỗi

```bash
# Cập nhật OpenCV source
cd ~/opencv_build/opencv
git pull origin 4.x

# Build lại
rm -rf build
mkdir build
cd build
# ... chạy lại CMake
```

## Kiểm tra OPENCV_ENABLE_NONFREE được kích hoạt

```bash
# Kiểm tra trong Python
python3 << 'EOF'
import cv2
import sys

# Kiểm tra version
print(f"OpenCV version: {cv2.__version__}")
print(f"Python: {sys.version}")

# Kiểm tra các thuật toán nonfree
nonfree_algorithms = {
    "SURF": hasattr(cv2.xfeatures2d, 'SURF_create'),
    "SIFT (old)": hasattr(cv2.xfeatures2d, 'SIFT_create') if hasattr(cv2, 'xfeatures2d') else False,
}

print("\nNonfree algorithms available:")
for name, available in nonfree_algorithms.items():
    status = "✓" if available else "✗"
    print(f"  {status} {name}")

# Test SURF
try:
    surf = cv2.xfeatures2d.SURF_create(400)
    print("\n✓ SURF works correctly!")
except Exception as e:
    print(f"\n✗ SURF error: {e}")
EOF
```

## Liên kết tham khảo

- OpenCV Build Guide: https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install_opencv.html
- OpenCV Extra Modules: https://github.com/opencv/opencv_contrib
- NONFREE Flag: https://docs.opencv.org/4.x/df/d4e/tutorial_nonfree.html

## Ghi chú

- Après build, OpenCV được cài vào `/usr/local/` (shared library và headers)
- Python bindings được cài vào site-packages
- Có thể build lại nếu cần cập nhật OpenCV phiên bản mới
