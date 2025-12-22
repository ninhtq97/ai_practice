#!/usr/bin/env python3
"""
Script chẩn đoán OpenCV SURF availability
Giúp xác định lý do SURF không khả dụng
"""

import sys
import cv2
from pathlib import Path


def check_opencv_info():
    """Kiểm tra thông tin OpenCV"""
    print("=" * 60)
    print("OpenCV Diagnostic Report")
    print("=" * 60)
    print()

    print("1. OpenCV Version:")
    print(f"   Version: {cv2.__version__}")
    print(f"   Location: {cv2.__file__}")
    print()

    print("2. Python Info:")
    print(f"   Python: {sys.version}")
    print(f"   Executable: {sys.executable}")
    print()


def check_xfeatures2d():
    """Kiểm tra xfeatures2d module"""
    print("3. xfeatures2d Module:")
    try:
        import cv2.xfeatures2d as xf2d
        print(f"   ✓ Module imported successfully")
        print(f"   Location: {xf2d.__file__}")
        print()

        print("4. Available algorithms in xfeatures2d:")
        algorithms = [
            'SURF_create', 'SIFT_create', 'BRIEF_create',
            'DAISY_create', 'FREAK_create', 'AKAZE_create'
        ]

        for algo in algorithms:
            has_algo = hasattr(xf2d, algo)
            status = "✓" if has_algo else "✗"
            print(f"   {status} {algo}")

        print()

        # Try to create SURF
        print("5. Testing SURF creation:")
        try:
            surf = cv2.xfeatures2d.SURF_create()
            print("   ✓ SURF created successfully!")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False

    except ImportError as e:
        print(f"   ✗ Failed to import xfeatures2d: {e}")
        print()
        print("   Possible solutions:")
        print("   1. OpenCV not built with OPENCV_ENABLE_NONFREE=ON")
        print("   2. opencv_contrib modules not compiled")
        print("   3. Python bindings not installed correctly")
        print()
        return False


def check_opencv_build_info():
    """Kiểm tra build configuration"""
    print("6. Build Configuration:")
    try:
        info = cv2.getBuildInformation()

        # Tìm NONFREE flag
        if 'OPENCV_ENABLE_NONFREE' in info:
            lines = [l for l in info.split('\n') if 'OPENCV_ENABLE_NONFREE' in l or 'nonfree' in l.lower()]
            for line in lines:
                print(f"   {line.strip()}")
        else:
            print("   ⚠ OPENCV_ENABLE_NONFREE info not found in build info")

        # Tìm xfeatures2d
        if 'xfeatures2d' in info:
            lines = [l for l in info.split('\n') if 'xfeatures2d' in l.lower()]
            for line in lines:
                print(f"   {line.strip()}")
        else:
            print("   ✗ xfeatures2d module not found in build info")

        print()

    except Exception as e:
        print(f"   Error reading build info: {e}")
        print()


def suggest_solutions():
    """Suggest solutions"""
    print("7. Recommended Solutions:")
    print()
    print("   If SURF is not available:")
    print()
    print("   Option A: Rebuild OpenCV properly")
    print("   -" * 30)
    print("   1. Remove old OpenCV:")
    print("      pip uninstall opencv-python opencv-contrib-python -y")
    print("      brew uninstall opencv opencv3 -y  # if installed via brew")
    print()
    print("   2. Clean build directory:")
    print("      rm -rf ~/opencv_build/opencv/build")
    print()
    print("   3. Run CMake with proper flags:")
    print("      mkdir ~/opencv_build/opencv/build && cd ~/opencv_build/opencv/build")
    print("      cmake -D CMAKE_BUILD_TYPE=Release \\")
    print("            -D OPENCV_ENABLE_NONFREE=ON \\")
    print("            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \\")
    print("            -D PYTHON3_EXECUTABLE=$(which python3) \\")
    print("            -D PYTHON3_PACKAGES_PATH=$(python3 -c 'import site; print(site.getsitepackages()[0])') \\")
    print("            -D BUILD_opencv_python3=ON \\")
    print("            -D WITH_TBB=ON ..")
    print()
    print("   4. Build:")
    print("      make -j$(sysctl -n hw.ncpu)")
    print()
    print("   5. Install:")
    print("      sudo make install")
    print()
    print("   Option B: Use alternative features")
    print("   -" * 30)
    print("   - SIFT is available by default (cv2.SIFT_create())")
    print("   - ORB is available by default (cv2.ORB_create())")
    print()
    print("   Option C: Verify build was successful")
    print("   -" * 30)
    print("   cd ~/opencv_build/opencv/build")
    print("   cmake -LA | grep NONFREE  # Check if NONFREE=ON")
    print("   make -j4  # Retry build")
    print("   sudo make install")
    print()


def main():
    check_opencv_info()
    has_surf = check_xfeatures2d()
    check_opencv_build_info()

    if has_surf:
        print("=" * 60)
        print("✓ SURF is available and working correctly!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("✗ SURF is not available")
        print("=" * 60)
        print()
        suggest_solutions()
        return 1


if __name__ == '__main__':
    exit(main())
