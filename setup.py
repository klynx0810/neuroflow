from setuptools import setup, find_packages

setup(
    name="neuroflow",
    version="0.1.0",
    author="Lê Trung Kiên",
    author_email="21011496@st.phenikaa-uni.edu.vn",
    description="Đây là thư viện neuroflow được viết theo cấu trúc của tensorflow," \
    "nhưng cả backend và high api đều được viết bằng python, xây dựng sử dụng duy nhất thư viện numpy",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
    ],
    python_requires=">=3.7",
)
