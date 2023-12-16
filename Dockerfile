# 使用 Python 3.10 的基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 将项目文件复制到容器中
COPY . .

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装项目依赖包
RUN pip install -r requirements.txt

# 暴露应用程序所需的端口（如果需要）
EXPOSE 8100

# 定义启动命令，这里假设您的主文件名为 app.py
CMD ["python", "api.py"]

