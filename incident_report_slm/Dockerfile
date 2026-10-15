FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY wualt_slm ./wualt_slm
RUN pip install --no-cache-dir . \
    && useradd --uid 10001 --create-home wualt \
    && mkdir /data && chown wualt:wualt /data
ENV WUALT_DATA_DIR=/data
USER wualt
EXPOSE 8000
CMD ["wualt-slm", "serve", "--host", "0.0.0.0", "--port", "8000"]
