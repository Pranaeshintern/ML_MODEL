FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md requirements-tested.txt ./
COPY wualt_slm ./wualt_slm
RUN pip install --no-cache-dir -c requirements-tested.txt . \
    && useradd --uid 10001 --create-home wualt
USER wualt
EXPOSE 8000
CMD ["wualt-slm", "serve", "--host", "0.0.0.0", "--port", "8000"]
