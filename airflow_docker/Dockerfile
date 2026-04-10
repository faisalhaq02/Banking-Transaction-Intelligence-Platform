FROM apache/airflow:2.9.3

# -------------------------
# 1) OS deps + Java (root)
# -------------------------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      procps \
      openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 2) Spark install (root)
# -------------------------
ENV SPARK_VERSION=3.5.1
RUN curl -L "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
  | tar -xz -C /opt \
  && ln -s "/opt/spark-${SPARK_VERSION}-bin-hadoop3" /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH="/opt/spark/bin:${PATH}"

# -------------------------
# 3) Iceberg runtime jar (root)
# -------------------------
ENV ICEBERG_VERSION=1.10.1
RUN curl -L -o "/opt/spark/jars/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar" \
  "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/${ICEBERG_VERSION}/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar"

# -------------------------
# 4) Python deps + Airflow provider (airflow user)
#    IMPORTANT: no pip as root on Airflow image
# -------------------------
USER airflow

# Install analytics deps 
RUN pip install --no-cache-dir duckdb pandas pyarrow scikit-learn joblib matplotlib

# Installs Spark provider with Airflow constraints 
RUN PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" && \
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-${PY_VER}.txt" && \
    echo "Using constraints: ${CONSTRAINT_URL}" && \
    pip install --no-cache-dir "apache-airflow-providers-apache-spark" --constraint "${CONSTRAINT_URL}"

# sanity check during build 
RUN airflow version