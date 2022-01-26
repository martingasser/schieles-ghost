FROM python:3.9


# Update default packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl

# Update new packages
RUN apt-get update

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

#RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
RUN pip install -U pip
#RUN pip install setuptools_rust docker-compose
RUN pip install transformers
RUN pip install -r requirements.txt

# # Copy app code and set working directory
COPY app.py app.py
COPY gpt2-schiele gpt2-schiele
COPY schiele_names.txt schiele_names.txt
WORKDIR .

EXPOSE 8501

# Run
#ENTRYPOINT ["streamlit", "run"]
#CMD ["app.py", "–-server.port=8080", "–-server.address=0.0.0.0"]
CMD ["streamlit", "run", "app.py", "–-server.port=8080", "–-server.address=0.0.0.0"]

