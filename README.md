# SUML

## Iris Streamlit app (Docker)

Run these commands from the `task_2` folder, which contains the `Dockerfile`.

Build the image:

```bash
docker build -t iris-streamlit .
```

Start the container:

```bash
docker run -p 8501:8501 iris-streamlit
```

Open http://localhost:8501
