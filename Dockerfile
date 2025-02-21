# Use the official Python image from the Docker Hub
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and the app file into the container
COPY style.css ./
COPY requirements.txt ./
COPY app.py ./

# Install the required packages
RUN pip install -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Build the Docker image #
# docker build -t my-streamlit-app .

# Run the Docker container #
# docker run -p 8501:8501 my-streamlit-app


