# Navigate to the project directory
cd path\to\sna_dashboard

# Stop and remove the existing container
docker stop sna_dashboard -ErrorAction SilentlyContinue
docker rm sna_dashboard -ErrorAction SilentlyContinue

# Build the Docker image
docker build -t sna-dashboard .

# Run the Docker container and auto-remove it
docker run --rm -d -p 8080:8080 --name sna_dashboard sna-dashboard
