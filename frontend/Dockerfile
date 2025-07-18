FROM node:18-alpine

# Install curl for health checks
RUN apk add --no-cache curl

WORKDIR /app

# Copy frontend files
COPY . .

# Install a simple HTTP server
RUN npm install -g http-server

# Expose port
EXPOSE 3000

# Start the frontend server
CMD ["http-server", "-p", "3000", "-a", "0.0.0.0"]
