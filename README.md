# Ticki - AI-Powered Face Search System

Ticki is a sophisticated content-based image retrieval (CBIR) system that allows users to search for images containing specific faces within a large repository. Built with computer vision and deep learning technologies, Ticki can process face images, extract facial features, and find similar faces across thousands of images with high accuracy.

<img width="1442" height="958" alt="Ticki - AI Face Search Interface" src="app/static/imgs/ticki.png" />

## Features

- **Face Detection & Extraction**: Automatically detects and extracts faces from uploaded images
- **Feature Embedding**: Uses Facenet512 deep learning model to generate high-quality facial embeddings
- **Fast Similarity Search**: Leverages FAISS for efficient vector similarity search across large datasets
- **Multiple Face Handling**: Supports images with multiple faces and allows selective face searching
- **Real-time Processing**: Asynchronous task processing with Celery for handling large image collections
- **Web Interface**: Clean, user-friendly web interface for image uploads and results visualization
- **Cloudinary Integration**: Webhook support for automatic processing of cloud-uploaded images
- **Scalable Architecture**: Redis-backed task queuing and caching for production scalability

## Architecture

### Core Components

- **Flask Web Application**: REST API and web interface (`app/routes.py`)
- **Image Processing Engine**: Face detection and feature extraction (`app/cbir.py`)
- **Embedding Management**: Pure FAISS-based vector database for similarity search (`app/embeddings.py`)
- **Asynchronous Tasks**: Celery workers for background processing (`app/tasks.py`)
- **Cloudinary Webhook**: External image processing pipeline (`cloudinary_wb.py`)

### Architecture Evolution

The system originally used HDF5 for embedding storage but was migrated to pure FAISS implementation due to thread-safety concerns. HDF5 file operations (open/close) in multithreaded environments could cause crashes, especially under concurrent load. The current FAISS-only approach provides:

- **Thread-safe operations** with proper locking mechanisms
- **In-memory persistence** with disk serialization for durability
- **Atomic rebuilds** when adding new embeddings
- **Better performance** with optimized vector operations

### Technology Stack

- **Backend**: Flask, Python 3.8+
- **Computer Vision**: DeepFace, RetinaFace, OpenCV
- **Machine Learning**: Facenet512, TensorFlow
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Task Queue**: Celery with Redis broker
- **Image Processing**: PIL, NumPy
- **Cloud Storage**: Cloudinary integration
- **Frontend**: HTML5, CSS3, JavaScript

## Installation

### Prerequisites

- Python 3.8 or higher
- Redis server
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ticki.git
   cd Ticki
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0
   CLOUDINARY_CLOUD_NAME=your_cloudinary_name
   CLOUDINARY_API_KEY=your_cloudinary_key
   CLOUDINARY_API_SECRET=your_cloudinary_secret
   CLOUDINARY_WEBHOOK_SECRET=your_webhook_secret
   ```

4. **Start Redis Server**
   ```bash
   redis-server
   ```

5. **Initialize Database Directories**
   The application will automatically create required directories:
   - `app/static/database/img_repo/`
   - `app/static/database/img_repo/extracted_faces/`
   - `app/static/database/img_repo/extracted_faces_embeddings/`

## Usage

### Quick Start

1. **Launch the Application**
   ```bash
   python launcher.py
   ```
   This starts Redis, Celery workers, and the Flask application automatically.

2. **Manual Launch** (Alternative)
   ```bash
   # Terminal 1: Start Celery worker
   celery -A app.celery worker --loglevel=info --concurrency=5 --pool threads
   
   # Terminal 2: Start Flask app
   flask run
   ```

3. **Access the Web Interface**
   Open your browser to `http://localhost:5000`

### Adding Images to Repository

1. Place images in `app/static/database/img_repo/img_data/`
2. The system will automatically process new images and extract faces
3. Facial embeddings will be stored for fast similarity search

### Searching for Faces

1. **Single Face Search**:
   - Upload an image through the web interface
   - System automatically detects faces and finds similar ones
   - Results show matched images with similarity scores

2. **Multiple Face Search**:
   - If uploaded image contains multiple faces, select specific faces to search
   - Submit selected faces for targeted search

### API Endpoints

- `GET/POST /` - Main search interface
- `POST /multiple-faces` - Multiple face selection endpoint
- `POST /cloudinary-webhook` - Cloudinary integration webhook
- `POST /upload` - Direct upload endpoint

## Configuration

### Image Processing Settings

The system supports various image formats (JPG, PNG, JPEG) and automatically:
- Resizes faces to 224x224 pixels for consistency
- Converts images to RGB format
- Normalizes facial embeddings for cosine similarity

### Performance Tuning

- **FAISS Index Type**: Choose between "flat" (exact) or "ivf" (approximate) indexing
- **Similarity Threshold**: Adjustable threshold for match sensitivity (default: 0.67)
- **Celery Concurrency**: Configure worker threads based on system resources
- **GPU Acceleration**: Enable GPU processing by modifying environment variables

## Project Structure

```
Ticki/
├── app/
│   ├── __init__.py              # Flask app initialization
│   ├── routes.py                # Web routes and API endpoints
│   ├── cbir.py                  # Image processing and face extraction
│   ├── embeddings.py            # FAISS-based embedding storage
│   ├── tasks.py                 # Celery background tasks
│   ├── celery.py                # Celery configuration
│   ├── static/
│   │   ├── css/main.css         # Styling
│   │   ├── js/input.js          # Frontend interactions
│   │   └── database/            # Image and embedding storage
│   └── templates/
│       └── main.html            # Web interface template
├── config.py                    # Application configuration
├── launcher.py                  # Multi-service launcher
├── main.py                      # Flask application entry point
├── cloudinary_wb.py             # Cloudinary webhook handler
├── wsgi.py                      # WSGI application server
└── requirements.txt             # Python dependencies
```

## Business Use Cases

### Photography Services
- **Event Photography**: Automatically organize and deliver client photos
- **Portrait Studios**: Quick client photo retrieval and categorization
- **Wedding Photography**: Efficient guest photo identification and delivery

### Security & Surveillance
- **Access Control**: Face-based authentication and monitoring
- **Event Security**: Real-time face recognition for VIP identification
- **Missing Person Search**: Search across large image databases

### Social Media & Content
- **Photo Tagging**: Automatic face-based photo organization
- **Content Moderation**: Identify and track specific individuals
- **User Experience**: Enhanced photo search and discovery

## Technical Details

### Face Detection
- Uses RetinaFace for robust face detection with alignment
- Handles multiple faces per image with individual extraction
- 30-pixel face area expansion for better feature capture

### Feature Extraction
- Facenet512 model generates 512-dimensional embeddings
- L2 normalization ensures consistent similarity calculations
- Features cached as NumPy arrays (.npy files) for fast loading
- BGR format conversion for DeepFace compatibility

### Similarity Search Evolution
The system evolved from HDF5-based storage to pure FAISS implementation:

**Previous HDF5 Approach:**
- Combined HDF5 file storage with FAISS indexing
- Thread-safety issues with concurrent file operations
- Risk of database corruption under heavy load

**Current FAISS-Only Approach:**
- Pure in-memory FAISS index with disk serialization
- Thread-safe operations with proper locking
- Eliminates file I/O bottlenecks during searches
- Automatic index rebuilding when adding new embeddings

### Embedding Management API

The `FaissEmbeddingsStore` class provides a comprehensive API for managing facial embeddings:

#### Public Methods

```python
from app.embeddings import embeddings_handler

# Load all face embeddings from files into FAISS index
embeddings_handler.load_all_embeddings_in_faiss(sync_mode=False, external=False)

# Add new feature vector to the index (with background rebuild)
embeddings_handler.add_feature(query_feature, image_path, sync_mode=True)

# Search for similar faces with threshold-based filtering
results = embeddings_handler.get_similar_faces(query_feature, threshold=0.67)

# Top-K similarity search
top_results = embeddings_handler.search_topk(query_feature, k=50)
```

#### Technical Implementation

**Index Types:**
- `flat`: Exact search using `IndexFlatIP` (Inner Product for cosine similarity)
- `ivf`: Approximate search using `IndexIVFFlat` with adaptive clustering

**Thread Safety:**
- `threading.Lock()` protects index rebuilds and writes
- Atomic operations ensure data consistency
- Background processing prevents UI blocking

**Memory Management:**
- In-memory index with disk persistence (`faiss_index.bin`, `faiss_names.pkl`)
- L2 normalization for consistent cosine similarity calculations
- Automatic index reconstruction when adding new embeddings

**Search Algorithms:**
- `range_search()`: Returns all matches above threshold (FLAT index only)
- `search()`: Top-K search with threshold filtering (IVF index)
- Batch query support for multiple face matching

### Scalability Features
- Redis-based task deduplication prevents duplicate processing
- Pure FAISS storage eliminates HDF5 thread-safety issues
- Thread-safe operations with proper locking mechanisms
- Background processing prevents UI blocking
- Automatic index rebuilding with atomic operations

## Development

### Adding New Features
1. Follow the existing architecture patterns
2. Use Celery tasks for CPU-intensive operations
3. Maintain thread-safety for shared resources
4. Update embeddings index after adding new images

### Testing
Place test images in appropriate directories and run:
```bash
python -c "from app.tasks import main; main(reprocess=True)"
```

### Deployment
For production deployment:
1. Use Gunicorn WSGI server (`wsgi.py` provided)
2. Configure Redis persistence
3. Set up proper logging and monitoring
4. Enable SSL/HTTPS for web interface

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues, feature requests, or questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include system information and error logs

---

**Note**: This system processes and stores facial embeddings. Ensure compliance with local privacy laws and regulations when deploying in production environments.