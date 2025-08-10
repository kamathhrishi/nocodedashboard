from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import json
import pandas as pd
import numpy as np
import logging
import traceback

# Import the existing CSV Dashboard Generator
from main import CSVDashboardGenerator

app = FastAPI(
    title="AI-Powered CSV Dashboard Generator API",
    description="Upload CSV files and generate intelligent dashboards with AI-powered insights",
    version="2.0.0"
)

# Add debug logging for all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"🔍 Incoming request: {request.method} {request.url}")
    print(f"🔍 Headers: {dict(request.headers)}")
    
    if request.method == "POST":
        try:
            body = await request.body()
            print(f"🔍 Request body length: {len(body)} bytes")
        except Exception as e:
            print(f"🔍 Could not read request body: {e}")
    
    response = await call_next(request)
    print(f"🔍 Response status: {response.status_code}")
    return response

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create temporary directory for file uploads
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create output directory for generated dashboards
OUTPUT_DIR = Path("generated_dashboards")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files for serving generated dashboards
# Use absolute path to ensure proper file serving
app.mount("/dashboards", StaticFiles(directory=str(OUTPUT_DIR.absolute())), name="dashboards")

# Add debug endpoint to check static file serving
@app.get("/debug/static-files")
async def debug_static_files():
    """Debug endpoint to check static file configuration"""
    dashboard_files = list(OUTPUT_DIR.glob("*.html"))
    return {
        "output_dir": str(OUTPUT_DIR.absolute()),
        "output_dir_exists": OUTPUT_DIR.exists(),
        "dashboard_files": [f.name for f in dashboard_files],
        "static_mount_path": "/dashboards"
    }

# Templates for the web interface
templates = Jinja2Templates(directory="templates")

# OpenAI API key configuration (set this in environment variable OPENAI_API_KEY)
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Home page with file upload form"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI-Powered CSV Dashboard Generator</title>
        <style>
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            .form-group {
                margin-bottom: 25px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                font-size: 1.1em;
            }
            input[type="file"] {
                width: 100%;
                padding: 15px;
                border: none;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.9);
                font-size: 16px;
                box-sizing: border-box;
                cursor: pointer;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            .info {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
                text-align: center;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .feature {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .feature h3 {
                margin-top: 0;
                color: #ffd700;
            }
            .api-note {
                background: rgba(255, 215, 0, 0.2);
                border: 1px solid #ffd700;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 25px;
                text-align: center;
                color: #ffd700;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AI-Powered CSV Dashboard Generator</h1>
            
            <div class="api-note">
                <strong>🔑 API Key Configured Server-Side</strong><br>
                Your OpenAI API key is securely configured on the server. No need to enter it here!
            </div>
            
            <form action="/upload" method="post" enctype="multipart/form-data" id="uploadForm">
                <div class="form-group">
                    <label for="csv_file">CSV File:</label>
                    <input type="file" id="csv_file" name="file" accept=".csv" required>
                </div>
                
                <button type="submit">Generate AI Dashboard</button>
            </form>
            
            <script>
                document.getElementById('uploadForm').addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData();
                    const fileInput = document.getElementById('csv_file');
                    
                    if (!fileInput.files[0]) {
                        alert('Please select a CSV file');
                        return;
                    }
                    
                    formData.append('file', fileInput.files[0]);
                    
                    console.log('Form data prepared:');
                    console.log('File:', fileInput.files[0] ? fileInput.files[0].name : 'Missing');
                    
                    fetch('/upload', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => {
                        console.log('Response status:', response.status);
                        if (!response.ok) {
                            return response.text().then(text => {
                                throw new Error(`HTTP ${response.status}: ${text}`);
                            });
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Success:', data);
                        alert('Dashboard generated successfully!');
                        if (data.dashboard_url) {
                            window.open(data.dashboard_url, '_blank');
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error: ' + error.message);
                    });
                });
            </script>
            
            <div class="info">
                <h3>📊 What This Tool Does</h3>
                <p>Upload your CSV file and get an intelligent, AI-powered dashboard with:</p>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h3>🤖 AI Analysis</h3>
                    <p>Smart insights and recommendations</p>
                </div>
                <div class="feature">
                    <h3>📈 Visualizations</h3>
                    <p>Interactive charts and graphs</p>
                </div>
                <div class="feature">
                    <h3>📋 Data Summary</h3>
                    <p>Comprehensive data overview</p>
                </div>
                <div class="feature">
                    <h3>🎨 Professional UI</h3>
                    <p>Beautiful, responsive dashboard</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    custom_prompt: str = Form(None)
):
    """Handle CSV file upload and generate dashboard"""
    
    print(f"🔍 Upload endpoint called")
    print(f"🔍 File type: {type(file)}")
    
    # Validate that we received the required parameters
    if not file:
        print(f"❌ No file received")
        raise HTTPException(status_code=400, detail="File is required")
    
    # Use server-side API key
    api_key = OPENAI_API_KEY
    print(f"🔑 Using server-side API key")
    
    try:
        # Debug logging
        print(f"📁 Received file upload: {file.filename}")
        print(f"📊 File size: {file.size} bytes")
        print(f"🔍 Content type: {file.content_type}")
        print(f"🔑 API key configured: {api_key[:10] if api_key else 'None'}...")
        print(f"🔍 Request received successfully - processing...")
    except Exception as e:
        print(f"❌ Error in debug logging: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in request processing: {str(e)}")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Validate API key (should not be empty at this point)
    if not api_key or (isinstance(api_key, str) and api_key.strip() == "") or api_key == "your-api-key-here":
        raise HTTPException(
            status_code=500, 
            detail="No valid API key configured on the server. Please contact the administrator to configure an OpenAI API key."
        )
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    temp_file_path = UPLOAD_DIR / f"{unique_id}_{file.filename}"
    output_file = OUTPUT_DIR / f"dashboard_{unique_id}.html"
    
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create dashboard generator instance with server-side API key
        generator = CSVDashboardGenerator(api_key, str(temp_file_path))
        
        # Set custom prompt if provided
        if custom_prompt and custom_prompt.strip():
            print(f"🔍 Custom prompt provided: {custom_prompt[:100]}...")
            # Store the custom prompt for use in dashboard generation
            generator.custom_prompt = custom_prompt.strip()
        
        # Generate dashboard
        success = generator.run_complete_analysis(str(output_file))
        
        if success:
            # Clean up temporary file
            temp_file_path.unlink()
            
            # Return success response with dashboard link
            dashboard_url = f"/dashboards/dashboard_{unique_id}.html"
            direct_url = f"/dashboard/{unique_id}"
            
            # Verify the file exists before returning the URL
            if not output_file.exists():
                raise HTTPException(status_code=500, detail="Dashboard file was not created")
            
            print(f"✅ Dashboard generated successfully: {output_file}")
            print(f"🌐 Dashboard URL (static): {dashboard_url}")
            print(f"🌐 Dashboard URL (direct): {direct_url}")
            print(f"📁 File exists: {output_file.exists()}")
            print(f"📁 File size: {output_file.stat().st_size if output_file.exists() else 'N/A'} bytes")
            
            return {
                "success": True,
                "message": "Dashboard generated successfully!",
                "dashboard_url": dashboard_url,
                "direct_url": direct_url,
                "download_url": f"/download/{unique_id}",
                "upload_id": unique_id
            }
        else:
            # Clean up on failure
            temp_file_path.unlink()
            raise HTTPException(status_code=500, detail="Failed to generate dashboard")
            
    except Exception as e:
        # Clean up on error
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/dashboard/{unique_id}")
async def serve_dashboard(unique_id: str):
    """Direct endpoint to serve dashboard files"""
    file_path = OUTPUT_DIR / f"dashboard_{unique_id}.html"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    # Read and return the HTML content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dashboard: {str(e)}")

@app.get("/download/{unique_id}")
async def download_dashboard(unique_id: str):
    """Download the generated dashboard file"""
    file_path = OUTPUT_DIR / f"dashboard_{unique_id}.html"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return FileResponse(
        path=file_path,
        filename=f"dashboard_{unique_id}.html",
        media_type="text/html"
    )

@app.get("/status/{unique_id}")
async def check_status(unique_id: str):
    """Check the status of dashboard generation"""
    file_path = OUTPUT_DIR / f"dashboard_{unique_id}.html"
    
    if file_path.exists():
        return {
            "status": "completed",
            "dashboard_url": f"/dashboards/dashboard_{unique_id}.html",
            "direct_url": f"/dashboard/{unique_id}",
            "download_url": f"/download/{unique_id}"
        }
    else:
        return {"status": "processing"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "CSV Dashboard Generator API"
    }

@app.get("/api/endpoints")
async def list_endpoints():
    """List available API endpoints"""
    return {
        "endpoints": {
            "GET /": "Home page with upload form",
            "POST /upload": "Upload CSV and generate dashboard",
            "GET /download/{unique_id}": "Download generated dashboard",
            "GET /status/{unique_id}": "Check generation status",
            "GET /api/health": "Health check",
            "GET /api/endpoints": "List endpoints"
        }
    }

@app.post("/test-upload")
async def test_upload(
    file: UploadFile = File(...)
):
    """Test endpoint for debugging upload issues"""
    print(f"🔍 Test upload endpoint called")
    
    # Use server-side API key
    api_key = OPENAI_API_KEY
    print(f"🔑 Using server-side API key for test")
    
    print(f"🔍 API key: {api_key[:10] if api_key else 'None'}...")
    print(f"🔍 File: {file.filename}")
    return {"message": "Test upload successful", "filename": file.filename, "api_key_configured": bool(api_key)}

@app.post("/simple-test")
async def simple_test():
    """Simple test endpoint without file upload"""
    print(f"🔍 Simple test endpoint called")
    
    # Use server-side API key
    api_key = OPENAI_API_KEY
    print(f"🔑 Using server-side API key for test")
    
    print(f"🔍 API key: {api_key[:10] if api_key else 'None'}...")
    return {"message": "Simple test successful", "api_key_configured": bool(api_key)}

@app.post("/preview-csv")
async def preview_csv(file: UploadFile = File(...)):
    """Preview CSV data - returns first few rows and basic info"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read CSV data
        df = pd.read_csv(file.file)
        
        # Get basic info
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "preview": df.head(10).to_dict('records'),  # First 10 rows
            "summary_stats": {}
        }
        
        # Add summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            info["summary_stats"][col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "median": float(df[col].median()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None
            }
        
        # Add value counts for categorical columns (first 10 unique values)
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            value_counts = df[col].value_counts().head(10)
            info["summary_stats"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": value_counts.to_dict()
            }
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting CSV Dashboard Generator FastAPI Server...")
    print("📁 Upload directory:", UPLOAD_DIR.absolute())
    print("📁 Output directory:", OUTPUT_DIR.absolute())
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔍 Debug mode enabled - all requests will be logged")
    
    # Check if a valid API key is configured
    if OPENAI_API_KEY == "your-api-key-here" or not OPENAI_API_KEY:
        print("⚠️  Warning: No valid OpenAI API key configured!")
        print("   Set the OPENAI_API_KEY environment variable to enable dashboard generation")
        print("   The server will not function without a valid API key")
    else:
        print(f"✅ OpenAI API key configured: {OPENAI_API_KEY[:10]}...")
        print("   Users can now upload CSV files without needing to provide API keys")
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
