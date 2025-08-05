from typing import Optional
import os
import uuid
import zipfile
import tempfile

# Configure matplotlib for web environment BEFORE any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query, Form  # noqa: E402
from fastapi.responses import HTMLResponse, FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
import numpy as np  # noqa: E402
from skimage import io  # noqa: E402
from PIL import Image  # noqa: E402
from pyxelate import Pyx  # noqa: E402

import utils  # noqa: E402
from metadata import MetadataDBManager  # noqa: E402

app = FastAPI(title="Pyxelate to Color", description="Convert images to pixel art with color grids")

# Create directories if they don't exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize metadata manager
metadata_manager = MetadataDBManager()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    downsample_by: int = Form(20),
    palette: int = Form(8),
    upscale: int = Form(100)
):
    # Debug: Print received parameters
    print("🔍 DEBUG - Received parameters:")
    print(f"  - downsample_by: {downsample_by} (type: {type(downsample_by)})")
    print(f"  - palette: {palette} (type: {type(palette)})")
    print(f"  - upscale: {upscale} (type: {type(upscale)})")

    # Debug: Print raw form data
    try:
        form_data = await request.form()
        print("🔍 DEBUG - Raw form data:")
        for key, value in form_data.items():
            if key != 'file':  # Don't print file content
                print(f"  - {key}: {value} (type: {type(value)})")
    except Exception as e:
        print(f"⚠️  Could not read raw form data: {e}")

    if not file.content_type.startswith('image/'):
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "File must be an image"
        })

    # Generate unique filename
    file_id = str(uuid.uuid4())
    upload_path = f"static/uploads/{file_id}_{file.filename}"

    # Save uploaded file
    with open(upload_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    try:
        # Process the image
        image = io.imread(upload_path)

        # Pixelate options
        pixelate_options = utils.PixelateOptions(
            downsample_by=downsample_by,
            upscale=upscale,
            palette=palette,
        )

        # Debug: Print the options being used
        print("🔍 DEBUG - PixelateOptions created:")
        print(f"  - downsample_by: {pixelate_options.downsample_by}")
        print(f"  - upscale: {pixelate_options.upscale}")
        print(f"  - palette: {pixelate_options.palette}")

        # Pixelate the image
        pyxelated_image, pyx = utils.pixelate_image(
            image=image,
            options=pixelate_options
        )

        # Save the pixelated version
        pixelated_path = f"static/outputs/{file_id}_pixelated.png"
        io.imsave(pixelated_path, pyxelated_image)

        # Save metadata for gallery
        file_size = os.path.getsize(pixelated_path)
        # Flatten colors to ensure proper structure [r, g, b] not [[r, g, b]]
        flattened_colors = pyx.colors.reshape(-1, 3).tolist()
        metadata_manager.add_image_metadata(
            file_id=file_id,
            original_filename=file.filename,
            downsample_by=downsample_by,
            palette=palette,
            upscale=upscale,
            colors=flattened_colors,
            file_size=file_size,
            original_size=image.shape[:2] if len(image.shape) >= 2 else None,
            pixelated_size=pyxelated_image.shape[:2] if len(pyxelated_image.shape) >= 2 else None
        )

        # Return template response for HTMX
        return templates.TemplateResponse("results.html", {
            "request": request,
            "file_id": file_id,
            "original_url": f"/{upload_path}",
            "pixelated_url": f"/{pixelated_path}",
            "result_url": None,  # No grid generated yet
            "colors": flattened_colors,
            "palette_size": palette,
            "has_grid": False
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}"
        })

def build_gallery_url(request: Request, **kwargs):
    """Helper function to build gallery URLs with query parameters."""
    from urllib.parse import urlencode

    # Start with current query parameters
    params = dict(request.query_params)

    # Update with new parameters
    params.update(kwargs)

    # Remove None values and convert to strings
    filtered_params = {k: str(v) for k, v in params.items() if v is not None}

    # Build URL
    query_string = urlencode(filtered_params)
    return f"/gallery?{query_string}" if query_string else "/gallery"

@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request,
                 page: int = Query(1, ge=1),
                 per_page: int = Query(12, ge=1, le=50),
                 sort_by: str = Query("timestamp", regex="^(timestamp|palette|downsample_by|upscale|original_filename)$"),
                 sort_order: str = Query("desc", regex="^(asc|desc)$"),
                 filter_palette: Optional[int] = Query(None, ge=2, le=32),
                 filter_downsample: Optional[int] = Query(None, ge=1, le=50),
                 search: Optional[str] = Query(None)):
    """Gallery page to browse all processed images."""
        # Get all metadata
    all_metadata = metadata_manager.get_all_metadata()

    # Apply filters
    filtered_metadata = all_metadata

    if filter_palette:
        filtered_metadata = [m for m in filtered_metadata if m.palette == filter_palette]

    if filter_downsample:
        filtered_metadata = [m for m in filtered_metadata if m.downsample_by == filter_downsample]

    if search:
        search_lower = search.lower()
        filtered_metadata = [m for m in filtered_metadata if search_lower in m.original_filename.lower()]

    # Apply sorting
    reverse = sort_order == "desc"
    if sort_by == "timestamp":
        filtered_metadata = sorted(filtered_metadata, key=lambda x: x.timestamp, reverse=reverse)
    elif sort_by == "palette":
        filtered_metadata = sorted(filtered_metadata, key=lambda x: x.palette, reverse=reverse)
    elif sort_by == "downsample_by":
        filtered_metadata = sorted(filtered_metadata, key=lambda x: x.downsample_by, reverse=reverse)
    elif sort_by == "upscale":
        filtered_metadata = sorted(filtered_metadata, key=lambda x: x.upscale, reverse=reverse)
    elif sort_by == "original_filename":
        filtered_metadata = sorted(filtered_metadata, key=lambda x: x.original_filename.lower(), reverse=reverse)

    # Pagination
    total_items = len(filtered_metadata)
    total_pages = (total_items + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_metadata = filtered_metadata[start_idx:end_idx]

    # Get stats and filter options
    stats = metadata_manager.get_stats()
    unique_palettes = sorted(list(set(m.palette for m in all_metadata)))
    unique_downsample = sorted(list(set(m.downsample_by for m in all_metadata)))

    # Build pagination URLs
    prev_url = build_gallery_url(request, page=page-1) if page > 1 else None
    next_url = build_gallery_url(request, page=page+1) if page < total_pages else None

    # Build page number URLs
    page_urls = {}
    start_page = max(1, page - 2)
    end_page = min(total_pages, page + 2)

    for page_num in range(start_page, end_page + 1):
        page_urls[page_num] = build_gallery_url(request, page=page_num)

    # Add first and last page URLs if needed
    if start_page > 1:
        page_urls[1] = build_gallery_url(request, page=1)
    if end_page < total_pages:
        page_urls[total_pages] = build_gallery_url(request, page=total_pages)

    # Build per_page URLs
    per_page_urls = {}
    for count in [12, 24, 48]:
        per_page_urls[count] = build_gallery_url(request, per_page=count, page=1)  # Reset to page 1 when changing per_page

    return templates.TemplateResponse("gallery.html", {
        "request": request,
        "images": page_metadata,
        "current_page": page,
        "total_pages": total_pages,
        "total_items": total_items,
        "per_page": per_page,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "filter_palette": filter_palette,
        "filter_downsample": filter_downsample,
        "search": search or "",
        "stats": stats,
        "unique_palettes": unique_palettes,
        "unique_downsample": unique_downsample,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < total_pages else None,
        "prev_url": prev_url,
        "next_url": next_url,
        "page_urls": page_urls,
        "per_page_urls": per_page_urls,
        "start_page": start_page,
        "end_page": end_page
    })

@app.get("/image/{file_id}", response_class=HTMLResponse)
async def view_image(request: Request, file_id: str):
    """View a specific processed image with details."""
    metadata = metadata_manager.get_metadata_by_id(file_id)
    print(f"🔍 DEBUG - File ID: {file_id}")
    print(f"🔍 DEBUG - Metadata: {metadata}")

    # Check if files exist first
    result_path = f"static/outputs/{file_id}_result.png"
    pixelated_path = f"static/outputs/{file_id}_pixelated.png"
    
    print("🔍 DEBUG - Checking paths:")
    print(f"  Result: {result_path} - Exists: {os.path.exists(result_path)}")
    print(f"  Pixelated: {pixelated_path} - Exists: {os.path.exists(pixelated_path)}")
    
    # Check if pixelated file exists (minimum requirement)
    if not os.path.exists(pixelated_path):
        raise HTTPException(status_code=404, detail="Pixelated image not found")
    
    # Check if grid exists
    has_grid = os.path.exists(result_path)

    # If no metadata exists, create default metadata
    if not metadata:
        file_size = os.path.getsize(result_path)
        metadata = metadata_manager.add_image_metadata(
            file_id=file_id,
            original_filename=f"unknown_{file_id}.png",
            downsample_by=20,
            palette=8,
            upscale=100,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                   [255, 0, 255], [0, 255, 255], [0, 0, 0], [255, 255, 255]],
            file_size=file_size
        )

    return templates.TemplateResponse("image_detail.html", {
        "request": request,
        "metadata": metadata,
        "result_url": f"/static/outputs/{file_id}_result.png" if has_grid else None,
        "pixelated_url": f"/static/outputs/{file_id}_pixelated.png",
        "has_grid": has_grid,
        "file_id": file_id
    })

@app.post("/generate-grid/{file_id}", response_class=HTMLResponse)
async def generate_grid(request: Request, file_id: str):
    """Generate color grid for an existing pixelated image."""
    # Get metadata
    metadata = metadata_manager.get_metadata_by_id(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Image metadata not found")
    
    # Check if pixelated image exists
    pixelated_path = f"static/outputs/{file_id}_pixelated.png"
    if not os.path.exists(pixelated_path):
        raise HTTPException(status_code=404, detail="Pixelated image not found")
    
    # Check if grid already exists
    result_path = f"static/outputs/{file_id}_result.png"
    result_url = f"/static/outputs/{file_id}_result.png"
    
    if not os.path.exists(result_path):
        try:
            # Load the pixelated image
            pixelated_image = io.imread(pixelated_path)
            
            # Recreate the Pyx object with the saved colors
            pyx = Pyx(
                factor=metadata['downsample_by'],
                palette=metadata['palette'],
                upscale=metadata['upscale']
            )
            # Set the colors from metadata
            pyx.colors = np.array(metadata['colors']).reshape(-1, 3)
            
            # Create grid with color indices
            result_image = utils.create_draw_grid(
                pixelated_image,
                cell_size=metadata['upscale'],
                pyx=pyx,
            )
            
            # Save result
            io.imsave(result_path, result_image)
            
        except Exception as e:
            # Return error as HTML
            return f"""
            <div class="bg-gray-100 rounded-lg p-8 text-center">
                <div class="text-red-500 mb-4">
                    <p class="text-lg font-medium mb-2">Failed to generate grid</p>
                    <p class="text-sm">{str(e)}</p>
                </div>
                <button hx-post="/generate-grid/{file_id}" 
                        hx-target="this" 
                        hx-swap="outerHTML"
                        class="bg-gradient-to-r from-pixel-blue to-pixel-purple text-white font-bold py-3 px-6 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all duration-200">
                    🔄 Try Again
                </button>
            </div>
            """
    
    # Return the grid section HTML
    return f"""
    <div class="bg-gray-100 rounded-lg p-4">
        <img src="{result_url}" alt="Grid" class="max-w-full h-auto mx-auto rounded-lg shadow-md">
    </div>
    <div class="mt-4 flex justify-center">
        <a href="/download/{file_id}"
           class="bg-gradient-to-r from-pixel-blue to-pixel-purple text-white font-bold py-2 px-4 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all duration-200">
            📥 Download Grid
        </a>
    </div>
    """

@app.post("/cleanup")
async def cleanup_gallery():
    """Clean up orphaned metadata entries."""
    cleaned_count = metadata_manager.cleanup_orphaned_metadata()
    return {"cleaned": cleaned_count, "message": f"Cleaned up {cleaned_count} orphaned entries"}

@app.delete("/image/{file_id}")
async def delete_image(file_id: str):
    """Delete a processed image and its metadata."""
    print(f"🗑️ DELETE request for image: {file_id}")
    
    # Delete files
    result_path = f"static/outputs/{file_id}_result.png"
    pixelated_path = f"static/outputs/{file_id}_pixelated.png"

    deleted_files = 0
    if os.path.exists(result_path):
        os.remove(result_path)
        deleted_files += 1
        print(f"  ✓ Deleted result file: {result_path}")
    if os.path.exists(pixelated_path):
        os.remove(pixelated_path)
        deleted_files += 1
        print(f"  ✓ Deleted pixelated file: {pixelated_path}")

    # Clean up upload files (they might have different original names)
    import glob
    upload_files = glob.glob(f"static/uploads/{file_id}_*")
    for upload_file in upload_files:
        os.remove(upload_file)
        deleted_files += 1
        print(f"  ✓ Deleted upload file: {upload_file}")

    # Delete metadata
    metadata_deleted = metadata_manager.delete_metadata(file_id)
    print(f"  {'✓' if metadata_deleted else '✗'} Metadata deletion: {metadata_deleted}")

    if not metadata_deleted:
        print(f"  ⚠️ WARNING: Metadata was not deleted for {file_id}")

    return {
        "deleted_files": deleted_files,
        "metadata_deleted": metadata_deleted,
        "message": f"Deleted {deleted_files} files and {'metadata' if metadata_deleted else 'NO METADATA'}"
    }

@app.get("/download/{file_id}")
async def download_result(file_id: str):
    file_path = f"static/outputs/{file_id}_result.png"
    if not os.path.exists(file_path):
        # If grid doesn't exist, download pixelated version instead
        pixelated_path = f"static/outputs/{file_id}_pixelated.png"
        if os.path.exists(pixelated_path):
            return FileResponse(pixelated_path, filename=f"pixelated_{file_id}.png")
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type='image/png',
        filename=f"pixelated_grid_{file_id}.png"
    )

@app.get("/download_quadrants/{file_id}")
async def download_quadrants(file_id: str):
    """Download the result image (with grid numbers) split into 4 quadrants as a zip file."""
    result_path = f"static/outputs/{file_id}_result.png"

    # Check if the result image exists
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result image not found")

    try:
        # Load the image using PIL
        img = Image.open(result_path)

        # Get dimensions and compute midpoints
        width, height = img.size
        mid_x = width // 2
        mid_y = height // 2

        # Check if image is large enough to split meaningfully
        if width < 4 or height < 4:
            raise HTTPException(status_code=400, detail="Image too small to split into quadrants")

        # Crop the image into 4 quadrants
        quadrants = {
            "top_left": img.crop((0, 0, mid_x, mid_y)),
            "top_right": img.crop((mid_x, 0, width, mid_y)),
            "bottom_left": img.crop((0, mid_y, mid_x, height)),
            "bottom_right": img.crop((mid_x, mid_y, width, height)),
        }

        # Get original filename for better naming
        metadata = metadata_manager.get_metadata_by_id(file_id)
        base_name = metadata.original_filename if metadata else f"image_{file_id}"
        # Remove extension if present
        base_name = os.path.splitext(base_name)[0]

        # Create a temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Save each quadrant to the zip
                for quadrant_name, quadrant_img in quadrants.items():
                    # Create temporary file for each quadrant
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                        quadrant_img.save(tmp_img.name, 'PNG')
                        tmp_img.flush()

                        # Add to zip with descriptive filename
                        zip_filename = f"{base_name}_{quadrant_name}.png"
                        zipf.write(tmp_img.name, zip_filename)

                        # Clean up temporary image file
                        os.unlink(tmp_img.name)

            # Return the zip file
            zip_filename = f"{base_name}_quadrants.zip"

            def cleanup_temp_file():
                """Clean up the temporary zip file after sending."""
                try:
                    os.unlink(tmp_zip.name)
                except OSError:
                    pass

            # Schedule cleanup after response is sent
            import atexit
            atexit.register(cleanup_temp_file)

            return FileResponse(
                tmp_zip.name,
                media_type='application/zip',
                filename=zip_filename,
                background=cleanup_temp_file
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating quadrants: {str(e)}")
