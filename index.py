from typing import Union
import os
import uuid
from pathlib import Path

# Configure matplotlib for web environment BEFORE any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from skimage import io
import utils

app = FastAPI(title="Pyxelate to Color", description="Convert images to pixel art with color grids")

# Create directories if they don't exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-alt")
async def upload_image_alternative(
    request: Request,
    file: UploadFile = File(...),
    downsample_by: int = Query(20),
    palette: int = Query(8),
    upscale: int = Query(100)
):
    """Alternative endpoint using Query parameters instead of Form parameters."""
    # Debug: Print received parameters
    print(f"🔍 DEBUG - ALTERNATIVE ENDPOINT - Received parameters:")
    print(f"  - downsample_by: {downsample_by} (type: {type(downsample_by)})")
    print(f"  - palette: {palette} (type: {type(palette)})")
    print(f"  - upscale: {upscale} (type: {type(upscale)})")

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
        # Process the image (same as original endpoint)
        image = io.imread(upload_path)

        pixelate_options = utils.PixelateOptions(
            downsample_by=downsample_by,
            upscale=upscale,
            palette=palette,
        )

        print(f"🔍 DEBUG - ALTERNATIVE - PixelateOptions created:")
        print(f"  - downsample_by: {pixelate_options.downsample_by}")
        print(f"  - upscale: {pixelate_options.upscale}")
        print(f"  - palette: {pixelate_options.palette}")

        pyxelated_image, pyx = utils.pixelate_image(
            image=image,
            options=pixelate_options
        )

        result_image = utils.create_draw_grid(
            pyxelated_image,
            cell_size=upscale,
            pyx=pyx,
        )

        output_path = f"static/outputs/{file_id}_result.png"
        io.imsave(output_path, result_image)

        pixelated_path = f"static/outputs/{file_id}_pixelated.png"
        io.imsave(pixelated_path, pyxelated_image)

        return templates.TemplateResponse("results.html", {
            "request": request,
            "file_id": file_id,
            "original_url": f"/{upload_path}",
            "pixelated_url": f"/{pixelated_path}",
            "result_url": f"/{output_path}",
            "colors": pyx.colors.tolist(),
            "palette_size": palette
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}"
        })


@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    downsample_by: int = Form(20),
    palette: int = Form(8),
    upscale: int = Form(100)
):
    # Debug: Print received parameters
    print(f"🔍 DEBUG - Received parameters:")
    print(f"  - downsample_by: {downsample_by} (type: {type(downsample_by)})")
    print(f"  - palette: {palette} (type: {type(palette)})")
    print(f"  - upscale: {upscale} (type: {type(upscale)})")

    # Debug: Print raw form data
    try:
        form_data = await request.form()
        print(f"🔍 DEBUG - Raw form data:")
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
        print(f"🔍 DEBUG - PixelateOptions created:")
        print(f"  - downsample_by: {pixelate_options.downsample_by}")
        print(f"  - upscale: {pixelate_options.upscale}")
        print(f"  - palette: {pixelate_options.palette}")

        # Pixelate the image
        pyxelated_image, pyx = utils.pixelate_image(
            image=image,
            options=pixelate_options
        )

        # Create grid with color indices
        result_image = utils.create_draw_grid(
            pyxelated_image,
            cell_size=upscale,
            pyx=pyx,
        )

        # Save result
        output_path = f"static/outputs/{file_id}_result.png"
        io.imsave(output_path, result_image)

        # Also save the pixelated version without grid
        pixelated_path = f"static/outputs/{file_id}_pixelated.png"
        io.imsave(pixelated_path, pyxelated_image)

        # Return template response for HTMX
        return templates.TemplateResponse("results.html", {
            "request": request,
            "file_id": file_id,
            "original_url": f"/{upload_path}",
            "pixelated_url": f"/{pixelated_path}",
            "result_url": f"/{output_path}",
            "colors": pyx.colors.tolist(),
            "palette_size": palette
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}"
        })

@app.get("/download/{file_id}")
async def download_result(file_id: str):
    file_path = f"static/outputs/{file_id}_result.png"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type='image/png',
        filename=f"pixelated_grid_{file_id}.png"
    )
