# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import tempfile

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.visual_track import visualize_tracks_on_images

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# Helper function to get image transformation info
# -------------------------------------------------------------------------
def get_image_transform_info(image_path_list):
    """
    Get transformation information for images (original size, model size, scale factors).
    Returns a list of dicts with transformation info for each image.
    """
    from PIL import Image
    from torchvision import transforms as TF
    
    target_size = 518
    mode = "crop"  # Match the mode used in load_and_preprocess_images
    transform_info = []
    
    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        
        orig_width, orig_height = img.size
        
        # Calculate model dimensions (same logic as load_and_preprocess_images)
        if mode == "pad":
            if orig_width >= orig_height:
                new_width = target_size
                new_height = round(orig_height * (new_width / orig_width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(orig_width * (new_height / orig_height) / 14) * 14
        else:  # crop
            new_width = target_size
            new_height = round(orig_height * (new_width / orig_width) / 14) * 14
        
        # Calculate scale factors
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height
        
        # Calculate crop offset (if height was cropped)
        crop_y = 0
        if mode == "crop" and new_height > target_size:
            crop_y = (new_height - target_size) // 2
            new_height = target_size
        
        # Calculate padding (if pad mode)
        pad_top = pad_left = 0
        if mode == "pad":
            h_padding = target_size - new_height
            w_padding = target_size - new_width
            pad_top = h_padding // 2
            pad_left = w_padding // 2
        
        transform_info.append({
            'orig_width': orig_width,
            'orig_height': orig_height,
            'model_width': target_size if mode == "pad" else new_width,
            'model_height': target_size,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'crop_y': crop_y,
            'pad_top': pad_top,
            'pad_left': pad_left,
        })
    
    return transform_info


def transform_points_to_model_space(points, transform_info, frame_idx=0):
    """
    Transform points from original image space to model space.
    
    Args:
        points: (N, 2) array of points in original image coordinates
        transform_info: List of transformation info dicts
        frame_idx: Index of the frame (default 0 for first frame)
    
    Returns:
        Transformed points in model coordinate space
    """
    if len(transform_info) == 0:
        return points
    
    info = transform_info[frame_idx]
    points = np.array(points, dtype=np.float32)
    
    # Scale
    points[:, 0] *= info['scale_x']
    points[:, 1] *= info['scale_y']
    
    # Apply crop offset (subtract crop_y from y)
    points[:, 1] -= info['crop_y']
    
    # Apply padding offset (add pad_left to x, pad_top to y)
    points[:, 0] += info['pad_left']
    points[:, 1] += info['pad_top']
    
    return points


def transform_points_to_original_space(points, transform_info, frame_idx=0):
    """
    Transform points from model space back to original image space.
    
    Args:
        points: (N, 2) or (S, N, 2) array of points in model coordinates
        transform_info: List of transformation info dicts
        frame_idx: Index of the frame (or array if points is 3D)
    
    Returns:
        Transformed points in original image coordinate space
    """
    if len(transform_info) == 0:
        return points
    
    points = np.array(points, dtype=np.float32)
    is_3d = points.ndim == 3
    
    if is_3d:
        S, N, _ = points.shape
        transformed = np.zeros_like(points)
        for s in range(S):
            info = transform_info[min(s, len(transform_info) - 1)]
            frame_points = points[s].copy()
            
            # Remove padding
            frame_points[:, 0] -= info['pad_left']
            frame_points[:, 1] -= info['pad_top']
            
            # Add crop offset back
            frame_points[:, 1] += info['crop_y']
            
            # Scale back
            frame_points[:, 0] /= info['scale_x']
            frame_points[:, 1] /= info['scale_y']
            
            transformed[s] = frame_points
        return transformed
    else:
        info = transform_info[frame_idx]
        transformed = points.copy()
        
        # Remove padding
        transformed[:, 0] -= info['pad_left']
        transformed[:, 1] -= info['pad_top']
        
        # Add crop offset back
        transformed[:, 1] += info['crop_y']
        
        # Scale back
        transformed[:, 0] /= info['scale_x']
        transformed[:, 1] /= info['scale_y']
        
        return transformed


# -------------------------------------------------------------------------
# 1) Core model inference with optional query_points
# -------------------------------------------------------------------------
def run_model(target_dir, model, query_points=None, transform_info=None) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    
    Args:
        target_dir: Directory containing images
        model: VGGT model instance
        query_points: Optional query points for tracking, shape (N, 2) in pixel coordinates
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Convert query_points to tensor if provided
            if query_points is not None:
                # Transform query points to model space if transform_info is provided
                if transform_info is not None and len(transform_info) > 0:
                    query_points_model = transform_points_to_model_space(query_points, transform_info, frame_idx=0)
                    print(f"Transformed query points from original to model space")
                    print(f"  Original: {query_points}")
                    print(f"  Model space: {query_points_model}")
                else:
                    query_points_model = query_points
                query_points_tensor = torch.tensor(query_points_model, dtype=torch.float32, device=device)
                print(f"Tracking {len(query_points)} query points")
            else:
                query_points_tensor = None
            
            predictions = model(images, query_points=query_points_tensor)
    print(f'predictions: {predictions.keys()}')

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            # Convert BFloat16 to float32 before numpy conversion (NumPy doesn't support BFloat16)
            tensor = predictions[key]
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            predictions[key] = tensor.cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Initial reconstruction (without tracking)
# -------------------------------------------------------------------------
def initial_reconstruction(target_dir):
    """
    Perform initial 3D reconstruction without tracking.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Get image paths and compute transformation info
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    transform_info = get_image_transform_info(image_names)

    print("Running initial reconstruction...")
    with torch.no_grad():
        predictions = run_model(target_dir, model, query_points=None, transform_info=transform_info)

    # Save predictions and transform info
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)
    
    # Save transform info
    transform_info_path = os.path.join(target_dir, "transform_info.npy")
    np.save(transform_info_path, transform_info, allow_pickle=True)

    # Get first image for query point selection
    first_image_path = image_names[0] if image_names else None

    end_time = time.time()
    print(f"Initial reconstruction took {end_time - start_time:.2f} seconds")
    log_msg = f"Reconstruction complete ({len(image_names)} frames). Click on the image below to select points to track."

    return first_image_path, log_msg, predictions, transform_info


# -------------------------------------------------------------------------
# 4) Track points based on user clicks
# -------------------------------------------------------------------------
def track_points_from_clicks(target_dir, evt: gr.SelectData, stored_predictions, current_query_points, stored_transform_info):
    """
    Track points based on user clicks on the image.
    
    Args:
        target_dir: Directory containing images
        evt: Gradio SelectData event from image click
        stored_predictions: Previously computed predictions (without tracking)
        current_query_points: List of current query points from state
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found.", None, None, []
    
    if stored_predictions is None:
        return None, "Please run reconstruction first.", None, None, []

    # Get transform info
    if stored_transform_info is None:
        # Try to load from disk
        transform_info_path = os.path.join(target_dir, "transform_info.npy")
        if os.path.exists(transform_info_path):
            stored_transform_info = np.load(transform_info_path, allow_pickle=True).tolist()
        else:
            # Compute it
            image_names = glob.glob(os.path.join(target_dir, "images", "*"))
            image_names = sorted(image_names)
            stored_transform_info = get_image_transform_info(image_names)

    # Get click coordinates (in original image space)
    x, y = evt.index[0], evt.index[1]  # Gradio uses (x, y) format
    print(f"User clicked at pixel coordinates (original space): ({x}, {y})")

    # Add new query point to the list (in original image space)
    if current_query_points is None:
        current_query_points = []
    current_query_points.append([x, y])
    query_points_original = np.array(current_query_points)
    
    print(f"Tracking {len(query_points_original)} points in original space: {query_points_original}")

    # Run tracking
    gc.collect()
    torch.cuda.empty_cache()
    
    start_time = time.time()
    with torch.no_grad():
        # Load images again for tracking
        image_names = glob.glob(os.path.join(target_dir, "images", "*"))
        image_names = sorted(image_names)
        images = load_and_preprocess_images(image_names).to(device)
        
        # Transform query points to model space
        query_points_model = transform_points_to_model_space(query_points_original, stored_transform_info, frame_idx=0)
        print(f"Query points in model space: {query_points_model}")
        
        # Run model with query points
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            query_points_tensor = torch.tensor(query_points_model, dtype=torch.float32, device=device)
            predictions = model(images, query_points=query_points_tensor)
    
    # Convert to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            # Convert BFloat16 to float32 before numpy conversion (NumPy doesn't support BFloat16)
            tensor = predictions[key]
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            predictions[key] = tensor.cpu().numpy().squeeze(0)
    
    # Extract tracks
    tracks = predictions.get("track")  # (S, N, 2) in model coordinate space
    vis = predictions.get("vis")  # (S, N)
    conf = predictions.get("conf")  # (S, N)
    
    if tracks is None:
        return None, "Tracking failed. Please try again.", None, None, current_query_points
    
    print(f"Tracks shape: {tracks.shape}, Vis shape: {vis.shape}")
    
    # Transform tracks from model space back to original image space
    tracks_original = transform_points_to_original_space(tracks, stored_transform_info)
    print(f"Transformed tracks from model space to original space")
    
    # Load original images (not preprocessed) for visualization
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    
    # Load original images using PIL
    from PIL import Image
    original_images = []
    original_shapes = []
    for img_path in image_names:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)  # (H, W, 3) in RGB, uint8
        original_images.append(img_array)
        original_shapes.append(img_array.shape[:2])  # (H, W)
    
    # Find max dimensions for padding
    max_h = max(shape[0] for shape in original_shapes)
    max_w = max(shape[1] for shape in original_shapes)
    
    # Pad images to same size for visualization and adjust track coordinates
    padded_images = []
    for s, img in enumerate(original_images):
        h, w = img.shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad image
        img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                           mode='constant', constant_values=0)
        padded_images.append(img_padded)
        
        # Adjust track coordinates for this frame's padding
        tracks_original[s, :, 0] += pad_left
        tracks_original[s, :, 1] += pad_top
    
    images_np = np.stack(padded_images).astype(np.float32) / 255.0  # (S, H, W, 3) in [0, 1]
    
    # Create visibility mask (points are visible if vis > threshold)
    vis_threshold = 0.2
    vis_mask = vis > vis_threshold
    
    # Visualize tracks
    temp_dir = tempfile.mkdtemp()
    try:
        # Images are in HWC format (S, H, W, 3), convert to CHW
        images_torch = torch.from_numpy(images_np.transpose(0, 3, 1, 2))  # (S, 3, H, W)
        
        # Use tracks in original space
        tracks_torch = torch.from_numpy(tracks_original)
        vis_mask_torch = torch.from_numpy(vis_mask)
        
        visualize_tracks_on_images(
            images_torch,
            tracks_torch,
            vis_mask_torch,
            out_dir=temp_dir,
            image_format="CHW",
            normalize_mode="[0,1]",
            cmap_name="hsv",
            frames_per_row=min(4, tracks.shape[0]),
            save_grid=True
        )
        
        # Load the grid image
        grid_path = os.path.join(temp_dir, "tracks_grid.png")
        if os.path.exists(grid_path):
            tracked_image = grid_path
        else:
            # Fallback to first frame
            frame_path = os.path.join(temp_dir, "frame_0000.png")
            tracked_image = frame_path if os.path.exists(frame_path) else None
        
        end_time = time.time()
        log_msg = f"Tracking complete! Tracked {len(query_points_original)} points across {tracks.shape[0]} frames in {end_time - start_time:.2f} seconds."
        
        return tracked_image, log_msg, query_points_original.tolist(), temp_dir, current_query_points
        
    except Exception as e:
        print(f"Error visualizing tracks: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", None, None, current_query_points


# -------------------------------------------------------------------------
# 5) Clear query points
# -------------------------------------------------------------------------
def clear_query_points(current_query_points):
    """Clear stored query points."""
    return "Query points cleared. Click on the image to add new points.", "", []


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    """
) as demo:
    gr.HTML(
        """
    <h1>🎯 VGGT: Point Tracking with Query Points</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction and track points across frames.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons to provide your input.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Select Points:</strong> Click on the first image to select points you want to track across all frames.</li>
        <li><strong>View Tracks:</strong> The tracked points will be visualized with colored trajectories.</li>
    </ol>
    <p><strong style="color: #0ea5e9;">Tip:</strong> <span style="color: #0ea5e9;">Click multiple times to track multiple points. Each point will be assigned a unique color based on its initial position.</span></p>
    </div>
    """
    )

    target_dir_state = gr.State(value=None)
    predictions_state = gr.State(value=None)
    transform_info_state = gr.State(value=None)  # Store transform info
    temp_dir_state = gr.State(value=None)
    query_points_state = gr.State(value=[])  # Store query points in state

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

            with gr.Row():
                reconstruct_btn = gr.Button("Reconstruct", variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, image_gallery],
                    scale=1,
                )

        with gr.Column(scale=3):
            log_output = gr.Markdown(
                "Please upload a video or images, then click Reconstruct.",
                elem_classes=["custom-log"]
            )

            # Image for query point selection
            query_image = gr.Image(
                label="Click on this image to select points to track",
                type="filepath",
                interactive=True,
                height=400
            )

            with gr.Row():
                clear_points_btn = gr.Button("Clear Points", variant="secondary")
                query_points_display = gr.Textbox(
                    label="Selected Points (x, y)",
                    interactive=False,
                    placeholder="No points selected yet. Click on the image above."
                )

            # Output for tracked visualization
            tracked_output = gr.Image(
                label="Tracked Points Visualization",
                type="filepath",
                height=500
            )

    # Handle uploads
    def update_gallery_on_upload(input_video, input_images):
        if not input_video and not input_images:
            return None, None, None, None
        target_dir, image_paths = handle_uploads(input_video, input_images)
        return target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin.", None

    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[target_dir_state, image_gallery, log_output, query_image],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[target_dir_state, image_gallery, log_output, query_image],
    )

    # Handle reconstruction
    def on_reconstruct(target_dir, current_query_points):
        if target_dir is None or target_dir == "None":
            return None, "Please upload images first.", None, None, "", None, []
        first_image, log_msg, predictions, transform_info = initial_reconstruction(target_dir)
        # Clear query points on new reconstruction
        return first_image, log_msg, predictions, transform_info, "", None, []

    reconstruct_btn.click(
        fn=on_reconstruct,
        inputs=[target_dir_state, query_points_state],
        outputs=[query_image, log_output, predictions_state, transform_info_state, query_points_display, tracked_output, query_points_state]
    )

    # Handle point selection and tracking
    def on_image_select(evt: gr.SelectData, target_dir, stored_predictions, current_query_points, stored_transform_info):
        if target_dir is None or target_dir == "None":
            return None, "Please upload and reconstruct first.", None, None, []
        tracked_img, log_msg, points_list, temp_dir, updated_query_points = track_points_from_clicks(
            target_dir, evt, stored_predictions, current_query_points, stored_transform_info
        )
        points_str = ", ".join([f"({p[0]}, {p[1]})" for p in points_list]) if points_list else ""
        return tracked_img, log_msg, points_str, temp_dir, updated_query_points

    query_image.select(
        fn=on_image_select,
        inputs=[target_dir_state, predictions_state, query_points_state, transform_info_state],
        outputs=[tracked_output, log_output, query_points_display, temp_dir_state, query_points_state]
    )

    # Handle clear points
    clear_points_btn.click(
        fn=clear_query_points,
        inputs=[query_points_state],
        outputs=[log_output, query_points_display, query_points_state]
    ).then(
        fn=lambda: None,
        inputs=[],
        outputs=[tracked_output]
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
