import os
import traceback
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from config import BASE_UPLOAD, DEVICE, VIS_CHANNELS, THRESHOLD
from model import load_model, predict
from utils.preprocess import preprocess_tif
from utils.labels import load_label_image
from utils.visualize import plot_to_base64

# ────── Flask Setup ──────
app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = BASE_UPLOAD
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ────── Load Model ──────
model = load_model()


# ────── Routes ──────
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or not file.filename.lower().endswith(('.tif', '.tiff')):
            return render_template("index.html", error="Please upload a .tif or .tiff file.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocessing
            input_tensor, input_tensor_vis = preprocess_tif(
                filepath, vis_channels=VIS_CHANNELS, device=DEVICE
            )

            # Prediction
            pred_mask = predict(model, input_tensor, threshold=THRESHOLD)

            # Load corresponding true mask
            true_mask = load_label_image(filename)

            # Visualization
            vis_for_plot = input_tensor_vis[0]  # remove batch dimension
            img_base64 = plot_to_base64(vis_for_plot, pred_mask, true_mask)

            return render_template("index.html", image=img_base64)

        except Exception as e:
            traceback.print_exc()
            return render_template("index.html", error=f"Error during processing: {str(e)}")

        finally:
            try:
                os.remove(filepath)
            except OSError:
                pass

    return render_template("index.html")


# ────── Entry Point ──────
if __name__ == '__main__':
    app.run(debug=True)
