# app_s3.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU inference in container
import io
import logging
import traceback
from threading import Lock
from collections import defaultdict

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np


try:
    import tflite_runtime.interpreter as tflite_runtime
    TFLITE_INTERPRETER = tflite_runtime.Interpreter
except Exception:
    try:
        import tensorflow as tf
        TFLITE_INTERPRETER = tf.lite.Interpreter
    except Exception:
        TFLITE_INTERPRETER = None
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
import boto3
import torch
from huggingface_hub import hf_hub_download
import traceback

# ---------- Config ----------
TMP_DIR = "/tmp"
PRODUCTS_KEY = "products.csv"               # S3 key where frontend uploads dataset
USER_EMB_KEY = "embeddings/user_recommender_embeddings.npy"
BRAND_EMB_KEY = "embeddings/brand_embeddings_chunk.npy"
TFLITE_MODEL_KEY = "models/sequential_Recommendation_system.tflite"

# YOUR HF repo and filenames (from your snippet)
HF_REPO = "oke39/ecommerce-recommender-models"
HF_PKL_FILENAME = "2025-08-25_14-19-26-363710.pkl"
HF_TFLITE_FILENAME = "sequential_Recommendation_system.tflite"

# Read configuration from env
S3_BUCKET = os.environ.get("S3_BUCKET")  # required if using S3
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
HF_TOKEN = os.environ.get("HF_TOKEN")    # required if HF repo is private

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")

# ---------- Flask app ----------
app = Flask(__name__)

# ---------- AWS S3 client ----------
# Do NOT pass explicit keys here so boto3 uses env vars or IAM role automatically.
s3_client = boto3.client("s3", region_name=AWS_REGION)

# ---------- Global lazy resources ----------
_resources_lock = Lock()
_resources = {
    "data": None,
    "product_id_encode": None,
    "category_encoder": None,
    "brand_encoder": None,
    "interpreter": None,
    "input_details": None,
    "output_details": None,
    "brand_embeddings": None,
    "tokenizer": None,
    "user_embeddings": None,
    "pkl_path": None
}

# ---------- Utility helpers ----------
def s3_upload_fileobj(file_obj, bucket, key):
    try:
        file_obj.seek(0)
        s3_client.upload_fileobj(file_obj, bucket, key)
        logger.info(f"Uploaded to s3://{bucket}/{key}")
        return True
    except Exception:
        logger.exception("S3 upload_fileobj failed")
        return False

def s3_download_to_path(bucket, key, dst_path):
    try:
        s3_client.download_file(bucket, key, dst_path)
        logger.info(f"Downloaded s3://{bucket}/{key} -> {dst_path}")
        return True
    except Exception:
        logger.warning(f"Failed to download s3://{bucket}/{key}")
        return False

def ensure_products_local():
    """Ensure /tmp/products.csv exists (download from S3 if needed). Returns local path or None."""
    local_path = os.path.join(TMP_DIR, PRODUCTS_KEY)
    if os.path.exists(local_path):
        return local_path
    if S3_BUCKET:
        ok = s3_download_to_path(S3_BUCKET, PRODUCTS_KEY, local_path)
        if ok:
            return local_path
    return None

# ---------- Loaders (lazy) ----------
def load_dataset(force_reload=False):
    with _resources_lock:
        if _resources["data"] is not None and not force_reload:
            return _resources["data"]

        local = ensure_products_local()
        if local is None:
            logger.error("Dataset not found in /tmp or S3")
            return None

        df = pd.read_csv(local)
        df['product_id'] = df['product_id'].astype(str)
        df['user_id'] = df['user_id'].astype(str)
        df['brand'] = df.get('brand', pd.Series(['unknown']*len(df))).fillna('unknown')
        df['category_code'] = df.get('category_code', pd.Series(['unknown']*len(df))).fillna('unknown')

        try:
            _resources['product_id_encode'] = LabelEncoder().fit(df['product_id'])
            _resources['category_encoder'] = LabelEncoder().fit(df['category_code'])
            _resources['brand_encoder'] = LabelEncoder().fit(df['brand'])
        except Exception:
            logger.exception("Failed to fit encoders")

        _resources['data'] = df
        logger.info("Dataset loaded into memory")
        return df

def load_user_embeddings():
    with _resources_lock:
        if _resources['user_embeddings'] is not None:
            return _resources['user_embeddings']
        local = os.path.join(TMP_DIR, os.path.basename(USER_EMB_KEY))
        # try S3 first
        if not os.path.exists(local) and S3_BUCKET:
            s3_download_to_path(S3_BUCKET, USER_EMB_KEY, local)
        # fallback to current working dir (if developer placed file there)
        if not os.path.exists(local) and os.path.exists("user_recommender_embeddings.npy"):
            local = "user_recommender_embeddings.npy"
        if not os.path.exists(local):
            logger.error("User embeddings not found")
            return None
        _resources['user_embeddings'] = np.load(local)
        return _resources['user_embeddings']

def load_brand_embeddings():
    with _resources_lock:
        if _resources['brand_embeddings'] is not None:
            return _resources['brand_embeddings']
        local = os.path.join(TMP_DIR, os.path.basename(BRAND_EMB_KEY))
        if not os.path.exists(local) and S3_BUCKET:
            s3_download_to_path(S3_BUCKET, BRAND_EMB_KEY, local)
        # fallback to local file if exists
        if not os.path.exists(local) and os.path.exists("brand_embeddings_chunk.npy"):
            local = "brand_embeddings_chunk.npy"
        if not os.path.exists(local):
            logger.error("Brand embeddings not found")
            return None
        arr = np.load(local)
        # keep compatibility with your old code which wrapped embeddings with torch.tensor
        try:
            _resources['brand_embeddings'] = torch.tensor(arr)
        except Exception:
            _resources['brand_embeddings'] = arr
        return _resources['brand_embeddings']

def load_pkl_from_hf():
    """Download the pkl (torch/cornac) artifact from HF if not present locally or in /tmp"""
    with _resources_lock:
        if _resources['pkl_path'] is not None:
            return _resources['pkl_path']
        local = os.path.join(TMP_DIR, HF_PKL_FILENAME)
        # try S3 first (if you saved the pkl there)
        if S3_BUCKET:
            s3_key = f"models/{HF_PKL_FILENAME}"
            if s3_download_to_path(S3_BUCKET, s3_key, local):
                _resources['pkl_path'] = local
                return local
        # fallback to Hugging Face hub
        if HF_REPO and HF_PKL_FILENAME and HF_TOKEN:
            try:
                hf_path = hf_hub_download(repo_id=HF_REPO, filename=HF_PKL_FILENAME, token=HF_TOKEN)
                # move to /tmp for consistent path
                os.replace(hf_path, local)
                _resources['pkl_path'] = local
                logger.info("Downloaded pkl from Hugging Face hub")
                return local
            except Exception:
                logger.exception("Failed to download pkl from HF hub")
        # fallback to local file if present
        if os.path.exists(HF_PKL_FILENAME):
            _resources['pkl_path'] = HF_PKL_FILENAME
            return HF_PKL_FILENAME
        logger.error("No pkl artifact available")
        return None

def load_tflite_interpreter():
    with _resources_lock:
        if _resources['interpreter'] is not None:
            return _resources['interpreter']

        # prefer S3 (maybe you uploaded the tflite there)
        local = os.path.join(TMP_DIR, os.path.basename(TFLITE_MODEL_KEY))
        if not os.path.exists(local) and S3_BUCKET:
            if s3_download_to_path(S3_BUCKET, TFLITE_MODEL_KEY, local):
                logger.info("TFLite downloaded from S3")
        # If not in S3, try Hugging Face Hub (your repo)
        if not os.path.exists(local):
            try:
                hf_path = hf_hub_download(repo_id=HF_REPO, filename=HF_TFLITE_FILENAME, token=HF_TOKEN)
                os.replace(hf_path, local)
                logger.info("TFLite downloaded from Hugging Face hub")
            except Exception:
                logger.exception("HF TFLite download failed or not configured")
        if not os.path.exists(local):
            logger.error("TFLite model not found (S3 or HF)")
            return None
        if TFLITE_INTERPRETER is None:
            logger.error("No TFLite runtime available in this environment")
            return None
        try:
            interp = TFLITE_INTERPRETER(model_path=local)
            interp.allocate_tensors()
            _resources['interpreter'] = interp
            _resources['input_details'] = interp.get_input_details()
            _resources['output_details'] = interp.get_output_details()
            logger.info("TFLite interpreter ready")
            return interp
        except Exception:
            logger.exception("Failed to init TFLite interpreter")
            return None

def ensure_tokenizer():
    with _resources_lock:
        if _resources.get("tokenizer") is not None:
            return _resources["tokenizer"]
        try:
            tok = BertTokenizer.from_pretrained('bert-base-uncased')
            _resources['tokenizer'] = tok
            return tok
        except Exception:
            logger.exception("Failed to init tokenizer")
            return None

# ---------- Endpoints ----------

@app.route("/")
def health():
    return jsonify({"status": "ok"}),200

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    """Receive file from Streamlit; save to /tmp and stream upload to S3."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save file in chunks to avoid memory overload
        local_path = os.path.join(TMP_DIR, PRODUCTS_KEY)
        with open(local_path, "wb") as f:
            for chunk in file.stream:
                f.write(chunk)

        # Upload to S3 if configured
        if S3_BUCKET:
            try:
                # Stream upload (no full in-memory read)
                s3_client.upload_file(
                    Filename=local_path,
                    Bucket=S3_BUCKET,
                    Key=PRODUCTS_KEY
                )
            except Exception as e:
                logger.exception("S3 upload failed")
                return jsonify({"error": f"S3 upload failed: {e}"}), 500

        # Clear cached dataset in memory
        with _resources_lock:
            _resources['data'] = None

        return jsonify({"message": "Dataset uploaded successfully"}), 200

    except Exception as e:
        logger.exception("Upload failed")
        return jsonify({"error": str(e)}), 500



def predict_tflite(padded_sequence):
    interp = load_tflite_interpreter()
    if interp is None:
        raise RuntimeError("TFLite interpreter not available")
    input_details = _resources['input_details']
    output_details = _resources['output_details']
    input_data = np.array(padded_sequence, dtype=np.float32)
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)
    interp.set_tensor(input_details[0]['index'], input_data)
    interp.invoke()
    preds = interp.get_tensor(output_details[0]['index'])
    return preds

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        payload = request.get_json()
        user_id = payload.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        df = load_dataset()
        if df is None:
            return jsonify({"error": "Dataset not available"}), 500

        product_id_encode = _resources.get("product_id_encode")
        if product_id_encode is None:
            product_id_encode = LabelEncoder().fit(df['product_id'])
            _resources['product_id_encode'] = product_id_encode

        user_history = df[df['user_id'].astype(str) == str(user_id)]
        if user_history.empty:
            return jsonify({"error": "User ID not found"}), 400

        product_sequence = user_history.sort_values("event_type")["product_id"].tolist()
        encoded_seq = product_id_encode.transform(product_sequence)
        padded = pad_sequences([encoded_seq], maxlen=9, padding='post')

        preds = predict_tflite(padded)
        top_indices = preds[0].argsort()[-5:][::-1]
        predicted_ids = product_id_encode.inverse_transform(top_indices)

        recommended = df[df['product_id'].isin(predicted_ids)].drop_duplicates("product_id").head(5)
        return jsonify({"recommendations": recommended.to_dict(orient="records")})
    except Exception as e:
        logger.exception("Error in /recommend")
        return jsonify({"error": str(e)}), 500

@app.route("/recommend_user_content_bert", methods=["POST"])
def recommend_user_content_bert():
    try:
        payload = request.get_json()
        user_id = payload.get("user_id")
        top_k = int(payload.get("top_k", 5))

        df = load_dataset()
        if df is None:
            return jsonify({"error": "Dataset not available"}), 500

        user_embeddings = load_user_embeddings()
        if user_embeddings is None:
            return jsonify({"error": "User embeddings missing"}), 500

        # Build product->embedding map for available embeddings
        valid_product_ids = df['product_id'].values[:len(user_embeddings)]
        product_id_to_embedding = {pid: emb for pid, emb in zip(valid_product_ids, user_embeddings)}

        # Build user profiles
        user_profiles = defaultdict(list)
        for _, row in df.iterrows():
            pid = row['product_id']
            if pid in product_id_to_embedding:
                user_profiles[row['user_id']].append(product_id_to_embedding[pid])

        user_id_to_profile = {uid: np.mean(embs, axis=0) for uid, embs in user_profiles.items() if len(embs) > 0}
        if user_id not in user_id_to_profile:
            return jsonify({"error": "User ID not found in profiles"}), 400

        user_vec = user_id_to_profile[user_id].reshape(1, -1)
        all_item_vecs = np.stack([product_id_to_embedding[pid] for pid in valid_product_ids])
        scores = cosine_similarity(user_vec, all_item_vecs).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_indices = [i for i in top_indices if i < len(df)]

        recommendations = df.iloc[top_indices][['product_id', 'brand', 'category_code', 'price']]
        return jsonify(recommendations.to_dict(orient="records"))
    except Exception as e:
        logger.exception("Error in recommend_user_content_bert")
        return jsonify({"error": str(e)}), 500

@app.route("/recommend_brand_similarity_bert", methods=["POST"])
def recommend_brand_similarity_bert():
    try:
        payload = request.get_json()
        brand = payload.get("brand")
        product_id = payload.get("product_id")
        top_k = int(payload.get("top_k", 5))
        min_price = payload.get("min_price")
        max_price = payload.get("max_price")
        same_category = payload.get("same_category", True)

        df = load_dataset()
        if df is None:
            return jsonify({"error": "Dataset not available"}), 500

        brand_embeddings = load_brand_embeddings()
        if brand_embeddings is None:
            return jsonify({"error": "Brand embeddings missing"}), 500

        if product_id:
            idxs = df.index[df['product_id'] == product_id]
            if len(idxs) == 0:
                return jsonify({"error": "Product not found"}), 400
            idx = idxs[0]
        elif brand:
            matches = df[df['brand'].str.lower() == brand.lower()]
            if matches.empty:
                return jsonify({"error": "Brand not found"}), 400
            idx = matches.index[0]
        else:
            return jsonify({"error": "brand or product_id required"}), 400

        product = df.loc[idx]
        # brand_embeddings may be torch tensor or numpy array
        be = _resources['brand_embeddings']
        if isinstance(be, torch.Tensor):
            product_vec = be[idx].detach().cpu().numpy().reshape(1, -1)
        else:
            product_vec = be[idx].reshape(1, -1)

        filtered = df.copy()
        if same_category:
            filtered = filtered[filtered['category_code'] == product['category_code']]
        if min_price is not None:
            filtered = filtered[filtered['price'] >= float(min_price)]
        if max_price is not None:
            filtered = filtered[filtered['price'] <= float(max_price)]
        filtered = filtered[filtered['product_id'] != product['product_id']]

        filtered_indices = [df.index.get_loc(i) for i in filtered.index]
        # ensure indices valid
        filtered_indices = [i for i in filtered_indices if i < (be.shape[0] if not isinstance(be, torch.Tensor) else be.size(0))]
        if not filtered_indices:
            return jsonify([])

        if isinstance(be, torch.Tensor):
            filtered_embeddings = be[filtered_indices].cpu().numpy()
        else:
            filtered_embeddings = be[filtered_indices]

        similarities = cosine_similarity(product_vec, filtered_embeddings).flatten()
        top_idx = similarities.argsort()[::-1][:top_k]
        top_data_indices = [filtered.index[i] for i in top_idx]

        results = df.loc[top_data_indices][['product_id', 'brand', 'category_code', 'price']]
        return jsonify(results.to_dict(orient="records"))
    except Exception as e:
        logger.exception("Error in recommend_brand_similarity_bert")
        return jsonify({"error": str(e)}), 500

@app.route("/vae_collaborative_recommender", methods=["POST"])
def collaborative_recommender():
    try:
        payload = request.get_json()
        user_id = payload.get("user_id")
        top_n = int(payload.get("top_n", 5))

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        df = load_dataset()
        if df is None:
            return jsonify({"error": "Dataset not available"}), 500

        # Prepare interactions
        interactions = list(zip(df['user_id'].astype(str), df['product_id'].astype(str),
                                df['event_type'].map({'view':1.0,'purchase':3.0}).fillna(1.0)))

        # Use Cornac RatioSplit to build train set mapping
        try:
            from cornac.eval_methods import RatioSplit
        except Exception:
            return jsonify({"error": "Cornac not installed on server"}), 500

        eval_method = RatioSplit(data=interactions, test_size=0.2, rating_threshold=0.5, exclude_unknowns=True)
        train_set = eval_method.train_set
        uid_map = train_set.uid_map
        iid_map = train_set.iid_map
        iid_map_inv = {v: k for k, v in iid_map.items()}

        if user_id not in uid_map:
            return jsonify({"error": "User not found in training set"}), 400

        uid = uid_map[user_id]

        # Load pretrained VAECF model from /tmp or S3 (this example expects pre-deployed model)
        vae_local = os.path.join(TMP_DIR, "vaecf_model")
        if not os.path.exists(vae_local):
            # (Optional) download pre-trained model dir from S3 if you stored it there
            pass

        try:
            from cornac.models.vaecf import VAECF
            vae_model = VAECF.load(vae_local)
        except Exception:
            return jsonify({"error": "Pretrained VAECF model not available"}, 500)

        # Score items
        scores = []
        for item_idx in range(train_set.num_items):
            try:
                s = vae_model.score(uid, item_idx)
                if hasattr(s, "cpu"):
                    s = s.cpu().item()
                elif hasattr(s, "item"):
                    s = s.item()
                else:
                    s = float(s)
                scores.append(s)
            except Exception:
                scores.append(0.0)

        seen = set(train_set.matrix[uid].indices)
        unseen_scores = [(i, scores[i]) for i in range(len(scores)) if i not in seen]
        top_items = sorted(unseen_scores, key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        max_score = max([s for _, s in top_items]) if top_items else 1.0
        for item_id, score in top_items:
            original_id = iid_map_inv[item_id]
            metadata = df[df['product_id'] == original_id].iloc[0].to_dict()
            metadata['score'] = round(score / max_score, 4) if max_score > 0 else 0.0
            results.append(metadata)

        return jsonify({"user_id": user_id, "recommendations": results, "count": len(results)})
    except Exception as e:
        logger.exception("Error in collaborative_recommender")
        return jsonify({"error": str(e)}), 500

# ---------- Run ----------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
