import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tflite_runtime.interpreter as tflite  
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
from collections import defaultdict
import torch
from cornac.eval_methods import RatioSplit
from huggingface_hub import hf_hub_download
import traceback

app = Flask(__name__)

token = os.getenv("HF_TOKEN")

# ---------- Globals (lazy init) ----------
data = None
product_id_encode = None
category_encoder = None
brand_encoder = None
interpreter = None
input_details = None
output_details = None
brand_embeddings = None
tokenizer = None
pkl_path = None


# ---------- confirmation Route ----------
@app.route("/")
def health():
    return {"status": "ok"}


# ---------- Lazy Initializer ----------
@app.before_first_request
def load_resources():
    global data, product_id_encode, category_encoder, brand_encoder
    global interpreter, input_details, output_details
    global brand_embeddings, tokenizer, pkl_path

    try:
        # Load dataset
        if data is None:
            data = pd.read_csv("products.csv")
            product_id_encode = LabelEncoder().fit(data['product_id'])
            category_encoder = LabelEncoder().fit(data['category_code'])
            brand_encoder = LabelEncoder().fit(data['brand'])

        # Hugging Face models
        if pkl_path is None:
            pkl_path = hf_hub_download(
                repo_id="oke39/ecommerce-recommender-models",
                filename="2025-08-25_14-19-26-363710.pkl",
                token=token
            )

        if interpreter is None:
            tflite_path = hf_hub_download(
                repo_id="oke39/ecommerce-recommender-models",
                filename="sequential_Recommendation_system.tflite",
                token=token
            )
            interpreter = tflite.Interpreter(model_path=tflite_path)  # ✅ tflite-runtime
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if brand_embeddings is None:
            brand_embeddings = np.load("brand_embeddings_chunk.npy")
            brand_embeddings = torch.tensor(brand_embeddings)  # keeps compatibility

        print("Resources loaded successfully")

    except Exception as e:
        print(f"Error loading resources: {e}")
        traceback.print_exc()


# ---------- Helper ----------
def predict_tflite(padded_sequence):
    global interpreter, input_details, output_details
    input_data = np.array(padded_sequence, dtype=np.float32)
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds


# ---------- Recommendation Endpoints ----------
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        req = request.get_json()
        user_id = req.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        user_history = data[data['user_id'].astype(str) == str(user_id)]
        if user_history.empty:
            return jsonify({"error": "User ID not found"}), 400

        product_sequence = user_history.sort_values("event_type")["product_id"].tolist()
        encoded_seq = product_id_encode.transform(product_sequence)
        padded = pad_sequences([encoded_seq], maxlen=9, padding='post')

        predictions = predict_tflite(padded)
        top_indices = predictions[0].argsort()[-5:][::-1]
        predicted_ids = product_id_encode.inverse_transform(top_indices)

        recommended = data[data['product_id'].isin(predicted_ids)].drop_duplicates("product_id").head(5)
        return jsonify({"recommendations": recommended.to_dict(orient="records")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend_user_content_bert", methods=["POST"])
def recommend_user_content_bert():
    try:
        user_embeddings = tf.convert_to_tensor(np.load("user_recommender_embeddings.npy"))
        
        user_id = request.json.get("user_id")
        top_k = int(request.json.get("top_k", 5))

        data = pd.read_csv("products.csv")
        scaler = MinMaxScaler()
        data["price_norm"] = scaler.fit_transform(data[['price']])

        data["brand"]= data['brand'].fillna('unknown')
        data["category_code"]= data['category_code'].fillna('unknown')

        data_text = data["brand"] + " " + data["category_code"] + "price" + data["price_norm"].astype(str)

# Map only the product_ids that have embeddings
        valid_product_ids = data['product_id'].values[:len(user_embeddings)]
        product_id_to_embedding = {
           pid: emb for pid, emb in zip(valid_product_ids, user_embeddings.numpy())
        }

# Build user profiles safely
        user_profiles = defaultdict(list)

        for _, row in data.iterrows():
            pid = row['product_id']
            if pid in product_id_to_embedding: 
                user_profiles[row['user_id']].append(product_id_to_embedding[pid])

# Compute mean embedding for each user
        user_id_to_profile = {
        user_id: np.mean(item_embs, axis=0)
        for user_id, item_embs in user_profiles.items()

        }

        if user_id not in user_id_to_profile:
            return jsonify({"error": "User ID not found"}), 400

        user_vec = user_id_to_profile[user_id].reshape(1, -1)
        all_item_vecs = user_embeddings.numpy()
        scores = cosine_similarity(user_vec, all_item_vecs).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_indices = filter_indices(top_indices, len(data))  # ensure valid indices
#
        if not top_indices:
            return jsonify({"error": "No valid recommendations found."}), 400

        recommendations = data.iloc[top_indices][['product_id', 'brand', 'category_code', 'price', 'text']]
        return jsonify(recommendations.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend_brand_similarity_bert", methods=["POST"])
def recommend_brand_similarity_bert():
    try:
        req = request.json
        brand = req.get("brand")
        product_id = req.get("product_id")
        top_k = int(req.get("top_k", 5))
        min_price = req.get("min_price")
        max_price = req.get("max_price")
        same_category = req.get("same_category", True)

        if product_id:
            idx = data.index[data['product_id'] == product_id][0]
        elif brand:
            matches = data[data['brand'].str.lower() == brand.lower()]
            if matches.empty:
                return jsonify({"error": "Brand not found"}), 400
            idx = matches.index[0]
        else:
            return jsonify({"error": "brand or product_id required"}), 400

        product = data.loc[idx]
        product_vec = brand_embeddings[idx].numpy().reshape(1, -1)

        filtered_data = data.copy()
        if same_category:
            filtered_data = filtered_data[filtered_data['category_code'] == product['category_code']]
        if min_price is not None:
            filtered_data = filtered_data[filtered_data['price'] >= float(min_price)]
        if max_price is not None:
            filtered_data = filtered_data[filtered_data['price'] <= float(max_price)]
        filtered_data = filtered_data[filtered_data['product_id'] != product['product_id']]

        filtered_indices = [data.index.get_loc(i) for i in filtered_data.index]
        if not filtered_indices:
            return jsonify([])
        #
        filtered_indices = filter_indices(filtered_indices, brand_embeddings.shape[0])  # prevent gather errors
        if not filtered_indices:
           return jsonify({"error": "No valid filtered product indices found."}), 400

        filtered_embeddings = tf.gather(brand_embeddings, filtered_indices)
        similarities = cosine_similarity(product_vec, filtered_embeddings.numpy()).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]
        top_data_indices = filter_indices(top_indices, len(data))

        results = data.loc[top_data_indices][['product_id', 'brand', 'category_code', 'price']]
        return jsonify(results.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/vae_collaborative_recommender', methods=['POST'])
def collaborative_recommender():
    try:
        # Force loading model on CPU and handle CUDA compatibility
        original_torch_load = torch.load

        def torch_load_cpu(*args, **kwargs):
            kwargs['map_location'] = torch.device('cpu')
            # Additional fix for CUDA models loaded on CPU
            kwargs['weights_only'] = False  # Allow loading of model objects
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_cpu

        from cornac.models.vaecf import VAECF
        
       
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache if available
        
        # Force CPU device for all operations
        device = torch.device('cpu')
        
        try:
            model = VAECF.load("model/collab_recommender/VAECF")
            # Ensure model is on CPU
            if hasattr(model, 'device'):
                model.device = device
            if hasattr(model, 'to'):
                model = model.to(device)
        except Exception as model_load_error:
            print(f"Model loading error: {model_load_error}")
            # Try alternative loading method
            try:
                # Load with explicit CPU mapping
                model_path = pkl_path
                model = torch.load(model_path, map_location='cpu')
                if hasattr(model, 'to'):
                    model = model.to(device)
            except Exception as fallback_error:
                print(f"Fallback loading failed: {fallback_error}")
                return jsonify({'error': 'Model loading failed. Please retrain model for CPU usage.'}), 500

        # Restore torch.load just in case
        torch.load = original_torch_load

        # Retrieve user data from request body (JSON)
        data = request.get_json()
        print("Received data:", data)

        # Ensure 'user_id' and 'top_n' are passed in the request
        user_id = data.get('user_id')
        top_n = data.get('top_n', 5)  # Default to 5 if top_n is not provided

        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        # Dynamically load and preprocess product data
        data_df = pd.read_csv("products.csv")
        
        # Map event types to weights
        event_mapping = {
            "view": 1.0,
            "purchase": 3.0
        }
        data_df["event_weight"] = data_df["event_type"].map(event_mapping).fillna(0.0)
        data_df["event_weight"] = data_df["event_weight"] / data_df["event_weight"].max()

        # Clean string types
        data_df['user_id'] = data_df['user_id'].astype(str).str.strip()
        data_df['product_id'] = data_df['product_id'].astype(str).str.strip()

        # Drop duplicates for product metadata
        metadata_df = data_df.drop_duplicates(subset=['product_id'])
        product_info = metadata_df.set_index('product_id').to_dict(orient='index')

        # Create interactions for training or evaluation
        interactions = list(zip(
            data_df['user_id'], data_df['product_id'], data_df['event_weight']
        ))

        # Evaluation method for the collaborative filtering model
        eval_method = RatioSplit(
            data=interactions,
            test_size=0.2,
            rating_threshold=0.5,
            exclude_unknowns=True,
            verbose=True
        )
        print("Loading....")
    
        def vae_collab_recommend(user_id_str, top_n=None, fallback_items=None): 
            train_set = eval_method.train_set
            uid_map = train_set.uid_map
            iid_map = train_set.iid_map
            iid_map_inv = {v: k for k, v in iid_map.items()}

            results = []

            if user_id_str not in uid_map:
                if fallback_items:
                    return [{"product_id": pid, "score": None} for pid in fallback_items[:top_n]]
                return []

            user_id = uid_map[user_id_str]
            n_items = train_set.num_items

            # Get item scores with error handling for CUDA issues
            try:
                scores = []
                for item_id in range(n_items):
                    try:
                        score = model.score(user_id, item_id)
                        # Ensure score is on CPU and convert to Python float
                        if hasattr(score, 'cpu'):
                            score = score.cpu().item() if hasattr(score.cpu(), 'item') else float(score.cpu())
                        elif hasattr(score, 'item'):
                            score = score.item()
                        else:
                            score = float(score)
                        scores.append(score)
                    except Exception as score_error:
                        print(f"Error scoring item {item_id}: {score_error}")
                        scores.append(0.0)  # Default score for failed items
            except Exception as scoring_error:
                print(f"Scoring error: {scoring_error}")
                return [{"error": "Scoring failed due to CUDA compatibility issues"}]

            seen_items = set(train_set.matrix[user_id].indices)
            unseen_scores = [(item_id, scores[item_id]) for item_id in range(len(scores)) if item_id not in seen_items]

            # Sort and take top items
            top_items = sorted(unseen_scores, key=lambda x: x[1], reverse=True)[:top_n]

            # Extract scores for normalization
            max_score = max(score for _, score in top_items) if top_items else 1

            for item_id, score in top_items:
                original_id = iid_map_inv[item_id]
                metadata = product_info.get(original_id, {}).copy()
                metadata['product_id'] = original_id
                # Normalize score to 0-1 range; avoid division by zero
                normalized_score = score / max_score if max_score > 0 else 0
                metadata['score'] = round(normalized_score, 4)
                results.append(metadata)

            return results

        # Call the recommendation function
        recommendations = vae_collab_recommend(user_id, top_n)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })

    except Exception as e:
        print(f"General error in collaborative_recommender: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500



# --- Start Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
