"""
Gambling Detector — Monitoring Dashboard (Backend)
====================================================

Flask-based backend that:
- Reads the existing final_dataset.csv for dataset statistics
- Serves dummy device/monitoring data (ready to swap for real API)
- Provides REST API endpoints for the frontend dashboard
- Serves the static HTML/CSS/JS frontend

Run with: python app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import random
import hashlib
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_from_directory, request

# Add parent directory to path so we can import from ml-training if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml-training'))

app = Flask(__name__, static_folder='static', static_url_path='')

# ─── Configuration ──────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'ml-training', 'dataset', 'final_dataset.csv'
)
MODEL_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'ml-training', 'model_weights.json'
)

# ─── Dataset Loader ─────────────────────────────────────────────────────────

_dataset_cache = None
_dataset_loaded_at = None


def load_dataset():
    """Load and cache the dataset CSV. Returns a list of dicts."""
    global _dataset_cache, _dataset_loaded_at

    if _dataset_cache is not None:
        return _dataset_cache, _dataset_loaded_at

    entries = []
    if os.path.exists(DATASET_PATH):
        try:
            import pandas as pd
            df = pd.read_csv(DATASET_PATH, sep=';', dtype=str, keep_default_na=False)
            # Only keep the columns we care about
            for _, row in df.iterrows():
                entries.append({
                    'id': row.get('id', ''),
                    'url': row.get('url', ''),
                    'label': int(row.get('label', 0)),
                    'title': row.get('title', '')[:200],  # truncate for dashboard
                })
            _dataset_loaded_at = datetime.now().isoformat()
        except Exception as e:
            print(f"[WARNING] Could not load dataset: {e}")
            _dataset_loaded_at = None
    else:
        print(f"[INFO] Dataset not found at {DATASET_PATH}")
        _dataset_loaded_at = None

    _dataset_cache = entries
    return _dataset_cache, _dataset_loaded_at


def load_model_info():
    """Load model weights metadata."""
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            with open(MODEL_WEIGHTS_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None


REAL_DEVICES = {}
REAL_NEW_WEBSITES = []
REAL_ACTIVITIES = []
# ─── API Routes ─────────────────────────────────────────────────────────────

@app.route('/api/sync', methods=['POST'])
def api_sync():
    """Receive real-time detection data from Android app."""
    print(f"--- [API SYNC] Request received from {request.remote_addr} ---")
    data = request.get_json(force=True, silent=True) or {}
    print(f"Payload: {data}")
    
    device_name = data.get('device_name', 'Unknown Device')
    os_version = data.get('os', 'Unknown OS')
    domain = data.get('domain', '')
    action = data.get('action', 'ALLOWED')
    score = data.get('score', 0.0)
    
    # Simple device ID based on name
    device_id = hashlib.md5(device_name.encode()).hexdigest()[:12]
    
    # Update device stats
    if device_id not in REAL_DEVICES:
        REAL_DEVICES[device_id] = {
            'id': device_id,
            'name': device_name,
            'ip': request.remote_addr,
            'mac': 'Unknown',
            'os': os_version,
            'model': 'App Device',
            'status': 'online',
            'last_seen': datetime.now().isoformat(),
            'app_version': '1.0.0',
            'vpn_active': True,
            'total_queries': 0,
            'blocked_queries': 0,
            'last_domain': '',
        }
    
    device = REAL_DEVICES[device_id]
    device['last_seen'] = datetime.now().isoformat()
    device['total_queries'] += 1
    if action == 'BLOCKED':
        device['blocked_queries'] += 1
    if domain:
        device['last_domain'] = domain
    
    # Record activity
    if domain:
        REAL_ACTIVITIES.insert(0, {
            'type': 'block' if action == 'BLOCKED' else 'allow',
            'domain': domain,
            'device': device_name,
            'time': datetime.now().isoformat(),
            'score': score
        })
        
        # Trim activities to keep memory usage bounded
        if len(REAL_ACTIVITIES) > 200:
            REAL_ACTIVITIES.pop()
            
        # Add to new websites if blocked and not in dataset
        if action == 'BLOCKED':
            dataset, _ = load_dataset()
            dataset_domains = {e['url'].replace('https://', '').replace('http://', '').split('/')[0] for e in (dataset or [])}
            
            if domain not in dataset_domains and not any(w['domain'] == domain for w in REAL_NEW_WEBSITES):
                REAL_NEW_WEBSITES.insert(0, {
                    'id': f'NW{len(REAL_NEW_WEBSITES)+1:03d}',
                    'domain': domain,
                    'full_url': f'https://{domain}',
                    'source_device': device_name,
                    'detected_at': datetime.now().isoformat(),
                    'ml_score': score,
                    'category': 'gambling',
                    'status': 'pending_review',
                    'title': 'Unknown (Detected by App)',
                })

    return jsonify({'success': True})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/stats')
def api_stats():
    """Return dashboard summary statistics."""
    dataset, loaded_at = load_dataset()
    model_info = load_model_info()

    devices_list = list(REAL_DEVICES.values())
    total_devices = len(devices_list)
    online_devices = sum(1 for d in devices_list if d['status'] == 'online')
    idle_devices = sum(1 for d in devices_list if d['status'] == 'idle')
    offline_devices = sum(1 for d in devices_list if d['status'] == 'offline')

    total_new_websites = len(REAL_NEW_WEBSITES)
    pending_review = sum(1 for w in REAL_NEW_WEBSITES if w['status'] == 'pending_review')
    total_queries = sum(d['total_queries'] for d in devices_list)
    total_blocked = sum(d['blocked_queries'] for d in devices_list)

    gambling_count = sum(1 for e in dataset if e['label'] == 1) if dataset else 0
    legit_count = sum(1 for e in dataset if e['label'] == 0) if dataset else 0

    return jsonify({
        'devices': {
            'total': total_devices,
            'online': online_devices,
            'idle': idle_devices,
            'offline': offline_devices,
        },
        'websites': {
            'total_new': total_new_websites,
            'pending_review': pending_review,
            'added_to_dataset': sum(1 for w in REAL_NEW_WEBSITES if w['status'] == 'added_to_dataset'),
            'verified_legit': sum(1 for w in REAL_NEW_WEBSITES if w['status'] == 'verified'),
        },
        'dataset': {
            'total': len(dataset) if dataset else 0,
            'gambling': gambling_count,
            'legitimate': legit_count,
            'loaded_at': loaded_at,
        },
        'queries': {
            'total': total_queries,
            'blocked': total_blocked,
            'allowed': total_queries - total_blocked,
            'block_rate': round(total_blocked / total_queries * 100, 1) if total_queries else 0,
        },
        'model': {
            'threshold': model_info.get('threshold') if model_info else None,
            'f1_score': model_info.get('training_info', {}).get('domain_only_f1') if model_info else None,
            'n_features': model_info.get('training_info', {}).get('n_features') if model_info else None,
        },
        'last_updated': datetime.now().isoformat(),
    })


@app.route('/api/devices')
def api_devices():
    """Return list of devices with optional filtering."""
    status_filter = request.args.get('status', '').lower()
    search = request.args.get('search', '').lower()

    devices = list(REAL_DEVICES.values())
    if status_filter and status_filter != 'all':
        devices = [d for d in devices if d['status'] == status_filter]
    if search:
        devices = [d for d in devices if
                   search in d['name'].lower() or
                   search in d['ip'].lower() or
                   search in d['mac'].lower() or
                   search in d['id'].lower() or
                   search in d.get('last_domain', '').lower()]

    sort_by = request.args.get('sort', 'last_seen')
    reverse = request.args.get('order', 'desc') == 'desc'
    if sort_by in ('name', 'ip', 'mac', 'status', 'last_seen', 'total_queries', 'blocked_queries'):
        devices = sorted(devices, key=lambda d: d.get(sort_by, ''), reverse=reverse)

    return jsonify({'devices': devices, 'total': len(devices)})


@app.route('/api/websites')
def api_websites():
    """Return list of new/unseen websites with optional filtering."""
    status_filter = request.args.get('status', '').lower()
    category_filter = request.args.get('category', '').lower()
    search = request.args.get('search', '').lower()

    websites = REAL_NEW_WEBSITES
    if status_filter and status_filter != 'all':
        websites = [w for w in websites if w['status'] == status_filter]
    if category_filter and category_filter != 'all':
        websites = [w for w in websites if w['category'] == category_filter]
    if search:
        websites = [w for w in websites if
                    search in w['domain'].lower() or
                    search in w.get('title', '').lower() or
                    search in w.get('source_device', '').lower()]

    sort_by = request.args.get('sort', 'detected_at')
    reverse = request.args.get('order', 'desc') == 'desc'
    if sort_by in ('domain', 'detected_at', 'ml_score', 'category', 'status', 'source_device'):
        websites = sorted(websites, key=lambda w: w.get(sort_by, ''), reverse=reverse)

    return jsonify({'websites': websites, 'total': len(websites)})


@app.route('/api/activities')
def api_activities():
    """Return recent activity feed."""
    limit = request.args.get('limit', 15, type=int)
    type_filter = request.args.get('type', '').lower()

    activities = REAL_ACTIVITIES
    if type_filter and type_filter != 'all':
        activities = [a for a in activities if a['type'] == type_filter]

    return jsonify({'activities': activities[:limit], 'total': len(activities)})


@app.route('/api/dataset/search')
def api_dataset_search():
    """Search through the existing dataset."""
    search = request.args.get('q', '').lower()
    label_filter = request.args.get('label', '')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)

    dataset, _ = load_dataset()
    results = dataset

    if label_filter in ('0', '1'):
        results = [e for e in results if e['label'] == int(label_filter)]

    if search:
        results = [e for e in results if
                   search in e['url'].lower() or
                   search in e['title'].lower()]

    paginated_results = results[offset:offset+limit]

    return jsonify({
        'results': paginated_results,
        'total': len(results),
        'showing': len(paginated_results),
    })


@app.route('/api/websites/<website_id>/status', methods=['POST'])
def api_update_website_status(website_id):
    """Update the status of a new website (mark as reviewed, etc.)."""
    data = request.get_json() or {}
    new_status = data.get('status', '')

    for w in REAL_NEW_WEBSITES:
        if w['id'] == website_id:
            if new_status in ('pending_review', 'added_to_dataset', 'verified', 'dismissed'):
                w['status'] = new_status
                return jsonify({'success': True, 'website': w})
            return jsonify({'error': 'Invalid status'}), 400

    return jsonify({'error': 'Website not found'}), 404


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Gambling Detector — Monitoring Dashboard")
    print("=" * 60)
    print(f"  Dataset path: {os.path.abspath(DATASET_PATH)}")
    print(f"  Model path:   {os.path.abspath(MODEL_WEIGHTS_PATH)}")
    print(f"  Dashboard:    http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
