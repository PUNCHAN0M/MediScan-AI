from flask import Flask, render_template, jsonify, send_from_directory
from pathlib import Path
import json

app = Flask(__name__)

# Global cache
umap_data_cache = None

def load_umap_data():
    """โหลดข้อมูลจากไฟล์ JSON ล่าสุด"""
    global umap_data_cache
    
    if umap_data_cache is not None:
        return umap_data_cache
    
    save_dir = Path("umap_vectors")
    json_files = list(save_dir.glob("umap_data_*.json"))
    
    if not json_files:
        raise FileNotFoundError("No UMAP data found. Please run extract_and_save.py first.")
    
    # หาไฟล์ latest หรือไฟล์ล่าสุด
    latest_file = save_dir / "umap_data_latest.json"
    if not latest_file.exists():
        latest_file = sorted(json_files)[-1]
    
    print(f"📂 Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    umap_data_cache = data
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """API ส่งข้อมูล UMAP"""
    try:
        data = load_umap_data()
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    """API ส่งสถิติ"""
    try:
        data = load_umap_data()
        return jsonify({
            'success': True,
            'stats': data['metadata']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reload', methods=['POST'])
def reload_data():
    """API โหลดข้อมูลใหม่"""
    global umap_data_cache
    umap_data_cache = None  # ล้าง cache
    try:
        data = load_umap_data()
        return jsonify({
            'success': True,
            'message': 'Data reloaded successfully',
            'stats': data['metadata']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting UMAP Visualization Server...")
    print("📂 Make sure to run extract_and_save.py first")
    print("🌐 Open http://localhost:5000 in your browser")
    print("📊 Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5000)