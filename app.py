from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Essay_Writer_Agent import run_agent

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_essay', methods=['POST'])
def generate_essay():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging line
        topic = data.get('topic', '')
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
        
        essay = run_agent(topic)
        print("Generated essay:", essay)  # Debugging line
        return jsonify({'essay': essay})
    except Exception as e:
        print("Error generating essay:", e)  # Debugging line
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
