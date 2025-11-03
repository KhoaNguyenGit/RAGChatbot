set -e

VENV_DIR="chatbot"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
else
    echo "Creating virtual environment '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."

source "$VENV_DIR/bin/activate"
echo "Installing required packages..."

pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt