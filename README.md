# 🧠 Charisma - Personal Memory Clone

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with Unsloth](https://img.shields.io/badge/Built%20with-Unsloth-orange)](https://github.com/unslothai/unsloth)
[![Powered by Gradio](https://img.shields.io/badge/Powered%20by-Gradio-yellow)](https://www.gradio.app/)

**Clone your memory and personality using AI** - A fully local application that fine-tunes large language models on your personal memories from Notion to create an AI representation of yourself.

## ✨ Features

- 🔗 **Notion Integration** - Browser-based OAuth or API key authentication to fetch all your notes
- 🎯 **Personalized Training** - Uses your personal details (name, age, location, hobbies, etc.) to create context
- 🚀 **Powered by Unsloth** - 2x faster fine-tuning with 60% less memory usage
- 💡 **LoRA & Full Fine-tuning** - Choose between fast LoRA fine-tuning or comprehensive full fine-tuning
- 🎨 **Beautiful UI** - Clean Gradio interface with tabs for training, settings, and logs
- 🔒 **100% Private** - All processing happens locally. NO data is collected on our servers
- 📤 **HuggingFace Integration** - Push your trained models directly to HuggingFace Hub
- ⚙️ **Fully Configurable** - All settings stored in `charisma.toml` and editable via UI
- 📊 **Real-time Console** - Live training logs displayed directly in the UI

## 🔒 Privacy Notice

**IMPORTANT:** Charisma is designed with privacy as the top priority:

- ✅ All data processing happens **locally on your machine**
- ✅ Your Notion data **never leaves your computer**
- ✅ Personal information is **only used for training**
- ✅ **No telemetry**, no analytics, no data collection
- ✅ You control where your models are saved

See [NOTICE.md](NOTICE.md) for complete privacy details.

## 📋 Requirements

- **Python:** 3.10 or higher
- **CUDA:** NVIDIA GPU with CUDA support (recommended)
- **RAM:** 8GB minimum, 16GB+ recommended
- **VRAM:** 4GB+ for default model, varies by model size
- **Storage:** 10GB+ free space

## 🚀 Installation & Setup

### 1. Install Charisma

```powershell
# Clone the repository
git clone https://github.com/muhammad-fiaz/charisma.git
cd charisma

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Create Notion Integration (Internal - API Key)

Charisma uses **Internal Integration** for secure, private access to your Notion workspace.

#### **Step 1: Create Integration**
1. Go to https://www.notion.so/profile/integrations
2. **Login to your Notion account** if not already logged in
3. Click **"+ New integration"**
4. Fill in the details:
   - **Name:** `Charisma` (or any name you prefer)
   - **Associated workspace:** Select your workspace/organization
5. Under **"Integration type"**, select **"Internal"**
   - ℹ️ This keeps your integration private - only you can use it
6. Click **"Submit"** to create the integration

#### **Step 2: Configure Integration (Configuration Tab)**
1. You'll land on the **Configuration** tab after creating the integration
2. Under **"Capabilities"**, make sure these are enabled:
   - ✅ **Read content** (REQUIRED - enable this!)
   - ✅ **Read comments** (optional)
   - ✅ **No user information** (recommended for privacy)
3. Copy your **Internal Integration Secret** (looks like: `secret_xxxxxxxxxxxxx`)
   - Save this securely - you'll paste it in Charisma Settings

#### **Step 3: Grant Access to Your Pages (Access Tab)**
1. Click on the **"Access"** tab at the top
2. Here you'll see which Notion pages/databases your integration can access
3. **Important:** You must manually allow access to your memory pages:
   
   **Method A (Recommended) - Share Individual Pages:**
   - Open each memory page in Notion
   - Click `•••` (three dots) at the top right
   - Select **"Add connections"**
   - Find and select your **Charisma** integration
   - Click **"Confirm"**
   - Repeat for ALL your memory pages
   
   **Method B - Share Parent Folder:**
   - Share the parent folder/database containing all memories
   - All child pages automatically get access
   - Easier if you have many pages

4. **Select ALL your memory pages** from your organization/workspace
5. Verify in the Access tab that all pages are listed

#### **Step 4: Verify Permissions**
Make sure you've granted access to:
- ✅ All daily memory pages (e.g., "Mem 30-10-2025", "Mem 29-10-2025", etc.)
- ✅ Any databases containing memories  
- ✅ Your workspace/organization if using private workspace

### 3. Get HuggingFace Token

Some models require authentication to download. Create a free token:

1. Go to https://huggingface.co/settings/tokens
2. **Create an account** or **login** if you haven't already
3. Click **"New token"**
4. Give it a name (e.g., "Charisma")
5. Select **"Read"** permission (or **"Write"** if you want to upload models later)
6. Click **"Generate"**
7. Copy the token (looks like: `hf_xxxxxxxxxxxxx`)
8. Save this - you'll paste it in Charisma Settings

**Why is this needed?**
- Some models on HuggingFace are gated (require agreement to terms)
- Token allows Charisma to download these models automatically
- Without it, some models may fail to download

### 4. Organize Your Notion Memories

**Important:** For best AI clone results, organize your memories properly!

**Memory Page Structure:**
- Each daily memory should be a **separate page** in Notion
- Use clear, date-based naming (any format works):
  - ✅ `Mem 30-10-2025`
  - ✅ `October 30, 2025 - Daily Journal`
  - ✅ `2025-10-30 Memories`
  - ✅ Any descriptive name you prefer
- **Do NOT** put all memories in one giant page - this confuses the AI

**Recommended Setup:**
```
📁 My Workspace (Private recommended)
  ├─ 📄 Mem 30-10-2025
  ├─ 📄 Mem 29-10-2025
  ├─ 📄 Mem 28-10-2025
  ├─ 📄 Mem 27-10-2025
  └─ ... (one page per day/memory)
```

**Tips:**
- Use a **private workspace** for personal memories (more secure)
- Write naturally - the AI learns from your writing style
- Include thoughts, experiences, opinions, and daily events
- More memories = better AI clone quality (recommend at least 10-20 pages)

## 🎮 Usage

### Launch the Application

```powershell
# Launch locally
charisma

# Launch with public URL (for Google Colab)
charisma --live

# Custom port
charisma --port 8080

# Or run directly with Python
uv run python launch.py
```

The UI will open in your browser at **http://127.0.0.1:7860**

### Step-by-Step Guide

#### **1. Configure Settings (⚙️ Settings Tab)**

#### **1. Configure Settings (⚙️ Settings Tab)**

1. Navigate to the **Settings** tab
2. Under **"Notion API Key"**:
   - Paste your **Internal Integration Secret** (from Step 2 above)
   - Format: `secret_xxxxxxxxxxxxx`
3. (Optional) Add your **HuggingFace Token**
   - Format: `hf_xxxxxxxxxxxxx`
4. Adjust training parameters if needed (defaults work great):
   - Max steps: 100
   - Learning rate: 2e-4
   - Batch size: 2
5. Click **"💾 Save All Settings"**

#### **2. Create Your AI Clone (🎯 Main Tab)**

**Enter Personal Information:**
- **Name:** Your full name
- **Age:** Your age
- **Country:** Your country
- **Location:** Your city
- **Hobbies:** Your hobbies (e.g., "Reading, Coding, Photography")
- **Favorites:** Your favorite things (e.g., "Pizza, Sci-fi movies, Python")

**Connect to Notion:**
1. Click **"🔗 Connect to Notion"**
2. Connection happens automatically using your API key
3. You'll see a success message with your workspace info:
   ```
   ✅ Connected to Notion
   
   Workspace: Your Workspace Name
   Pages: 25
   ```
4. **Important:** Only pages you shared with the integration (in Step 3 above) will be visible
5. If you see "0 pages", make sure you've shared your memory pages with the Charisma integration

**Select Memories:**
- All accessible memory pages are listed with checkboxes
   - By default, all are selected
   - Uncheck any pages you don't want to include in training
   - **Tip:** Include at least 10-20 memory pages for best results

**Choose Model & Configure Training:**
- **Model Selection:**
  - Default: `unsloth/gemma-3-270m-it` (270M params, ~4GB VRAM)
  - Or choose from 10+ pre-configured models
  - Or enter any HuggingFace model ID
- **Training Mode:**
  - ✅ **LoRA Fine-tune** - Fast, efficient (recommended)
  - ⬜ **Full Fine-tune** - Thorough but slower
- **Output Model Name:** 
  - Enter a name for your model (e.g., `my-memory-clone`)

**Generate Your Clone:**
1. Click **"✨ Generate AI Clone"**
2. Watch real-time training progress in the **console output** below
3. Training logs show:
   - Data processing steps
   - Model loading progress
   - Training metrics (loss, learning rate)
   - Completion status
4. Wait for completion (typically 5-30 minutes depending on model size)

#### **3. Monitor Training (📋 Logs Tab)**

- View detailed training logs
- Select different log files from the dropdown
- Monitor progress and debug any issues
- Logs are automatically saved to the `logs/` directory

## 🎯 Available Models

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `unsloth/gemma-3-270m-it` | 270M | ~4GB | **Default** - Fast & efficient |
| `unsloth/gemma-2-2b-it` | 2B | ~6GB | Balanced performance |
| `unsloth/Llama-3.2-1B-Instruct` | 1B | ~4GB | Compact & fast |
| `unsloth/Llama-3.2-3B-Instruct` | 3B | ~8GB | Better quality |
| `unsloth/Meta-Llama-3.1-8B-Instruct` | 8B | ~16GB | High quality |
| `unsloth/Qwen2.5-7B-Instruct` | 7B | ~14GB | Excellent reasoning |
| `unsloth/Phi-3.5-mini-instruct` | 3.8B | ~8GB | Microsoft Phi |
| `unsloth/mistral-7b-instruct-v0.3` | 7B | ~14GB | Strong general model |
| `unsloth/Ministral-8B-Instruct-2410` | 8B | ~16GB | Latest Ministral |
| `unsloth/Llama-3.3-70B-Instruct` | 70B | ~40GB | Best quality (needs large GPU) |

You can also use **any custom model** from HuggingFace!

## ⚙️ Configuration

All settings are stored in `charisma.toml` (created automatically):

```toml
[project]
name = "charisma"
version = "0.1.0"

[model]
max_seq_length = 2048
load_in_4bit = true

[training]
batch_size = 2
learning_rate = 0.0002
num_epochs = 1
max_steps = 60

[lora]
r = 16
lora_alpha = 16
lora_dropout = 0

[notion]
api_key = ""

[huggingface]
token = ""
default_repo = "my-charisma-model"
private = true
```

Edit these values in the **Settings tab** or directly in the file.

## 📁 Project Structure

```
charisma/
├── charisma/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config/                 # Configuration management
│   │   ├── config_manager.py
│   │   └── models.py
│   ├── core/                   # Core training logic
│   │   ├── data_processor.py
│   │   ├── model_manager.py
│   │   └── trainer.py
│   ├── integrations/           # External integrations
│   │   ├── notion_client.py
│   │   └── huggingface_client.py
│   ├── ui/                     # Gradio UI
│   │   ├── app.py
│   │   └── tabs/
│   │       ├── main_tab.py
│   │       ├── settings_tab.py
│   │       └── logs_tab.py
│   └── utils/                  # Utilities
│       ├── logger.py
│       └── validators.py
├── outputs/                    # Trained models (created at runtime)
├── logs/                       # Application logs (created at runtime)
├── charisma.toml              # Configuration file
├── pyproject.toml             # Project metadata
├── NOTICE.md                  # Privacy notice
└── README.md                  # This file
```

## 🛠️ Development

### Running from Source

```powershell
# Install in editable mode
uv sync

# Or with pip
pip install -e .

# Run directly
python -m charisma.main
```

### Command-Line Options

```powershell
charisma --help

Options:
  --live              Create public URL (for Colab)
  --port PORT         Port number (default: 7860)
  --config PATH       Config file path (default: charisma.toml)
  --server-name IP    Server IP (default: 127.0.0.1)
  --debug             Enable debug mode
```

## 🐛 Troubleshooting

### "CUDA out of memory" error
- Use a smaller model (e.g., `unsloth/Llama-3.2-1B-Instruct`)
- Reduce `batch_size` in Settings
- Enable `load_in_4bit` in Settings
- Use LoRA fine-tuning instead of full fine-tuning

### "Notion connection failed"
- Verify your API token in Settings
- Ensure you've shared pages with your Notion integration
- Test connection using the "🧪 Test" button

### Training is slow
- LoRA fine-tuning is much faster than full fine-tuning
- Reduce `max_steps` or `num_epochs`
- Use a smaller model
- Ensure you have a CUDA-capable GPU

### No memories found
- Check that your Notion pages are shared with the integration
- Ensure the integration has read permissions
- Refresh the connection

## 📚 How It Works

1. **Data Collection:** Fetches your notes from Notion via the Notion API
2. **Data Processing:** Converts memories into conversation format with your personal context
3. **Model Loading:** Loads an Unsloth FastLanguageModel with optional LoRA adapters
4. **Training:** Fine-tunes the model on your memories using supervised fine-tuning (SFT)
5. **Saving:** Saves the trained model locally (and optionally to HuggingFace)

The training uses the **Gemma-3 chat template** format:

```
System: You are [your name], [your details]...
User: Tell me about [date/topic]
Assistant: [Your memory content]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Unsloth](https://github.com/unslothai/unsloth)** - For 2x faster and more memory-efficient LLM fine-tuning
- **[Gradio](https://www.gradio.app/)** - For the amazing UI framework
- **[HuggingFace](https://huggingface.co/)** - For transformers and model hosting
- **[Notion](https://www.notion.so/)** - For the API that makes memory collection possible

## 📧 Contact

**Muhammad Fiaz**
- Email: contact@muhammadfiaz.com
- GitHub: [@muhammad-fiaz](https://github.com/muhammad-fiaz)

## ⚠️ Disclaimer

This tool is for personal use or educational purposes only. By using Charisma:
- You are responsible for your Notion data and API usage
- You agree that all processing is done locally at your own risk
- The authors are not responsible for any data loss or misuse
- Ensure you comply with Notion's and HuggingFace's terms of service

---


