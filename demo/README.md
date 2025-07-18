# doc2lora Demo - Software Developer Assistant

This demo showcases how to use **doc2lora** to create a custom LoRA adapter from documents and deploy it to Cloudflare Workers AI. The demo creates an AI assistant that can answer questions about software development practices based on training data.

## 📁 Demo Structure

```
demo/
├── data/                           # Training documents
│   ├── developer_duties.md         # Daily/weekly/monthly developer responsibilities
│   ├── best_practices.txt          # Development best practices and guidelines
│   ├── developer_profile.json      # Developer skills and current projects
│   └── tech_stack.yaml            # Technical stack and infrastructure details
├── scripts/                        # Automation scripts
│   ├── train_lora.sh/.bat          # Train LoRA adapter from data
│   ├── deploy_to_r2.sh/.bat        # Deploy adapter to R2 bucket
│   └── wrangler_deploy.sh/.bat     # Deploy Cloudflare Worker
├── output/                         # Generated LoRA adapter (created after running)
│   ├── software_dev_adapter.json   # Adapter metadata
│   └── software_dev_adapter/       # Actual adapter weights
├── worker.js                       # Cloudflare Worker implementation
├── wrangler.toml                   # Wrangler configuration
├── index.html                      # Demo web interface
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

1. **doc2lora installed**: Install from the parent directory

   ```bash
   cd .. && pip install -e .
   ```

2. **Cloudflare account**: Sign up at [dash.cloudflare.com](https://dash.cloudflare.com)

3. **Wrangler CLI**: Install the Cloudflare CLI

   ```bash
   npm install -g wrangler
   ```

4. **Environment variables**: Set up your credentials

   ```bash
   # Required for ML training (optional but recommended)
   export HF_API_KEY=your_huggingface_token

   # Required for Cloudflare deployment
   export CLOUDFLARE_ACCOUNT_ID=your_account_id
   export CLOUDFLARE_API_KEY=your_api_key  # Or use CLOUDFLARE_API_TOKEN
   export R2_BUCKET_NAME=doc2lora-adapters  # Optional: defaults to 'doc2lora-adapters'
   ```

### Step 1: Generate LoRA Adapter

**Important**: Run scripts from the demo directory:

```bash
cd demo
./scripts/train_lora.sh    # Linux/Mac
# or
.\scripts\train_lora.bat   # Windows
```

This script will:

- ✅ Check that the `data` directory exists
- ✅ Verify doc2lora is installed
- 🚀 Run doc2lora on the training documents
- 📁 Create the adapter in `output/software_dev_adapter`

### Step 2: Deploy to R2 Bucket

Deploy the adapter to Cloudflare R2:

```bash
cd demo  # Make sure you're in the demo directory
./scripts/deploy_to_r2.sh    # Linux/Mac
# or
.\scripts\deploy_to_r2.bat   # Windows
```

This script will:

- ✅ Verify the adapter was generated
- ✅ Check R2 credentials from .env file
- ☁️ Upload adapter to Cloudflare R2 bucket
- 🚀 Make adapter available for Workers AI

### Step 3: Deploy the Worker

Deploy the demo Worker:

```bash
cd demo  # Make sure you're in the demo directory
./scripts/wrangler_deploy.sh    # Linux/Mac
# or
.\scripts\wrangler_deploy.bat   # Windows
```

### Step 4: Test the Demo

1. Open `index.html` in your browser
2. Update the `WORKER_URL` to your deployed Worker URL
3. Ask questions about software development!

## 🤖 What the Assistant Can Answer

The assistant has been trained on documentation covering:

### 📋 Daily Responsibilities

- Code development and review processes
- Project management and collaboration
- Technical documentation practices
- Quality assurance and testing

### 🛠️ Technical Skills

- Programming languages: Python, JavaScript, TypeScript, Java, Go
- Frameworks: React, Node.js, Django, Flask, Express.js
- Databases: PostgreSQL, MongoDB, Redis, MySQL
- Tools: Docker, Kubernetes, Jenkins, AWS

### 💡 Best Practices

- Version control with Git
- Code quality standards
- Testing methodologies
- Security practices
- Performance optimization

### 🏗️ Current Projects

- E-commerce platform development
- Analytics dashboard creation
- Microservices architecture
- Real-time data visualization

## 💬 Example Questions

Try asking these questions:

- "What are my daily responsibilities as a software developer?"
- "What technologies and frameworks do I work with?"
- "How do I approach problem-solving in development?"
- "What are the best practices I follow for code quality?"
- "What is my current tech stack and development environment?"
- "How do I handle security in my applications?"
- "What testing methodologies do I use?"

## 🔧 Customization

### Adding Your Own Data

1. Replace files in the `data/` directory with your own documentation
2. Run `./generate_adapter.sh` to create a new adapter
3. Deploy with `./deploy_to_cloudflare.sh`

### Modifying the Worker

Edit `worker.js` to:

- Change the system prompt
- Adjust response parameters (temperature, max_tokens)
- Add custom logic or validation
- Integrate with other Cloudflare services

### Training Parameters

Modify `generate_adapter.sh` to adjust:

- `--epochs`: Number of training iterations
- `--batch-size`: Training batch size (reduce if you have memory issues)
- `--learning-rate`: Learning rate for training
- `--lora-r` and `--lora-alpha`: LoRA configuration parameters

## 🌐 Using Different Models

You can modify the scripts to use different Cloudflare AI models:

### Mistral (Default)

```javascript
'@cf/mistralai/mistral-7b-instruct-v0.2-lora'
```

### Gemma

```javascript
'@cf/google/gemma-7b-it-lora'
```

### Llama

```javascript
'@cf/meta-llama/llama-2-7b-chat-hf-lora'
```

## 🐛 Troubleshooting

### Common Issues

1. **"doc2lora not found"**
   - Install doc2lora: `cd .. && pip install -e .`

2. **"data directory not found"**
   - **Make sure you're running scripts from the `demo/` directory**
   - Use: `cd demo && .\scripts\train_lora.bat`

3. **"You are trying to access a gated repo" (HuggingFace 401 error)**
   - **Set your HuggingFace API key**: `set HF_API_KEY=your_token_here` (Windows) or `export HF_API_KEY=your_token_here` (Linux/Mac)
   - **Login to HuggingFace**: `huggingface-cli login`
   - **Request access** to the Mistral model at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
   - **Alternative**: Use a non-gated model like `microsoft/DialoGPT-medium`

4. **"wrangler not found"**
   - Install Node.js from: https://nodejs.org/
   - Install Wrangler: `npm install -g wrangler`

5. **"CLOUDFLARE_ACCOUNT_ID not set"**
   - Set environment variables as shown above
   - Or use: `wrangler login` to authenticate

6. **"Adapter directory not found"**
   - Run `.\scripts\train_lora.bat` first from the demo directory

7. **Memory errors during training**
   - Reduce `--batch-size` to 1
   - Use a smaller model
   - Ensure you have enough RAM/VRAM

### Getting Help

- Check the main doc2lora documentation in the parent directory
- Review Cloudflare Workers AI documentation
- Ensure your environment variables are correctly set

## 📈 Performance Tips

1. **For faster training**: Set `HF_API_KEY` environment variable
2. **For production use**: Increase epochs and use more training data
3. **For memory efficiency**: Reduce batch size and LoRA parameters
4. **For better responses**: Tune temperature and max_tokens in the Worker

## 🎯 Next Steps

After getting the demo working:

1. **Add more training data**: Include more comprehensive documentation
2. **Fine-tune parameters**: Experiment with different training settings
3. **Enhance the Worker**: Add features like conversation history
4. **Create a better UI**: Build a more sophisticated frontend
5. **Monitor usage**: Add analytics and logging to your Worker

---

This demo shows the complete workflow from document to deployed AI assistant using doc2lora and Cloudflare Workers AI. Happy experimenting! 🚀
