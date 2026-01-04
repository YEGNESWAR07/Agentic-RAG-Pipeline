# Agentic RAG Pipeline

A **local, privacy-friendly** Agentic RAG (Retrieval-Augmented Generation) system using LangChain, ChromaDB, and local language models. This system intelligently decides when to search your documents and when to answer directly, providing accurate answers with confidence scores.

## 🌟 Features

- **🔒 100% Local & Private**: All processing happens on your machine - no data sent to external servers
- **🤖 Intelligent Agent**: Automatically decides whether to search documents or answer directly
- **🧮 Smart Arithmetic**: Computes math questions directly with 100% accuracy (e.g., "What is 2+2?" → "The answer is 4.")
- **📊 Accuracy Metrics**: Provides confidence scores and performance metrics for each answer
- **📚 Multi-Document Support**: Process multiple PDFs and query across all of them
- **⚡ Performance Tracking**: Shows retrieval time, generation time, and sources used
- **🎯 Sample Data Included**: 10 diverse sample PDFs to get started immediately


## 📋 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 10GB free (for models and data)
- **GPU** (optional): CUDA-compatible GPU for faster processing

## 🚀 Quick Start

### 1. Setup Environment

**Windows PowerShell:**
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

The project includes a script to generate 10 diverse sample PDFs for testing:

```powershell
python generate_sample_pdfs.py
```

This creates sample documents including:
- 📄 Invoice
- 📖 User Manual
- 📊 Technical Report
- 📝 Meeting Minutes
- 🛍️ Product Catalog
- 🔬 Research Paper
- 📋 Policy Document
- 🎓 Training Guide
- 💰 Financial Summary
- ❓ FAQ Document

### 3. Build the Index

Process your PDFs and create a searchable vector store:

```powershell
python rag_agent.py build --data-dir ./data
```

**Options:**
- `--data-dir`: Folder containing PDF files (default: `./data`)
- `--persist-dir`: Where to save the index (default: `./chroma_db`)
- `--chunk-size`: Size of text chunks (default: 500)
- `--chunk-overlap`: Overlap between chunks (default: 80)

### 4. Query the System

**One-off query:**
```powershell
python rag_agent.py query --q "What is the total amount in the invoice?"
```

**Interactive mode:**
```powershell
python rag_agent.py query
```

**Query options:**
- `--q`: Your question (omit for interactive mode)
- `--k`: Number of document chunks to retrieve (default: 3)
- `--verbose`: Show detailed metadata (confidence, timing, sources)
- `--persist-dir`: Location of the index (default: `./chroma_db`)

## 📊 Understanding the Output

Each answer includes accuracy and performance metrics:

```
==================================================================
ANSWER:
==================================================================
The total amount in the invoice is $2,653.57
==================================================================

Metadata:
  Action: SEARCH
  Confidence: 85.0%
  Sources Used: 3
  Retrieval Time: 0.15s
  Generation Time: 0.42s
  Total Time: 0.57s
```

**Metrics Explained:**
- **Action**: `SEARCH` (used documents) or `DIRECT` (answered without searching)
- **Confidence**: Estimated accuracy of the answer (0-100%)
  - 100% for computed arithmetic answers
  - 70-90% for document-based answers
  - Lower for uncertain answers
- **Sources Used**: Number of document chunks used (0 for direct/arithmetic answers)
- **Retrieval Time**: Time to find relevant documents
- **Generation Time**: Time to generate the answer (0.00s for arithmetic)
- **Total Time**: Complete processing time

### Arithmetic Questions

The system automatically detects and computes simple math questions:

```
Query: "What is 2+2?"
Answer: The answer is 4.
Confidence: 100.0%
Generation Time: 0.00s
```

Supported operations: `+`, `-`, `*`, `/`


## 📁 Adding Your Own Data

### Using Custom PDFs

1. **Place your PDF files** in the `./data` folder (or any folder you prefer)
2. **Rebuild the index** to include your documents:
   ```powershell
   python rag_agent.py build --data-dir ./data
   ```
3. **Query your documents**:
   ```powershell
   python rag_agent.py query --q "Your question about your documents"
   ```

### Best Practices for Your Data

✅ **DO:**
- Use text-based PDFs (not scanned images)
- Organize related documents in the same folder
- Use clear, descriptive filenames
- Keep individual PDFs under 100 pages for best performance

❌ **DON'T:**
- Use password-protected PDFs
- Mix unrelated document types in one index
- Use corrupted or damaged PDF files

### Multiple Document Collections

You can maintain separate indexes for different projects:

```powershell
# Build index for project A
python rag_agent.py build --data-dir ./project_a_docs --persist-dir ./index_a

# Build index for project B
python rag_agent.py build --data-dir ./project_b_docs --persist-dir ./index_b

# Query project A
python rag_agent.py query --persist-dir ./index_a

# Query project B
python rag_agent.py query --persist-dir ./index_b
```

## 🎯 Example Queries

Try these questions with the sample data:

```
What is the total amount in the invoice?
Summarize the user manual in 5 points
What were the key findings in the technical report?
List the action items from the meeting minutes
What products are available in the catalog?
What is the main conclusion of the research paper?
Explain the remote work policy
What are the goals for new employees in the first 90 days?
What was the Q4 2023 revenue?
How do I troubleshoot installation errors?
```

## 🔧 Customization

### Using a Different Language Model

Edit `rag_agent.py` and change the `DEFAULT_LLM_MODEL` constant:

```python
DEFAULT_LLM_MODEL = "google/flan-t5-large"  # Larger, more accurate
# or
DEFAULT_LLM_MODEL = "google/flan-t5-small"  # Smaller, faster
```

Browse available models at [Hugging Face](https://huggingface.co/models?pipeline_tag=text2text-generation).

### Using a Different Embedding Model

Change the `DEFAULT_EMBEDDING_MODEL` constant:

```python
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"  # More accurate
# or
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Faster (default)
```

### Adjusting Chunk Size

Smaller chunks = more precise but may miss context
Larger chunks = more context but less precise

```powershell
python rag_agent.py build --chunk-size 1000 --chunk-overlap 100
```

## 🐛 Troubleshooting

### Installation Issues

**Problem**: `pip install` fails
```powershell
# Update pip first
pip install --upgrade pip
# Try installing again
pip install -r requirements.txt
```

**Problem**: CUDA/GPU errors
```powershell
# Install CPU-only version of PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Runtime Issues

**Problem**: "No PDF files found"
- Check that PDFs are in the correct folder
- Verify the `--data-dir` path is correct
- Ensure files have `.pdf` extension

**Problem**: "Memory error"
- Close other applications
- Reduce chunk size: `--chunk-size 300`
- Process fewer documents at once
- Use a smaller model

**Problem**: "Answers seem irrelevant"
- Rebuild index with different chunk size
- Increase retrieval: `--k 5`
- Rephrase your question
- Verify the information exists in your documents

**Problem**: Slow performance
- First query is always slower (model loading)
- Use GPU if available
- Reduce number of retrieved chunks: `--k 2`
- Use a smaller model

## 📚 Project Structure

```
Agentic RAG Pipeline/
├── data/                      # PDF documents (sample data included)
│   ├── sample_invoice.pdf
│   ├── sample_manual.pdf
│   └── ... (8 more samples)
├── chroma_db/                 # Vector store (created after build)
├── rag_agent.py              # Main application
├── generate_sample_pdfs.py   # Sample data generator
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔐 Privacy & Security

- **100% Local**: All data stays on your machine
- **No API Calls**: No external services used
- **No Telemetry**: No usage data collected
- **Open Source**: Full code transparency

## ⚙️ Technical Details

- **Framework**: LangChain
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Flan-T5-Base (local)
- **PDF Processing**: PyPDF
- **Agent Logic**: Custom decision-making based on query analysis

## 🤝 Contributing

This is a local project, but feel free to:
- Customize for your needs
- Share improvements
- Report issues
- Suggest features

## 📄 License

This project is provided as-is for educational and personal use.

## 📞 Support

For questions or issues:
- Check the Troubleshooting section above
- Review the code comments in `rag_agent.py`
- Consult LangChain documentation: https://python.langchain.com/

## 🎓 Learn More

- **LangChain**: https://python.langchain.com/
- **ChromaDB**: https://www.trychroma.com/
- **Hugging Face**: https://huggingface.co/
- **RAG Concepts**: https://arxiv.org/abs/2005.11401

---

**Note**: This project currently runs on sample data. To use with your own documents, simply replace the PDFs in the `./data` folder and rebuild the index. The system will work with any text-based PDF documents you provide.

**Happy Querying! 🚀**
