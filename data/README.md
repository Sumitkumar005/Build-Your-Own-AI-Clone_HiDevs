# Data Sources for AI Clone Chatbot

## Current Data Structure
```
data/
├── pdfs/           # PDF documents
├── texts/          # Plain text files
└── unstructured/   # Unstructured data (reviews, articles, etc.)
```

## How to Add More Data

### 1. PDF Documents (`data/pdfs/`)
Download PDFs from these sources:

#### Academic & Research Papers:
- **arXiv.org**: Free research papers in AI/ML
  - Search for: "large language models", "RAG systems", "AI applications"
  - Download: https://arxiv.org/

- **Papers with Code**: Research papers with implementations
  - Visit: https://paperswithcode.com/
  - Topics: NLP, Computer Vision, AI

#### Technical Documentation:
- **Hugging Face Docs**: Transformer models, datasets
  - Download: https://huggingface.co/docs
- **LangChain Documentation**: RAG and LLM frameworks
  - Download: https://python.langchain.com/docs/
- **OpenAI Research**: GPT and AI research papers
  - Visit: https://openai.com/research/

#### Educational Resources:
- **MIT OpenCourseWare**: AI and ML course materials
  - Visit: https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/
- **Stanford CS Courses**: AI course notes and papers
  - Visit: https://cs.stanford.edu/

### 2. Text Files (`data/texts/`)
Create or download text content:

#### Online Sources:
- **Project Gutenberg**: Public domain books
  - Visit: https://www.gutenberg.org/
  - Great for literature and classic texts

- **Wikipedia Data**: Export articles as text
  - Use tools like: https://en.wikipedia.org/wiki/Special:Export
  - Topics: Technology, Science, History

#### Create Custom Content:
- Write technical guides and tutorials
- Document your own projects and experiences
- Create FAQ documents for specific domains

### 3. Unstructured Data (`data/unstructured/`)
Add real-world data:

#### Product Reviews & Feedback:
- **Amazon Product Reviews**: Download datasets from:
  - https://nijianmo.github.io/amazon/index.html
- **Yelp Reviews**: Academic datasets available at:
  - https://www.yelp.com/dataset

#### Social Media & Forums:
- **Reddit Data**: Use Pushshift.io for Reddit archives
  - Visit: https://pushshift.io/
- **Twitter Academic API**: Research access to tweets
  - Visit: https://developer.twitter.com/en/products/twitter-api/academic-research

#### News & Articles:
- **News APIs**: Get articles from:
  - NewsAPI.org, Google News API
- **Common Crawl**: Web crawl data
  - Visit: https://commoncrawl.org/

## Quick Start Data Sources

### For AI/ML Domain:
1. **LangChain Documentation PDF**: Download from python.langchain.com
2. **Hugging Face Transformers Guide**: Download from huggingface.co/docs
3. **Research Papers**: 3-5 recent papers on RAG from arXiv

### For General Knowledge:
1. **Wikipedia Articles**: Export 5-10 articles on AI topics
2. **MIT AI Course Notes**: Download lecture notes
3. **Technical Blogs**: Save articles from Towards Data Science, Medium

### For Product Reviews:
1. **Amazon Review Datasets**: Download electronics/product reviews
2. **Yelp Academic Dataset**: Restaurant/business reviews
3. **Custom Reviews**: Create your own product review data

## Adding Data to Your Project

### Step 1: Download/Collect Data
```bash
# Create directories if needed
mkdir -p data/pdfs data/texts data/unstructured

# Download example PDFs
# wget https://example.com/ai-paper.pdf -O data/pdfs/ai_research_paper.pdf
```

### Step 2: Process Data
The data ingestion pipeline will automatically process:
- **PDFs**: Extract text using PyPDF
- **Text files**: Direct processing
- **Unstructured data**: Parse with unstructured library

### Step 3: Test Ingestion
```bash
# Run data ingestion
python -c "from src.data_ingestion import DataIngestionPipeline; pipeline = DataIngestionPipeline(); pipeline.ingest_all_data()"
```

## Recommended Data Strategy

### Minimum Viable Dataset (for competition):
- 5-10 PDF documents (research papers, technical docs)
- 3-5 text files (guides, tutorials)
- 2-3 unstructured datasets (reviews, articles)

### Optimal Dataset (for production):
- 20+ PDF documents across multiple domains
- 10+ text files with diverse content
- 5+ unstructured datasets for real-world scenarios

## Data Quality Tips

1. **Diversity**: Include content from different domains
2. **Recency**: Use recent papers and current information
3. **Quality**: Prefer authoritative sources (academic papers, official docs)
4. **Relevance**: Choose content related to your target use cases
5. **Size**: Start with smaller datasets, expand gradually

## Example Data Collection Commands

```bash
# Download AI research papers
curl -L "https://arxiv.org/pdf/2305.14314.pdf" -o "data/pdfs/llm_survey.pdf"
curl -L "https://arxiv.org/pdf/2310.11511.pdf" -o "data/pdfs/rag_systems.pdf"

# Create sample text content
echo "Your custom AI guide content here" > "data/texts/custom_guide.txt"

# Download review datasets (if available)
# wget https://example.com/reviews.json -O "data/unstructured/reviews.json"
```

## Next Steps

1. Start with 5-10 documents from the sources above
2. Run the data ingestion pipeline
3. Test chatbot responses
4. Add more data based on performance
5. Focus on your competition domain expertise

This will give your AI Clone Chatbot rich, diverse knowledge to draw from and significantly improve its responses!
