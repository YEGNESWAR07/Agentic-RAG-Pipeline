"""
Generate diverse sample PDF documents for testing the Agentic RAG Pipeline.
Creates 10 different types of documents with realistic content.
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import os
import argparse


def create_pdf(filename: str, title: str, content: str, output_dir: str = "data"):
    """Create a PDF file with the given title and content."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, title)
    
    # Content
    c.setFont("Helvetica", 11)
    y = height - 110
    
    for line in content.split("\n"):
        if y < 72:  # Start new page if needed
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 72
        c.drawString(72, y, line)
        y -= 16
    
    c.save()
    print(f"✓ Created: {filename}")


def generate_sample_pdfs(output_dir: str = "data"):
    """Generate all sample PDF documents."""
    
    samples = [
        # 1. Invoice
        ("sample_invoice.pdf", "Invoice #INV-2024-001", 
         """Acme Corporation
123 Business Street, Suite 100
New York, NY 10001
Phone: (555) 123-4567

BILL TO:
Tech Solutions Inc.
456 Innovation Drive
San Francisco, CA 94102

Invoice Date: January 4, 2024
Due Date: February 4, 2024
Payment Terms: Net 30

ITEMS:
1. Premium Widget Model X - Qty: 10 - Unit Price: $99.99 - Total: $999.90
2. Standard Widget Model Y - Qty: 25 - Unit Price: $49.99 - Total: $1,249.75
3. Widget Maintenance Kit - Qty: 5 - Unit Price: $29.99 - Total: $149.95

Subtotal: $2,399.60
Tax (8.5%): $203.97
Shipping: $50.00
TOTAL DUE: $2,653.57

Payment Instructions:
Please remit payment via check or wire transfer to:
Bank: First National Bank
Account: 123456789
Routing: 987654321

Thank you for your business!"""),

        # 2. User Manual
        ("sample_manual.pdf", "Agentic RAG System - User Manual",
         """Version 1.0 - January 2024

TABLE OF CONTENTS:
1. Introduction
2. System Requirements
3. Installation
4. Building the Index
5. Querying the System
6. Troubleshooting
7. Support

1. INTRODUCTION
The Agentic RAG (Retrieval-Augmented Generation) system is a powerful tool
for document search and question answering. It uses advanced AI to understand
your documents and provide accurate answers to your questions.

2. SYSTEM REQUIREMENTS
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Internet connection for initial model download

3. INSTALLATION
Step 1: Create a virtual environment
Step 2: Install dependencies from requirements.txt
Step 3: Prepare your PDF documents in the data folder

4. BUILDING THE INDEX
Use the build command to process your documents and create a searchable index.
This needs to be done once, or whenever you add new documents.

5. QUERYING THE SYSTEM
You can query in two modes:
- One-off mode: For single questions
- Interactive mode: For conversational queries

6. TROUBLESHOOTING
If you encounter issues, check the following:
- Ensure all dependencies are installed
- Verify PDF files are not corrupted
- Check available disk space
- Review error messages carefully

7. SUPPORT
For assistance, contact: support@agenticrag.example.com
Documentation: https://docs.agenticrag.example.com"""),

        # 3. Technical Report
        ("sample_tech_report.pdf", "Q4 2023 Technical Performance Report",
         """Executive Summary
This report analyzes the technical performance metrics for Q4 2023.

KEY FINDINGS:
- System uptime: 99.97%
- Average response time: 145ms
- Total API calls processed: 15.2 million
- Error rate: 0.03%

INFRASTRUCTURE METRICS:
Server Performance:
- CPU utilization: Average 42%, Peak 78%
- Memory usage: Average 6.2GB, Peak 11.8GB
- Network throughput: 2.3 TB total data transferred
- Storage: 450GB used of 2TB capacity

Database Performance:
- Query response time: Average 23ms
- Concurrent connections: Peak 1,250
- Data growth: 45GB added in Q4
- Backup success rate: 100%

SECURITY METRICS:
- Zero security incidents reported
- 12 security patches applied
- 3 penetration tests conducted
- All compliance requirements met

RECOMMENDATIONS:
1. Increase server capacity by 25% to handle growth
2. Implement additional caching layer
3. Upgrade database to latest version
4. Schedule quarterly security audits

Prepared by: Technical Operations Team
Date: January 4, 2024"""),

        # 4. Meeting Minutes
        ("sample_meeting_minutes.pdf", "Product Development Meeting - Minutes",
         """Date: January 3, 2024
Time: 2:00 PM - 3:30 PM
Location: Conference Room B / Virtual

ATTENDEES:
- Sarah Johnson (Product Manager)
- Mike Chen (Lead Developer)
- Emily Rodriguez (UX Designer)
- David Kim (QA Manager)
- Lisa Wang (Marketing)

AGENDA:
1. Q1 2024 Product Roadmap
2. User Feedback Review
3. Feature Prioritization
4. Timeline Discussion

DISCUSSION POINTS:

1. Q1 2024 Product Roadmap
Sarah presented the proposed roadmap for Q1. Key features include:
- Enhanced search functionality
- Mobile app improvements
- New analytics dashboard
- API rate limiting improvements

2. User Feedback Review
Emily shared insights from recent user surveys:
- 87% satisfaction rate
- Top request: Better mobile experience
- 23% want dark mode
- Performance improvements needed for large datasets

3. Feature Prioritization
Team voted on priority features:
HIGH: Mobile optimization, Performance improvements
MEDIUM: Analytics dashboard, Dark mode
LOW: Additional integrations

4. Timeline Discussion
Agreed timeline:
- Week 1-2: Design phase
- Week 3-6: Development
- Week 7-8: Testing
- Week 9: Release

ACTION ITEMS:
- Mike: Create technical specifications by Jan 10
- Emily: Complete UX mockups by Jan 12
- David: Prepare test plan by Jan 15
- Lisa: Draft release announcement by Jan 20

NEXT MEETING: January 17, 2024 at 2:00 PM"""),

        # 5. Product Catalog
        ("sample_product_catalog.pdf", "2024 Product Catalog",
         """Welcome to Our 2024 Product Line

CATEGORY: ELECTRONICS

Smart Home Hub Pro
Model: SHH-2024-PRO
Price: $299.99
Features:
- Voice control compatible
- Supports 100+ devices
- Energy monitoring
- 2-year warranty

Wireless Security Camera 4K
Model: WSC-4K-360
Price: $179.99
Features:
- 4K resolution
- 360-degree rotation
- Night vision up to 50ft
- Cloud storage included

CATEGORY: ACCESSORIES

Premium Charging Station
Model: PCS-5PORT
Price: $49.99
Features:
- 5 USB ports
- Fast charging support
- Compact design
- Surge protection

Smart LED Bulbs (4-pack)
Model: SLB-RGB-4PK
Price: $39.99
Features:
- 16 million colors
- App controlled
- Energy efficient
- 10-year lifespan

CATEGORY: SERVICES

Premium Support Plan
Price: $99.99/year
Includes:
- 24/7 phone support
- Priority service
- Free shipping on repairs
- Extended warranty

Installation Service
Price: $149.99 per device
Includes:
- Professional installation
- Network setup
- Training session
- 30-day support

For orders or inquiries:
Phone: 1-800-TECH-HELP
Email: sales@techproducts.example.com
Website: www.techproducts.example.com"""),

        # 6. Research Paper Abstract
        ("sample_research_paper.pdf", "Machine Learning in Document Retrieval",
         """Abstract

Title: Enhancing Document Retrieval Through Agentic AI Systems
Authors: Dr. Jane Smith, Prof. Robert Johnson, Dr. Maria Garcia
Institution: Institute of Advanced Computing
Date: December 2023

ABSTRACT:
This paper presents a novel approach to document retrieval using agentic
artificial intelligence systems. We introduce a framework that combines
retrieval-augmented generation (RAG) with intelligent agent-based decision
making to improve the accuracy and relevance of document search results.

INTRODUCTION:
Traditional document retrieval systems rely on keyword matching and basic
semantic search. Our approach enhances this by incorporating an intelligent
agent that can decide when to search, what to search for, and how to
synthesize information from multiple sources.

METHODOLOGY:
We developed a three-stage pipeline:
1. Document Processing: PDFs are chunked and embedded using transformer models
2. Intelligent Routing: An agent decides whether to search or answer directly
3. Response Generation: Context-aware answer generation using local LLMs

RESULTS:
Our experiments on a dataset of 10,000 technical documents showed:
- 34% improvement in answer relevance
- 28% reduction in response time
- 92% user satisfaction rate
- Significant reduction in hallucinations

EXPERIMENTS:
We tested three configurations:
- Baseline: Traditional keyword search
- RAG-only: Standard RAG without agent
- Agentic RAG: Our proposed system

The agentic RAG system outperformed both baselines across all metrics.

CONCLUSION:
Agentic AI systems represent a significant advancement in document retrieval
technology. By incorporating intelligent decision-making, we can provide more
accurate, relevant, and efficient search results.

FUTURE WORK:
- Multi-modal document support
- Real-time learning from user feedback
- Distributed processing for large-scale deployments

Keywords: RAG, Agentic AI, Document Retrieval, NLP, Machine Learning"""),

        # 7. Policy Document
        ("sample_policy_document.pdf", "Remote Work Policy - 2024",
         """Company Policy: Remote Work Guidelines
Effective Date: January 1, 2024
Version: 2.0

PURPOSE:
This policy establishes guidelines for remote work arrangements to ensure
productivity, security, and work-life balance for all employees.

SCOPE:
This policy applies to all full-time and part-time employees who are
approved for remote work arrangements.

ELIGIBILITY:
Employees must meet the following criteria:
- Completed 90-day probationary period
- Satisfactory performance reviews
- Role suitable for remote work
- Manager approval required

WORK ARRANGEMENTS:

Fully Remote:
- Work from home 5 days per week
- Quarterly in-office meetings required
- Must be available during core hours (10 AM - 3 PM local time)

Hybrid:
- Minimum 2 days in office per week
- Flexible scheduling with manager approval
- Team collaboration days: Tuesdays and Thursdays

EQUIPMENT AND TECHNOLOGY:
Company Provided:
- Laptop computer
- Monitor (if requested)
- Keyboard and mouse
- VPN access

Employee Responsibility:
- Reliable internet connection (minimum 25 Mbps)
- Quiet, dedicated workspace
- Proper lighting for video calls

SECURITY REQUIREMENTS:
- Use company VPN for all work activities
- Enable full disk encryption
- Use strong passwords and 2FA
- Never share credentials
- Report security incidents immediately

COMMUNICATION EXPECTATIONS:
- Respond to emails within 4 hours during work hours
- Attend all scheduled meetings
- Keep calendar updated
- Use video for team meetings

PERFORMANCE METRICS:
Remote employees will be evaluated on:
- Quality of work output
- Meeting deadlines
- Communication responsiveness
- Team collaboration

TERMINATION OF REMOTE WORK:
Remote work privileges may be revoked if:
- Performance declines
- Security violations occur
- Business needs change
- Policy violations

For questions, contact HR at hr@company.example.com"""),

        # 8. Training Guide
        ("sample_training_guide.pdf", "New Employee Onboarding Guide",
         """Welcome to the Team!

WEEK 1: ORIENTATION

Day 1: Getting Started
- 9:00 AM: Welcome meeting with HR
- 10:00 AM: IT setup (computer, accounts, access)
- 11:00 AM: Office tour
- 1:00 PM: Team introduction lunch
- 2:00 PM: Review employee handbook
- 3:00 PM: Benefits enrollment

Day 2-3: Company Overview
- Company history and mission
- Organizational structure
- Products and services overview
- Customer base and market position
- Core values and culture

Day 4-5: Department Training
- Meet your team members
- Review department goals
- Understand your role and responsibilities
- Set up initial projects
- Schedule regular 1-on-1s with manager

WEEK 2: ROLE-SPECIFIC TRAINING

Technical Training:
- System access and tools
- Software and platforms used
- Development environment setup
- Code repositories and documentation
- Testing procedures

Process Training:
- Project management methodology
- Communication protocols
- Time tracking and reporting
- Meeting schedules
- Escalation procedures

WEEK 3-4: HANDS-ON LEARNING

Shadowing:
- Observe experienced team members
- Participate in team meetings
- Review past projects
- Ask questions freely

Initial Projects:
- Start with small, guided tasks
- Gradually increase complexity
- Regular check-ins with mentor
- Feedback sessions

RESOURCES:

Internal Documentation:
- Company wiki: wiki.company.internal
- Training portal: training.company.internal
- HR portal: hr.company.internal

Key Contacts:
- IT Support: ext. 1234
- HR: ext. 5678
- Facilities: ext. 9012

FIRST 90 DAYS GOALS:
- Complete all required training modules
- Successfully complete 3 starter projects
- Build relationships with team members
- Understand company processes and culture
- Receive positive feedback in 30/60/90 day reviews

Remember: Everyone was new once. Don't hesitate to ask questions!"""),

        # 9. Financial Summary
        ("sample_financial_summary.pdf", "Q4 2023 Financial Summary",
         """Quarterly Financial Report
Period: October 1 - December 31, 2023

REVENUE SUMMARY:

Total Revenue: $4,850,000
- Product Sales: $3,200,000 (66%)
- Service Revenue: $1,350,000 (28%)
- Licensing: $300,000 (6%)

Year-over-Year Growth: +18%
Quarter-over-Quarter Growth: +7%

REVENUE BY REGION:

North America: $2,425,000 (50%)
Europe: $1,455,000 (30%)
Asia Pacific: $728,000 (15%)
Other: $242,000 (5%)

EXPENSE SUMMARY:

Total Expenses: $3,640,000
- Personnel: $2,100,000 (58%)
- Operations: $910,000 (25%)
- Marketing: $364,000 (10%)
- R&D: $266,000 (7%)

PROFITABILITY:

Gross Profit: $1,210,000
Operating Income: $1,210,000
Net Income: $1,089,000
Profit Margin: 22.5%

BALANCE SHEET HIGHLIGHTS:

Assets:
- Cash and Equivalents: $2,500,000
- Accounts Receivable: $1,200,000
- Inventory: $800,000
- Total Assets: $6,800,000

Liabilities:
- Accounts Payable: $600,000
- Short-term Debt: $400,000
- Total Liabilities: $1,500,000

Equity: $5,300,000

KEY PERFORMANCE INDICATORS:

Customer Metrics:
- New Customers: 450
- Customer Retention Rate: 94%
- Average Contract Value: $8,500
- Customer Lifetime Value: $42,000

Operational Metrics:
- Gross Margin: 66%
- Operating Margin: 25%
- Return on Equity: 20.5%
- Current Ratio: 2.8

OUTLOOK FOR Q1 2024:

Expected Revenue: $5,100,000 (+5% QoQ)
Expected Net Income: $1,150,000
Focus Areas:
- Expand into new markets
- Launch 2 new product lines
- Increase marketing spend by 15%
- Hire 10 additional team members

RISKS AND OPPORTUNITIES:

Risks:
- Economic uncertainty
- Increased competition
- Supply chain challenges

Opportunities:
- Growing market demand
- New partnership opportunities
- Technology advancements

Prepared by: Finance Department
Approved by: CFO Sarah Martinez
Date: January 4, 2024"""),

        # 10. FAQ Document
        ("sample_faq.pdf", "Frequently Asked Questions - Agentic RAG System",
         """Frequently Asked Questions

GENERAL QUESTIONS:

Q1: What is an Agentic RAG system?
A: Agentic RAG (Retrieval-Augmented Generation) is an AI system that
intelligently decides when to search documents and when to answer directly.
It combines document retrieval with language generation for accurate answers.

Q2: What types of documents can I use?
A: Currently, the system supports PDF documents. We recommend using text-based
PDFs rather than scanned images for best results.

Q3: How many documents can I process?
A: There's no hard limit, but performance is optimal with up to 1,000 documents.
For larger collections, consider organizing into separate indexes.

Q4: Is my data secure?
A: Yes! The system runs entirely locally on your machine. No data is sent to
external servers. All processing happens on your computer.

INSTALLATION QUESTIONS:

Q5: What are the system requirements?
A: You need Python 3.8+, 8GB RAM (16GB recommended), and 10GB free disk space.
The first run will download AI models (approximately 2GB).

Q6: How long does installation take?
A: Initial setup takes 10-15 minutes, including downloading dependencies and
AI models. Subsequent runs are much faster.

Q7: I'm getting installation errors. What should I do?
A: Ensure you're using a virtual environment and have the latest pip version.
Try: pip install --upgrade pip, then reinstall requirements.

USAGE QUESTIONS:

Q8: How do I add new documents?
A: Place PDF files in the data folder, then run the build command to reindex.
The system will process all PDFs in the folder.

Q9: How accurate are the answers?
A: Accuracy depends on document quality and question clarity. The system
provides confidence scores to help you assess answer reliability.

Q10: Can I use this for multiple projects?
A: Yes! Use different persist-dir paths for different document collections.
Each project can have its own index.

Q11: What's the difference between search and direct mode?
A: Search mode retrieves relevant document chunks before answering. Direct
mode answers without searching. The agent automatically chooses based on
your question.

PERFORMANCE QUESTIONS:

Q12: Why is the first query slow?
A: The AI model loads into memory on first use. Subsequent queries are faster.
Consider keeping the system running for multiple queries.

Q13: How can I improve answer quality?
A: Use clear, specific questions. Ensure your documents are well-formatted.
Increase the number of retrieved chunks (k parameter) for complex questions.

Q14: Can I use GPU acceleration?
A: Yes! If you have a CUDA-compatible GPU, the system will automatically use
it for faster processing. Ensure you have the GPU version of PyTorch installed.

TROUBLESHOOTING:

Q15: The system says no documents found. Why?
A: Check that PDF files are in the correct folder and the path is correct.
Ensure PDFs are not corrupted or password-protected.

Q16: I'm getting memory errors. What should I do?
A: Reduce the number of documents or use a smaller AI model. Close other
applications to free up RAM. Consider processing documents in batches.

Q17: Answers seem irrelevant. How to fix?
A: Rebuild the index with different chunk sizes. Try rephrasing your question.
Check that the relevant information exists in your documents.

ADVANCED QUESTIONS:

Q18: Can I customize the AI model?
A: Yes! Edit the model name in rag_agent.py. You can use any Hugging Face
text2text-generation model compatible with the transformers library.

Q19: How do I backup my index?
A: Simply copy the entire persist-dir folder (default: chroma_db). Restore
by copying it back.

Q20: Can I integrate this into my application?
A: Yes! The code is modular. Import the functions you need and integrate
them into your Python application.

SUPPORT:

For additional help:
- Email: support@agenticrag.example.com
- Documentation: https://docs.agenticrag.example.com
- GitHub Issues: https://github.com/agenticrag/issues

Last Updated: January 4, 2024
Version: 1.0""")
    ]
    
    print(f"\nGenerating {len(samples)} sample PDF documents...\n")
    
    for filename, title, content in samples:
        try:
            create_pdf(filename, title, content, output_dir)
        except Exception as e:
            print(f"✗ Error creating {filename}: {e}")
    
    print(f"\n✓ Successfully created {len(samples)} sample PDFs in ./{output_dir}/")
    print("\nGenerated files:")
    for filename, _, _ in samples:
        print(f"  - {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample PDF documents for Agentic RAG Pipeline testing"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for generated PDFs (default: data)"
    )
    
    args = parser.parse_args()
    generate_sample_pdfs(args.output_dir)


if __name__ == "__main__":
    main()
