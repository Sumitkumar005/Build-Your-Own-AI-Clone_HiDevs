#!/usr/bin/env python3
"""
Deployment verification script for AI Clone Chatbot
Checks all components before deployment
"""

import os
import sys
from pathlib import Path

def check_deployment_readiness():
    """Check if the project is ready for deployment"""
    print("🚀 Checking AI Clone Chatbot Deployment Readiness...")
    
    checks = []
    
    # 1. Check required files
    required_files = [
        'app.py',
        'requirements.txt', 
        'README.md',
        '.gitignore',
        'Procfile',
        'runtime.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            checks.append(f"✅ {file}")
        else:
            checks.append(f"❌ {file} - MISSING")
    
    # 2. Check source code
    src_files = ['src/chatbot.py', 'src/evaluation.py', 'src/prompt.py']
    for file in src_files:
        if os.path.exists(file):
            checks.append(f"✅ {file}")
        else:
            checks.append(f"❌ {file} - MISSING")
    
    # 3. Check data directory
    if os.path.exists('data'):
        data_count = len(list(Path('data').rglob('*.*')))
        checks.append(f"✅ data/ directory with {data_count} files")
    else:
        checks.append("❌ data/ directory - MISSING")
    
    # 4. Check vector database
    if os.path.exists('chroma_db'):
        checks.append("✅ chroma_db/ vector database")
    else:
        checks.append("❌ chroma_db/ - Run: python initialize_kb.py")
    
    # 5. Check configuration
    if os.path.exists('.streamlit/config.toml'):
        checks.append("✅ Streamlit configuration")
    else:
        checks.append("❌ Streamlit config - MISSING")
    
    # Print results
    print("\n📋 Deployment Checklist:")
    for check in checks:
        print(f"  {check}")
    
    # Summary
    passed = len([c for c in checks if c.startswith('✅')])
    total = len(checks)
    
    print(f"\n📊 Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 READY FOR DEPLOYMENT!")
        print("\n🚀 Next Steps:")
        print("1. git add .")
        print("2. git commit -m 'Complete AI Clone Chatbot'")
        print("3. git push origin main")
        print("4. Deploy on Streamlit Cloud")
        return True
    else:
        print("⚠️ Fix missing components before deployment")
        return False

if __name__ == "__main__":
    check_deployment_readiness()