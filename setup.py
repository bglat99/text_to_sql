#!/usr/bin/env python3
"""
Setup script for Text-to-SQL Fine-tuning Framework
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n" + "=" * 60)
    print("Installing Dependencies")
    print("=" * 60)
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def setup_database():
    """Setup the financial database"""
    print("\n" + "=" * 60)
    print("Setting up Database")
    print("=" * 60)
    
    try:
        from database_setup import create_database_with_real_data, verify_database
        create_database_with_real_data(use_real_data=True)
        verify_database()
        return True
    except Exception as e:
        print(f"✗ Database setup failed: {e}")
        return False

def generate_training_data():
    """Generate training data"""
    print("\n" + "=" * 60)
    print("Generating Training Data")
    print("=" * 60)
    
    try:
        from data_generator import main as generate_data
        generate_data()
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n" + "=" * 60)
    print("Creating Directories")
    print("=" * 60)
    
    directories = [
        "outputs",
        "models",
        "data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def check_system_requirements():
    """Check system requirements"""
    print("\n" + "=" * 60)
    print("System Requirements Check")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check for Apple Silicon (M1/M2/M3)
    import platform
    if platform.machine() == 'arm64':
        print("✓ Apple Silicon (M1/M2/M3) detected - optimal for this project")
    else:
        print("⚠ Non-Apple Silicon system detected - may have slower performance")
    
    # Check available memory
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"✓ Available memory: {memory_gb:.1f} GB")
    
    if memory_gb < 8:
        print("⚠ Warning: Less than 8GB RAM detected. Training may be slow or fail.")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    print(f"✓ Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 10:
        print("⚠ Warning: Less than 10GB free space detected. Consider freeing up space.")
    
    return True

def create_quick_start_script():
    """Create a quick start script"""
    script_content = '''#!/bin/bash
# Quick start script for Text-to-SQL Fine-tuning

echo "🚀 Starting Text-to-SQL Fine-tuning Framework"

# Check if database exists
if [ ! -f "financial_data.db" ]; then
    echo "📊 Setting up database..."
    python database_setup.py
fi

# Check if training data exists
if [ ! -f "train_data.json" ]; then
    echo "📝 Generating training data..."
    python data_generator.py
fi

# Start training
echo "🤖 Starting model training..."
python finetune_model.py

# Evaluate model
echo "📈 Evaluating model..."
python evaluate_model.py --comprehensive

echo "✅ Setup complete! You can now use the model interactively:"
echo "python evaluate_model.py --interactive"
'''
    
    with open("quick_start.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("quick_start.sh", 0o755)
    print("✓ Created quick_start.sh script")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Text-to-SQL Fine-tuning Framework Setup")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n✗ System requirements not met. Please check the requirements.")
        return False
    
    # Create directories
    if not create_directories():
        print("\n✗ Failed to create directories.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Failed to install dependencies.")
        return False
    
    # Setup database
    if not setup_database():
        print("\n✗ Failed to setup database.")
        return False
    
    # Generate training data
    if not generate_training_data():
        print("\n✗ Failed to generate training data.")
        return False
    
    # Create quick start script
    create_quick_start_script()
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run training: python finetune_model.py")
    print("2. Evaluate model: python evaluate_model.py --comprehensive")
    print("3. Interactive mode: python evaluate_model.py --interactive")
    print("4. Or use quick start: ./quick_start.sh")
    
    print("\nExpected timeline:")
    print("- Training: 3-4 hours on M3 MacBook Air")
    print("- Evaluation: 10-15 minutes")
    print("- Total setup time: ~8-12 hours for first working model")
    
    print("\nTarget performance:")
    print("- Syntax accuracy: >90%")
    print("- Execution rate: >85%")
    print("- Semantic accuracy: >80%")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 