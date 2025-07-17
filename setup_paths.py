#!/usr/bin/env python3
"""
Setup script to configure directory paths for AutoEncoderScreen application
"""

import os
import re

def update_directory_path():
    """
    Interactive setup to update directory paths in AutoEncoderScreen.py
    """
    print("🔧 AutoEncoder Screen Setup")
    print("=" * 40)
    
    # Get current directory
    current_dir = os.getcwd()
    stock_data_dir = os.path.join(current_dir, 'stock_data', 'Data')
    
    print(f"📁 Current directory: {current_dir}")
    print(f"🔍 Looking for stock data in: {stock_data_dir}")
    
    # Check if stock_data directory exists
    if os.path.exists(stock_data_dir):
        print("✅ Found stock_data/Data directory!")
        use_default = input(f"Use this directory? (y/n): ").lower().strip()
        if use_default == 'y':
            new_directory = stock_data_dir + '/'
        else:
            new_directory = get_custom_directory()
    else:
        print("❌ stock_data/Data directory not found.")
        new_directory = get_custom_directory()
    
    # Update the AutoEncoderScreen.py file
    update_file_paths(new_directory)
    
    print(f"✅ Updated directory path to: {new_directory}")
    print("\n🚀 You can now run: python AutoEncoderScreen.py")

def get_custom_directory():
    """Get custom directory path from user"""
    while True:
        custom_path = input("📂 Enter the full path to your stock data directory: ").strip()
        
        if os.path.exists(custom_path):
            if not custom_path.endswith('/'):
                custom_path += '/'
            return custom_path
        else:
            print(f"❌ Directory '{custom_path}' does not exist. Please try again.")

def update_file_paths(new_directory):
    """Update directory paths in AutoEncoderScreen.py"""
    
    # Files to update
    files_to_update = ['AutoEncoderScreen.py']
    
    for filename in files_to_update:
        if not os.path.exists(filename):
            print(f"⚠️  Warning: {filename} not found")
            continue
            
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
        
        # Update the DIRECTORY path
        # Look for pattern: DIRECTORY = '/path/to/directory/'
        pattern = r"DIRECTORY = ['\"][^'\"]*['\"]"
        replacement = f"DIRECTORY = '{new_directory}'"
        
        updated_content = re.sub(pattern, replacement, content)
        
        # Write back to file
        with open(filename, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {filename}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'statsmodels', 
        'tensorflow', 'plotly', 'dash', 'dash_bootstrap_components',
        'tqdm', 'hdbscan'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("OR")
        print("pip install -r requirements.txt")
    else:
        print("\n✅ All required packages are installed!")

def check_encoder_file():
    """Check if encoder.pkl exists"""
    print("\n🔍 Checking for encoder file...")
    
    if os.path.exists('encoder.pkl'):
        print("✅ encoder.pkl found!")
        return True
    else:
        print("❌ encoder.pkl not found!")
        print("📋 To create encoder.pkl:")
        print("   1. Run: python AutoEncoderPairScreenerV2.py")
        print("   2. Wait for training to complete")
        print("   3. The encoder.pkl file will be created automatically")
        return False

def main():
    """Main setup function"""
    print("🚀 AUTOENCODER SCREEN SETUP")
    print("=" * 50)
    
    # Update directory paths
    update_directory_path()
    
    # Check dependencies
    check_dependencies()
    
    # Check encoder file
    encoder_exists = check_encoder_file()
    
    print("\n" + "=" * 50)
    print("🎯 SETUP SUMMARY")
    print("=" * 50)
    
    if encoder_exists:
        print("✅ Ready to run AutoEncoderScreen!")
        print("🚀 Execute: python AutoEncoderScreen.py")
    else:
        print("⚠️  Setup incomplete - missing encoder.pkl")
        print("📋 Next steps:")
        print("   1. Run training script first")
        print("   2. Then run AutoEncoderScreen.py")
    
    print("\n📖 For detailed help, see README_AutoEncoderScreen.md")

if __name__ == "__main__":
    main() 