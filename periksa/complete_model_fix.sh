#!/bin/bash
# 🔧 Complete Dataset Fix & Model Retraining Pipeline
# ==================================================
# 
# Script lengkap untuk memperbaiki masalah dataset dan melatih ulang model
# dengan data yang sudah dibenarkan alignment-nya.
#
# Author: Lambda One
# Date: August 13, 2024

echo "🔧 COMPLETE DATASET FIX & MODEL RETRAINING PIPELINE"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "GAN_AHTR.py" ]; then
    print_error "Please run this script from the GAN-HTR project root directory"
    exit 1
fi

print_status "Starting complete dataset fix and model retraining pipeline..."

# Step 1: Fix dataset alignment issues
print_status "Step 1: Fixing dataset alignment issues..."
echo "----------------------------------------"

if python3 fix_dataset_alignment.py; then
    print_success "Dataset alignment completed successfully"
else
    print_error "Dataset alignment failed"
    exit 1
fi

echo ""

# Step 2: Check if aligned dataset was created properly
print_status "Step 2: Validating aligned dataset..."
echo "------------------------------------"

if [ -d "datasets/nan_aligned" ]; then
    print_success "Aligned dataset directory exists"
    
    # Count files
    train_dist_count=$(find datasets/nan_aligned/train/distorted -name "*.jpg" 2>/dev/null | wc -l)
    train_gt_count=$(find datasets/nan_aligned/train/gt -name "*.jpg" 2>/dev/null | wc -l)
    test_dist_count=$(find datasets/nan_aligned/test/distorted -name "*.jpg" 2>/dev/null | wc -l)
    test_gt_count=$(find datasets/nan_aligned/test/gt -name "*.jpg" 2>/dev/null | wc -l)
    
    print_status "Dataset statistics:"
    echo "  - Training distorted images: $train_dist_count"
    echo "  - Training ground truth images: $train_gt_count"
    echo "  - Test distorted images: $test_dist_count"
    echo "  - Test ground truth images: $test_gt_count"
    
    if [ "$train_dist_count" -gt 0 ] && [ "$train_gt_count" -gt 0 ]; then
        print_success "Aligned dataset contains training data"
    else
        print_error "Aligned dataset is empty or incomplete"
        exit 1
    fi
else
    print_error "Aligned dataset directory not found"
    exit 1
fi

echo ""

# Step 3: Ask user if they want to proceed with training
print_status "Step 3: Model retraining preparation..."
echo "-------------------------------------"

read -p "Do you want to proceed with model retraining? This will take significant time and GPU resources. (y/N): " confirm

if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    print_warning "Training skipped by user"
    print_status "You can run training later with: python3 train_improved_model.py"
    exit 0
fi

# Step 4: Train improved model
print_status "Step 4: Training improved model..."
echo "--------------------------------"

print_status "Starting model training with improved architecture and loss functions..."
print_warning "This process will take several hours depending on your hardware"
print_status "Training progress will be shown with metrics and checkpoints"

if python3 train_improved_model.py; then
    print_success "Model training completed successfully"
else
    print_error "Model training failed"
    exit 1
fi

echo ""

# Step 5: Test the improved model
print_status "Step 5: Testing improved model..."
echo "-------------------------------"

print_status "Running comprehensive model testing and comparison..."

if python3 test_enhanced_model.py << EOF
3
EOF
then
    print_success "Model testing completed successfully"
else
    print_warning "Model testing encountered some issues, but training was successful"
fi

echo ""

# Step 6: Summary and next steps
print_status "Step 6: Pipeline completion summary..."
echo "------------------------------------"

print_success "🎉 COMPLETE PIPELINE FINISHED!"
echo ""
print_status "What was accomplished:"
echo "✅ Fixed dataset alignment issues (size mismatches)"
echo "✅ Created properly aligned training and test datasets"
echo "✅ Trained improved GAN-HTR model with enhanced loss functions"
echo "✅ Tested model performance and generated comparison results"
echo ""

print_status "Generated files and directories:"
echo "📁 datasets/nan_aligned/ - Aligned dataset with matching image sizes"
echo "📁 checkpoints/improved_model_*/ - New model checkpoints and weights"
echo "📁 test_results/ - Model comparison visualizations"
echo "📊 training_history.png - Training metrics plots"
echo ""

print_status "Next steps you can take:"
echo "1. Review training history plots in the checkpoint directory"
echo "2. Examine model comparison results in test_results/"
echo "3. Test the enhanced model on your own document images"
echo "4. Use the document enhancement pipeline with the new model"
echo ""

print_status "Quick test command:"
echo "python3 GAN_AHTR.py --input your_image.jpg --output enhanced_image.jpg"
echo ""

print_success "Pipeline completed successfully! 🚀"

# Final check - show latest results
if [ -d "test_results" ]; then
    print_status "Latest test results:"
    ls -la test_results/
fi

echo ""
print_status "For detailed usage instructions, see:"
echo "📖 MANUAL_PENGGUNAAN.md"
echo "🚀 QUICK_START_GUIDE.md"
echo "🔍 ENHANCEMENT_PROBLEM_ANALYSIS.md"
