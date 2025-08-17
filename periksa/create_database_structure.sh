#!/bin/bash
# Script untuk membuat struktur direktori database

echo "🔧 Membuat struktur direktori untuk database..."

# Buat struktur direktori untuk dataset
echo "📁 Membuat direktori dataset..."
mkdir -p datasets/nan_raw_biner/{train,test,validation}/images
mkdir -p Sets

# Buat direktori hasil training (contoh untuk beberapa scenario)
echo "📁 Membuat direktori hasil training..."
mkdir -p ResultGanS_iam_OP_debug/{epoch0,epoch1,epoch2,epoch3,epoch4,epoch5}
mkdir -p ResultS_iam_OP_debug/{epoch0,epoch1,epoch2,epoch3,epoch4,epoch5}
mkdir -p ResultGanS_iam_OP_stable/{epoch0,epoch1,epoch2,epoch3,epoch4,epoch5}

# Tampilkan struktur yang dibuat
echo ""
echo "✅ Struktur direktori berhasil dibuat:"
echo "📦 datasets/"
echo "   └── nan_raw_biner/"
echo "       ├── train/images/"
echo "       ├── test/images/"
echo "       └── validation/images/"
echo ""
echo "📦 Sets/ (untuk file lists)"
echo ""
echo "📦 Result directories:"
echo "   ├── ResultGanS_iam_OP_debug/"
echo "   ├── ResultS_iam_OP_debug/"
echo "   └── ResultGanS_iam_OP_stable/"
echo ""
echo "💡 Langkah selanjutnya:"
echo "1. Copy gambar ground truth ke datasets/nan_raw_biner/[train|test|validation]/images/"
echo "2. Jalankan: poetry run python periksa/create_file_lists.py"
echo "3. Validasi: poetry run python periksa/validate_database.py"
echo "4. Mulai training: poetry run python jnm_GAN_AHTR.py --scenario S_iam_OP_debug"
