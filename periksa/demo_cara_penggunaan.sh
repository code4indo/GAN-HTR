#!/bin/bash
# 🎯 DEMO SCRIPT: Memperbaiki Dokumen Rusak dengan GAN-HTR
# Jalankan: bash demo_cara_penggunaan.sh

echo "🎯 DEMO: Cara Memperbaiki Dokumen Rusak dengan GAN-HTR"
echo "=========================================================="
echo ""

echo "📋 File yang tersedia untuk testing:"
ls -la *.png *.jpg 2>/dev/null | head -5
echo ""

echo "🤖 Model yang tersedia:"
if [ -f "./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5" ]; then
    echo "✅ Model utama: ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
else
    echo "⚠️ Mencari model alternatif..."
    find . -name "generator.weights.h5" -type f | head -3
fi
echo ""

echo "🚀 PERINTAH UNTUK MEMPERBAIKI DOKUMEN:"
echo "======================================"
echo ""

echo "1️⃣ METODE SIMPLE (Recommended untuk pemula):"
echo "   python simple_enhancement_test.py"
echo ""

echo "2️⃣ METODE CLI (Manual control):"
echo "   python full_document_enhancement.py \\"
echo "     --input a.png \\"
echo "     --output a_diperbaiki.png \\"
echo "     --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
echo ""

echo "3️⃣ BATCH PROCESSING (Multiple files):"
echo "   for file in *.jpg; do"
echo "     python full_document_enhancement.py \\"
echo "       --input \"\$file\" \\"
echo "       --output \"enhanced_\$file\" \\"
echo "       --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
echo "   done"
echo ""

echo "📁 OUTPUT YANG DIHASILKAN:"
echo "========================="
echo "• enhanced_[nama_file] - Dokumen yang sudah diperbaiki"
echo "• comparison_[nama_file] - Perbandingan sebelum vs sesudah"
echo "• segments/ - File intermediate (jika --save-intermediates)"
echo ""

echo "⚡ QUICK TEST:"
echo "============="
echo "Jalankan perintah ini untuk test cepat:"
echo "python simple_enhancement_test.py"
echo ""

echo "📖 DOKUMENTASI LENGKAP:"
echo "======================="
echo "• Manual Penggunaan: MANUAL_PENGGUNAAN.md"
echo "• Quick Start Guide: QUICK_START_GUIDE.md" 
echo "• Table of Contents: tableofcontent.md"
echo "• Success Summary: DOCUMENT_ENHANCEMENT_SUCCESS_SUMMARY.md"
echo ""

echo "✅ STATUS: READY TO USE!"
echo "========================"
echo "Sistem GAN-HTR Document Enhancement siap digunakan untuk"
echo "memperbaiki dokumen rusak, scan berkualitas rendah, dan"
echo "handwritten text yang tidak jelas."
