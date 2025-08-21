#!/usr/bin/env python3
"""
Hyperparameter Optimization Strategy for GAN-HTR
Panduan lengkap untuk menemukan hyperparameter optimal
"""

import os
import sys
import argparse

# Add parent directory untuk imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from periksa.wandb_integration import WANDBHyperparameterSweep


class HyperparameterOptimizer:
    """
    Class untuk manage hyperparameter optimization strategy
    """
    
    def __init__(self, project_base_name="gan-htr-optimization"):
        self.project_base = project_base_name
        self.optimization_phases = []
    
    def phase_1_coarse_search(self):
        """
        Phase 1: Coarse hyperparameter search
        Explore wide range of parameters untuk identify promising regions
        """
        
        print("🔍 PHASE 1: Coarse Hyperparameter Search")
        print("="*60)
        
        sweep_config = {
            'name': 'gan-htr-coarse-search',
            'method': 'bayes',  # Bayesian optimization
            'metric': {
                'name': 'val/g_loss',
                'goal': 'minimize'
            },
            'program': 'sweep_train.py',
            'parameters': {
                # Wide learning rate range
                'learning-rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 1e-2  # Extended range
                },
                
                # Multiple batch sizes
                'batch-size': {
                    'values': [1, 2, 4, 8]  # Extended range
                },
                
                # Critical loss weights - wide exploration
                'adv-weight': {
                    'distribution': 'uniform',
                    'min': 0.01,
                    'max': 5.0  # Very wide range
                },
                
                'content-weight': {
                    'distribution': 'uniform', 
                    'min': 0.1,
                    'max': 10.0  # Very wide range
                },
                
                'recognition-weight': {
                    'distribution': 'uniform',
                    'min': 0.01,
                    'max': 3.0  # Wide range
                },
                
                # Training stability
                'patience': {
                    'values': [5, 10, 15, 20, 30]
                },
                
                'epochs': {
                    'value': 10  # Short epochs untuk quick exploration
                },
                
                'scenario': {
                    'value': 'S_coarse_search'
                }
            },
            
            # Aggressive early termination
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 2,
                'eta': 3  # More aggressive
            }
        }
        
        return sweep_config
    
    def phase_2_fine_tuning(self, best_params_phase1):
        """
        Phase 2: Fine-tuning around best parameters dari Phase 1
        """
        
        print("🎯 PHASE 2: Fine-tuning Optimization")
        print("="*60)
        
        # Narrow ranges around best parameters
        lr_center = best_params_phase1.get('learning_rate', 1e-5)
        adv_center = best_params_phase1.get('adv_weight', 1.0)
        content_center = best_params_phase1.get('content_weight', 1.5)
        recognition_center = best_params_phase1.get('recognition_weight', 0.5)
        
        sweep_config = {
            'name': 'gan-htr-fine-tuning',
            'method': 'bayes',
            'metric': {
                'name': 'val/g_loss',
                'goal': 'minimize'
            },
            'program': 'sweep_train.py',
            'parameters': {
                # Narrow learning rate range around best
                'learning-rate': {
                    'distribution': 'log_normal',
                    'mu': lr_center,
                    'sigma': 0.3  # ±30% variation
                },
                
                # Best batch size dari phase 1 + neighbors
                'batch-size': {
                    'values': [1, 2, 4]  # Focus on stable sizes
                },
                
                # Fine-tune loss weights
                'adv-weight': {
                    'distribution': 'normal',
                    'mu': adv_center,
                    'sigma': adv_center * 0.2  # ±20%
                },
                
                'content-weight': {
                    'distribution': 'normal',
                    'mu': content_center,
                    'sigma': content_center * 0.2  # ±20%
                },
                
                'recognition-weight': {
                    'distribution': 'normal',
                    'mu': recognition_center,
                    'sigma': recognition_center * 0.15  # ±15%
                },
                
                'patience': {
                    'values': [10, 15, 20]  # Stable values
                },
                
                'epochs': {
                    'value': 20  # Longer untuk better evaluation
                },
                
                'scenario': {
                    'value': 'S_fine_tuning'
                }
            },
            
            # Less aggressive early termination
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 5,
                'eta': 2
            }
        }
        
        return sweep_config
    
    def phase_3_production_validation(self, best_params_phase2):
        """
        Phase 3: Production validation dengan best parameters
        """
        
        print("🏭 PHASE 3: Production Validation")
        print("="*60)
        
        sweep_config = {
            'name': 'gan-htr-production-validation',
            'method': 'grid',  # Deterministic validation
            'metric': {
                'name': 'val/g_loss',
                'goal': 'minimize'
            },
            'program': 'periksa/sweep_production.py',  # Use production script
            'parameters': {
                # Fixed best parameters
                'learning-rate': {
                    'value': best_params_phase2['learning_rate']
                },
                'batch-size': {
                    'value': best_params_phase2['batch_size']
                },
                'adv-weight': {
                    'value': best_params_phase2['adv_weight']
                },
                'content-weight': {
                    'value': best_params_phase2['content_weight']
                },
                'recognition-weight': {
                    'value': best_params_phase2['recognition_weight']
                },
                'patience': {
                    'value': best_params_phase2['patience']
                },
                
                # Production settings
                'epochs': {
                    'value': 100  # Full training
                },
                'scenario': {
                    'value': 'S_production_validation'
                }
            }
        }
        
        return sweep_config
    
    def run_optimization_strategy(self, phases=['coarse', 'fine', 'production']):
        """
        Run complete optimization strategy
        """
        
        print("🚀 STARTING COMPLETE HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        
        results = {}
        
        if 'coarse' in phases:
            print("\n" + "="*50)
            print("🔍 PHASE 1: COARSE SEARCH")
            print("="*50)
            
            # Create and run coarse search
            coarse_config = self.phase_1_coarse_search()
            
            print("📋 Coarse Search Configuration:")
            print(f"   • Learning Rate: 1e-6 to 1e-2 (log-uniform)")
            print(f"   • Batch Size: [1, 2, 4, 8]")
            print(f"   • ADV Weight: 0.01 to 5.0")
            print(f"   • Content Weight: 0.1 to 10.0")
            print(f"   • Recognition Weight: 0.01 to 3.0")
            print(f"   • Epochs: 10 (quick exploration)")
            print(f"   • Runs: 30-50 recommended")
            
            print("\n💡 To start Phase 1:")
            print(f"   poetry run python periksa/start_sweep.py --project '{self.project_base}-phase1' --count 30")
            
        if 'fine' in phases:
            print("\n" + "="*50)
            print("🎯 PHASE 2: FINE TUNING")
            print("="*50)
            
            print("📋 Fine Tuning Strategy:")
            print("   • Analyze Phase 1 results first")
            print("   • Narrow parameter ranges around best performers")
            print("   • Use longer epochs (20) untuk better evaluation")
            print("   • Focus on parameter interactions")
            
            print("\n💡 Steps for Phase 2:")
            print("   1. Analyze Phase 1 results di WandB")
            print("   2. Extract best parameter ranges")
            print("   3. Update fine-tuning config")
            print("   4. Run: poetry run python periksa/start_sweep.py --project '{}-phase2' --count 20")
            
        if 'production' in phases:
            print("\n" + "="*50)
            print("🏭 PHASE 3: PRODUCTION VALIDATION")
            print("="*50)
            
            print("📋 Production Validation:")
            print("   • Use best parameters dari Phase 2")
            print("   • Run full training (100+ epochs)")
            print("   • Multiple seeds untuk statistical significance")
            print("   • Final model selection")
            
            print("\n💡 Steps for Phase 3:")
            print("   1. Select best hyperparameters dari Phase 2")
            print("   2. Update production config")
            print("   3. Run: poetry run python periksa/start_sweep.py --project '{}-phase3' --count 5")


def analyze_sweep_results(project_name):
    """
    Function untuk analyze sweep results dan extract best parameters
    """
    
    print(f"📊 ANALYZING SWEEP RESULTS: {project_name}")
    print("="*60)
    
    print("💡 Manual Analysis Steps:")
    print("1. Visit WandB dashboard:")
    print(f"   https://wandb.ai/your-username/{project_name}")
    
    print("\n2. Sort runs by 'val/g_loss' (ascending)")
    
    print("\n3. Look at top 5-10 performers:")
    print("   • Check parameter combinations")
    print("   • Look for patterns/trends")
    print("   • Note any outliers")
    
    print("\n4. Analyze parameter correlations:")
    print("   • Use WandB parallel coordinates plot")
    print("   • Check parameter importance plot")
    print("   • Look at scatter plots")
    
    print("\n5. Extract best parameter ranges:")
    print("   • Learning rate: [min_best, max_best]")
    print("   • Loss weights: optimal ratios")
    print("   • Batch size: most stable size")
    
    print("\n6. Check for overfitting:")
    print("   • Compare train vs val metrics")
    print("   • Look at learning curves")
    print("   • Validate on different scenarios")


def main():
    """Main optimization workflow"""
    
    parser = argparse.ArgumentParser(description='GAN-HTR Hyperparameter Optimization Strategy')
    parser.add_argument('--phase', choices=['coarse', 'fine', 'production', 'all'], 
                       default='all', help='Optimization phase to run')
    parser.add_argument('--project-base', type=str, default='gan-htr-optimization',
                       help='Base project name for optimization')
    parser.add_argument('--analyze', type=str, help='Analyze results dari project name')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_sweep_results(args.analyze)
        return
    
    optimizer = HyperparameterOptimizer(args.project_base)
    
    if args.phase == 'all':
        phases = ['coarse', 'fine', 'production']
    else:
        phases = [args.phase]
    
    optimizer.run_optimization_strategy(phases)
    
    print("\n" + "="*70)
    print("🎯 OPTIMIZATION STRATEGY SUMMARY")
    print("="*70)
    
    print("\n📈 Expected Timeline:")
    print("   Phase 1 (Coarse): 2-4 hours (30-50 runs)")
    print("   Phase 2 (Fine): 1-2 hours (15-20 runs)")
    print("   Phase 3 (Production): 8-12 hours (3-5 full runs)")
    
    print("\n🎯 Success Metrics:")
    print("   • Reduced validation loss")
    print("   • Stable training (no divergence)")
    print("   • Consistent performance across seeds")
    print("   • Good train/val balance (no overfitting)")
    
    print("\n📊 Analysis Tools:")
    print("   • WandB parallel coordinates")
    print("   • Parameter importance plots")
    print("   • Learning curve comparisons")
    print("   • Loss correlation analysis")


if __name__ == "__main__":
    main()
