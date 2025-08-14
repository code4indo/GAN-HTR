"""
Real-time Performance Monitor untuk Full Optimization GAN-HTR
============================================================

Script untuk monitoring performa real-time selama training
"""

import psutil
import GPUtil
import threading
import time
import os
from datetime import datetime

class FullOptimizationMonitor:
    """Monitor performa real-time untuk training yang dioptimasi"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = []
        self.start_time = None
        self.log_file = "performance_log.txt"
        
    def start_monitoring(self):
        """Mulai monitoring performa"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("🔍 Performance monitoring started...")
        print(f"📊 Logs will be saved to: {self.log_file}")
        
    def stop_monitoring(self):
        """Stop monitoring dan tampilkan summary"""
        if self.monitoring:
            self.monitoring = False
            self.monitor_thread.join(timeout=5)
            self._save_and_show_summary()
    
    def _monitor_loop(self):
        """Loop monitoring utama"""
        with open(self.log_file, 'w') as f:
            f.write("Timestamp,CPU%,Memory%,GPU0_Util%,GPU0_Mem_MB,GPU1_Util%,GPU1_Mem_MB,Temp_C\\n")
        
        while self.monitoring:
            try:
                # CPU dan Memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU stats
                gpus = GPUtil.getGPUs()
                gpu_stats = []
                
                for gpu in gpus:
                    gpu_stats.extend([
                        gpu.load * 100,  # GPU utilization %
                        gpu.memoryUsed,  # Memory used in MB
                    ])
                
                # Temperature (jika available)
                try:
                    temp = psutil.sensors_temperatures()
                    cpu_temp = temp.get('coretemp', [{'current': 0}])[0]['current']
                except:
                    cpu_temp = 0
                
                # Simpan stats
                timestamp = time.time() - self.start_time
                stats_entry = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'gpu_stats': gpu_stats,
                    'temperature': cpu_temp
                }
                
                self.stats.append(stats_entry)
                
                # Log ke file
                with open(self.log_file, 'a') as f:
                    gpu_str = ','.join(map(str, gpu_stats)) if gpu_stats else "0,0,0,0"
                    f.write(f"{timestamp:.1f},{cpu_percent:.1f},{memory.percent:.1f},{gpu_str},{cpu_temp:.1f}\\n")
                
                # Print real-time stats
                if len(self.stats) % 10 == 0:  # Every 10 seconds
                    self._print_realtime_stats(stats_entry)
                
                time.sleep(1)  # Monitor setiap detik
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def _print_realtime_stats(self, stats):
        """Print statistik real-time"""
        uptime = stats['timestamp']
        
        print(f"\\n📊 Performance Stats (Uptime: {uptime//60:.0f}m {uptime%60:.0f}s)")
        print(f"🖥️  CPU: {stats['cpu_percent']:.1f}% | Memory: {stats['memory_percent']:.1f}%")
        
        gpu_stats = stats['gpu_stats']
        if len(gpu_stats) >= 4:
            print(f"🎮 GPU0: {gpu_stats[0]:.1f}% util, {gpu_stats[1]:.0f}MB mem")
            print(f"🎮 GPU1: {gpu_stats[2]:.1f}% util, {gpu_stats[3]:.0f}MB mem")
        
        if stats['temperature'] > 0:
            print(f"🌡️  CPU Temp: {stats['temperature']:.1f}°C")
    
    def _save_and_show_summary(self):
        """Simpan dan tampilkan summary performa"""
        if not self.stats:
            return
        
        total_time = time.time() - self.start_time
        
        # Kalkulasi rata-rata
        avg_cpu = sum(s['cpu_percent'] for s in self.stats) / len(self.stats)
        avg_memory = sum(s['memory_percent'] for s in self.stats) / len(self.stats)
        
        # GPU averages
        gpu_data = [s['gpu_stats'] for s in self.stats if s['gpu_stats']]
        if gpu_data:
            gpu_avg = [sum(col) / len(gpu_data) for col in zip(*gpu_data)]
        else:
            gpu_avg = [0, 0, 0, 0]
        
        # Generate summary report
        summary = f"""
🎯 PERFORMANCE SUMMARY - Full Optimization Training
{'='*60}
⏱️  Total Runtime: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s
🖥️  Average CPU Usage: {avg_cpu:.1f}%
💾 Average Memory Usage: {avg_memory:.1f}%
🎮 GPU 0 - Avg Utilization: {gpu_avg[0]:.1f}%, Avg Memory: {gpu_avg[1]:.0f}MB
🎮 GPU 1 - Avg Utilization: {gpu_avg[2]:.1f}%, Avg Memory: {gpu_avg[3]:.0f}MB

📈 OPTIMIZATION ANALYSIS:
{'='*30}
"""
        
        # Analysis
        if avg_cpu > 80:
            summary += "✅ CPU: Excellent utilization (>80%)\\n"
        elif avg_cpu > 60:
            summary += "🟡 CPU: Good utilization (60-80%)\\n" 
        else:
            summary += "🔴 CPU: Low utilization (<60%) - Room for improvement\\n"
            
        if gpu_avg[0] > 80 and gpu_avg[2] > 80:
            summary += "✅ GPU: Excellent dual-GPU utilization (>80% both)\\n"
        elif gpu_avg[0] > 60 or gpu_avg[2] > 60:
            summary += "🟡 GPU: Good utilization (>60% at least one GPU)\\n"
        else:
            summary += "🔴 GPU: Low utilization - Check batch size or model complexity\\n"
            
        if avg_memory > 90:
            summary += "⚠️  Memory: Very high usage (>90%) - Monitor for OOM\\n"
        elif avg_memory > 70:
            summary += "✅ Memory: Good utilization (70-90%)\\n"
        else:
            summary += "💡 Memory: Low usage (<70%) - Can increase batch size\\n"
        
        # Performance recommendations
        summary += f"""
💡 RECOMMENDATIONS:
{'='*20}
"""
        
        if gpu_avg[0] < 70 or gpu_avg[2] < 70:
            summary += "• Consider increasing batch size for better GPU utilization\\n"
        
        if avg_cpu < 70:
            summary += "• Increase data loading parallelism (num_parallel_calls)\\n"
            
        if avg_memory < 60:
            summary += "• Can safely increase batch size or enable more aggressive caching\\n"
            
        summary += f"""
📁 Detailed logs saved to: {self.log_file}
📊 Use this data to further optimize your training setup!
"""
        
        print(summary)
        
        # Save summary to file
        with open("performance_summary.txt", "w") as f:
            f.write(summary)
        
        print("💾 Summary saved to: performance_summary.txt")

def start_monitoring_in_background():
    """Helper function untuk memulai monitoring dari script utama"""
    monitor = FullOptimizationMonitor()
    monitor.start_monitoring()
    return monitor

if __name__ == "__main__":
    print("🚀 Full Optimization Performance Monitor")
    print("This script should be imported and used in your training script")
    print("\\nUsage:")
    print("from periksa.performance_monitor import start_monitoring_in_background")
    print("monitor = start_monitoring_in_background()")
    print("# ... your training code ...")
    print("monitor.stop_monitoring()")
    
    # Demo monitoring
    print("\\n🔍 Starting demo monitoring for 30 seconds...")
    monitor = FullOptimizationMonitor()
    monitor.start_monitoring()
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\\n⏹️  Monitoring stopped by user")
    
    monitor.stop_monitoring()
